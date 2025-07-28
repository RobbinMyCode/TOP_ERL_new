import copy
import torch
import mprl.util as util
from mprl.rl.agent import AbstractAgent
from mprl.rl.critic import TopErlCritic
from mprl.rl.policy import TopErlPolicy
from mprl.rl.replay_buffer import TopErlReplayBuffer
from mprl.rl.sampler import TopErlSampler
from mprl.util import autocast_if

from torch.cuda.amp import GradScaler
import numpy as np

class TopErlAgent(AbstractAgent):
    def __init__(self,
                 policy: TopErlPolicy,
                 critic: TopErlCritic,
                 sampler: TopErlSampler,
                 replay_buffer: TopErlReplayBuffer,
                 projection=None,
                 dtype=torch.float32,
                 device=torch.device("cpu"),
                 **kwargs):
        self.load_pretrained_agent = False
        self.log_now = False
        self.betas = kwargs.get("betas", (0.9, 0.999))

        super().__init__(policy, critic, sampler, projection,
                         dtype=dtype, device=device, **kwargs)

        self.traj_length = kwargs.get("traj_length")

        self.evaluation_interval = kwargs.get("evaluation_interval", 1)

        #For reference splitting
        self.reference_split_args = kwargs.get("reference_split",
                                               {'split_strategy': 'n_equal_splits', 'n_splits': 1})


        # For off-policy learning
        self.replay_buffer = replay_buffer
        self.batch_size = kwargs.get("batch_size")
        self.critic_update_from = kwargs.get("critic_update_from", 0)
        self.policy_update_from = kwargs.get("policy_update_from", 0)
        assert self.critic_update_from <= self.policy_update_from, \
            "Critic update should be earlier than policy update"
        self.use_old_policy = kwargs.get("use_old_policy", False)
        if self.use_old_policy:
            self.old_policy_update_rate = kwargs.get("old_policy_update_rate", 1)
            self.policy_old = copy.deepcopy(self.policy)
        self.traj_has_downsample = (
                self.sampler.traj_downsample_factor is not None)

        # Float 16 for faster computation
        self.use_mix_precision = kwargs.get("use_mix_precision", False)
        self.critic_grad_scaler = [GradScaler(), GradScaler()]

    def get_optimizer(self, policy, critic):
        """
        Get the policy and critic network optimizers

        Args:
            policy: policy network
            critic: critic network

        Returns:
            two optimizers
        """
        self.policy_net_params = policy.parameters
        policy_optimizer = torch.optim.Adam(params=self.policy_net_params,
                                            lr=self.lr_policy,
                                            weight_decay=self.wd_policy)
        critic_opt1, critic_opt2 = self.critic.configure_optimizer(
            weight_decay=self.wd_critic, learning_rate=self.lr_critic,
            betas=self.betas)

        return policy_optimizer, (critic_opt1, critic_opt2)

    def step(self):
        # Update total step count
        self.num_iterations += 1

        # If logging data in the current step
        self.log_now = self.evaluation_interval == 1 or \
                       self.num_iterations % self.evaluation_interval == 1
        update_critic_now = self.num_iterations >= self.critic_update_from
        update_policy_now = self.num_iterations >= self.policy_update_from

        if self.load_pretrained_agent:
            buffer_is_ready = self.replay_buffer.is_full()
            self.num_iterations -= 1  # iteration only for collecting data
        else:
            buffer_is_ready = True
        # Collect dataset
        dataset, num_env_interation = \
            self.sampler.run(training=True, policy=self.policy, critic=None)
        self.num_global_steps += num_env_interation

        # Process dataset and save to RB
        self.replay_buffer.add(dataset)

        # Update agent, if continue training, collect dataset first
        if update_critic_now and buffer_is_ready:
            critic_loss_dict = self.update_critic()

            if self.schedule_lr_critic:
                lr_schedulers = util.make_iterable(self.critic_lr_scheduler)
                for scheduler in lr_schedulers:  # one or two critic nets
                    scheduler.step()
        else:
            critic_loss_dict = {}

        if update_policy_now and buffer_is_ready:
            policy_loss_dict = self.update_policy()

            if self.schedule_lr_policy:
                self.policy_lr_scheduler.step()
        else:
            policy_loss_dict = {}

        # Log data
        if self.log_now and buffer_is_ready:
            # Generate statistics for environment rollouts
            dataset_stats = \
                util.generate_many_stats(dataset, "exploration", to_np=True,
                                         exception_keys=["episode_init_idx"])

            # Prepare result metrics
            result_metrics = {
                **dataset_stats,
                "num_global_steps": self.num_global_steps,
                **critic_loss_dict, **policy_loss_dict,
                "lr_policy": self.policy_lr_scheduler.get_last_lr()[0]
                if self.schedule_lr_policy else self.lr_policy,
                "lr_critic1": self.critic_lr_scheduler[0].get_last_lr()[0]
                if self.schedule_lr_critic else self.lr_critic,
                "lr_critic2": self.critic_lr_scheduler[1].get_last_lr()[0]
                if self.schedule_lr_critic else self.lr_critic
            }

            # Evaluate agent
            evaluate_metrics = util.generate_many_stats(
                self.evaluate()[0], "evaluation", to_np=True,
                exception_keys=["episode_init_idx"])
            result_metrics.update(evaluate_metrics),
        else:
            result_metrics = {}

        return result_metrics

    def get_random_segments(self, pad_additional=False):
        """
        Get random segments, the number is between 1 and 25
        Args:
            pad_additional: if True, pad an additional segment that beyond the
            traj length. This helps to learn the value of last several actions
            of the trajectory

        Returns:
            idx_in_segments: [num_segments, seg_length + 1], where the last idx
            is the beginning of the next segment
        """
        possible_num_segments = torch.arange(1, 26, device=self.device)
        segment_lengths = self.traj_length // possible_num_segments
        segment_lengths_unique = segment_lengths.unique()
        possible_num_segments_after_unique\
            = self.traj_length // segment_lengths_unique
        # random choose the number of segments
        num_seg = possible_num_segments_after_unique[torch.randint(
            0, len(possible_num_segments_after_unique), [])]

        seg_length = self.traj_length // num_seg
        if num_seg == 1:
            start_idx = 0
        else:
            start_idx = torch.randint(0, seg_length, [],
                                      dtype=torch.long, device=self.device)
        num_seg_actual = (self.traj_length - start_idx) // seg_length
        if pad_additional:
            num_seg_actual += 1
        idx_in_segments = torch.arange(0, num_seg_actual * seg_length,
                                       device=self.device) + start_idx
        idx_in_segments = idx_in_segments.view(-1, seg_length)
        idx_in_segments = torch.cat([idx_in_segments,
                                     idx_in_segments[:, -1:] + 1], dim=-1)
        if pad_additional and idx_in_segments[-1][0] == self.traj_length:
            return idx_in_segments[:-1]
        else:
            return idx_in_segments

    def make_new_pred(self, dataset, **kwargs):
        compute_trust_region_loss = (
            kwargs.get("compute_trust_region_loss", True))
        states = dataset["step_states"]
        episode_init_idx = dataset["episode_init_idx"]
        num_traj = states.shape[0]

        # Decision state
        # [num_traj, n_splits, dim_state]
        d_state = states[torch.arange(num_traj, device=self.device).unsqueeze(-1), dataset["split_start_indexes"]]

        if self.use_old_policy:
            with torch.no_grad():
                params_mean_old, params_L_old = self.policy_old.policy(d_state)
        else:
            params_mean_old = dataset["episode_params_mean"]
            params_L_old = dataset["episode_params_L"]

        # entropy decay
        if self.projection.initial_entropy is None:
            self.projection.initial_entropy = \
                self.policy.entropy([params_mean_old, params_L_old]).mean()

        # Make prediction
        params_mean_new, params_L_new = self.policy.policy(d_state)

        # Projection
        flatter_mean_new = torch.reshape(params_mean_new, (np.prod(params_mean_new.size()[:-1]), params_mean_new.size()[-1]))
        flatter_mean_old = torch.reshape(params_mean_old,
                                         (np.prod(params_mean_old.size()[:-1]), params_mean_old.size()[-1]))
        flatter_L_new = torch.reshape(params_L_new,
                                         (np.prod(params_L_new.size()[:-2]), *params_L_new.size()[-2:]))
        flatter_L_old = torch.reshape(params_L_old,
                                      (np.prod(params_L_old.size()[:-2]), *params_L_old.size()[-2:]))
        proj_mean, proj_L = self.projection(self.policy,
                                            (flatter_mean_new, flatter_L_new),
                                            (flatter_mean_old, flatter_L_old),
                                            self.num_iterations)

        proj_mean = torch.reshape(proj_mean, params_mean_new.size())
        proj_L = torch.reshape(proj_L, params_L_new.size())

        # Trust Region loss
        if compute_trust_region_loss:
            trust_region_loss = \
                self.projection.get_trust_region_loss(self.policy,
                                                      (params_mean_new,
                                                       params_L_new),
                                                      (proj_mean, proj_L))
        else:
            trust_region_loss = None


        info = {"trust_region_loss": trust_region_loss}

        return proj_mean, proj_L, info

    def segments_n_step_return_vf(self, dataset, idx_in_segments):
        """
        Segment-wise n-step return using Value function

        Use Q-func as the target of the V-func prediction
        Use N-step return + V-func as the target of the Q-func prediction

        We compute everything in a batch manner.

        Args:
            dataset:
            idx_in_segments:

        Returns:
            n_step_returns: [num_traj, num_segments, 1 + num_seg_actions]

        """
        states = dataset["step_states"]  # [num_traj, traj_length, dim_state]
        rewards = dataset["step_rewards"]

        # [num_traj, traj_length, dim_action]
        traj_init_pos = dataset["step_desired_pos"] #desired_pos = end_pos of old trajectory at index
        traj_init_vel = dataset["step_desired_vel"]

        #TODONE: truncation not optimal, but cant just select incorrect start point for param 2
        #TODONE 0) check truncation in V is the problem (check everything else)
        #TODONE: 1) i) get parameters of policy 2 given the correct initial conditions from traj_init at its corr timestep    ii) use those params to continue from the last pos
        #TODONE: [[ 2) use policy outside its range of knowledge ]] (perhaps)
        # -> truncation seems the best

        num_segments = idx_in_segments.shape[0]
        num_seg_actions = idx_in_segments.shape[-1] - 1
        seg_start_idx = idx_in_segments[..., 0]

        num_traj, traj_length = states.shape[0], states.shape[1]

        with torch.no_grad():
            # params: [num_traj, num_weights]
            params_mean_new, params_L_new, _ \
                = self.make_new_pred(dataset, compute_trust_region_loss=False)

        # [num_traj, num_weights] -> [num_traj, num_segments, num_weights]
        params_mean_new = util.add_expand_dim(params_mean_new, [1],
                                              [num_segments])
        params_L_new = util.add_expand_dim(params_L_new, [1],
                                           [num_segments])

        # [num_traj, traj_length]
        times = self.sampler.get_times(dataset["episode_init_time"],
                                       self.sampler.num_times,
                                       self.traj_has_downsample)

        ref_time = times[0]
        # [num_traj, traj_length] -> [num_traj, num_segments]
        times = times[:, seg_start_idx]

        # [num_traj, num_segments] -> [num_traj, num_segments, num_seg_actions]
        action_times = util.add_expand_dim(times, [-1], [num_seg_actions])
        time_evo = torch.linspace(0, self.sampler.dt * (num_seg_actions-1),
                                  num_seg_actions, device=self.device).float()
        action_times = action_times + time_evo

        # Init time, shape: [num_traj, num_segments]
        init_time = times - self.sampler.dt

        # [num_traj, num_segments, dim_action]
        init_pos = traj_init_pos[:, seg_start_idx]
        init_vel = traj_init_vel[:, seg_start_idx]

        # Get the new MP trajectory using the current policy and condition on
        # the buffered desired pos and vel as initial conditions
        # util.set_global_random_seed(0)

        params_mean_new = self.policy.fix_relative_goal_for_segments(
            params=params_mean_new, traj_init_pos=traj_init_pos[:, 0],
            segments_init_pos=init_pos)

        # [num_traj, num_segments, num_seg_actions, dim_action] or
        # [num_traj, num_segments, num_smps, num_seg_actions, dim_action]

        sampling_args_value_func = self.reference_split_args.copy()
        sampling_args_value_func["q_loss_strategy"] = sampling_args_value_func["v_func_estimation"]
        actions = self.policy.sample(require_grad=False,
                                     params_mean=params_mean_new,
                                     params_L=params_L_new, times=action_times,
                                     init_time=init_time,
                                     init_pos=init_pos, init_vel=init_vel,
                                     use_mean=False,
                                     split_args=sampling_args_value_func,
                                     use_case = "agent",
                                     ref_time_list = self.sampler.get_times(dataset["episode_init_time"],
                                       self.sampler.num_times,
                                       self.traj_has_downsample)[0])

        ########################################################################
        ########## Compute Q-func in the future, i.e. Eq(7) second row #########
        ########################################################################

        # [num_traj, num_segments, 1 + num_seg_actions]
        future_returns = torch.zeros([num_traj, num_segments,
                                      1 + num_seg_actions], device=self.device)

        # [num_traj, num_segments, dim_state]
        c_state = states[:, seg_start_idx]

        # [num_traj, num_segments]
        c_idx = util.add_expand_dim(seg_start_idx, [0], [num_traj])

        # [num_segments, num_seg_actions]
        # -> [num_traj, num_segments, num_seg_actions]
        a_idx = util.add_expand_dim(idx_in_segments[..., :-1],
                                    [0], [num_traj])

        # Use mix precision for faster computation
        with autocast_if(self.use_mix_precision):
            # [num_traj, num_segments, (num_smp,) 1 + num_seg_actions]
            future_q1 = self.critic.critic(self.critic.target_net1,
                                           state=c_state,
                                           actions=actions,
                                           idx_s=c_idx, idx_a=a_idx)
            if not self.critic.single_q:
                future_q2 = self.critic.critic(self.critic.target_net2,
                                               state=c_state,
                                               actions=actions,
                                               idx_s=c_idx, idx_a=a_idx)
            else:
                future_q2 = future_q1

        future_q = torch.minimum(future_q1, future_q2)

        # Use last q as the target of the V-func
        # [num_traj, num_segments, 1 + num_seg_actions]
        # -> [num_traj, num_segments]
        # Tackle the Q-func beyond the length of the trajectory
        future_returns[:, :-1, 0] = future_q[:, :-1, -1]

        # Find the idx where the action is the last action in the valid trajectory
        last_valid_q_idx = idx_in_segments[-1] == traj_length
        future_returns[:, -1, 0] = (
            future_q[:, -1, last_valid_q_idx].squeeze(-1))

        ########################################################################
        ######### Compute V-func in the future, i.e. Eq(8) RHS 2nd term ########
        ########################################################################

        # state after executing the action
        # [num_traj, traj_length]
        c_idx = torch.arange(traj_length, device=self.device).long()
        c_idx = util.add_expand_dim(c_idx, [0], [num_traj])

        # [num_traj, traj_length, dim_state]
        c_state = states

        # Use mix precision for faster computation
        with autocast_if(self.use_mix_precision):
            # [num_traj, traj_length]
            future_v1 = self.critic.critic(self.critic.target_net1,
                                           state=c_state,
                                           actions=None,
                                           idx_s=c_idx, idx_a=None).squeeze(-1)
            if not self.critic.single_q:
                future_v2 = self.critic.critic(self.critic.target_net2,
                                               state=c_state,
                                               actions=None,
                                               idx_s=c_idx, idx_a=None).squeeze(-1)
            else:
                future_v2 = future_v1

        future_v = torch.minimum(future_v1, future_v2)

        # Pad zeros to the states which go beyond the traj length
        future_v_pad_zero_end \
            = torch.nn.functional.pad(future_v, (0, num_seg_actions))

        # [num_segments, num_seg_actions]
        v_idx = idx_in_segments[:, 1:]
        # assert v_idx.max() <= traj_length

        # [num_traj, traj_length] -> [num_traj, num_segments, num_seg_actions]
        future_returns[..., 1:] = future_v_pad_zero_end[:, v_idx]

        ########################################################################
        ######### Compute rewards in the future, i.e. Eq(8) RHS 1st term #######
        ########################################################################

        # discount_seq as: [1, gamma, gamma^2..., ]
        discount_idx = torch.arange(1 + num_seg_actions, device=self.device)

        # [num_trajs, 1 + num_seg_actions]
        discount_seq = torch.pow(self.discount_factor, discount_idx)
        discount_seq = util.add_expand_dim(discount_seq, [0], [num_traj])

        # Apply discount to all rewards and returns w.r.t the traj start
        # [num_traj, num_segments, 1 + num_seg_actions]
        discount_return = future_returns * discount_seq[:, None, :]

        # [num_traj, traj_length] -> [num_traj, traj_length + num_seg_actions]
        future_r_pad_zero_end \
            = torch.nn.functional.pad(rewards, (0, num_seg_actions))

        # -> [num_traj, num_segments, traj_length + num_seg_actions]
        future_r_pad_zero_end = util.add_expand_dim(future_r_pad_zero_end,
                                                    [1], [num_segments])

        # -> [num_traj, num_segments, 1 + num_seg_actions]
        seg_reward_idx = util.add_expand_dim(idx_in_segments, [0],
                                             [num_traj])

        # torch.gather shapes
        # input: [num_traj, num_segments, traj_length + num_seg_actions]
        # index: [num_traj, num_segments, 1 + num_seg_actions]
        # result: [num_traj, num_segments, 1 + num_seg_actions]
        seg_r = torch.gather(input=future_r_pad_zero_end, dim=-1,
                             index=seg_reward_idx)

        seg_discount_r = seg_r * discount_seq[:, None, :]

        # [num_traj, num_segments, 1 + num_seg_actions] ->
        # [num_traj, num_segments, 1 + num_seg_actions, 1 + num_seg_actions]
        seg_discount_r = util.add_expand_dim(seg_discount_r, [-2],
                                             [1 + num_seg_actions])

        # Get a lower triangular mask with off-diagonal elements as 1
        # [num_seg_actions + 1, num_seg_actions + 1]
        reward_tril_mask = torch.tril(torch.ones(1 + num_seg_actions,
                                                 1 + num_seg_actions,
                                                 device=self.device),
                                      diagonal=-1)

        # [num_traj, num_segments, num_seg_actions + 1, num_seg_actions + 1]
        tril_discount_rewards = seg_discount_r * reward_tril_mask

        # N-step return as target
        # V(s0) -> R0
        # Q(s0, a0) -> r0 + \gam * R1
        # Q(s0, a0, a1) -> r0 + \gam * r1 + \gam^2 * R2
        # Q(s0, a0, a1, a2) -> r0 + \gam * r1 + \gam^2 * r2 + \gam^3 * R3

        # [num_traj, num_segments, 1 + num_seg_actions]
        n_step_returns = (tril_discount_rewards.sum(dim=-1) + discount_return)
        return n_step_returns

    def update_critic(self):
        self.critic.train()
        self.critic.requires_grad(True)

        critic_loss_list = []

        for grad_idx in range(self.epochs_critic):
            # Sample from replay buffer
            dataset = self.replay_buffer.sample(self.batch_size)
            states = dataset["step_states"]
            actions = dataset["step_actions"]
            num_traj = states.shape[0]
            traj_length = states.shape[1]

            idx_in_segments = self.get_random_segments(pad_additional=True)
            seg_start_idx = idx_in_segments[..., 0]
            assert seg_start_idx[-1] < self.traj_length
            seg_actions_idx = idx_in_segments[..., :-1]
            num_seg_actions = seg_actions_idx.shape[-1]

            # [num_traj, num_segments, dim_state]
            c_state = states[:, seg_start_idx]
            seg_start_idx = util.add_expand_dim(seg_start_idx, [0], [num_traj])

            padded_actions = torch.nn.functional.pad(
                actions, (0, 0, 0, num_seg_actions), "constant", 0)

            # [num_traj, num_segments, num_seg_actions, dim_action]
            seg_actions = padded_actions[:, seg_actions_idx]

            # [num_traj, num_segments, num_seg_actions]
            seg_actions_idx = util.add_expand_dim(seg_actions_idx, [0],
                                                  [num_traj])

            # [num_traj, num_segments, 1 + num_seg_actions]
            targets = self.segments_n_step_return_vf(dataset, idx_in_segments)

            for net, target_net, opt, scaler in self.critic_nets_and_opt():
                # Use mix precision for faster computation
                with autocast_if(self.use_mix_precision):
                    # [num_traj, num_segments, 1 + num_seg_actions]
                    vq_pred = self.critic.critic(
                        net=net, state=c_state, actions=seg_actions,
                        idx_s=seg_start_idx, idx_a=seg_actions_idx)

                    # Mask out the padded actions
                    # [num_traj, num_segments, num_seg_actions]
                    valid_mask = seg_actions_idx < self.traj_length

                    # [num_traj, num_segments, num_seg_actions]
                    vq_pred[..., 1:] = vq_pred[..., 1:] * valid_mask
                    targets[..., 1:] = targets[..., 1:] * valid_mask

                    # Loss
                    critic_loss = torch.nn.functional.mse_loss(vq_pred, targets)

                # Update critic net parameters
                opt.zero_grad(set_to_none=True)

                # critic_loss.backward()
                scaler.scale(critic_loss).backward()

                # opt.step()
                scaler.step(opt)
                scaler.update()

                # Logging
                critic_loss_list.append(critic_loss.item())

                # Update target network
                self.critic.update_target_net(net, target_net)

        if self.log_now:
            # Get critic update statistics
            critic_info_dict = {
                **util.generate_stats(critic_loss_list, "critic_loss")}
        else:
            critic_info_dict = {}
        return critic_info_dict

    def update_policy(self, *args, **kwargs):

        policy_loss_list = []
        trust_region_loss_list = []
        q_loss_list = []
        entropy_list = []

        for grad_idx in range(self.epochs_policy):
            # Sample from replay buffer
            dataset = self.replay_buffer.sample(self.batch_size)

            # Make a new prediction
            pred_mean, pred_L, info = self.make_new_pred(dataset)

            if self.projection is not None:
                trust_region_loss = info["trust_region_loss"]
            else:
                trust_region_loss = torch.zeros([1], device=self.device)

            entropy = self.policy.entropy([pred_mean, pred_L]).mean().item()

            q_loss = self.q_loss(dataset, pred_mean, pred_L)
            policy_loss = q_loss + trust_region_loss

            # Update policy net parameters
            self.policy_optimizer.zero_grad(set_to_none=True)
            with torch.autograd.set_detect_anomaly(True):
                policy_loss.backward()
            self.policy_optimizer.step()

            # Logging
            policy_loss_list.append(policy_loss.item())
            trust_region_loss_list.append(trust_region_loss.item())
            entropy_list.append(entropy)
            q_loss_list.append(q_loss.item())

        if self.use_old_policy:
            self.policy_old.copy_(self.policy, self.old_policy_update_rate)

        if self.log_now:
            policy_info_dict = {
                **util.generate_stats(q_loss_list, "q_loss_raw"),
                **util.generate_stats(trust_region_loss_list,
                                      "trust_region_loss"),
                **util.generate_stats(policy_loss_list, "policy_loss"),
                **util.generate_stats(entropy_list, "entropy")}
        else:
            policy_info_dict = {}

        return policy_info_dict

    def q_loss(self, dataset, pred_mean, pred_L):
        self.critic.eval()  # disable dropout
        self.critic.requires_grad(False)

        states = dataset["step_states"]
        init_time = dataset["episode_init_time"]
        init_pos = dataset["episode_init_pos"]
        init_vel = dataset["episode_init_vel"]
        num_traj = states.shape[0]
        times = self.sampler.get_times(init_time, self.sampler.num_times,
                                       self.traj_has_downsample)

        # Shape of idx_in_segments [num_segments, num_seg_actions + 1]
        idx_in_segments = self.get_random_segments()
        num_segments = idx_in_segments.shape[0]
        seg_start_idx = idx_in_segments[..., 0]
        seg_actions_idx = idx_in_segments[..., :-1]

        # Broadcasting data to the correct shape
        pred_mean = util.add_expand_dim(pred_mean, [-3], [num_segments])
        pred_L = util.add_expand_dim(pred_L, [-4], [num_segments])
        pred_at_times = times[:, idx_in_segments[..., :-1]]

        # Note, here the init condition of the traj is used by all segments
        #  We found it is better than using the init condition of the segment
        init_time = util.add_expand_dim(init_time, [-1], [num_segments])
        init_pos = util.add_expand_dim(init_pos, [-2], [num_segments])
        init_vel = util.add_expand_dim(init_vel, [-2], [num_segments])

        # Get the trajectory segments
        # [num_trajs, num_segments, num_seg_actions, num_dof]
        pred_seg_actions = self.policy.sample(
            require_grad=True, params_mean=pred_mean,
            params_L=pred_L, times=pred_at_times, init_time=init_time,
            init_pos=init_pos, init_vel=init_vel, use_mean=False, split_args=self.reference_split_args,
            use_case="agent",
            segment_wise_init_pos= dataset["segment_wise_init_pos"],
            segment_wise_init_vel= dataset["segment_wise_init_vel"],
            ref_time_list=self.sampler.get_times(dataset["episode_init_time"],
                                                 self.sampler.num_times,
                                                 self.traj_has_downsample)[0]

        )
        #pred_seg_actions = pred_seg_actions.unsqueeze(-3)

        # Current state
        # [num_traj, num_segments, dim_state]
        c_state = states[:, seg_start_idx]

        # [num_segments] -> [num_traj, num_segments]
        seg_start_idx = util.add_expand_dim(seg_start_idx, [0],
                                            [num_traj])
        # [num_segments, num_actions] -> [num_traj, num_segments, num_actions]
        seg_actions_idx = util.add_expand_dim(seg_actions_idx, [0],
                                              [num_traj])

        # [num_traj, num_segments, num_seg_actions]
        # vq -> q
        q1 = self.critic.critic(net=self.critic.net1, state=c_state,
                                actions=pred_seg_actions,
                                idx_s=seg_start_idx,
                                idx_a=seg_actions_idx)[..., 1:]
        if not self.critic.single_q:
            q2 = self.critic.critic(net=self.critic.net2, state=c_state,
                                    actions=pred_seg_actions,
                                    idx_s=seg_start_idx,
                                    idx_a=seg_actions_idx)[..., 1:]
        else:
            q2 = q1

        q = torch.minimum(q1.sum(dim=-1), q2.sum(dim=-1))

        # Mean over trajs and segments
        # [num_traj, num_segments] -> scalar
        q_loss = -q.mean()
        return q_loss

    def critic_nets_and_opt(self):
        if self.critic.single_q:
            return zip(util.make_iterable(self.critic.net1, "list"),
                       util.make_iterable(self.critic.target_net1, "list"),
                       util.make_iterable(self.critic_optimizer[0], "list"),
                       util.make_iterable(self.critic_grad_scaler[0], "list"))
        else:
            return zip([self.critic.net1, self.critic.net2],
                       [self.critic.target_net1, self.critic.target_net2],
                       self.critic_optimizer, self.critic_grad_scaler)

    def save_agent(self, log_dir: str, epoch: int):
        super().save_agent(log_dir, epoch)
        self.sampler.save_rms(log_dir, epoch)

    def load_agent(self, log_dir: str, epoch: int):
        super().load_agent(log_dir, epoch)
        self.sampler.load_rms(log_dir, epoch)
        self.load_pretrained_agent = True
