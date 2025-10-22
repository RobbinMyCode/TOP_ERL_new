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
        self.update_policy_based_on_dataset_splits = "rand" in self.reference_split_args[
            "split_strategy"] and not self.reference_split_args.get("use_top_erl_splits_policy", False)
        self.update_critic_based_on_dataset_splits = "rand" in self.reference_split_args[
            "split_strategy"] and not self.reference_split_args.get("use_top_erl_splits_critic", False)

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

    def get_random_segments(self, pad_additional=False, fit_splits=False, split_starts=None):
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
        if fit_splits:
            num_seg_actual += split_starts.shape[-1]
        idx_in_segments = torch.arange(0, num_seg_actual * seg_length,
                                       device=self.device) + start_idx
        idx_in_segments = idx_in_segments.view(-1, seg_length)
        idx_in_segments = torch.cat([idx_in_segments,
                                     idx_in_segments[:, -1:] + 1], dim=-1)


        if fit_splits:
            assert  self.reference_split_args['split_strategy'] == 'n_equal_splits', "exact split hit only supported for n equal splits"
            #in n_equal case all split_starts are the same, else this does not work
            split_starts = split_starts[0]
            n_th_split_start = 1
            for idx, idx_val_row in enumerate(idx_in_segments):
                if idx_val_row[0] >= split_starts[n_th_split_start]:
                    idx_in_segments[idx:] += split_starts[n_th_split_start] - idx_in_segments[idx, 0]
                    n_th_split_start += 1
                    if n_th_split_start > split_starts.shape[0] - 1:
                        break
            relevant_idx = 0 if pad_additional else -1
            while idx_in_segments[-1, relevant_idx] >= self.traj_length:
                idx_in_segments = idx_in_segments[:-1]
            #TODO: THIS ONLY WORKS FOR IN POLICYUPDATE / QLOSS CORRECTLY --> makes strategy useless in v_func_est
            if "include_0" in self.reference_split_args["q_loss_strategy"] and not num_seg==1 and idx_in_segments[0,0]!=0:
                idx_in_segments = torch.cat([torch.arange(seg_length+1, device=self.device)[None, :], idx_in_segments], dim=0)
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

        if not self.update_critic_based_on_dataset_splits:
            num_segments = idx_in_segments.shape[0]
        else:
            num_segments = idx_in_segments.shape[1]

        num_seg_actions = idx_in_segments.shape[-1] - 1
        seg_start_idx = idx_in_segments[..., 0]

        num_traj, traj_length = states.shape[0], states.shape[1]

        with torch.no_grad():
            # params: [num_traj, num_weights]
            params_mean_new, params_L_new, _ \
                = self.make_new_pred(dataset, compute_trust_region_loss=False)

        if not self.update_critic_based_on_dataset_splits:
            # [num_traj, num_weights] -> [num_traj, num_segments, num_weights]
            params_mean_new = util.add_expand_dim(params_mean_new, [1],
                                                  [num_segments])
            params_L_new = util.add_expand_dim(params_L_new, [1],
                                               [num_segments])

        # [num_traj, traj_length]
        times = self.sampler.get_times(dataset["episode_init_time"],
                                       self.sampler.num_times,
                                       self.traj_has_downsample)

        ref_time_list = times[0]

        ref_time = times[0]
        # [num_traj, traj_length] -> [num_traj, num_segments]
        #random_range case
        if times.shape[0] == seg_start_idx.shape[0]:
            times = times[np.arange(times.shape[0])[:, None],  seg_start_idx]
        #all fixed sizes
        else:
            times = times[:, seg_start_idx]

        # [num_traj, num_segments] -> [num_traj, num_segments, num_seg_actions]
        action_times = util.add_expand_dim(times, [-1], [num_seg_actions])
        time_evo = torch.linspace(0, self.sampler.dt * (num_seg_actions-1),
                                  num_seg_actions, device=self.device).float()
        action_times = action_times + time_evo

        # Init time, shape: [num_traj, num_segments]
        init_time = times - self.sampler.dt

        # [num_traj, num_segments, dim_action]
        # random_range case
        if traj_init_pos.shape[0] == seg_start_idx.shape[0]:
            init_pos = traj_init_pos[np.arange(traj_init_pos.shape[0])[:, None], seg_start_idx]
            init_vel = traj_init_vel[np.arange(traj_init_pos.shape[0])[:, None], seg_start_idx]
        else: #all fixed sizes
            init_pos = traj_init_pos[:, seg_start_idx]
            init_vel = traj_init_vel[:, seg_start_idx]

        # Get the new MP trajectory using the current policy and condition on
        # the buffered desired pos and vel as initial conditions
        # util.set_global_random_seed(0)

        params_mean_new = self.policy.fix_relative_goal_for_segments(
            params=params_mean_new, traj_init_pos=traj_init_pos[:, 0],
            segments_init_pos=init_pos)


        sampling_args_value_func = self.reference_split_args.copy()
        if self.update_critic_based_on_dataset_splits:
            use_case = "aa"
        else:
            sampling_args_value_func["q_loss_strategy"] = sampling_args_value_func["v_func_estimation"]
            if "enforce_no_overlap_overconf" in sampling_args_value_func["q_loss_strategy"]:
                sampling_args_value_func["q_loss_strategy"] = "overconfident"
            use_case = "agent"

        # [num_traj, num_segments, num_seg_actions, dim_action] or
        # [num_traj, num_segments, num_smps, num_seg_actions, dim_action]
        actions = self.policy.sample(require_grad=False,
                                     params_mean=params_mean_new,
                                     params_L=params_L_new, times=action_times,
                                     init_time=init_time,
                                     init_pos=init_pos, init_vel=init_vel,
                                     use_mean=False,
                                     split_args=sampling_args_value_func,
                                     use_case = use_case,
                                     split_start_indexes=dataset["split_start_indexes"],
                                     segment_wise_init_pos=dataset["segment_wise_init_pos"],
                                     segment_wise_init_vel=dataset["segment_wise_init_vel"],
                                     mp_distr_rel_pos = dataset["mp_distr_rel_pos"] if self.reference_split_args.get("re_use_rand_coord_from_sampler_for_updates",False) else None,
                                     ref_time_list = ref_time_list)

        ########################################################################
        ########## Compute Q-func in the future, i.e. Eq(7) second row #########
        ########################################################################

        # [num_traj, num_segments, 1 + num_seg_actions]
        future_returns = torch.zeros([num_traj, num_segments,
                                      1 + num_seg_actions], device=self.device)

        # [num_traj, num_segments, dim_state]
        if states.shape[0] == seg_start_idx.shape[0]:
            c_state = states[np.arange(states.shape[0])[:, None], seg_start_idx]
            c_idx = seg_start_idx
            a_idx = idx_in_segments[..., :-1]
        else:
            c_state = states[:, seg_start_idx]
            # [num_traj, num_segments]
            c_idx = util.add_expand_dim(seg_start_idx, [0], [num_traj])
            # [num_segments, num_seg_actions]
            # -> [num_traj, num_segments, num_seg_actions]
            a_idx = util.add_expand_dim(idx_in_segments[..., :-1],
                                        [0], [num_traj])

        kwargs = {}
        if self.reference_split_args.get("add_d_state_to_critic", False):
            if self.update_critic_based_on_dataset_splits:
                kwargs["idx_d"] = c_idx
                kwargs["d_state"] = c_state
            else:
                kwargs["idx_d"] = torch.sum(c_idx[..., None] >= dataset["split_start_indexes"][..., None, :], dim=-1)
                kwargs["d_state"] = states[np.arange(states.shape[0])[:, None], kwargs["idx_d"]]

        # Use mix precision for faster computation
        with autocast_if(self.use_mix_precision):
            # [num_traj, num_segments, (num_smp,) 1 + num_seg_actions]
            future_q1 = self.critic.critic(self.critic.target_net1,
                                           state=c_state,
                                           actions=actions,
                                           idx_s=c_idx, idx_a=a_idx, kwargs=kwargs)
            if not self.critic.single_q:
                future_q2 = self.critic.critic(self.critic.target_net2,
                                               state=c_state,
                                               actions=actions,
                                               idx_s=c_idx, idx_a=a_idx, kwargs=kwargs)
            else:
                future_q2 = future_q1

        future_q = torch.minimum(future_q1, future_q2)

        # Find the idx where the action is the last action in the valid trajectory
        if not self.update_critic_based_on_dataset_splits:
            # Use last q as the target of the V-func
            # [num_traj, num_segments, 1 + num_seg_actions]
            # -> [num_traj, num_segments]
            # Tackle the Q-func beyond the length of the trajectory
            future_returns[:, :-1, 0] = future_q[:, :-1, -1]

            #if we cut out segments we may not reach the full traj_length
            corrected_traj_length = min(traj_length, idx_in_segments[-1, -1])
            last_valid_q_idx = idx_in_segments[-1] == corrected_traj_length
            future_returns[:, -1, 0] = (
                future_q[:, -1, last_valid_q_idx].squeeze(-1))

        else:
            next_seg_start_idx = seg_start_idx[..., 1:]
            #segment goes from [0, next_start] -> start_idx of next is end of previous
            last_valid_q_idx_split = next_seg_start_idx[..., None] == idx_in_segments[:, :-1, :]

            last_valid_q_idx_max_len = (idx_in_segments[:, -1] == traj_length)[:, None, :]
            last_valid_mask = torch.cat([last_valid_q_idx_split, last_valid_q_idx_max_len], dim=1)

            #last_valid is a single index -> multiplication is basically getting the index
            future_returns[:, :, 0] = (
                torch.sum(future_q * last_valid_mask, axis=-1)
            )

        ########################################################################
        ######### Compute V-func in the future, i.e. Eq(8) RHS 2nd term ########
        ########################################################################

        # state after executing the action
        # [num_traj, traj_length]
        c_idx = torch.arange(traj_length, device=self.device).long()
        c_idx = util.add_expand_dim(c_idx, [0], [num_traj]) #512, 100

        # [num_traj, traj_length, dim_state]
        c_state = states

        if self.reference_split_args.get("add_d_state_to_critic", False):
            if self.update_critic_based_on_dataset_splits:
                kwargs["idx_d"] = c_idx
                kwargs["d_state"] = c_state
            else:
                kwargs["idx_d"] = torch.sum(c_idx[..., None] >= dataset["split_start_indexes"][..., None, :], dim=-1)
                kwargs["d_state"] = states[np.arange(states.shape[0])[:, None], kwargs["idx_d"]]

        # Use mix precision for faster computation
        with autocast_if(self.use_mix_precision):
            # [num_traj, traj_length]
            future_v1 = self.critic.critic(self.critic.target_net1,
                                           state=c_state,
                                           actions=None,
                                           idx_s=c_idx, idx_a=None, kwargs=kwargs).squeeze(-1)
            if not self.critic.single_q:
                future_v2 = self.critic.critic(self.critic.target_net2,
                                               state=c_state,
                                               actions=None,
                                               idx_s=c_idx, idx_a=None, kwargs=kwargs).squeeze(-1)
            else:
                future_v2 = future_v1

        future_v = torch.minimum(future_v1, future_v2)

        # Pad zeros to the states which go beyond the traj length
        future_v_pad_zero_end \
            = torch.nn.functional.pad(future_v, (0, num_seg_actions))


        if not self.update_critic_based_on_dataset_splits:
            # [num_segments, num_seg_actions]
            v_idx = idx_in_segments[:, 1:] #segment wise indexes 0..end -> 1...end, 100 instead of 101 indexes
            # assert v_idx.max() <= traj_length

            # [num_traj, traj_length] -> [num_traj, num_segments, num_seg_actions]
            future_returns[..., 1:] = future_v_pad_zero_end[:, v_idx]
        else:
            '''
            Parallized version of the following loop
                rewards_reshaped = torch.zeros_like(future_returns)
                for j in range(future_returns.shape[1]):
                    start = dataset["split_start_indexes"][:, j]
                    end = dataset["split_start_indexes"][:, j + 1] if j != actions.shape[1] - 1 else \
                    future_v.shape[-1] * torch.ones_like(start)

                    for b in range(future_returns.shape[0]):
                        s = start[b]
                        e = end[b]
                        if e-s!=0: #edgecase e.g. both=0 -> [..., 1:0] = [..., 0:-1] would be incorrect
                            future_returns[b, j, 1:e - s] = future_v_pad_zero_end[b, s:e-1]
                            rewards_reshaped[b, j, :e - s] = rewards[b, s:e]
                return future_returns, rewards_reshaped
            '''
            start = dataset["split_start_indexes"]

            last_idx_append = future_v.shape[-1]* torch.ones((*start.shape[:-1], 1), dtype=torch.int, device=self.device)
            end = torch.cat((dataset["split_start_indexes"][:, 1:], last_idx_append), dim=-1)

            length = end - start

            #create a mask for indexes where values shall be applied -> ensures only values in current split range are considered
            pos = torch.arange(future_returns.shape[-1], device=self.device)[None, None, :]

            #<= because state + actions -> 1+length*actions (last action (-> resulting pos) = init of next segment)
            segment_length_mask = pos <= length.unsqueeze(-1)

            #reward sorted by splits, idx clamped so that rewards can get assigned properly (and invalids get masked out by segment_length_mask)
            indices = torch.clamp(idx_in_segments, min=0, max=future_v.shape[-1])



            #one zero in front as first entry = initial state does not yield a reward, in the end to make the masking convenient
            rewards_zero_pad \
                = torch.nn.functional.pad(rewards, (0, 1))
            rewards_reshaped = torch.where(
                segment_length_mask,
                torch.gather(rewards_zero_pad[:, None, :].expand(-1, dataset["split_start_indexes"].shape[1], -1), 2, indices),
                torch.zeros_like(indices, dtype=rewards.dtype)
            )
            #initial state of each action sequence does not yield a reward (only actions do)
            #rewards_reshaped[..., 0] = 0
            #returns sorted by splits, first dublicate index does not matter as we only use action pos > 0
            indices_v = torch.clamp(idx_in_segments, min=0, max=future_v.shape[-1])
            g_v = torch.gather(
                future_v_pad_zero_end.unsqueeze(1).expand(-1, dataset["split_start_indexes"].shape[1], -1),  # [B,J,T_v]
                2,
                indices_v
            )
            g_v[idx_in_segments > future_v.shape[-1]-1] = 0.
            write_mask = (pos > 0) & segment_length_mask

            #last action can not get a proper return as there is no state after ("index 100") to evaluate -> 0
            future_returns[write_mask] = g_v.to(torch.float)[write_mask]

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



        # -> [num_traj, num_segments, 1 + num_seg_actions]
        if not self.update_critic_based_on_dataset_splits:
            # [num_traj, traj_length] -> [num_traj, traj_length + num_seg_actions]
            future_r_pad_zero_end \
                = torch.nn.functional.pad(rewards, (0, num_seg_actions))

            # -> [num_traj, num_segments, traj_length + num_seg_actions]
            future_r_pad_zero_end = util.add_expand_dim(future_r_pad_zero_end,
                                                        [1], [num_segments])
            seg_reward_idx = util.add_expand_dim(idx_in_segments, [0],
                                             [num_traj])

            # torch.gather shapes
            # input: [num_traj, num_segments, traj_length + num_seg_actions]
            # index: [num_traj, num_segments, 1 + num_seg_actions]
            # result: [num_traj, num_segments, 1 + num_seg_actions]
            seg_r = torch.gather(input=future_r_pad_zero_end, dim=-1,
                                 index=seg_reward_idx)

        else:
            #valid in splits -> use reshaped one
            seg_r = rewards_reshaped


        seg_discount_r = seg_r * discount_seq[:, None, :]

        # [num_traj, num_segments, 1 + num_seg_actions] ->
        # [num_traj, num_segments, 1 + num_seg_actions, 1 + num_seg_actions]
        seg_discount_r = util.add_expand_dim(seg_discount_r, [-2],
                                             [1 + num_seg_actions])

        # Get a lower triangular segment_length_mask with off-diagonal elements as 1
        # [num_seg_actions + 1, num_seg_actions + 1]
        reward_tril_mask = torch.tril(torch.ones(1 + num_seg_actions,
                                                 1 + num_seg_actions,
                                                 device=self.device),
                                      diagonal=-1)

        # [num_traj, num_segments, num_seg_actions + 1, num_seg_actions + 1]
        tril_discount_rewards = seg_discount_r * reward_tril_mask
        mc_returns_resample = tril_discount_rewards.sum(dim=-1)

        # N-step return as target
        # V(s0) -> R0
        # Q(s0, a0) -> r0 + \gam * R1
        # Q(s0, a0, a1) -> r0 + \gam * r1 + \gam^2 * R2
        # Q(s0, a0, a1, a2) -> r0 + \gam * r1 + \gam^2 * r2 + \gam^3 * R3

        # [num_traj, num_segments, 1 + num_seg_actions]
        n_step_returns = (tril_discount_rewards.sum(dim=-1) + discount_return)


        #filter out invalid commulativ returns (for steps not actually done in this split)
        if self.update_critic_based_on_dataset_splits:
            next_seg_start_idx = seg_start_idx[..., 1:]
            valid_q_idx_split = idx_in_segments[:, :-1, :] <= next_seg_start_idx[..., None]
            # last split will never collide with the next split_start -> add True here
            valid_q_idx_split_full = torch.cat([valid_q_idx_split, torch.ones(
                (*(valid_q_idx_split.shape[:-2]), 1, valid_q_idx_split.shape[-1]), dtype=torch.bool,
                device=self.device)], dim=-2)

            valid_q_idx_max_len = idx_in_segments[..., :] <= self.traj_length
            # only take in actions, first of valid_q_idx_split is the start state of each segment -> and statement with absolute length
            valid_mask = valid_q_idx_split_full * valid_q_idx_max_len

            n_step_returns = n_step_returns * valid_mask
            mc_returns_resample = mc_returns_resample * valid_mask

        return n_step_returns, mc_returns_resample

    def update_critic(self):
        self.critic.train()
        self.critic.requires_grad(True)

        mc_returns_list = []
        mc_returns_resample_list = []
        critic_loss_list = []
        targets_list = []
        targets_bias_list = []
        targets_resample_bias_list = []


        for grad_idx in range(self.epochs_critic):
            # Sample from replay buffer
            dataset = self.replay_buffer.sample(self.batch_size)
            states = dataset["step_states"]
            actions = dataset["step_actions"]
            num_traj = states.shape[0]
            traj_length = states.shape[1]

            if not self.update_critic_based_on_dataset_splits:
                split_start_as_ind0 = "enforce_no_overlap" in self.reference_split_args["q_loss_strategy"]
                idx_in_segments = self.get_random_segments(pad_additional=True, fit_splits=split_start_as_ind0, split_starts = dataset["split_start_indexes"])
                last_valid_start = self.reference_split_args.get("ignore_top_erl_updates_after_index", idx_in_segments[-1, -1])
                while idx_in_segments[-1, 0] > last_valid_start:
                    idx_in_segments = idx_in_segments[:-1]
                seg_start_idx = idx_in_segments[..., 0]
                assert seg_start_idx[-1] < self.traj_length

                seg_actions_idx = idx_in_segments[..., :-1]
                num_seg_actions = seg_actions_idx.shape[-1]
                used_split_args = self.reference_split_args
            else:
                seg_start_idx = dataset["split_start_indexes"]
                if self.reference_split_args["split_strategy"] == "random_size_range":
                    max_diff = self.reference_split_args["size_range"][1]
                elif self.reference_split_args["split_strategy"] == "fixed_size_rand_start":
                    max_diff = self.reference_split_args["fixed_size"]
                elif self.reference_split_args["split_strategy"] == "intra_episode_fixed_inter_rand_size":
                    max_diff = self.reference_split_args["inter_fixed_size_range"][-1]
                else:
                    raise NotImplementedError

                idx_in_segments = seg_start_idx[..., None] + torch.arange(max_diff+1, device=self.device)
                seg_actions_idx = idx_in_segments[..., :-1]
                num_seg_actions = seg_actions_idx.shape[-1]



            # [num_traj, num_segments, dim_state]

            if states.shape[0] == seg_start_idx.shape[0]:
                c_state = states[np.arange(states.shape[0])[:, None], seg_start_idx]
                padded_actions = torch.nn.functional.pad(
                    actions, (0, 0, 0, num_seg_actions), "constant", 0)

                # [num_traj, num_segments, num_seg_actions, dim_action]
                seg_actions = padded_actions[np.arange(states.shape[0])[:, None, None], seg_actions_idx]

            else:
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
            targets, mc_returns_resample = self.segments_n_step_return_vf(dataset, idx_in_segments)

            mc_returns_mean = util.compute_mc_return(
                dataset["step_rewards"].mean(dim=0),
                self.discount_factor).mean().item()

            mc_returns_resample_list.append(mc_returns_resample.mean().item())
            mc_returns_list.append(mc_returns_mean)


            #for relativ indexing: adapting idx_s, idx_a
            #seg_start_idx_pred = seg_start_idx[..., 0][..., None] * torch.ones_like(seg_start_idx)
            kwargs={}
            if self.reference_split_args.get("add_d_state_to_critic", False):
                if self.update_critic_based_on_dataset_splits:
                    kwargs["idx_d"] = seg_start_idx
                    kwargs["d_state"] = c_state
                else:
                    kwargs["idx_d"] = torch.sum(seg_start_idx[..., None] >= dataset["split_start_indexes"][..., None, :],
                                                dim=-1)
                    kwargs["d_state"] = states[np.arange(states.shape[0])[:, None], kwargs["idx_d"]]
            #seg_actions_idx_pred = seg_actions_idx[..., 0, :][..., None, :] * torch.ones_like(seg_actions_idx)
            for net, target_net, opt, scaler in self.critic_nets_and_opt():
                # Use mix precision for faster computation
                with autocast_if(self.use_mix_precision):
                    # [num_traj, num_segments, 1 + num_seg_actions]
                    vq_pred = self.critic.critic(
                        net=net, state=c_state, actions=seg_actions,
                        idx_s=seg_start_idx, idx_a=seg_actions_idx, kwargs=kwargs)


                    # Mask out the padded actions
                    # [num_traj, num_segments, num_seg_actions]
                    #as top_erl_based actions already correspond
                    if not self.update_critic_based_on_dataset_splits:
                        valid_mask = seg_actions_idx < self.traj_length

                        #in case of truncation mask out all actions -> v/q estimations that are truncated
                        if used_split_args["v_func_estimation"] == "truncated":
                            times = self.sampler.get_times(dataset["episode_init_time"],
                                                           self.sampler.num_times,
                                                           self.traj_has_downsample)
                            ref_time_list = times[0]
                            ref_time = times[0]
                            # [num_traj, traj_length] -> [num_traj, num_segments]
                            # random_range case
                            if times.shape[0] == seg_start_idx.shape[0]:
                                times = times[np.arange(times.shape[0])[:, None], seg_start_idx]
                            # all fixed sizes
                            else:
                                times = times[:, seg_start_idx]


                            # [num_traj, num_segments] -> [num_traj, num_segments, num_seg_actions]
                            action_times = util.add_expand_dim(times, [-1], [num_seg_actions])
                            time_evo = torch.linspace(0, self.sampler.dt * (num_seg_actions - 1),
                                                      num_seg_actions, device=self.device).float()
                            action_times = action_times + time_evo
                            policy_indexes = self.policy.get_param_indexes(action_times,
                                                                    dataset["split_start_indexes"],
                                                                    ref_time_list)
                            init_time = times - self.sampler.dt
                            policy_start_indexes = self.policy.get_param_indexes(init_time[..., None], dataset["split_start_indexes"],
                                                   ref_time_list)[..., 0]
                            overlap_mask = policy_start_indexes.unsqueeze(-1) == policy_indexes
                            valid_mask = overlap_mask & valid_mask
                    else: #also mask out actions that belong to the next segment
                        next_seg_start_idx = seg_start_idx[..., 1:]
                        valid_q_idx_split = idx_in_segments[:, :-1, :] <= next_seg_start_idx[..., None]
                        valid_q_idx_split_full = torch.cat([valid_q_idx_split, torch.ones((*(valid_q_idx_split.shape[:-2]), 1, valid_q_idx_split.shape[-1]), dtype=torch.bool, device=self.device)], dim=-2)

                        #we only count actions -> one step [=init pos] less
                        valid_q_idx_max_len = seg_actions_idx <= self.traj_length - 1
                        #only take in actions, first of valid_q_idx_split is the start state of each segment -> and statement with absolute length
                        valid_mask = valid_q_idx_split_full[..., 1:] * valid_q_idx_max_len
                    # [num_traj, num_segments, num_seg_actions]
                    vq_pred[..., 1:] = vq_pred[..., 1:] * valid_mask
                    targets[..., 1:] = targets[..., 1:] * valid_mask

                    # Loss
                    critic_loss = torch.nn.functional.mse_loss(vq_pred, targets)
                    #print(f"critic loss: {critic_loss} \t vq_pred: {vq_pred.mean().item()} \t targets: {targets.mean().item()} \t mc_return: {mc_returns_resample.mean().item()}")
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

            targets_mean = targets.mean().item()
            targets_list.append(targets_mean)
            targets_bias_list.append(targets_mean - mc_returns_mean)
            targets_resample_bias_list.append((targets - mc_returns_resample).mean().item())
        if self.log_now:
            # Get critic update statistics
            critic_info_dict = {
                **util.generate_stats(critic_loss_list, "critic_loss"),
                **util.generate_stats(mc_returns_list, "mc_returns"),
                **util.generate_stats(mc_returns_resample_list, "mc_returns_resample"),
                **util.generate_stats(targets_resample_bias_list, "target_bias_resample"),
                **util.generate_stats(targets_list, "targets"),
                **util.generate_stats(targets_bias_list, "targets_bias")
            }
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
        ref_time = times[0]
        # Shape of idx_in_segments [num_segments, num_seg_actions + 1]
        if not self.update_policy_based_on_dataset_splits:
            split_start_as_ind0 = "enforce_no_overlap" in self.reference_split_args["q_loss_strategy"]
            if self.reference_split_args.get("policy_use_all_action_indexes", False):
                idx_in_segments = self.get_random_segments(pad_additional=True, fit_splits=split_start_as_ind0, split_starts=dataset["split_start_indexes"])
            else:
                idx_in_segments = self.get_random_segments(pad_additional=False, fit_splits=split_start_as_ind0, split_starts=dataset["split_start_indexes"])

            #drop some updates if wanted
            last_valid_start = self.reference_split_args.get("ignore_top_erl_updates_after_index",
                                                             idx_in_segments[-1, -1])
            while idx_in_segments[-1, 0] > last_valid_start:
                idx_in_segments = idx_in_segments[:-1]

            num_segments = idx_in_segments.shape[0]
            seg_start_idx = idx_in_segments[..., 0]
            seg_actions_idx = idx_in_segments[..., :-1]
            # Broadcasting data to the correct shape
            pred_mean = util.add_expand_dim(pred_mean, [-3], [num_segments])
            pred_L = util.add_expand_dim(pred_L, [-4], [num_segments])
            pred_at_times = times[:, torch.clip(idx_in_segments[..., :-1], max=times.size(-1)-1)]
            used_split_args = self.reference_split_args
            if "enforce_no_overlap_overconf" in used_split_args["q_loss_strategy"]:
                used_split_args["q_loss_strategy"] = "overconfident"
            use_case = "agent"
        else:
            num_segments = dataset["split_start_indexes"].size(1)
            seg_start_idx = dataset["split_start_indexes"]
            if self.reference_split_args["split_strategy"] == "random_size_range":
                max_diff = self.reference_split_args["size_range"][1]
            elif self.reference_split_args["split_strategy"] == "fixed_size_rand_start":
                max_diff = self.reference_split_args["fixed_size"]
            elif self.reference_split_args["split_strategy"] == "intra_episode_fixed_inter_rand_size":
                max_diff = self.reference_split_args["inter_fixed_size_range"][-1]

            #1 extra entry as it is [s_0, a_0, ...,a_{l-1}]
            idx_in_segments = seg_start_idx[..., None] + torch.arange(max_diff +1, device=self.device)
            seg_actions_idx = idx_in_segments[..., :-1]

            #idx_in_segments goes to index > 100 -> clip so that "unused" values are all at last timestep
            pred_at_times = times[torch.arange(states.shape[0], device=self.device)[:, None, None], torch.clip(idx_in_segments[..., :-1], max=times.size(-1)-1)]
            used_split_args = self.reference_split_args.copy()
            used_split_args["q_loss_strategy"] = "truncated"
            use_case = "just_parallel_sample_lul_this_string_does_not_matter_it_just_is_not_agent"


        # Note, here the init condition of the traj is used by all segments
        #  We found it is better than using the init condition of the segment
        init_time = util.add_expand_dim(init_time, [-1], [num_segments]).contiguous()
        init_pos = util.add_expand_dim(init_pos, [-2], [num_segments]).contiguous()
        init_vel = util.add_expand_dim(init_vel, [-2], [num_segments]).contiguous()

        # UPDATE WE DO NOT WANT ALL INITIAL CONDITION IDENTICAL FOR SPLITS
        if self.reference_split_args["n_splits"] != 1:
            #init condition for next segment is last step of prev action
            clipped_indexes = torch.clip(idx_in_segments[..., 0:-1, -1], max=dataset["step_actions"].size(1)-1)
            if not self.update_policy_based_on_dataset_splits:
                init_time[:, 1:] = times[:, clipped_indexes]
                init_pos[:, 1:] = dataset["step_actions"][:, clipped_indexes, :7]
                init_vel[:, 1:] = dataset["step_actions"][:, clipped_indexes, 7:]
            else:
                init_time[:, 1:] = times[np.arange(times.size(0))[:, None], clipped_indexes]
                init_pos[:, 1:] = dataset["step_actions"][np.arange(times.size(0))[:, None], clipped_indexes, :7]
                init_vel[:, 1:] = dataset["step_actions"][np.arange(times.size(0))[:, None], clipped_indexes, 7:]
        # Get the trajectory segments
        # [num_trajs, num_segments, num_seg_actions, num_dof]
        pred_seg_actions = self.policy.sample(
            require_grad=True, params_mean=pred_mean,
            params_L=pred_L, times=pred_at_times, init_time=init_time,
            init_pos=init_pos, init_vel=init_vel, use_mean=False, split_args=used_split_args,
            use_case=use_case,
            segment_wise_init_pos= dataset["segment_wise_init_pos"],
            segment_wise_init_vel= dataset["segment_wise_init_vel"],
            split_start_indexes = dataset["split_start_indexes"],
            mp_distr_rel_pos=dataset["mp_distr_rel_pos"] if self.reference_split_args.get(
                "re_use_rand_coord_from_sampler_for_updates", False) else None,
            ref_time_list=ref_time

        )

        # Current state
        # [num_traj, num_segments, dim_state]
        if not self.update_policy_based_on_dataset_splits:
            c_state = states[:, seg_start_idx]
            # [num_segments] -> [num_traj, num_segments]
            seg_start_idx = util.add_expand_dim(seg_start_idx, [0],
                                                [num_traj])
            # [num_segments, num_actions] -> [num_traj, num_segments, num_actions]
            seg_actions_idx = util.add_expand_dim(seg_actions_idx, [0],
                                                  [num_traj])
        else:
            c_state = states[np.arange(states.shape[0])[:, None], seg_start_idx]

            # for relativ indexing: adapting idx_s, idx_a
            #seg_start_idx_rel = seg_start_idx[..., 0][..., None] * torch.ones_like(seg_start_idx)
            #seg_actions_idx_rel = seg_actions_idx[..., 0, :][..., None, :] * torch.ones_like(seg_actions_idx)
        kwargs = {}
        if self.reference_split_args.get("add_d_state_to_critic", False):
            if self.update_critic_based_on_dataset_splits:
                kwargs["idx_d"] = seg_start_idx
                kwargs["d_state"] = c_state
            else:
                kwargs["idx_d"] = torch.sum(seg_start_idx[..., None] >= dataset["split_start_indexes"][..., None, :],
                                            dim=-1)
                kwargs["d_state"] = states[np.arange(states.shape[0])[:, None], kwargs["idx_d"]]
        # [num_traj, num_segments, num_seg_actions]
        # vq -> q
        q1 = self.critic.critic(net=self.critic.net1, state=c_state,
                                actions=pred_seg_actions,
                                idx_s=seg_start_idx,
                                idx_a=seg_actions_idx, kwargs=kwargs)[..., 1:]
        if not self.critic.single_q:
            q2 = self.critic.critic(net=self.critic.net2, state=c_state,
                                    actions=pred_seg_actions,
                                    idx_s=seg_start_idx,
                                    idx_a=seg_actions_idx, kwargs=kwargs)[..., 1:]
        else:
            q2 = q1

        #do top erl style sample till index 99 [separate for speed purpose]
        if self.reference_split_args.get("policy_use_all_action_indexes", False) and not self.update_policy_based_on_dataset_splits:
            seg_actions_idx = idx_in_segments[-1, :-1] #only last one as we assert previous stops before
            valid_mask = seg_actions_idx < self.traj_length
            q1[:, -1, :] = q1[:, -1, :] * valid_mask
            q2[:, -1, :] = q2[:, -1, :] * valid_mask

        #mask out values from "invalid" actions from q -> set to 0 (only required for the "oversampling" in random_size_range)
        if self.update_policy_based_on_dataset_splits:
            #last segment ends at index 99 as we cut out the init states, e.g. else 19 actions for splitlength 20 (=state+19 actions)
            #problem: last start at 99 -> no action generated ==> exception to the rule
            #not required for prior states as the last action leads to the initial state of the next segment -> still valid
            #also all segment lengths != 0 should sample an action
            seg_end_idx = torch.cat([seg_start_idx[:, 1:],
                                     (states.size(1)) * torch.ones((seg_start_idx.size(0) , 1), dtype=torch.int,
                                                                 device=self.device)], axis=-1)
            seg_len_idx = seg_end_idx - seg_start_idx
            index_list = torch.arange(q2.size(-1), device=self.device)
            mask = index_list[None, None, :] < seg_len_idx[:, :, None]
            q2 = q2 * mask
            q1 = q1 * mask



        #truncated update needs to mask out out-truncated steps
        elif used_split_args["q_loss_strategy"] == "truncated":
            seg_actions_idx = idx_in_segments[..., :-1]
            num_seg_actions = seg_actions_idx.shape[-1]
            valid_mask = seg_actions_idx < self.traj_length
            times = self.sampler.get_times(dataset["episode_init_time"],
                                           self.sampler.num_times,
                                           self.traj_has_downsample)
            ref_time_list = times[0]
            ref_time = times[0]
            # [num_traj, traj_length] -> [num_traj, num_segments]
            # random_range case
            if times.shape[0] == seg_start_idx.shape[0]:
                times = times[np.arange(times.shape[0])[:, None], seg_start_idx]
            # all fixed sizes
            else:
                times = times[:, seg_start_idx]


            # [num_traj, num_segments] -> [num_traj, num_segments, num_seg_actions]
            action_times = util.add_expand_dim(times, [-1], [num_seg_actions])
            time_evo = torch.linspace(0, self.sampler.dt * (num_seg_actions - 1),
                                      num_seg_actions, device=self.device).float()
            action_times = action_times + time_evo
            policy_indexes = self.policy.get_param_indexes(action_times,
                                                    dataset["split_start_indexes"],
                                                    ref_time_list)
            init_time = times - self.sampler.dt
            policy_start_indexes = self.policy.get_param_indexes(init_time[..., None], dataset["split_start_indexes"],
                                   ref_time_list)[..., 0]
            overlap_mask = policy_start_indexes.unsqueeze(-1) == policy_indexes
            valid_mask = overlap_mask & valid_mask
            q2 = q2 * valid_mask
            q1 = q1 * valid_mask

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
        #self.sampler.load_rms(log_dir, epoch)
        self.load_pretrained_agent = True
