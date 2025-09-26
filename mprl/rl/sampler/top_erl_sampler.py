import numpy as np
import torch

import mprl.util as util
from mprl.util import assert_shape
from mprl.util import to_np
from mprl.util import to_ts
from mprl.rl.sampler import BlackBoxSampler
import copy

class TopErlSampler(BlackBoxSampler):
    def __init__(self,
                 env_id: str,
                 num_env_train: int = 1,
                 num_env_test: int = 1,
                 episodes_per_train_env: int = 1,
                 episodes_per_test_env: int = 1,
                 dtype: str = "torch.float32",
                 device: str = "cpu",
                 seed: int = 1,
                 **kwargs):
        super().__init__(env_id, num_env_train, num_env_test,
                         episodes_per_train_env, episodes_per_test_env, dtype,
                         device, seed, **kwargs)

        # Get time step and episode length
        self._dt = self.debug_env.envs[0].dt
        self._num_times = self.debug_env.envs[0].spec.max_episode_steps

        #get reference split args
        self.reference_split_args = kwargs.get("reference_split", {'split_strategy': 'fixed_max_size', 'split_size': 1e100})

        #decide maximum info for forcing terms
        self.include_pos_in_forcing_terms = kwargs.get("include_pos_in_forcing_terms", False)


        # If episode is too long, such as Metaworld, we can down sample the
        # trajectory when update the policy and critic
        self.traj_downsample_factor = kwargs.get("traj_downsample_factor", None)
        if self.traj_downsample_factor:
            assert self._num_times % self.traj_downsample_factor == 0, \
                "down sample can't be divided by num_times"
            self.reward_scaling = kwargs.get("reward_scaling",
                                             1.0 / self.traj_downsample_factor)
        else:
            self.reward_scaling = 1

    @property
    def dt(self):
        if self.traj_downsample_factor is None:
            return self._dt
        else:
            return self._dt * self.traj_downsample_factor

    @property
    def num_times(self):
        if self.traj_downsample_factor is None:
            return self._num_times
        else:
            return int(self._num_times / self.traj_downsample_factor)

    def get_times(self, init_time, num_times, down_sample=False):
        """
        Get time steps for traj generation, using low level time step length

        Args:
            init_time: initial time
            num_times: number of time steps
            down_sample: if the time sequence is downsampled or not

        Returns:
            time sequence in a tensor
        """

        dt = self.dt if down_sample else self._dt

        # Low level time steps
        return util.tensor_linspace(start=init_time + dt,
                                    end=init_time + num_times * dt,
                                    steps=num_times).T

    @torch.no_grad()
    def run(self,
            training: bool,
            policy,
            critic=None,
            deterministic: bool = False,
            render: bool = False,
            task_specified_metrics: list = None):
        """
        Sample trajectories

        Args:
            training: True for training, False for evaluation
            policy: policy model to get actions from
            deterministic: evaluation only, if the evaluation is deterministic
            render: evaluation only, whether render the environment
            task_specified_metrics: task specific metrics

        Returns:
            rollout results
        """
        # Training or evaluation
        if training:
            assert deterministic is False and render is False
            envs = self.train_envs
            episode_init_state = envs.reset()
            num_env = self.num_env_train
            ep_per_env = self.episodes_per_train_env
        else:
            envs = self.test_envs
            episode_init_state = envs.reset()
            num_env = self.num_env_test
            if render and num_env == 1:
                envs.render()
            ep_per_env = self.episodes_per_test_env

        # Determine the dimensions
        dim_obs = self.observation_space.shape[-1]
        dim_mp_params = policy.dim_out

        num_times = self._num_times
        num_dof = policy.num_dof

        # Storage for rollout results
        list_episode_init_time = list()
        list_episode_init_pos = list()
        list_episode_init_vel = list()


        list_segment_wise_init_pos = list()
        list_segment_wise_init_vel = list()


        list_episode_reward = list()

        list_step_states = list()
        list_step_actions = list()
        list_step_rewards = list()
        list_step_dones = list()
        list_step_desired_pos = list()
        list_step_desired_vel = list()
        list_split_indexes = list()
        list_episode_init_idx = list()

        # Storage for policy results
        list_episode_params_mean = list()  # Policy mean
        list_episode_params_L = list()  # Policy covariance cholesky

        # Storage task specified metrics
        if self.task_specified_metrics is not None:
            dict_task_specified_metrics = dict()
            for metric in self.task_specified_metrics:
                dict_task_specified_metrics[metric] = list()
        else:
            dict_task_specified_metrics = dict()

        # Env interaction steps
        num_total_env_steps = 0

        # Main rollout loop
        for ep_idx in range(ep_per_env):

            # Initial conditions
            episode_init_state = to_ts(episode_init_state,
                                       self.dtype, self.device)

            episode_init_time = episode_init_state[..., -num_dof * 2 - 1]
            episode_init_pos = episode_init_state[..., -num_dof * 2: -num_dof]
            episode_init_vel = episode_init_state[..., -num_dof:]
            assert_shape(episode_init_time, [num_env])
            assert_shape(episode_init_pos, [num_env, num_dof])
            assert_shape(episode_init_vel, [num_env, num_dof])

            list_episode_init_time.append(episode_init_time)
            list_episode_init_pos.append(episode_init_pos)
            list_episode_init_vel.append(episode_init_vel)

            # We put the robot pos and vel at t=0 into the state, because
            # constraint of Gym Reset API does not have info in return value
            # Storage for policy results
            list_episode_params_mean = list()  # Policy mean
            list_episode_params_L = list()  # Policy covariance cholesky
            list_segment_wise_init_pos_it = list()
            list_segment_wise_init_vel_it = list()

            # Get time steps for traj generation
            step_times = self.get_times(episode_init_time, num_times,
                                        down_sample=False)
            '''
                Currently all envs in parallel get same split_index
                -- should be fine (its only 4 in parallel vs 512 per update)
            '''
            split_list = util.get_splits(step_times, self.reference_split_args)
            # Sample a trajectory using the predicted MP parameters
            #actual start
            split_start_idx = 0
            #referenced one for replanning
            ref_split_start_idx = 0
            list_episode_init_idx.append(
                torch.zeros(num_env, dtype=torch.long, device=self.device)
            )

            list_split_indexes = copy.deepcopy(split_list)
            list_split_indexes[0] = 0
            for i in range(1, len(split_list)):
                list_split_indexes[i] = split_list[i-1] + list_split_indexes[i-1] #steps from previous steps + sum of steps before

                #avoid out of bounds for tailing zero splits
                if list_split_indexes[i] == np.sum(split_list):
                    list_split_indexes[i] = list_split_indexes[i] -1
            assert list_split_indexes[-1] == min(num_times - split_list[-1], num_times - 1)


            for split_iteration, split in enumerate(split_list):
                #repetition of episode_inits from new assessment further down
                #and torchify it for use
                #its a repetition the first time, but then neccesary
                episode_init_state = to_ts(episode_init_state,
                                           self.dtype, self.device)

                episode_init_time = episode_init_state[..., -num_dof * 2 - 1]
                episode_init_pos = episode_init_state[..., -num_dof * 2: -num_dof]
                episode_init_vel = episode_init_state[..., -num_dof:]
                if split_iteration == 0:
                    segment_init_vel = torch.zeros((len(split_list), *episode_init_vel.size()),
                                                   device=self.device)
                    segment_init_pos = torch.zeros((len(split_list), *episode_init_pos.size()),
                                                   device=self.device)

                if split_iteration < len(split_list) - 1:
                    segment_init_vel[split_iteration] = episode_init_vel
                    segment_init_pos[split_iteration] = episode_init_pos
                else:
                    list_segment_wise_init_vel_it.append(segment_init_vel)
                    list_segment_wise_init_pos_it.append(segment_init_pos)
                    segment_init_vel = torch.zeros((len(split_list), *episode_init_state[..., -num_dof * 2 - 1].size()), device=self.device)
                    segment_init_pos = torch.zeros((len(split_list), *episode_init_state[..., -num_dof * 2 - 1].size()), device=self.device)

                # Policy prediction, we remove the desired position and velocity
                # from observations
                #FIRST TWO CASES ONLY IF GRIPPER POSITION IS USED AS VARIABLE FOR GENERATED FORCES (if no pos given: set to 0, else take last)
                if split_iteration == 0 and ep_idx == 0 and self.include_pos_in_forcing_terms:
                    episode_params_mean, episode_params_L = \
                        policy.policy(torch.cat([episode_init_state[..., :-num_dof * 2],
                                                torch.zeros((*episode_init_state.size()[:-1], num_dof), device=self.device)],
                                                dim=-1))
                elif self.include_pos_in_forcing_terms:
                    episode_params_mean, episode_params_L = \
                        policy.policy(torch.cat([episode_init_state[..., :-num_dof * 2],
                                                 step_actions[..., -1,  :step_actions.size(-1)//2]], dim=-1))
                else:
                    episode_params_mean, episode_params_L = \
                        policy.policy(episode_init_state[..., :-num_dof * 2])

                assert_shape(episode_params_mean, [num_env, dim_mp_params])
                assert_shape(episode_params_L,
                             [num_env, dim_mp_params, dim_mp_params])
                if split_iteration == 0:
                    episode_params_mean_tensor = torch.ones((num_env, len(split_list), dim_mp_params))
                    episode_params_L_tensor = torch.zeros((num_env, len(split_list), dim_mp_params, dim_mp_params))
                episode_params_mean_tensor[:, split_iteration, :] = episode_params_mean
                episode_params_L_tensor[:, split_iteration, :, :] = episode_params_L

                if split_iteration == len(split_list) - 1:
                    list_episode_params_mean.append(episode_params_mean_tensor)
                    list_episode_params_L.append(episode_params_L_tensor)

                if split != 0:
                    step_actions = policy.sample(require_grad=False,
                                                 params_mean=episode_params_mean,
                                                 params_L=episode_params_L,
                                                 times=step_times[..., ref_split_start_idx : ref_split_start_idx+split],
                                                 init_time=episode_init_time if split_iteration==0 or self.reference_split_args["correction_completion"] == "as_zero" \
                                                     else step_times[..., ref_split_start_idx-1],
                                                 init_pos=episode_init_pos,
                                                 init_vel=episode_init_vel,
                                                 use_mean=deterministic,
                                                 ref_time = step_times[0],
                                                 split_args={"split_strategy": "n_equal_splits", "n_splits": 1, "random_permute_splits": False})

                # STEP_ACTION = NO MOVEMENT / DEFAULT POSITION
                else:
                    step_actions = torch.zeros((*step_times.size()[:-1], 1, num_dof*2), device=self.device)



                if self.reference_split_args["correction_completion"] == "current_idx":
                    ref_split_start_idx += split
                elif self.reference_split_args["correction_completion"] == "as_zero":
                    pass
                else:
                    raise ValueError(f"Invalid Correction completion {self.reference_split_args['correction_completion']}. "
                                              f"Valid are 'current_idx' to use time steps as they are assigned for replanning"
                                              " or 'as_zero' to set the replanning reference point at timestep 0 "
                                              "[so created trajectory full would be > total steps] and gets cut accordingly")
                assert_shape(step_actions, [num_env, max(split,1), num_dof * 2])

                if split_iteration == 0:
                    step_actions_tensor = torch.zeros(*step_actions.size()[:-2], num_times, step_actions.size(-1))

                #need to check split == 0 everywhere, as minimum one step_action is sampled
                if split != 0:
                    step_actions_tensor[..., split_start_idx: split_start_idx+split, :] = step_actions


                # Observation, reward, done, info
                # Here, the gymnasium step() API get suppressed by stable-baseline3
                # So we get 4 return elements rather than 5
                #cant do that if we have a 0 split
                if split != 0:
                    next_episode_init_state, episode_reward, _, step_infos = \
                        envs.step(to_np(step_actions))

                # Step states and values, only useful when an actual split is done, not an empty one
                if split != 0:
                    step_states = util.get_item_from_dicts(step_infos, "step_states")
                    step_states = to_ts(np.asarray(step_states),
                                        self.dtype, self.device)
                    assert_shape(step_states, [num_env, split, dim_obs])

                    #Optional augment step_states to include step_actions -> condition parameters(-> forces) on initial position
                    #only position --> first half of step_actions (second half = velocities)
                    if self.include_pos_in_forcing_terms:
                        step_states = torch.cat([step_states, step_actions[..., :step_actions.size(-1)//2]], dim=-1)

                    # Exclude robot desired pos and vel from state (only add at valid splits, else no information gained)

                    if np.sum(split_list[:split_iteration]) == 0 and split != 0:
                        # Include the initial state into step states
                        if self.include_pos_in_forcing_terms:
                            init_add = torch.cat([episode_init_state, step_actions[..., 0,  :step_actions.size(-1)//2]], dim=-1)
                        else:
                            init_add = episode_init_state

                        step_states = torch.cat([init_add[:, None], step_states], dim=-2)
                        step_states_tensor = torch.zeros(*step_states.size()[:-2], num_times+1, step_states.size(-1))
                        step_states_tensor[..., split_start_idx:split_start_idx + split + 1, :] = step_states
                    else:
                        step_states_tensor[..., split_start_idx:split_start_idx + split, :] = step_states




                # Overwrite the initial state (if any valid steps are done)
                if split != 0:
                    episode_init_state = next_episode_init_state

                    # Step rewards
                    step_rewards = util.get_item_from_dicts(step_infos, "step_rewards")
                    step_rewards = to_ts(np.asarray(step_rewards),
                                         self.dtype, self.device)
                    assert_shape(step_rewards, [num_env, max(split,1)])

                    # Turn Non-MDP rewards into MDP rewards if necessary
                    step_rewards = util.make_mdp_reward(task_id=self.env_id,
                                                        step_rewards=step_rewards,
                                                        step_infos=step_infos,
                                                        dtype=self.dtype,
                                                        device=self.device)

                    # Scale the reward, only when trajectory is down sampled
                    # during policy and critic update
                    step_rewards = step_rewards * self.reward_scaling

                # Store step rewards
                if np.sum(split_list[:split_iteration]) == 0 and split != 0:
                    step_rewards_tensor = torch.zeros(*step_rewards.size()[:-1], num_times)

                if split != 0:
                    step_rewards_tensor[..., split_start_idx:split_start_idx+split] = step_rewards

                    #only if split != 0, bcause else there will not be a reward
                    # Episode rewards
                    #episode reward only exists for reaching goal -> sufficient as reward of last segment
                    assert_shape(episode_reward, [num_env])
                    episode_reward = to_ts(np.asarray(episode_reward),
                                           self.dtype, self.device) * int(split!=0) * int(split != 0)

                if np.sum(split_list[:split_iteration]) == 0 and split != 0: #first non-zero element
                    list_episode_reward.append(episode_reward)
                elif split != 0:
                    list_episode_reward[-1] += episode_reward

                # Step dones, adapt to new gymnasium interface, only makes sense for nonzero splits
                if split != 0:
                    step_terminations = util.get_item_from_dicts(step_infos,
                                                                 "step_terminations")
                    step_truncations = util.get_item_from_dicts(step_infos,
                                                                "step_truncations")

                    step_terminations = to_ts(np.asarray(step_terminations),
                                              torch.bool, self.device)
                    step_truncations = to_ts(np.asarray(step_truncations),
                                             torch.bool, self.device)

                    step_dones = torch.logical_or(step_terminations, step_truncations)

                    assert_shape(step_dones, [num_env, split])

                if np.sum(split_list[:split_iteration]) == 0 and split != 0:
                    step_dones_tensor = torch.zeros(*step_dones.size()[:-1], num_times)

                if split != 0:
                    step_dones_tensor[..., split_start_idx:split_start_idx+split] = step_dones

                #####################################
                #START INDEX FOR NEXT SPLIT ITERATION
                split_start_idx += split
                #####################################

                # Update training steps (0 for split==0, but step_info not given
                if split != 0:
                    episode_length = util.get_item_from_dicts(
                        step_infos, "segment_length")
                    num_total_env_steps += np.asarray(episode_length).sum()

                # Task specified metrics
                if split != 0:
                    if self.task_specified_metrics is not None:
                        for metric in self.task_specified_metrics:
                            metric_value = \
                                util.get_item_from_dicts(step_infos,
                                                         metric, lambda x: x[-1])

                            metric_value = \
                                to_ts(metric_value, self.dtype, self.device)
                            #task specific metrics are usually only dependend on last bit (success, ...)
                            if split_iteration == len(split_list) - 1 or np.sum(split_list[split_iteration+1:]) == 0:
                                dict_task_specified_metrics[metric].append(metric_value)

            list_step_states.append(step_states_tensor[..., :-num_dof * 2])
            list_step_desired_pos.append(
                step_states_tensor[..., -num_dof * 2:-num_dof])
            list_step_desired_vel.append(step_states_tensor[..., -num_dof:])
            list_step_actions.append(step_actions_tensor)
            list_step_rewards.append(step_rewards_tensor)
            list_step_dones.append(step_dones_tensor)
            list_segment_wise_init_pos.append(torch.cat(list_segment_wise_init_pos_it, dim=0).permute(1,0,2) )
            list_segment_wise_init_vel.append(torch.cat(list_segment_wise_init_vel_it, dim=0).permute(1, 0, 2))

        # Step-wise data
        step_actions = torch.cat(list_step_actions, dim=0)
        step_states = torch.cat(list_step_states, dim=0)[:, :-1]
        step_desired_pos = torch.cat(list_step_desired_pos, dim=0)[:, :-1]
        step_desired_vel = torch.cat(list_step_desired_vel, dim=0)[:, :-1]
        step_rewards = torch.cat(list_step_rewards, dim=0)
        step_dones = torch.cat(list_step_dones, dim=0)
        step_time_limit_dones = torch.zeros_like(step_dones)

        # Down sample the trajectory to make it efficient for very long sequence
        if self.traj_downsample_factor is not None:
            step_actions = step_actions[:, ::self.traj_downsample_factor]
            step_states = step_states[:, ::self.traj_downsample_factor]
            step_desired_pos = step_desired_pos[:,
                               ::self.traj_downsample_factor]
            step_desired_vel = step_desired_vel[:,
                               ::self.traj_downsample_factor]
            step_rewards = (
                step_rewards.reshape(num_env, -1,
                                     self.traj_downsample_factor).sum(dim=-1))
            step_dones = (
                step_dones.reshape(num_env, -1,
                                   self.traj_downsample_factor).any(dim=-1))
            step_time_limit_dones = step_time_limit_dones[:,
                                    ::self.traj_downsample_factor]

        # Form up return dictionary
        results = dict()
        results["step_actions"] = step_actions
        results["step_states"] = step_states
        results["step_desired_pos"] = step_desired_pos
        results["step_desired_vel"] = step_desired_vel
        results["step_rewards"] = step_rewards
        results["step_dones"] = step_dones
        results["step_time_limit_dones"] = step_time_limit_dones

        results["split_start_indexes"] = torch.tensor(list_split_indexes)[None, :]  ### these are start-indexes of the specific split (e.g. [0, 25, 50, 75]) for a const splitsize of 25
        results["total_time_list"] = step_times.to("cpu").numpy()


        results["episode_reward"] = torch.cat(list_episode_reward, dim=0)

        results["episode_init_time"] = torch.cat(list_episode_init_time, dim=0)
        results["episode_init_pos"] = torch.cat(list_episode_init_pos, dim=0)
        results["episode_init_vel"] = torch.cat(list_episode_init_vel, dim=0)

        results["segment_wise_init_pos"] = torch.cat(list_segment_wise_init_pos, dim=0)
        results["segment_wise_init_vel"] = torch.cat(list_segment_wise_init_vel, dim=0)

        results["episode_init_idx"] = torch.cat(list_episode_init_idx, dim=0)
        results["episode_params_mean"] = \
            torch.cat(list_episode_params_mean, dim=0)
        results["episode_params_L"] = torch.cat(list_episode_params_L, dim=0)

        if self.task_specified_metrics:
            for metric in self.task_specified_metrics:
                results[metric] = torch.cat(dict_task_specified_metrics[metric],
                                            dim=0)

        return results, num_total_env_steps
