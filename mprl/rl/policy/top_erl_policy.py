import torch
from mprl.util.util_mp import *
import mprl.util as util
from .black_box_policy import BlackBoxPolicy
import numpy as np
import copy

class TopErlPolicy(BlackBoxPolicy):
    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 mean_net_args: dict,
                 variance_net_args: dict,
                 init_method: str,
                 out_layer_gain: float,
                 act_func_hidden: str,
                 act_func_last: str,
                 dtype: str = "torch.float32",
                 device: str = "cpu",
                 **kwargs):
        super().__init__(dim_in,
                         dim_out,
                         mean_net_args,
                         variance_net_args,
                         init_method,
                         out_layer_gain,
                         act_func_hidden,
                         act_func_last,
                         dtype,
                         device,
                         **kwargs)

        self.mp: ProDMP = get_mp(**kwargs["mp"])
        # self.mp.show_scaled_basis(True)
        self.num_dof = self.mp.num_dof

    def get_param_indexes(self, times, split_list, ref_time_list, split_starts=[]):
        if len(split_starts)==0:
            split_starts = np.array([0]*len(split_list))
            for i in range(1, len(split_list)):
                split_starts[i] = split_starts[i-1] + split_list[i-1]

        #invalid values get assigned first index here, then invalid times get added (total time + 1) -> never < sth in times
        valid = split_starts < ref_time_list.shape[-1]
        split_start_time_steps = ref_time_list[split_starts * valid].to(self.device) + torch.tensor(~valid, device=self.device) * (max(ref_time_list) + 1)

        if len(split_start_time_steps.shape) == 1:
            condition_reached =  times[..., None] >= split_start_time_steps
        else:
            condition_reached = times[..., None] >= split_start_time_steps[..., None, None, :]

        policy_indexes = torch.sum(condition_reached, axis=-1) - 1
        policy_indexes[policy_indexes < 0] = 0
        return policy_indexes

    def sample_splitted_partial_times(self, times,
                                      sample_func,
                                      splitting={"split_strategy": "n_equal_splits", "n_splits": 1},
                                      ref_time_list=torch.tensor(np.linspace(0,2,101)),
                                      **sample_func_kwargs):

        if "ref_time_list" in splitting:
            ref_time_list = splitting["ref_time_list"].to(self.device)
        times = times.to(self.device)

        split_size_list = util.get_splits(ref_time_list, splitting)
        split_start_indexes = [0] * len(split_size_list)
        for i in range(1, len(split_start_indexes)):
            split_start_indexes[i] = split_start_indexes[i-1] + split_size_list[i-1]

        if splitting["split_strategy"] != "random_size_range":
            policy_indexes = self.get_param_indexes(times, split_size_list, ref_time_list)
            policy_start_indexes = self.get_param_indexes(sample_func_kwargs["init_time"], split_size_list, ref_time_list)

            params = sample_func_kwargs.get("params")
            #
            #param_at_start = sample_func_kwargs.get("params")[..., policy_start_indexes, :]
            i = np.arange(params.size()[0])[:, None]
            j = np.arange(params.size()[1])[None, :]
            param_at_start = params[i,j, policy_start_indexes, :]

            #get indexes->timesteps I want to adress my parameter arry with shape (512, 3, 69) with the corresponding referenceat which the policy changes --> cut off point
            mask = policy_start_indexes.unsqueeze(-1) == policy_indexes

            iteration_sample_func_kwargs = dict()
            for key in sample_func_kwargs:
                iteration_sample_func_kwargs[key] = sample_func_kwargs[key]
            iteration_sample_func_kwargs["params"] = param_at_start


            if "params_L" in sample_func_kwargs:
                params_L = sample_func_kwargs.get("params_L")
                param_L_at_start = params_L[i, j, policy_start_indexes, :, :]
                iteration_sample_func_kwargs["params_L"] = param_L_at_start

            smp_pos_1, smp_vel_1 = \
                sample_func(times=times, **iteration_sample_func_kwargs)

            # zero out timesteps that should not have been used
            smp_pos = smp_pos_1.squeeze(2) * mask.unsqueeze(-1)
            smp_vel = smp_vel_1.squeeze(2) * mask.unsqueeze(-1)
        else:
            smp_pos, smp_vel = sample_func(times=times, **sample_func_kwargs)



        ####################################################################################################
        # create positions + velocities after timestep which corresponds to first change in parameter
        ####################################################################################################
        # strategies:   1) continue from replay_buffer initial conditions
        #                   [-> params of trajectory are guaranteed to make sense but path may have jumps]
        #               2) continue from last position reached so far
        #                   [-> smooth path but params may not be good for the current area]
        #               3) truncate -> just ignore everything to "not do anything incorrect" but limit information used
        #               4) overconfidence -> first set of parameters that is used in the sequence evaluates the rest aswell
        ####################################################################################################

        if splitting.get("q_loss_strategy", "truncated") == "overconfident":
            return smp_pos_1.squeeze(2), smp_vel_1.squeeze(2)

        if splitting.get("q_loss_strategy", "truncated") == "truncated":
            #repeat the last valid entry for the rest of the tensor in the respective axis
            #--> pos(time)=total angular position of robot(time) --> same = no change
            counted_mask = torch.arange(smp_pos.size(-2), device=self.device)[None, None, :] * mask
            maxed_mask = torch.where(counted_mask == torch.argmax(counted_mask, dim=-1).unsqueeze(-1), 1.0, 0.0)

            #residual to add = last valid entry of respective quantity
            last_entry_pos = smp_pos * maxed_mask.unsqueeze(-1)
            addition_pos = torch.sum(last_entry_pos, dim=-2)[..., None, :]

            #analogous for velocity
            last_entry_vel = smp_vel * maxed_mask.unsqueeze(-1)
            addition_vel = torch.sum(last_entry_vel, dim=-2)[..., None, :]

            #add residual to all entries that are not filled in already
            smp_pos = smp_pos + addition_pos * ~mask.unsqueeze(-1)
            smp_vel = smp_vel + addition_vel * ~mask.unsqueeze(-1)
            return smp_pos, smp_vel

        #limit number of iterations to not waste too much recources
        #example: splits [25,25,25,25] -> need 26 steps to be able to have 2 changes of reference policy parameters (just to save time with leaving out unnecessary iterations)
        max_remaining_iterations = max(times.size(-1) // max(1, np.min(split_size_list)) + 1, len(split_size_list)-1) #max(np.min(..), 1) so we do not divide by 0)
        for iteration in range(max_remaining_iterations):
            # new index = +=1 or max index in that axis if index+1 > max valid index
            policy_start_indexes = torch.minimum(policy_start_indexes + 1,
                                                 (len(split_size_list) -1) * torch.ones(policy_start_indexes.size(), device=self.device)).to(torch.int32)


            if splitting.get("q_loss_strategy", "truncated") == "start_unchanged":
                segment_init_indexes = policy_start_indexes.unsqueeze(-1).expand(-1, -1, splitting["segment_wise_init_pos"].size(-1))
                iteration_sample_func_kwargs["init_pos"] = torch.gather(splitting["segment_wise_init_pos"], index=segment_init_indexes.to(torch.int64), dim=-2)
                iteration_sample_func_kwargs["init_vel"] = torch.gather(splitting["segment_wise_init_vel"], index=segment_init_indexes.to(torch.int64), dim=-2)


            init_idx = torch.tensor(split_start_indexes, device=self.device)[policy_start_indexes]
            #not first timestep of sequence but one timestep of equal size before that
            iteration_sample_func_kwargs["init_time"] = ref_time_list[init_idx] - (ref_time_list[1] - ref_time_list[0])

            if splitting.get("q_loss_strategy", "truncated") == "continuing":
                counted_mask = torch.arange(smp_pos.size(-2), device=self.device)[None, None, :] * mask
                maxed_mask = torch.where(counted_mask == torch.argmax(counted_mask, dim=-1).unsqueeze(-1), 1.0, 0.0)

                # residual to add = last valid entry of respective quantity
                last_entry_pos = smp_pos * maxed_mask.unsqueeze(-1)
                iteration_sample_func_kwargs["init_pos"] = torch.sum(last_entry_pos, dim=-2)

                # analogous for velocity
                last_entry_vel = smp_vel * maxed_mask.unsqueeze(-1)
                iteration_sample_func_kwargs["init_vel"] = torch.sum(last_entry_vel, dim=-2)

            param_idx = policy_start_indexes.unsqueeze(-1).unsqueeze(-1).to(torch.int64).expand(-1, -1, 1, params.size(-1))
            param_update = torch.gather(params, dim=-2, index=param_idx).squeeze(-2)
            iteration_sample_func_kwargs["params"] = param_update

            if "params_L" in sample_func_kwargs:
                param_idx = policy_start_indexes.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(torch.int64).expand(-1, -1, 1,
                                                                                                    params.size(-1), params.size(-1))
                param_L_update = torch.gather(params_L, dim=-3, index=param_idx).squeeze(-3)
                iteration_sample_func_kwargs["params_L"] = param_L_update

            #basically overkill in calculation, everytime whole sequence
            smp_pos_add_i, smp_vel_add_i = \
                sample_func(times=times, **iteration_sample_func_kwargs)

            #mask off all entries, where a different params/params_L should have been used as 0 -> can be added on top
            non_zero_mask_results = policy_start_indexes.unsqueeze(-1) == policy_indexes
            #for next iteration if "continuing" methodology
            mask = non_zero_mask_results

            smp_pos_add = smp_pos_add_i.squeeze(2) * non_zero_mask_results.unsqueeze(-1)
            smp_vel_add = smp_vel_add_i.squeeze(2) * non_zero_mask_results.unsqueeze(-1)


            smp_pos = smp_pos + smp_pos_add
            smp_vel = smp_vel + smp_vel_add


        return smp_pos, smp_vel


    def sample_splitted(self, times, sample_func, splitting={"split_strategy": "n_equal_splits", "n_splits": 1}, **sample_func_kwargs):
        '''
            called in sample to sample with changing "reference points" (defined in times)
        Args:
            times: timepoints to evaluate pos+vel
            sample_func: function to sample trajectories from
            splitting: dict: [split_strategy:str, split options (different args for different split_stragegies]
                split_strategy:
                    "fixed_max_size":       split in segments, maximally as big as the arg
                    "n_equal_splits":       split into n equal parts (+one for the rest if num_samples is not a multiple)
                    "random_size_range":    randomized segment sizes, uniformly distributed from x to y (different values each call)
                    "random_gauss":         randomized segment sizes, gaussian distributed around mean with std

                    -> for every strategy if it does not fully cover the data and the next segment would "overcover" it a smaller segment is added in the end
                args:
                    split_size: int         #required for split strategy "fixed_max_size"
                    n_splits: int           #required for split_strategy "n_equal_splits"
                    size_range: [int,int]   #required for split_strategy "random_size_range"
                    mean_std: [int,int]     #required for split_strategy "random_gauss"
            **sample_func_kwargs: all arguments of sample func (except times) [params, params_L, .... ]

        Returns:
            pos and vel trajectories
            shape 2 x [*add_dim, num_samples, num_times, num_dof]
        '''
        init_time = sample_func_kwargs["init_time"]
        num_smp = sample_func_kwargs.get("num_smp", 1)

        split_size_list = util.get_splits(times, splitting)
        smp_pos, smp_vel = \
            sample_func(times=times[..., :split_size_list[0]], **sample_func_kwargs)

        iteration_sample_func_kwargs = sample_func_kwargs.copy()
        iteration_sample_func_kwargs.pop('init_pos', None)
        iteration_sample_func_kwargs.pop('init_vel', None)
        iteration_sample_func_kwargs.pop('init_time', None)
        if "num_smp" in iteration_sample_func_kwargs.keys():
            iteration_sample_func_kwargs['num_smp'] = 1 #only 1 sample per intermediate point/step, else we have exponentially many


        # if we have multiple stops --> sample sub trajectories from there
        # initial conditions: time and pos+vel after first n time steps (after first sample_trajectories call)
        smp_pos_xn = smp_pos
        smp_vel_xn = smp_vel

        #curr_init_time = init_time
        curr_time_idx = 0 #will be increased in loop before usage
        for time_split_idx in range(1, len(split_size_list)):
            #give just large enough tensors for pos and vel (no additional empty entries)
            curr_split_size = split_size_list[time_split_idx]
            if curr_split_size == 0:
                continue
            #pos_xn = torch.empty_like(smp_pos)[..., :curr_split_size, :]
            #vel_xn = torch.empty_like(smp_vel)[..., :curr_split_size, :]
            
            # start conditions for current loop
            prev_pos_tensor = smp_pos_xn #last timestep, all samples
            prev_vel_tensor = smp_vel_xn
            curr_time_idx += split_size_list[time_split_idx-1]

            for sample_n in range(num_smp):
                smp_pos_xi, smp_vel_xi = \
                    sample_func(times=times[..., curr_time_idx : curr_time_idx + curr_split_size ] if splitting["correction_completion"]== "current_idx" else times[..., 0 : 0 + curr_split_size ],
                                init_pos=prev_pos_tensor[..., -1, :].squeeze(-2) if num_smp == 1 \
                                    else prev_pos_tensor[..., sample_n, -1, :].squeeze(-2),
                                init_vel=prev_vel_tensor[..., -1, :].squeeze(-2) if num_smp == 1 \
                                    else prev_vel_tensor[..., sample_n, -1, :].squeeze(-2),
                                init_time=times[..., curr_time_idx - 1].squeeze(-1) if splitting["correction_completion"]== "current_idx" else init_time,
                                **iteration_sample_func_kwargs)

                # most likely if case not needed as output shape will be [..., 1, num_times, num_dof]
                if num_smp == 1:
                    smp_pos_xn = smp_pos_xi
                    smp_vel_xn = smp_vel_xi
                else:
                    #adapt the positions to not add exra steps that are not used // keep all states if previous was smaller
                    if curr_split_size != smp_pos_xn.size(-2) and sample_n == 0:
                        smp_pos_xn = torch.zeros(*smp_pos_xn.size()[:-2], curr_split_size, smp_pos_xn.size(-1))
                        smp_vel_xn = torch.zeros(*smp_vel_xn.size()[:-2], curr_split_size, smp_vel_xn.size(-1)) #smp_vel_xn[..., :curr_split_size, :]
                    smp_pos_xn[..., sample_n, :, :] = smp_pos_xi
                    smp_vel_xn[..., sample_n, :, :] = smp_vel_xi

            smp_pos = torch.cat([smp_pos, smp_pos_xn], dim=-2)
            smp_vel = torch.cat([smp_vel, smp_vel_xn], dim=-2)

        return smp_pos, smp_vel

    def sample_once(self, times, sample_func, splitting=None, **sample_func_kwargs):
        smp_pos, smp_vel = \
            sample_func(times=times, **sample_func_kwargs)

        return smp_pos, smp_vel

    def sample(self, require_grad, params_mean, params_L,
               times, init_time, init_pos, init_vel, use_mean=False,
               num_samples=1, split_args={"split_strategy": "n_equal_splits", "n_splits": 1}, **kwargs):
        """
        Given a segment-wise state, rsample an action
        Args:
            require_grad: require gradient from the samples
            params_mean: mean of the ProDMP parameters
            params_L: cholesky decomposition of the ProDMP parameters covariance

            times: trajectory times points --> tensor "list" of timepoints

            init_time: initial condition time
            init_pos: initial condition pos
            init_vel: initial condition vel
            use_mean: if True, return the mean action

            Shape of params_mean:
            [*add_dim, num_dof * num_basis_g]

            Shape of params_L:
            [*add_dim, num_dof * num_basis_g, num_dof * num_basis_g]

            Shape of times:
            [*add_dim, num_times]

            Shape of init_time:
            [*add_dim]

            Shape of init_pos:
            [*add_dim, num_dof]

            Shape of init_vel:
            [*add_dim, num_dof]

        Returns:
            smp_pos: sampled traj pos
            smp_vel: sampled traj vel

            Shape of smp_traj if num_samples == 1:
            [*add_dim, num_times, num_dof * 2]
            else if num_samples > 1:
            [*add_dim, num_samples, num_times, num_dof * 2]

        """
        for key in kwargs:
            split_args[key] = kwargs[key]
        use_case = split_args.get("use_case", "sampler")
        if use_case == "agent":
            split_sample_func = self.sample_splitted_partial_times
        #elif split_args["split_strategy"] == "n_equal_splits" and  split_args["n_splits"]== 1:
        else:
            split_sample_func = self.sample_once
        #else:
        #    split_sample_func = self.sample_splitted


        if not use_mean:
            smp_pos, smp_vel = split_sample_func(times=times,
                                                    sample_func=self.mp.sample_trajectories,
                                                    params=params_mean,
                                                    params_L=params_L,
                                                    init_time=init_time,
                                                    init_pos=init_pos,
                                                    init_vel=init_vel,
                                                    num_smp=num_samples,
                                                    flat_shape=False,
                                                    splitting=split_args)
            '''
            # Sample trajectory
            smp_pos, smp_vel = \
                self.mp.sample_trajectories(times=times[0], params=params_mean,
                                            params_L=params_L,
                                            init_time=init_time,
                                            init_pos=init_pos,
                                            init_vel=init_vel,
                                            num_smp=num_samples,
                                            flat_shape=False)

            # squeeze the dimension of sampling
            '''
            if num_samples == 1 and use_case != "agent":
                smp_pos, smp_vel = smp_pos.squeeze(-3), smp_vel.squeeze(-3)
            else:
                pass

        else:
            #smp_pos = self.mp.get_traj_pos(times=times, params=params_mean,
            #                               init_time=init_time,
            #                               init_pos=init_pos,
            #                               init_vel=init_vel, flat_shape=False)
            # smp_vel = self.mp.get_traj_vel()
            pos_vel_func = lambda times, params, init_time, init_pos, init_vel, flat_shape: \
                [
                    self.mp.get_traj_pos(times, params, init_time, init_pos, init_vel, flat_shape), 
                    self.mp.get_traj_vel(times, params, init_time, init_pos, init_vel, flat_shape)
                ]
            smp_pos, smp_vel = split_sample_func(times=times,
                                                    sample_func=pos_vel_func,
                                                    params=params_mean,
                                                    init_time=init_time,
                                                    init_pos=init_pos,
                                                    init_vel=init_vel,
                                                    flat_shape=False,
                                                    splitting=split_args
                                                    )

            if num_samples != 1:
                smp_pos = util.add_expand_dim(smp_pos, [-3], [num_samples])
                smp_vel = util.add_expand_dim(smp_vel, [-3], [num_samples])

        # Remove gradient if necessary
        if not require_grad:
            smp_pos = smp_pos.detach()
            smp_vel = smp_vel.detach()

        # Concatenate position and velocity
        smp_traj = torch.cat([smp_pos, smp_vel], dim=-1)

        return smp_traj

    def fix_relative_goal_for_segments(self,
                                       params: torch.Tensor,
                                       traj_init_pos: torch.Tensor,
                                       segments_init_pos: torch.Tensor):
        """

        Args:
            params: ProDMP parameters, [*add_dim, num_segments, num_weights]
            traj_init_pos: [*add_dim, num_dof]
            segments_init_pos: [*add_dim, num_segments, num_dof]

        Returns:

        """
        relative_goal = self.mp.relative_goal

        if relative_goal:
            # [*add_dim, num_segments, num_dof]
            delta_relative_goal \
                = segments_init_pos - traj_init_pos[..., None, :]
            num_basis_g = self.mp.num_basis_g

            # As, abs_goal = rel_goal + traj_init_pos
            # Set: delta = seg_init_pos - traj_init_pos
            # -> traj_init_pos = seg_init_pos - delta
            # So: abs_goal = rel_goal + seg_init_pos - delta
            #              = fix_rel_goal + seg_init_pos
            # So, fix_rel_goal = rel_goal - delta

            params = params.clone()
            # [*add_dim, num_segments, num_dof]
            params[..., num_basis_g - 1::num_basis_g] \
                = (params[...,
                   num_basis_g - 1::num_basis_g] - delta_relative_goal)

            return params

        else:
            return params
