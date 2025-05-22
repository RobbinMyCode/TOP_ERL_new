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

    def get_param_indexes(self, times, split_list, ref_time_list):
        split_starts = [0]*len(split_list)
        for i in range(1, len(split_list)):
            split_starts[i] = split_starts[i-1] + split_list[i]

        split_start_time_steps = ref_time_list[split_starts].to(self.device)

        condition_reached =  times[..., None] >= split_start_time_steps
        policy_indexes = torch.sum(condition_reached, axis=-1) - 1
        return policy_indexes

    def sample_splitted_partial_times(self, times,
                                      sample_func,
                                      splitting={"split_strategy": "n_equal_splits", "n_splits": 1},
                                      ref_time_list=torch.tensor(np.linspace(0,2,101)),
                                      **sample_func_kwargs):

        if "ref_time_list" in sample_func_kwargs:
            ref_time_list = sample_func_kwargs["ref_time_list"].to(self.device)
        times = times.to(self.device)

        split_size_list = util.get_splits(times, splitting)
        policy_indexes = self.get_param_indexes(times, split_size_list, ref_time_list)
        policy_start_indexes = self.get_param_indexes(sample_func_kwargs["init_time"], split_size_list, ref_time_list)

        params = sample_func_kwargs.get("params")
        #

        #param_at_start = sample_func_kwargs.get("params")[..., policy_start_indexes, :]
        i = np.arange(params.size()[0])[:, None]
        j = np.arange(params.size()[1])[None, :]
        param_at_start = params[i,j, policy_start_indexes, :]



        #get indexes->timesteps at which the policy changes --> cut off point
        mask = policy_start_indexes.unsqueeze(-1) < policy_indexes
        first_diff = torch.argmax(mask.int(), dim=-1) #argmax returns first largest entry (either 0=False or 1=True) --> first match
        first_diff[~mask.any(dim=-1)] = policy_indexes.size(-1)


        iteration_sample_func_kwargs = copy.deepcopy(sample_func_kwargs)
        iteration_sample_func_kwargs["params"] = param_at_start


        if "params_L" in sample_func_kwargs:
            params_L = sample_func_kwargs.get("params_L")
            param_L_at_start = params_L[i, j, policy_start_indexes, :, :]
            iteration_sample_func_kwargs["params_L"] = param_L_at_start

        smp_pos, smp_vel = \
            sample_func(times=times, **iteration_sample_func_kwargs)

        #zero out timesteps that should not have been used
        smp_pos = smp_pos.squeeze(2) * ~mask.unsqueeze(-1)
        smp_vel= smp_vel.squeeze(2) * ~mask.unsqueeze(-1)


        for time_split in split_size_list:
            #SOLUTION OF DGL SHOULD BE INDEPENDEND ON WHIH START CONDITION ONE USES, SO WE CAN JUST REUSE THE PREVIOUS ONES
            #new breaking point 1 index higher, else the same
            policy_start_indexes += 1
            mask = policy_start_indexes.unsqueeze(-1) < policy_indexes

            param_update = params[i, j, policy_start_indexes, :]
            iteration_sample_func_kwargs["params"] = param_update
            if "params_L" in sample_func_kwargs:
                param_L_update = params_L[i, j, policy_start_indexes, :, :] #i,j dummy vars only ensure correct indexing
                iteration_sample_func_kwargs["params_L"] = param_L_update

            times_masked = torch.where(mask, times, times.size(-1))
            first_valid_times, _ = times_masked.min(dim=2)

            #inital conditions stay the same, only mean and var (param, param_L)
            #basically overkill in calculation, everytime whole sequence
            smp_pos_add, smp_vel_add = \
                sample_func(times=times, **iteration_sample_func_kwargs)

            #mask off all entries, where a different params/params_L should have been used as 0 -> can be added on top
            non_zero_mask_results = policy_start_indexes.unsqueeze(-1) == policy_indexes
            smp_pos_add = smp_pos_add.squeeze(2) * non_zero_mask_results.unsqueeze(-1)
            smp_vel_add = smp_vel_add.squeeze(2) * non_zero_mask_results.unsqueeze(-1)


            smp_pos += smp_pos_add
            smp_vel += smp_vel_add


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

        iteration_sample_func_kwargs = sample_func_kwargs
        iteration_sample_func_kwargs.pop('init_pos', None)
        iteration_sample_func_kwargs.pop('init_vel', None)
        iteration_sample_func_kwargs.pop('init_time', None)
        if "num_smp" in iteration_sample_func_kwargs.keys():
            iteration_sample_func_kwargs['num_smp'] = 1 #only 1 sample per intermediate point/step, else we have exponentially many


        # if we have multiple stops --> sample sub trajectories from there #TODO pararellize in case
        # initial conditions: time and pos+vel after first n time steps (after first sample_trajectories call)
        smp_pos_xn = smp_pos
        smp_vel_xn = smp_vel

        #curr_init_time = init_time
        curr_time_idx = 0 #will be increased in loop before usage
        for time_split_idx in range(1, len(split_size_list)):
            #give just large enough tensors for pos and vel (no additional empty entries)
            curr_split_size = split_size_list[time_split_idx]
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
        else:
            split_sample_func = self.sample_splitted


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
