import torch
from mprl.util.util_mp import *
from .black_box_policy import BlackBoxPolicy


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

    def sample(self, require_grad, params_mean, params_L,
               times, init_time, init_pos, init_vel, use_mean=False,
               num_samples=1):
        """
        Given a segment-wise state, rsample an action
        Args:
            require_grad: require gradient from the samples
            params_mean: mean of the ProDMP parameters
            params_L: cholesky decomposition of the ProDMP parameters covariance
            times: trajectory times points
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
        if not use_mean:
            # Sample trajectory
            smp_pos, smp_vel = \
                self.mp.sample_trajectories(times=times, params=params_mean,
                                            params_L=params_L,
                                            init_time=init_time,
                                            init_pos=init_pos,
                                            init_vel=init_vel,
                                            num_smp=num_samples,
                                            flat_shape=False)

            # squeeze the dimension of sampling
            if num_samples == 1:
                smp_pos, smp_vel = smp_pos.squeeze(-3), smp_vel.squeeze(-3)
            else:
                pass

        else:
            smp_pos = self.mp.get_traj_pos(times=times, params=params_mean,
                                           init_time=init_time,
                                           init_pos=init_pos,
                                           init_vel=init_vel, flat_shape=False)
            smp_vel = self.mp.get_traj_vel()
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
