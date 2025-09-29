import torch.linalg
from mp_pytorch.mp import * #ProDMP

class ProDMPReuseSample(ProDMP):
    def __init__(self,
                 basis_gn: ProDMPBasisGenerator,
                 num_dof: int,
                 weights_scale: Union[float, Iterable] = 1.,
                 goal_scale: float = 1.,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = 'cpu',
                 **kwargs):

        super().__init__(basis_gn, num_dof, weights_scale, goal_scale, dtype, device, **kwargs)
        self.sample_relative_pos_in_distr = None

    def sample_trajectories(self, times=None, params=None, params_L=None,
                            init_time=None, init_pos=None, init_vel=None,
                            num_smp=1, flat_shape=False, re_use_pos_from_prev_distr=False,
                            **kwargs):
        """
        Sample trajectories from MP

        Args:
            times: time points
            params: learnable parameters
            params_L: learnable parameters' variance
            init_time: initial condition time
            init_pos: initial condition position
            init_vel: initial condition velocity
            num_smp: num of trajectories to be sampled
            flat_shape: if flatten the dimensions of Dof and time

        Returns:
            sampled trajectories
        """

        tanh_squash = kwargs.get("tanh_squash", False)  # Used for SAC policy
        tanh_scale_after_squash = kwargs.get("tanh_scale_after_squash", 1.0)

        # Shape of pos_smp
        # [*add_dim, num_smp, num_times, num_dof]
        # or [*add_dim, num_smp, num_dof * num_times]

        if all([data is None for data in {times, params, params_L, init_time,
                                          init_pos, init_vel}]):
            times = self.times
            params = self.params
            params_L = self.params_L
            init_time = self.init_time
            init_pos = self.init_pos
            init_vel = self.init_vel

        num_add_dim = params.ndim - 1

        # Add additional sample axis to time
        # Shape [*add_dim, num_smp, num_times]
        times_smp = util.add_expand_dim(times, [num_add_dim], [num_smp])

        # Sample parameters, shape [num_smp, *add_dim, num_mp_params]
        if not re_use_pos_from_prev_distr:
            if len(params.shape) >= 3:
                params_smp_0 = MultivariateNormal(loc=params[:, 0],
                                                scale_tril=params_L[:, 0],
                                                validate_args=False).rsample([num_smp])[:, :, None, ...]
                #projection of samples onto unit gauss
                self.sample_relative_pos_in_distr = torch.linalg.solve_triangular(params_L[: , 0, ...][None, :, None, ...],
                                                                             (params_smp_0 - params[:, 0,  ...][None, :, None, ...])[..., :, None],
                                                                             upper=False)[..., 0]
                #projection back according to params of the other splits
                params_smp_rest = params[None, :, 1:]  + (params_L[None, :, 1:] @ self.sample_relative_pos_in_distr[..., None])[..., 0]

                params_smp = torch.cat((params_smp_0, params_smp_rest), dim=-2)
            else:
                params_smp = MultivariateNormal(loc=params,
                                                scale_tril=params_L,
                                                validate_args=False).rsample([num_smp])
                self.sample_relative_pos_in_distr = \
                torch.linalg.solve_triangular(params_L[:, ...][None, ...],
                                              (params_smp - params[:,  ...][None,  ...])[..., :, None],
                                              upper=False)[..., 0]
        else:
            params_smp = params[None, ...] + (params_L[None, ...] @ self.sample_relative_pos_in_distr[..., None])[..., 0]

        if tanh_squash:
            params_smp = torch.tanh(params_smp)
            params_smp = tanh_scale_after_squash * params_smp

        # Switch axes to [*add_dim, num_smp, num_mp_params]
        params_smp = torch.einsum('i...j->...ij', params_smp)

        params_super = self.basis_gn.get_params()
        if params_super.nelement() != 0:
            params_super_smp = util.add_expand_dim(params_super, [-2],
                                                   [num_smp])
            params_smp = torch.cat([params_super_smp, params_smp], dim=-1)

        # Add additional sample axis to initial condition
        if init_time is not None:
            init_time_smp = util.add_expand_dim(init_time, [num_add_dim], [num_smp])
            init_pos_smp = util.add_expand_dim(init_pos, [num_add_dim], [num_smp])
            init_vel_smp = util.add_expand_dim(init_vel, [num_add_dim], [num_smp])
        else:
            init_time_smp = None
            init_pos_smp = None
            init_vel_smp = None

        # Update inputs
        self.reset()
        self.update_inputs(times_smp, params_smp, None,
                           init_time_smp, init_pos_smp, init_vel_smp)

        # Get sample trajectories
        pos_smp = self.get_traj_pos(flat_shape=flat_shape)
        vel_smp = self.get_traj_vel(flat_shape=flat_shape)

        # Recover old inputs
        if params_super.nelement() != 0:
            params = torch.cat([params_super, params], dim=-1)
        self.reset()
        self.update_inputs(times, params, None, init_time, init_pos, init_vel)

        return pos_smp, vel_smp
