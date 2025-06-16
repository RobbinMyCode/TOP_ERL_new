"""
This is a multiprocessing version of the TopErlAgent, which collect data and
update the model in parallel. This can significantly speed up the training,
especially for the metaworld environments.

The idea is to put the agent and sampler in two different processes, and
communicate between them. The agent process sends the policy parameters to the
sampler and the sampler sends the dataset back to the agent.

"""

import multiprocessing.connection
import multiprocessing
import torch

import mprl.rl.policy.abstract_policy as abs_policy
import mprl.rl.sampler.abstract_sampler as abs_sampler
import mprl.util as util
from mprl.rl.critic import TopErlCritic
from mprl.rl.replay_buffer import TopErlReplayBuffer

from .top_erl_agent import TopErlAgent


class TopErlAgentMultiProcessing(TopErlAgent):
    def __init__(self,
                 policy: abs_policy.AbstractGaussianPolicy,
                 critic: TopErlCritic,
                 sampler: abs_sampler.AbstractSampler,
                 conn: multiprocessing.connection.Connection,
                 replay_buffer: TopErlReplayBuffer,
                 projection=None,
                 dtype=torch.float32,
                 device=torch.device("cpu"),
                 **kwargs):

        super().__init__(
            policy,
            critic,
            sampler,
            replay_buffer,
            projection,
            dtype,
            device,
            **kwargs,
        )
        self.conn = conn

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

        # Note: Collect dataset until buffer size is greater than batch size
        while len(self.replay_buffer) < self.batch_size:
            self.conn.send((self.policy.parameters))
            dataset, num_env_interation = self.conn.recv()
            self.num_global_steps += num_env_interation
            self.replay_buffer.add(dataset)

        # NOTE: Update parameter of policy in the subprocess
        self.conn.send((self.policy.parameters))
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

        # NOTE: Wait for data from subprocess
        dataset, num_env_interation = self.conn.recv()
        self.num_global_steps += num_env_interation

        # Process dataset and save to RB
        self.replay_buffer.add(dataset)

        # Log data
        if self.log_now and buffer_is_ready:
            # Generate statistics for environment rollouts
            dataset_stats = \
                util.generate_many_stats(dataset, "exploration", to_np=True,
                                         exception_keys=["episode_init_idx", "split_start_indexes"])

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
                exception_keys=["episode_init_idx", "split_start_indexes"])
            result_metrics.update(evaluate_metrics),
        else:
            result_metrics = {}

        return result_metrics
