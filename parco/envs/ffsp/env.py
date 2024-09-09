from math import factorial
from typing import Optional
from einops import rearrange, reduce
import torch

from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)

from rl4co.data.dataset import FastTdDataset
from rl4co.envs.common.base import RL4COEnvBase

from .generator import FFSPGenerator


class FFSPEnv(RL4COEnvBase):
    """Flexible Flow Shop Problem (FFSP) environment.
    The goal is to schedule a set of jobs on a set of machines such that the makespan is minimized.

    Observations:
        - time index
        - sub time index
        - batch index
        - machine index
        - schedule
        - machine wait step
        - job location
        - job wait step
        - job duration

    Constraints:
        - each job has to be processed on each machine in a specific order
        - the machine has to be available to process the job
        - the job has to be available to be processed

    Finish Condition:
        - all jobs are scheduled

    Reward:
        - (minus) the makespan of the schedule

    Args:
        generator: FFSPGenerator instance as the data generator
        generator_params: parameters for the generator
    """

    name = "ffsp"

    def __init__(
        self,
        generator: FFSPGenerator = None,
        generator_params: dict = {},
        **kwargs,
    ):
        super().__init__(check_solution=False, dataset_cls=FastTdDataset, **kwargs)
        if generator is None:
            generator = FFSPGenerator(**generator_params)
        self.generator = generator

        self.num_stage = generator.num_stage
        self.num_machine = generator.num_machine
        self.num_job = generator.num_job
        self.num_machine_total = generator.num_machine_total
        self.tables = None
        self.step_cnt = None
        self.stage_table = torch.tensor(
            [ma for ma in list(range(self.num_stage)) for _ in range(self.num_machine)],
            device=self.device,
            dtype=torch.long
        )

    def get_num_starts(self, td):
        return factorial(self.num_machine)

    def select_start_nodes(self, td, num_starts):
        raise NotImplementedError("Shdsu")


    def pre_step(self, td: TensorDict) -> TensorDict:
        self.stage_table = self.stage_table.to(td.device)
        # update action mask and stage machine indx
        td = self._update_step_state(td)

        # return updated td
        return td

    def _update_step_state(self, td: TensorDict):

        batch_size = td.batch_size

        mask = torch.full(
            size=(*batch_size, self.num_machine_total, self.num_job),
            fill_value=False,
            dtype=torch.bool,
            device=td.device
        )

        # shape: (batch, job)
        job_loc = td["job_location"][:, :self.num_job]
        # shape: (batch, 1, job)
        job_finished = (job_loc >= self.num_stage).unsqueeze(-2).expand_as(mask)

        stage_table_expanded = self.stage_table[None, :, None].expand_as(mask)
        job_not_in_machines_stage = job_loc[:, None] != stage_table_expanded

        mask.add_(job_finished)
        mask.add_(job_not_in_machines_stage)

        mask = rearrange(mask, "b (s m) j -> b s m j", s=self.num_stage)
        # add mask for wait, which is allowed if machine cannot process any job
        mask = torch.cat(
            (mask, ~reduce(mask, "... j -> ... 1", "all")), 
            dim=-1
        )
        mask = rearrange(mask, "b s m j -> b (s m) j")
        
        td.update({
            "full_action_mask": ~mask
        })

        return td


    def _step(self, td: TensorDict) -> TensorDict:

        batch_size = td.batch_size
        batch_idx = torch.arange(*batch_size, dtype=torch.long, device=td.device)
        actions = td["action"].split(1, dim=-1)

        for action in actions:
            job_idx = torch.flatten(action["jobs"].squeeze(-1))
            machine_idx = torch.flatten(action["mas"].squeeze(-1))
            skip = job_idx == self.num_job
            if skip.all():
                continue

            b_idx = batch_idx[~skip]

            job_idx = job_idx[~skip]
            machine_idx = machine_idx[~skip]

            t_job = td["t_job_ready"][b_idx, job_idx]
            t_ma = td["t_ma_idle"][b_idx, machine_idx]
            t = torch.maximum(t_job, t_ma)

            td["schedule"][b_idx, machine_idx, job_idx] = t

            # shape: (batch)
            job_length = td["job_duration"][b_idx, job_idx, machine_idx]

            # shape: (batch, machine)
            td["t_ma_idle"][b_idx, machine_idx] = t + job_length
            td["t_job_ready"][b_idx, job_idx] = t + job_length
            # shape: (batch, job+1)
            td["job_location"][b_idx, job_idx] += 1
            # shape: (batch)
            td["done"] = (td["job_location"][:, :self.num_job] >= self.num_stage).all(dim=-1)

        ####################################
        all_done = td["done"].all()

        if all_done:
            pass  # do nothing. do not update step_state, because it won't be used anyway
        else:
            self._update_step_state(td)

        if all_done:
            reward = -self._get_makespan(td)  # Note the MINUS Sign ==> We want to MAXIMIZE reward
            # shape: (batch, pomo)
        else:
            reward = None

        td["reward"] = reward

        return td


    def _reset(
        self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None
    ) -> TensorDict:

        device = td.device

        self.step_cnt = 0

        # Scheduling status information
        schedule = torch.full(
            size=(*batch_size, self.num_machine_total, self.num_job + 1),
            dtype=torch.long,
            device=device,
            fill_value=-999999,
        )
        job_location = torch.zeros(
            size=(*batch_size, self.num_job + 1),
            dtype=torch.long,
            device=device,
        )
        job_duration = torch.empty(
            size=(*batch_size, self.num_job + 1, self.num_machine * self.num_stage),
            dtype=torch.long,
            device=device,
        )
        job_duration[..., : self.num_job, :] = td["run_time"]
        job_duration[..., self.num_job, :] = 0
        # time information
        t_job_ready = torch.zeros(
            size=(*batch_size, self.num_job+1), 
            dtype=torch.long,
            device=device
        )
        t_ma_idle = torch.zeros(
            size=(*batch_size, self.num_machine_total), 
            dtype=torch.long,
            device=device)
        
        # Finish status information
        reward = torch.full(
            size=(*batch_size,),
            dtype=torch.float32,
            device=device,
            fill_value=float("-inf"),
        )
        done = torch.full(
            size=(*batch_size,),
            dtype=torch.bool,
            device=device,
            fill_value=False,
        )

        return TensorDict(
            {
                # Index information
                "t_job_ready": t_job_ready,
                "t_ma_idle": t_ma_idle,
                # Scheduling status information
                "schedule": schedule,
                "job_location": job_location,
                "job_duration": job_duration,
                # Finish status information
                "reward": reward,
                "done": done
            },
            batch_size=batch_size,
        )


    def _get_makespan(self, td):

        # shape: (batch, machine, job+1)
        job_durations_perm = td["job_duration"].permute(0, 2, 1)
        # shape: (batch, machine, job+1)
        end_schedule = td["schedule"] + job_durations_perm

        # shape: (batch, machine)
        end_time_max, _ = end_schedule[:, :, :self.num_job].max(dim=-1)
        # shape: (batch)
        end_time_max, _ = end_time_max.max(dim=-1)

        return end_time_max.float()
    

    def _get_reward(self, td, actions) -> TensorDict:
        return td["reward"].float()

    def render(self, td: TensorDict, idx: int):
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt

        total_machine_cnt = self.num_machine_total
        num_job = self.num_job

        # shape: (job, machine)
        job_durations = td["job_duration"][idx, :, :]
        # shape: (machine, job)
        schedule = td["schedule"][idx, :, :]

        makespan = -td["reward"][idx].item()

        # Create figure and axes
        fig, ax = plt.subplots(figsize=(makespan / 3, 5))
        cmap = self._get_cmap(num_job)

        plt.xlim(0, makespan)
        plt.ylim(0, total_machine_cnt)
        ax.invert_yaxis()

        plt.plot([0, makespan], [4, 4], "black")
        plt.plot([0, makespan], [8, 8], "black")

        for machine_idx in range(total_machine_cnt):
            duration = job_durations[:, machine_idx]
            # shape: (job)
            machine_schedule = schedule[machine_idx, :]
            # shape: (job)

            for job_idx in range(num_job):
                job_length = duration[job_idx].item()
                job_start_time = machine_schedule[job_idx].item()
                if job_start_time >= 0:
                    # Create a Rectangle patch
                    rect = patches.Rectangle(
                        (job_start_time, machine_idx),
                        job_length,
                        1,
                        facecolor=cmap(job_idx),
                    )
                    ax.add_patch(rect)

        ax.grid()
        ax.set_axisbelow(True)
        plt.show()

    @staticmethod
    def _get_cmap(color_cnt):
        from random import shuffle

        from matplotlib.colors import CSS4_COLORS, ListedColormap

        color_list = list(CSS4_COLORS.keys())
        shuffle(color_list)
        cmap = ListedColormap(color_list, N=color_cnt)
        return cmap