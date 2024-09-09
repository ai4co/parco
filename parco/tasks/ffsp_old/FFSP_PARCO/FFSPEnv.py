
from dataclasses import dataclass
import torch
from einops import reduce, rearrange
from FFSProblemDef import get_random_problems

# For Gantt Chart
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib.patches as patches


@dataclass
class Reset_State:
    problems_list: list


@dataclass
class MachineState:
    stg_cnt: int
    wait_step: torch.Tensor

    def __getitem__(self, idx: int):
        assert isinstance(idx, int)
        return self.wait_step.chunk(self.stg_cnt, dim=-1)[idx]

@dataclass
class JobState:
    curr_stage: torch.Tensor
    wait_step: torch.Tensor


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    # shape: (batch, pomo)
    stg_cnt: int
    mask: torch.Tensor = None
    # shape: (batch, pomo, ma_tot, job+1)

    finished: torch.Tensor = None
    # shape: (batch, pomo)

    machine_state: MachineState = None
    job_state: JobState = None

    def get_stage_mask(self, idx: int):
        assert isinstance(idx, int)
        return self.mask.chunk(self.stg_cnt, dim=-2)[idx]
        

class FFSPEnv:
    def __init__(self, **env_params):
        # Const @INIT
        ####################################
        self.stage_cnt = len(env_params['machine_cnt_list'])
        self.machine_cnt_list = env_params['machine_cnt_list']
        self.total_machine_cnt = sum(self.machine_cnt_list)
        self.job_cnt = env_params['job_cnt']
        self.process_time_params = env_params['process_time_params']
        self.pomo_size = env_params['pomo_size']

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.problems_list = None
        # len(problems_list) = stage_cnt
        # problems_list[current_stage].shape: (batch, job, machine_cnt_list[current_stage])
        self.job_durations = None
        # shape: (batch, job+1, total_machine)
        # last job means NO_JOB ==> duration = 0
        self.skip_ratio = []
        # Dynamic
        ####################################

        self.schedule = None
        # shape: (batch, pomo, machine, job+1)
        # records start time of each job at each machine
        self.t_ma_idle = None
        # shape: (batch, pomo, machine)
        # How many time steps each machine needs to run, before it become available for a new job
        self.job_location = None
        # shape: (batch, pomo, job+1)
        # index of stage each job can be processed at. if stage_cnt, it means the job is finished (when job_wait_step=0)
        self.job_wait_step = None
        self.t_job_ready = None
        # shape: (batch, pomo, job+1)
        # how many time steps job needs to wait, before it is completed and ready to start at job_location
        self.finished = None  # is scheduling done?
        # shape: (batch, pomo)
        self.stage_table = torch.tensor([
            ma for ma, rep in zip(list(range(self.stage_cnt)), self.machine_cnt_list) for _ in range(rep)
        ])
        # self.stage_table = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.long)

    def load_problems(self, batch_size):
        self.batch_size = batch_size
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        problems_INT_list = get_random_problems(
            batch_size, 
            self.machine_cnt_list,
            self.job_cnt, 
            self.process_time_params
        )

        problems_list = []
        for stage_num in range(self.stage_cnt):
            stage_problems_INT = problems_INT_list[stage_num]
            stage_problems = stage_problems_INT.clone().type(torch.float)
            problems_list.append(stage_problems)
        self.problems_list = problems_list

        self.job_durations = torch.empty(size=(self.batch_size, self.job_cnt+1, self.total_machine_cnt),
                                         dtype=torch.long)
        # shape: (batch, job+1, total_machine)
        self.job_durations[:, :self.job_cnt, :] = torch.cat(problems_INT_list, dim=2)
        self.job_durations[:, self.job_cnt, :] = 0

    def load_problems_manual(self, problems_INT_list):
        # problems_INT_list[current_stage].shape: (batch, job, machine_cnt_list[current_stage])

        self.batch_size = problems_INT_list[0].size(0)
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        problems_list = []
        for stage_num in range(self.stage_cnt):
            stage_problems_INT = problems_INT_list[stage_num]
            stage_problems = stage_problems_INT.clone().type(torch.float)
            problems_list.append(stage_problems)
        self.problems_list = problems_list

        self.job_durations = torch.empty(size=(self.batch_size, self.job_cnt+1, self.total_machine_cnt),
                                         dtype=torch.long)
        # shape: (batch, job+1, total_machine)
        self.job_durations[:, :self.job_cnt, :] = torch.cat(problems_INT_list, dim=2)
        self.job_durations[:, self.job_cnt, :] = 0

    def reset(self):

        self.schedule = torch.full(size=(self.batch_size, self.pomo_size, self.total_machine_cnt, self.job_cnt+1),
                                   dtype=torch.long, fill_value=-999999)
        # shape: (batch, pomo, machine, job+1)

        self.t_ma_idle = torch.zeros(size=(self.batch_size, self.pomo_size, self.total_machine_cnt),
                                             dtype=torch.long)
        # shape: (batch, pomo, machine)
        self.job_location = torch.zeros(size=(self.batch_size, self.pomo_size, self.job_cnt+1), dtype=torch.long)
        # shape: (batch, pomo, job+1)
        self.job_wait_step = torch.zeros(size=(self.batch_size, self.pomo_size, self.job_cnt+1), dtype=torch.long)
        self.t_job_ready = torch.zeros(size=(self.batch_size, self.pomo_size, self.job_cnt+1), dtype=torch.long)
        # shape: (batch, pomo, job+1)
        self.finished = torch.full(size=(self.batch_size, self.pomo_size), dtype=torch.bool, fill_value=False)
        # shape: (batch, pomo)

        self.step_state = Step_State(self.BATCH_IDX, self.POMO_IDX, self.stage_cnt)

        reward = None
        done = None

        return Reset_State(self.problems_list), reward, done

    def pre_step(self):
        self._update_step_state()
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, job_indices, machine_indices):
        job_indices = job_indices.split(1, dim=-1)
        machine_indices = machine_indices.split(1, dim=-1)
        
        # job_idx.shape: (batch, pomo)
        for job_idx, machine_idx in zip(job_indices, machine_indices):

            job_idx = torch.flatten(job_idx.squeeze(-1))
            machine_idx = torch.flatten(machine_idx.squeeze(-1))
            skip = job_idx == self.job_cnt

            self.skip_ratio.append(float(skip[~self.finished.reshape(-1)].sum() / skip[~self.finished.reshape(-1)].numel()))

            b_idx = torch.flatten(self.BATCH_IDX)[~skip]
            p_idx = torch.flatten(self.POMO_IDX)[~skip]

            job_idx = job_idx[~skip]
            machine_idx = machine_idx[~skip]

            t_job = self.t_job_ready[b_idx, p_idx, job_idx]
            t_ma = self.t_ma_idle[b_idx, p_idx, machine_idx]
            t = torch.maximum(t_job, t_ma)

            self.schedule[b_idx, p_idx, machine_idx, job_idx] = t

            job_length = self.job_durations[b_idx, job_idx, machine_idx]
            # shape: (batch, pomo)

            self.t_ma_idle[b_idx, p_idx, machine_idx] = t + job_length
            self.t_job_ready[b_idx, p_idx, job_idx] = t + job_length
            # shape: (batch, pomo, machine)
            self.job_location[b_idx, p_idx, job_idx] += 1
            # shape: (batch, pomo, job+1)
            self.job_wait_step[b_idx, p_idx, job_idx] = job_length
            # shape: (batch, pomo, job+1)
            self.finished = (self.job_location[:, :, :self.job_cnt] >= self.stage_cnt).all(dim=2)
            # shape: (batch, pomo)

        ####################################
        done = self.finished.all()

        if done:
            pass  # do nothing. do not update step_state, because it won't be used anyway
        else:
            # self._move_to_next_time()
            self._update_step_state()

        if done:
            reward = -self._get_makespan()  # Note the MINUS Sign ==> We want to MAXIMIZE reward
            # shape: (batch, pomo)
        else:
            reward = None

        return self.step_state, reward, done
 
    def _update_step_state(self):

        mask = torch.full(
            size=(self.batch_size, self.pomo_size, self.total_machine_cnt, self.job_cnt),
            fill_value=False,
            dtype=torch.bool
        )

        job_loc = self.job_location[:, :, :self.job_cnt]
        # shape: (batch, pomo, job)
        job_finished = (job_loc >= self.stage_cnt).unsqueeze(-2).expand_as(mask)
        # shape: (batch, pomo, 1, job)

        stage_table_expanded = self.stage_table[None, None, :, None].expand_as(mask)
        job_not_in_machines_stage = job_loc[:, :, None] != stage_table_expanded

        mask.add_(job_finished)
        mask.add_(job_not_in_machines_stage)

        mask = rearrange(mask, "b p (s m) j -> b p s m j", s=self.stage_cnt)
        # add mask for wait, which is allowed if machine cannot process any job
        mask = torch.cat(
            (mask, ~reduce(mask, "... j -> ... 1", "all")), 
            dim=-1
        )
        mask = rearrange(mask, "b p s m j -> b p (s m) j")
        
        job_state = JobState(job_loc, self.t_job_ready[:, :, :self.job_cnt])
        machine_state = MachineState(self.stage_cnt, self.t_ma_idle)

        self.step_state = Step_State(
            self.BATCH_IDX, 
            self.POMO_IDX,
            self.stage_cnt,
            mask=mask,
            finished=self.finished,
            job_state=job_state,
            machine_state=machine_state
        )
    

    def _get_makespan(self):

        job_durations_perm = self.job_durations.permute(0, 2, 1)
        # shape: (batch, machine, job+1)
        end_schedule = self.schedule + job_durations_perm[:, None, :, :]
        # shape: (batch, pomo, machine, job+1)

        end_time_max, _ = end_schedule[:, :, :, :self.job_cnt].max(dim=3)
        # shape: (batch, pomo, machine)
        end_time_max, _ = end_time_max.max(dim=2)
        # shape: (batch, pomo)

        return end_time_max

    def draw_Gantt_Chart(self, batch_i, pomo_i):

        job_durations = self.job_durations[batch_i, :, :]
        # shape: (job, machine)
        schedule = self.schedule[batch_i, pomo_i, :, :]
        # shape: (machine, job)

        total_machine_cnt = self.total_machine_cnt
        makespan = self._get_makespan()[batch_i, pomo_i].item()

        # Create figure and axes
        fig,ax = plt.subplots(figsize=(makespan/3, 5))
        cmap = self._get_cmap(self.job_cnt)

        plt.xlim(0, makespan)
        plt.ylim(0, total_machine_cnt)
        ax.invert_yaxis()

        plt.plot([0, makespan], [4, 4], 'black')
        plt.plot([0, makespan], [8, 8], 'black')

        for machine_idx in range(total_machine_cnt):

            duration = job_durations[:, machine_idx]
            # shape: (job)
            machine_schedule = schedule[machine_idx, :]
            # shape: (job)

            for job_idx in range(self.job_cnt):

                job_length = duration[job_idx].item()
                job_start_time = machine_schedule[job_idx].item()
                if job_start_time >= 0:
                    # Create a Rectangle patch
                    rect = patches.Rectangle((job_start_time,machine_idx),job_length,1, facecolor=cmap(job_idx), alpha=0.8)
                    ax.add_patch(rect)

        ax.grid()
        ax.set_axisbelow(True)
        plt.show()

    def _get_cmap(self, color_cnt):

        colors_list = ['red', 'orange', 'yellow', 'green', 'blue',
                       'purple', 'aqua', 'aquamarine', 'black',
                       'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chocolate',
                       'coral', 'cornflowerblue', 'darkblue', 'darkgoldenrod', 'darkgreen',
                       'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange',
                       'darkorchid', 'darkred', 'darkslateblue', 'darkslategrey', 'darkturquoise',
                       'darkviolet', 'deeppink', 'deepskyblue', 'dimgrey', 'dodgerblue',
                       'forestgreen', 'gold', 'goldenrod', 'gray', 'greenyellow',
                       'hotpink', 'indianred', 'khaki', 'lawngreen', 'magenta',
                       'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid',
                       'mediumpurple',
                       'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue',
                       'navy', 'olive', 'olivedrab', 'orangered',
                       'orchid',
                       'palegreen', 'paleturquoise', 'palevioletred', 'pink', 'plum', 'powderblue',
                       'rebeccapurple',
                       'rosybrown', 'royalblue', 'saddlebrown', 'sandybrown', 'sienna',
                       'silver', 'skyblue', 'slateblue',
                       'springgreen',
                       'steelblue', 'tan', 'teal', 'thistle',
                       'tomato', 'turquoise', 'violet', 'yellowgreen']

        cmap = ListedColormap(colors_list, N=color_cnt)

        return cmap
