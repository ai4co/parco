
import torch


def get_random_problems(batch_size, machine_cnt_list, job_cnt, process_time_params):

    time_low = process_time_params['time_low']
    time_high = process_time_params['time_high']
    stage_cnt = len(machine_cnt_list)
    problems_INT_list = []
    for stage_num in range(stage_cnt):
        machine_cnt = machine_cnt_list[stage_num]
        stage_problems_INT = torch.randint(low=time_low, high=time_high, size=(batch_size, job_cnt, machine_cnt))
        problems_INT_list.append(stage_problems_INT)


    return problems_INT_list


def load_problems_from_file(filename, device=torch.device('cpu')):
    data = torch.load(filename)

    problems_INT_list = data['problems_INT_list']

    for stage_idx in range(data['stage_cnt']):
        problems_INT_list[stage_idx] = problems_INT_list[stage_idx].to(device)

    return problems_INT_list

def get_random_problems_by_random_state(rand, batch_size, machine_cnt_list, job_cnt, **process_time_params):
    distribution = process_time_params['distribution']
    same_process_time_within_stage = process_time_params['same_process_time_within_stage']
    min_process_time_list = process_time_params['min_process_time_list']
    max_process_time_list = process_time_params['max_process_time_list']

    if same_process_time_within_stage:
        if distribution == 'uniform':
            return [
                torch.tensor(rand.randint(low=min_time, high=max_time, size=(batch_size, job_cnt, 1)),
                             dtype=torch.float32).expand((batch_size, job_cnt, m_cnt))
                for min_time, max_time, m_cnt in zip(min_process_time_list,
                                                     max_process_time_list,
                                                     machine_cnt_list)]
        elif distribution == 'normal':
            return [
                torch.tensor(rand.normal(loc=max_time - min_time,
                                         scale=(max_time - min_time) / 3,
                                         size=(batch_size, job_cnt, 1)
                                         ).clip(min_time, max_time).astype(int),
                             dtype=torch.float32).expand((batch_size, job_cnt, m_cnt))
                for min_time, max_time, m_cnt in zip(min_process_time_list,
                                                     max_process_time_list,
                                                     machine_cnt_list)]
    else:
        if distribution == 'uniform':
            return [torch.tensor(rand.randint(low=min_time, high=max_time, size=(batch_size, job_cnt, m_cnt)),
                                 dtype=torch.float32)
                    for min_time, max_time, m_cnt in zip(min_process_time_list,
                                                         max_process_time_list,
                                                         machine_cnt_list)]
        elif distribution == 'normal':
            return [torch.tensor(rand.normal(loc=max_time - min_time,
                                             scale=(max_time - min_time) / 3,
                                             size=(batch_size, job_cnt, m_cnt)).clip(min_time, max_time).astype(int),
                                 dtype=torch.float32)
                    for min_time, max_time, m_cnt in zip(min_process_time_list,
                                                         max_process_time_list,
                                                         machine_cnt_list)]
    raise NotImplementedError


def load_ONE_problem_from_file(filename, device=torch.device('cpu'), index=0):
    data = torch.load(filename)

    problems_INT_list = data['problems_INT_list']
    problems_list = data['problems_list']

    for stage_idx in range(data['stage_cnt']):
        problems_INT_list[stage_idx] = problems_INT_list[stage_idx][[index], :, :]
        problems_INT_list[stage_idx] = problems_INT_list[stage_idx].to(device)

        problems_list[stage_idx] = problems_list[stage_idx][[index], :, :]
        problems_list[stage_idx] = problems_list[stage_idx].to(device)

    return problems_INT_list, problems_list
  
