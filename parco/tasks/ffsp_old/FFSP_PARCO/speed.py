from hydra import compose, initialize
from omegaconf import OmegaConf
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def

import timeit
import torch
from FFSPEnv import FFSPEnv as Env
from FFSPModel import FFSPModel
from FFSProblemDef import load_problems_from_file, get_random_problems


with initialize(version_base=None, config_path="../../configs"):
    cfg = compose(config_name="config", overrides=["env=ffsp20", "model.use_comm_layer=True"])

print(OmegaConf.to_yaml(cfg))



device_num = 6

env_params = cfg["env"]
model_params = cfg["model"]
optimizer_params = cfg["optimizer"]
trainer_params = cfg["train"]
tester_params = cfg["test"]
logger_params = cfg["logger"]


use_cuda = torch.cuda.is_available()
if use_cuda:
    cuda_device_num = device_num
    torch.cuda.set_device(cuda_device_num)
    device = torch.device('cuda', cuda_device_num)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device('cpu')
    torch.set_default_tensor_type('torch.FloatTensor')


model = FFSPModel(**model_params)
env = Env(**env_params)

#ffsp20
checkpoint_fullname = "FFSP/FFSP_PARCO/result/20240814_230605_matnet_train/checkpoint-100.pt"
# #ffsp50 comm 
# checkpoint_fullname = "FFSP/FFSP_PARCO/result/20240813_213230_matnet_train/checkpoint-150.pt"
# #ffsp100 comm 
# checkpoint_fullname = "FFSP/FFSP_PARCO/result/20240814_021241_matnet_train/checkpoint-200.pt"
# ffsp20 no comm
# checkpoint_fullname = "FFSP/FFSP_PARCO/result/20240816_192812_matnet_train/checkpoint-100.pt"



checkpoint = torch.load(checkpoint_fullname, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])


saved_problem_folder = tester_params['saved_problem_folder']
saved_problem_filename = tester_params['saved_problem_filename']
filename = os.path.join(saved_problem_folder, saved_problem_filename)
try:
    ALL_problems_INT_list = load_problems_from_file(filename, device=device)
except:
    ALL_problems_INT_list = get_random_problems(
        tester_params["problem_count"],
        env_params["machine_cnt_list"],
        env_params["job_cnt"],
        env_params["process_time_params"]
    )


def solve_one_instance(episode=0):
    batch_size = 1
    problems_INT_list = []
    for stage_idx in range(env.stage_cnt):
        problems_INT_list.append(ALL_problems_INT_list[stage_idx][episode:episode+batch_size])
    model.eval()
    with torch.inference_mode():
        env.load_problems_manual(problems_INT_list)
        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)

        # POMO Rollout
        ###############################################
        state, reward, done = env.pre_step()
        while not done:
            jobs, machines, _ = model(state)
            # shape: (batch, pomo)
            state, reward, done = env.step(jobs, machines)


if __name__ == "__main__":
    import numpy as np
    nums = 20
    res = timeit.repeat(f"for i in range({nums}): solve_one_instance(i)", "from __main__ import solve_one_instance", number=1)
    # exclude first for warmup (gpu)
    if isinstance(res, float):
        print(res / nums)
    else:
        print(np.array(res[1:]).mean() / nums)
