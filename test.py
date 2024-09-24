import argparse
import os
import time
import warnings

import torch

from rl4co.data.utils import load_npz_to_tensordict
from tqdm.auto import tqdm

from parco.models import PARCORLModule
from parco.tasks.eval import get_dataloader

warnings.filterwarnings("ignore", category=FutureWarning)

# Tricks for faster inference
try:
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
except AttributeError:
    pass
torch.set_float32_matmul_precision("medium")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--problem", type=str, default="hcvrp", help="Problem name: hcvrp, omdcpdp, etc."
    )
    parser.add_argument(
        "--datasets",
        help="Filename of the dataset(s) to evaluate. Defaults to all under data/{problem}/ dir",
        default=None,
    )
    parser.add_argument(
        "--decode_type",
        type=str,
        default="greedy",
        help="Decoding type. Available: greedy, sampling",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of samples to use for sampling decoding",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")

    # Use load_from_checkpoint with map_location, which is handled internally by Lightning
    # Suppress FutureWarnings related to torch.load and weights_only
    warnings.filterwarnings("ignore", message=".*weights_only.*", category=FutureWarning)

    opts = parser.parse_args()

    batch_size = opts.batch_size
    sample_size = opts.sample_size
    decode_type = opts.decode_type
    checkpoint_path = opts.checkpoint
    problem = opts.problem
    if "cuda" in opts.device and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    if checkpoint_path is None:
        assert (
            problem is not None
        ), "Problem must be specified if checkpoint is not provided"
        checkpoint_path = f"./checkpoints/{problem}/parco.ckpt"
    if decode_type == "greedy":
        assert (sample_size == 1 or sample_size is None), "Greedy decoding only uses 1 sample"
    if opts.datasets is None:
        assert problem is not None, "Problem must be specified if dataset is not provided"
        data_paths = [f"./data/{problem}/{f}" for f in os.listdir(f"./data/{problem}")]
    else:
        data_paths = [opts.datasets] if isinstance(opts.datasets, str) else opts.datasets
    if decode_type == "sampling":
        assert sample_size is not None, "Sample size must be specified for sampling decoding with --sample_size"
        assert batch_size == 1, "Only batch_size=1 is supported for sampling decoding currently"
    data_paths = sorted(data_paths)  # Sort for consistency

    # Load the checkpoint as usual
    print("Loading checkpoint from ", checkpoint_path)
    model = PARCORLModule.load_from_checkpoint(
        checkpoint_path, map_location="cpu", strict=False
    )
    env = model.env
    policy = model.policy.to(device).eval()  # Use mixed precision if supported

    for dataset in data_paths:
        costs = []
        inference_times = []
        eval_steps = []

        print(f"Loading {dataset}")
        td_test = load_npz_to_tensordict(dataset)
        dataloader = get_dataloader(td_test, batch_size=batch_size)

        with torch.cuda.amp.autocast() if "cuda" in opts.device else torch.inference_mode():  # Use mixed precision if supported
            with torch.inference_mode():
                for td_test_batch in tqdm(dataloader):
                    td_reset = env.reset(td_test_batch).to(device)
                    start_time = time.time()
                    out = policy(
                        td_reset,
                        env,
                        decode_type=decode_type,
                        num_samples=sample_size,
                        return_actions=False,
                    )
                    end_time = time.time()
                    inference_time = end_time - start_time
                    if decode_type == "greedy":
                        costs.append(-out["reward"].mean().item())
                    else:
                        costs.extend(
                            -out["reward"].reshape(-1, sample_size).max(dim=-1)[0]
                        )
                    inference_times.append(inference_time)
                    eval_steps.append(out["steps"])

            print(f"Average cost: {sum(costs)/len(costs):.4f}")
            print(
                f"Per step inference time: {sum(inference_times)/len(inference_times):.4f}s"
            )
            print(f"Total inference time: {sum(inference_times):.4f}s")
            print(f"Average eval steps: {sum(eval_steps)/len(eval_steps):.2f}")
