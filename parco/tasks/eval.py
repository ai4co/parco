import time

import torch

from rl4co.data.dataset import TensorDictDataset
from rl4co.data.transforms import StateAugmentation
from rl4co.utils.ops import batchify, gather_by_index, unbatchify
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from parco.models.agent_handlers import RandomAgentHandler
from parco.models.augmentations import DilationAugmentation


def get_dataloader(td, batch_size=4):
    """Get a dataloader from a TensorDictDataset"""
    # Set up the dataloader
    dataloader = DataLoader(
        TensorDictDataset(td.clone()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=TensorDictDataset.collate_fn,
    )
    return dataloader


def check_unused_kwargs(class_, kwargs):
    # if len(kwargs) > 0 and not (len(kwargs) == 1 and "progress" in kwargs):
    #     print(f"Warning: {class_.__class__.__name__} does not use kwargs {kwargs}")
    pass


class EvalBase:
    """Base class for evaluation

    Args:
        env: Environment
        progress: Whether to show progress bar
        **kwargs: Additional arguments (to be implemented in subclasses)
    """

    name = "base"

    def __init__(
        self, env, progress=True, verbose=True, reset_env=True, batch_size=4, **kwargs
    ):
        check_unused_kwargs(self, kwargs)
        self.env = env
        self.progress = progress
        self.verbose = verbose
        self.reset_env = reset_env
        self.batch_size = batch_size

    def __call__(self, policy, td, **kwargs):
        """Evaluate the policy on the given data with **kwargs parameter
        self._inner is implemented in subclasses and returns actions and rewards
        """

        if torch.cuda.is_available():
            # Collect timings for evaluation (more accurate than timeit)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_time = time.time()

        dataloader = get_dataloader(td, batch_size=100)

        with torch.inference_mode():
            rewards_list = []
            actions_list = []

            for batch in tqdm(
                dataloader, disable=not self.progress, desc=f"Running {self.name}"
            ):
                td = batch.to(next(policy.parameters()).device)
                if self.reset_env:
                    td = self.env.reset(td)
                actions, rewards = self._inner(policy, td, self.env, **kwargs)
                rewards_list.extend(rewards)
                actions_list.extend(actions)

            # Padding: pad actions to the same length with zeros
            max_length = max(action.size(-1) for action in actions_list)
            actions = torch.stack(
                [
                    torch.nn.functional.pad(action, (0, max_length - action.size(-1)))
                    for action in actions_list
                ],
                0,
            )
            # actions = pad_actions(actions_list)
            rewards = torch.stack(rewards_list, 0)

        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()
            inference_time = start_event.elapsed_time(end_event)
        else:
            inference_time = time.time() - start_time

        if self.verbose:
            tqdm.write(f"Mean reward for {self.name}: {rewards.mean():.4f}")
            tqdm.write(f"Time: {inference_time/1000:.4f}s")

        # Empty cache
        torch.cuda.empty_cache()

        return {
            "actions": actions.cpu(),
            "reward": rewards.cpu(),
            "inference_time": inference_time,
            "avg_reward": rewards.cpu().mean(),
        }

    def _inner(self, policy, td, env=None, **kwargs):
        """Inner function to be implemented in subclasses.
        This function returns actions and rewards for the given policy
        """
        raise NotImplementedError("Implement in subclass")

    def _get_reward(self, td, actions):
        """Note: actions already count for the depots"""
        next_td = td.clone()  # .to("cpu")
        with torch.inference_mode():
            # take actions from the last dimension
            for i in range(actions.shape[-1]):
                cur_a = actions[:, :, i]
                next_td.set("action", cur_a)
                next_td = self.env.step(next_td)["next"]

        reward = self.env.get_reward(next_td.clone(), actions)
        return reward


class GreedyEval(EvalBase):
    """Evaluates the policy using greedy decoding and single trajectory"""

    name = "greedy"

    def __init__(self, env, **kwargs):
        check_unused_kwargs(self, kwargs)
        super().__init__(env, **kwargs)

    def _inner(self, policy, td, env, **kwargs):
        out = policy(
            td.clone(),  # note: we need to
            env,
            decode_type="greedy",
            return_actions=True,
        )

        return out["actions"], self._get_reward(td, out["actions"])


class SamplingEval(EvalBase):
    """Evaluates the policy via N samples from the policy

    Args:
        samples (int): Number of samples to take
        softmax_temp (float): Temperature for softmax sampling. The higher the temperature, the more random the sampling
    """

    name = "sampling"

    def __init__(self, env, samples, softmax_temp=None, **kwargs):
        check_unused_kwargs(self, kwargs)
        super().__init__(env, kwargs.get("progress", True), kwargs.get("verbose", True))

        self.samples = samples
        self.softmax_temp = softmax_temp

    def _inner(self, policy, td, env, **kwargs):
        td_init = td.clone()
        td = batchify(td, self.samples)
        out = policy(
            td.clone(),
            env,
            decode_type="sampling",
            return_actions=True,
            softmax_temp=self.softmax_temp,
        )

        # Move into batches and compute rewards
        rewards = self._get_reward(batchify(td_init, self.samples), out["actions"])
        rewards = unbatchify(rewards, self.samples)
        actions = unbatchify(out["actions"], self.samples)

        # Get best reward and corresponding action
        rewards, max_idxs = rewards.max(dim=1)
        actions = gather_by_index(actions, max_idxs, dim=1)
        return actions, rewards


class AugmentationEval(EvalBase):
    """Evaluates the policy via N state augmentations
    `force_dihedral_8` forces the use of 8 augmentations (rotations and flips) as in POMO
    https://en.wikipedia.org/wiki/Examples_of_groups#dihedral_group_of_order_8

    Args:
        num_augment (int): Number of state augmentations
        force_dihedral_8 (bool): Whether to force the use of 8 augmentations
    """

    name = "augmentation"

    def __init__(self, env, num_augment=8, force_dihedral_8=False, **kwargs):
        check_unused_kwargs(self, kwargs)
        super().__init__(env, kwargs.get("progress", True), kwargs.get("verbose", True))
        self.augmentation = StateAugmentation(
            num_augment=num_augment,
            augment_fn="dihedral_8" if force_dihedral_8 else "symmetric",
        )  # augment with tsp cuz its the same

    def _inner(self, policy, td, env, num_augment=None):
        if num_augment is None:
            num_augment = self.augmentation.num_augment
        td_init = td.clone()
        td = self.augmentation(td)
        out = policy(td.clone(), env, decode_type="greedy", return_actions=True)

        # Move into batches and compute rewards
        rewards = self._get_reward(batchify(td_init, num_augment), out["actions"])
        rewards = unbatchify(rewards, num_augment)
        actions = unbatchify(out["actions"], num_augment)

        # Get best reward and corresponding action
        rewards, max_idxs = rewards.max(dim=1)
        actions = gather_by_index(actions, max_idxs, dim=1)
        return actions, rewards

    @property
    def num_augment(self):
        return self.augmentation.num_augment


class DilationEval(EvalBase):
    """Evaluates the policy with Dilation

    Args:
        num_augment (int): Number of state augmentations
    """

    name = "augmentation"

    def __init__(self, env, num_augment=8, min_s=0.5, max_s=1.0, **kwargs):
        check_unused_kwargs(self, kwargs)
        super().__init__(env, kwargs.get("progress", True), kwargs.get("verbose", True))
        self.augmentation = DilationAugmentation(
            env.name, num_augment=num_augment, min_s=min_s, max_s=max_s
        )

    def _inner(self, policy, td, env, num_augment=None):
        if num_augment is None:
            num_augment = self.augmentation.num_augment
        td_init = td.clone()
        td = self.augmentation(td)[0]
        out = policy(td.clone(), env, decode_type="greedy", return_actions=True)

        # Move into batches and compute rewards
        # NOTE: need to use initial td, since it is not augmented with different scales
        rewards = self._get_reward(batchify(td_init, num_augment), out["actions"])
        rewards = unbatchify(rewards, num_augment)
        actions = unbatchify(out["actions"], num_augment)

        # Get best reward and corresponding action
        rewards, max_idxs = rewards.max(dim=1)
        actions = gather_by_index(actions, max_idxs, dim=1)
        return actions, rewards  # self._get_reward(td_init, actions)

    @property
    def num_augment(self):
        return self.augmentation.num_augment


class DilationSymEval(EvalBase):
    """Evaluates the policy with Dilation and Symmetric Augmentation

    Args:
        num_augment (int): Number of state augmentations
    """

    name = "augmentation"

    def __init__(
        self, env, num_augment_dil=8, num_augment_sym=8, min_s=0.5, max_s=1.0, **kwargs
    ):
        check_unused_kwargs(self, kwargs)
        super().__init__(env, kwargs.get("progress", True), kwargs.get("verbose", True))
        self.augmentation_dil = DilationAugmentation(
            env.name, num_augment=num_augment_dil, min_s=min_s, max_s=max_s
        )
        self.augmentation_sym = StateAugmentation(
            env.name, num_augment=num_augment_sym, use_dihedral_8=False
        )

    def _inner(self, policy, td, env, num_augment=None):
        if num_augment is None:
            num_augment = (
                self.augmentation_dil.num_augment * self.augmentation_sym.num_augment
            )
        td_init = td.clone()
        td = self.augmentation_dil(td)[0]
        td = self.augmentation_sym(td)

        out = policy(td.clone(), env, decode_type="greedy", return_actions=True)

        # Move into batches and compute rewards
        # NOTE: need to use initial td, since it is not augmented with different scales
        rewards = self._get_reward(batchify(td_init, num_augment), out["actions"])
        rewards = unbatchify(rewards, num_augment)
        actions = unbatchify(out["actions"], num_augment)

        # Get best reward and corresponding action
        rewards, max_idxs = rewards.max(dim=1)
        actions = gather_by_index(actions, max_idxs, dim=1)
        return actions, rewards  # self._get_reward(td_init, actions)

    @property
    def num_augment(self):
        return self.augmentation.num_augment


class GreedyRandomAgentSampling(EvalBase):
    """Evaluates the policy via N samples from the policy with random agent conflict handler

    Args:
        samples (int): Number of samples to take
    """

    name = "greedy_random_agent_sampling"

    def __init__(self, env, samples, **kwargs):
        check_unused_kwargs(self, kwargs)
        super().__init__(env, kwargs.get("progress", True), kwargs.get("verbose", True))

        self.samples = samples

    def _inner(self, policy, td, env, **kwargs):
        td_init = td.clone()
        try:
            policy.decoder.agent_handler = RandomAgentHandler()
        except Exception:
            raise Exception("RandomAgentHandler not implemented for this policy")
        td = batchify(td, self.samples)
        out = policy(
            td.clone(),
            env,
            decode_type="greedy",
            return_actions=True,
        )

        # Move into batches and compute rewards
        rewards = self._get_reward(batchify(td_init, self.samples), out["actions"])
        rewards = unbatchify(rewards, self.samples)
        actions = unbatchify(out["actions"], self.samples)

        # Get best reward and corresponding action
        rewards, max_idxs = rewards.max(dim=1)
        actions = gather_by_index(actions, max_idxs, dim=1)
        return actions, rewards
