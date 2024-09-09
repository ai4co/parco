import abc

from typing import Callable, List, Union

import torch

from rl4co.utils.ops import gather_by_index
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class AgentHandler(abc.ABC):
    """Base class for agent handlers. Handles conflicts between agents.
    By default, one occurrence is always kept (i.e. one agent among agents
    that selected the same action is selected).

    Args:
        mask_all: If True, the all occurrences of the same value will be masked, i.e. no agent will select it.
        exclude_values: If provided, the values in the actions that are in this list will not be masked.
        return_none_mask: If True, the mask will be None. This may be useful for loss computation.
    """

    def __init__(
        self,
        mask_all: bool = False,
        exclude_values: Union[torch.Tensor, int, List] = None,
        return_none_mask: bool = False,
    ):
        super(AgentHandler, self).__init__()
        self.mask_all = mask_all
        self.exclude_values = exclude_values
        self.return_none_mask = return_none_mask

    @abc.abstractmethod
    def _preprocess_actions(
        self, actions: torch.Tensor, td, probs: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Preprocesses actions such that first action in order to appear with an index will be selected"""
        raise NotImplementedError("Subclasses must implement this method.")

    def __call__(
        self,
        actions: torch.Tensor,
        replacement_value: Union[torch.Tensor, int] = -1,
        td=None,
        exclude_values: Union[torch.Tensor, int, List] = None,
        probs: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        exclude_values = self.exclude_values if exclude_values is None else exclude_values
        if isinstance(exclude_values, int):
            exclude_values = [exclude_values]
        if not isinstance(exclude_values, torch.Tensor) and exclude_values is not None:
            exclude_values = torch.tensor(exclude_values, device=actions.device)

        # First reordering of actions based on the type of handler (highest probability, closest, etc.)
        sorted_actions1, indices1 = self._preprocess_actions(
            actions, td, probs=probs, **kwargs
        )

        # Second reordering of actions based on the index of selected nodes, for masking non-first occurrences
        sorted_actions2, indices2 = sorted_actions1.sort(dim=1)

        # Create a mask for non-first occurrences on the sorted actions
        mask_sorted = torch.zeros_like(actions, dtype=torch.bool)
        mask_sorted[:, 1:] = sorted_actions2[:, 1:] == sorted_actions2[:, :-1]

        # Mask first occurrences on the sorted actions if needed
        if self.mask_all:
            mask_sorted[:, :-1] |= sorted_actions2[:, :-1] == sorted_actions2[:, 1:]

        # Recover the mask_sorted based on the second reordering of actions
        mask_sorted = mask_sorted.gather(1, indices2.argsort(dim=1))

        # Recover the mask_sorted based on the first reordering of actions
        mask = mask_sorted.gather(1, indices1.argsort(dim=1))

        # If exclude_values is provided, we set their mask to False so that they are not replaced
        if self.exclude_values is not None:
            self.exclude_values = self.exclude_values.to(actions.device)
            mask = mask & ~torch.isin(actions, self.exclude_values)

        # Replace values in the original actions using the mask
        if isinstance(replacement_value, int):
            actions[mask] = replacement_value
        else:
            actions[mask] = replacement_value[mask]

        # Calculate num of conflicts (sum of true values in mask / total)
        halting_ratio = mask.sum().float() / (mask.numel())
        return actions, mask if not self.return_none_mask else None, halting_ratio


class FirstPrecedenceAgentHandler(AgentHandler):
    """First agent in dim 1 is selected in case of conflicts"""

    def _preprocess_actions(self, actions: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        indices = torch.arange(actions.size(1), device=actions.device).repeat(
            actions.size(0), 1
        )
        return actions, indices


class RandomAgentHandler(AgentHandler):
    """Random agent in dim 1 is selected in case of conflicts"""

    def _preprocess_actions(self, actions: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        indices_ = torch.randperm(actions.size(1), device=actions.device).repeat(
            actions.size(0), 1
        )
        shuffled_actions = torch.gather(actions, 1, indices_)
        return shuffled_actions, indices_


class ClosestAgentHandler(AgentHandler):
    """Closest agent to target node in dim 1 is selected in case of conflicts"""

    def _preprocess_actions(
        self, actions: torch.Tensor, td, *args, **kwargs
    ) -> torch.Tensor:
        current_loc = gather_by_index(td["locs"], td["current_node"])
        target_loc = gather_by_index(td["locs"], actions)
        distances = torch.norm(current_loc - target_loc, dim=-1)
        _, indices = torch.sort(distances, dim=-1, descending=False, stable=True)
        sorted_actions = gather_by_index(actions, indices)
        return sorted_actions, indices


class HighestProbabilityAgentHandler(AgentHandler):
    """Highest probability agent in dim 1 is selected in case of conflicts"""

    def _preprocess_actions(self, actions: torch.Tensor, td, probs) -> torch.Tensor:
        # sort indices by probability
        action_probs = gather_by_index(probs, actions, dim=-1)
        _, indices = torch.sort(action_probs, dim=-1, descending=True, stable=True)
        sorted_actions = gather_by_index(actions, indices)
        return sorted_actions, indices


class SmallestPathToClosure(AgentHandler):
    """Use agent that has the lowest current path"""

    def __init__(self, *args, count_depot=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.count_depot = count_depot

    def _preprocess_actions(self, actions: torch.Tensor, td, probs) -> torch.Tensor:
        # sort indices by probability
        current_loc = gather_by_index(td["locs"], td["current_node"])
        target_loc = gather_by_index(td["locs"], actions)
        distances_to_target = torch.norm(current_loc - target_loc, dim=-1)
        if self.count_depot:
            distance_target_depot = torch.norm(
                target_loc - td["locs"][..., 0:1, :], dim=-1
            )
        else:
            distance_target_depot = 0
        distance_target_depot = torch.norm(
            target_loc - td["locs"][..., 0:1, :], dim=-1
        )  # just singled depot
        current_traveled = td["current_length"]
        total_distance = distances_to_target + current_traveled + distance_target_depot
        _, indices = torch.sort(total_distance, dim=-1, descending=False, stable=True)
        sorted_actions = gather_by_index(actions, indices)
        return sorted_actions, indices


class LowestLateness(SmallestPathToClosure):
    def _preprocess_actions(self, actions: torch.Tensor, td, probs) -> torch.Tensor:
        # TODO
        raise NotImplementedError("Not implemented yet.")


class NoHandler(AgentHandler):
    """No handler is used, i.e. all agents can select the same action. The mask is always None."""

    def __call__(self, actions: torch.Tensor, *args, **kwargs):
        return actions, None


AGENT_HANDLER_REGISTRY = {
    "first": FirstPrecedenceAgentHandler,
    "random": RandomAgentHandler,
    "closest": ClosestAgentHandler,
    "highprob": HighestProbabilityAgentHandler,
    "smallestpath": SmallestPathToClosure,
    "none": NoHandler,
}


def get_agent_handler(
    name: str, registry: dict = AGENT_HANDLER_REGISTRY, **config
) -> Callable:
    return registry[name](**config)
