from typing import Optional

import torch

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index
from rl4co.utils.pylogger import get_pylogger
from tensordict.tensordict import TensorDict
from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec

from .generator import OMDCPDPGenerator
from .render import render

log = get_pylogger(__name__)


class OMDCPDPEnv(RL4COEnvBase):
    """Open Multi-Depot Capacitated Pickup and Delivery Problem(OMDCPDP)

    Problem description:
    - Open: agents do not need to go back to the depot after finishing all orders (optional)
    - Multi-depot: each agent has its own depot (i.e., its own starting node)
    - Capacitated: each agent has a capacity constraint (cannot hold more than a certain number of orders)
    - Vehicle routing: the agents need to visit all nodes (pickup and delivery) while minimizing an objective function
    - Pickup and delivery: the orders need to be picked up and delivered in pairs (first pickup, then delivery)

    The stepping is performed in parallel for all agents.
    Note that this environment has capacity constraints.
    We support two reward modes: minmax and minsum.

    Args:
        num_loc (int): number of locations (including depots)
        num_agents (int): number of agents
        min_loc (float): minimum value for the location coordinates
        max_loc (float): maximum value for the location coordinates
        capacity_min (int): minimum capacity for each agent
        capacity_max (int): maximum capacity for each agent
        min_lateness_weight (float): minimum lateness weight for each agent. If 1, the reward is the same as the lateness.
        max_lateness_weight (float): maximum lateness weight for each agent. If 1, the reward is the same as the lateness.
        dist_norm (str): distance norm. Either L1 or L2.
        reward_mode (str): reward mode. Either minmax or minsum.
        problem_mode (str): problem mode. Either close or open.
        use_different_depot_locations (bool): whether to use different depot locations for each agent
        num_agents_override (bool): whether to override the number of agents in the TensorDict if it is provided in :meth:`_reset`
        td_params (TensorDict): TensorDict of parameters for generating the data
        check_conflicts (bool): whether to check for conflicts (i.e., if an agent visits the same node twice)
    """

    name = "omdcpdp"
    stepping = "parallel"

    def __init__(
        self,
        generator: OMDCPDPGenerator = None,
        generator_params: dict = {},
        dist_norm: str = "L2",
        reward_mode: str = "lateness",
        problem_mode: str = "open",
        td_params: TensorDict = None,
        check_conflicts: bool = False,
        check_solution: bool = False,
        **kwargs,
    ):
        kwargs["check_solution"] = check_solution
        super().__init__(**kwargs)
        if generator is None:
            generator = OMDCPDPGenerator(**generator_params)
        self.generator = generator

        self.dist_norm = dist_norm
        assert reward_mode in [
            "minmax",
            "minsum",
            "lateness",
            "lateness_square",
        ], "Invalid reward mode. Must be minmax, minsum, lateness or lateness_square."
        self.reward_mode = reward_mode
        assert problem_mode in [
            "close",
            "open",
        ], "Invalid problem mode. Must be close or open."
        self.problem_mode = problem_mode
        self.check_conflicts = check_conflicts
        # raise warning if check conflicts
        if self.check_conflicts:
            log.warning("Checking conflicts is enabled. This may slow down the code.")
        self._make_spec(td_params)

    def _reset(
        self,
        td: Optional[TensorDict] = None,
        batch_size: Optional[int] = None,
    ) -> TensorDict:
        device = td.device

        # TODO: check
        num_agents = (
            td["depots"].size(-2)
            if "depots" in td.keys()
            else td["num_agents"].max().item()
        )

        # Check if depots is in keys. If not, it is the first location
        if "depots" not in td.keys():
            depots = td["locs"][..., 0:1, :]
            cities = td["locs"][..., 1:, :]
        else:
            depots = td["depots"]
            cities = td["locs"]

        # Pad depot if only one
        if depots.shape[-2] == 1 or depots.ndim == 2:
            depots = depots.unsqueeze(-2) if depots.ndim == 2 else depots
            depots = depots.repeat(1, num_agents, 1)

        # Remove padding depots if more than num_agents
        depots = depots[..., :num_agents, :]

        num_cities = cities.shape[-2]
        # If num_cities is odd, decrease it by 1
        if num_cities % 2 == 1:
            cities = cities[..., :-1, :]
            num_cities -= 1
        num_loc_tot = num_agents + num_cities
        num_p_d = num_cities // 2

        # Each agent starts in their respective node with index equal to their agent index
        depot_node = torch.arange(num_agents, dtype=torch.int64, device=device)[
            None, ...
        ].repeat(*batch_size, 1)
        current_node = depot_node.clone()

        # Last outer node, used for open problem
        last_outer_node = depot_node.clone()

        # Seperate the unvisited_node and the mask, 1-unvisisted, 0-visited
        # available still include the depot for the calculation convenience,
        # so the size will be [B, num_loc+1]
        available = torch.ones(
            (*batch_size, num_loc_tot), dtype=torch.bool, device=device
        )

        # Depots are always unavailable
        available[..., :num_agents] = 0

        # Only pickup nodes are available at the initial state.
        # num_pickup = int(td["locs"].size(-2) / 2) + num_agents # bug!
        action_mask = torch.cat(
            (
                torch.zeros(
                    (*batch_size, num_agents, num_agents),
                    dtype=torch.bool,
                    device=device,
                ),  # depot is not available
                torch.ones(
                    (*batch_size, num_agents, num_p_d),
                    dtype=torch.bool,
                    device=device,
                ),  # pickup nodes are available
                torch.zeros(
                    (*batch_size, num_agents, num_p_d),
                    dtype=torch.bool,
                    device=device,
                ),  # delivery nodes are not available
            ),
            dim=-1,
        )  # 1-available, 0-not available

        # Variable to record the delivery node for each agent,
        delivery_record = torch.zeros(
            (*batch_size, num_agents, num_loc_tot), dtype=torch.int64, device=device
        )

        # Number of orders for each agent
        num_orders = torch.zeros(
            (*batch_size, num_agents), dtype=torch.int64, device=device
        )

        return TensorDict(
            {
                "locs": torch.cat([depots, cities], dim=-2),
                "current_length": torch.zeros(
                    *batch_size, num_agents, dtype=torch.float32, device=device
                ),
                "arrivetime_record": torch.zeros(
                    *batch_size, num_loc_tot, dtype=torch.float32, device=device
                ),
                "current_node": current_node,
                "depot_node": depot_node,  # depot node is the first node for each agent
                "last_outer_node": last_outer_node,  # last outer node is the last node for each agent except the depot
                "delivery_record": delivery_record,
                "available": available,
                "action_mask": action_mask,
                "i": torch.zeros(*batch_size, dtype=torch.int64, device=device),
                # Capacity or max orders
                "num_orders": num_orders,
                "capacity": td["capacity"][
                    ..., :num_agents
                ],  # remove padding capacity if any
                "lateness_weight": td["lateness_weight"],
            },
            batch_size=batch_size,
        )

    def _step(self, td: TensorDict) -> TensorDict:
        """Note: here variables like the actions are of size [B, num_agents]"""

        # Initial variables
        selected = td["action"]
        num_agents = td["current_node"].size(-1)

        num_cities = td["locs"].shape[-2] - num_agents
        num_pickup = num_agents + int(num_cities / 2)
        arrivetime = td["arrivetime_record"]

        # Use for debugging only
        if self.check_conflicts:
            self._check_conflicts(selected, num_agents)

        # Get the locations of the current node and the previous node and the depot
        cur_loc = gather_by_index(td["locs"], selected)

        # Update the current length
        backtodepot_flag = selected < num_agents

        if self.problem_mode == "open":
            # Update last outer node
            last_outer_node = td["last_outer_node"].clone()
            prev_loc = gather_by_index(td["locs"], last_outer_node)
            current_length = (
                td["current_length"]
                + self.get_distance(prev_loc, cur_loc) * (~backtodepot_flag).float()
            )
            # last_outer_node[~backtodepot_flag] = selected[~backtodepot_flag]
            ##
            last_outer_node = torch.where(backtodepot_flag, last_outer_node, selected)
            # current_length = td["current_length"] + self.get_distance(prev_loc, cur_loc) * (~backtodepot_flag).float()
        else:
            prev_loc = gather_by_index(
                td["locs"], td["current_node"]
            )  # current_node is the previous node
            current_length = td["current_length"] + self.get_distance(prev_loc, cur_loc)

        # Update the arrival time
        arrivetime = torch.scatter(arrivetime, -1, selected, current_length)

        # Update the visited node (available node)
        available = torch.scatter(td["available"], -1, selected, 0)

        stay_flag = selected == td["current_node"]

        # Update number of orders of agents, note this number is the current pickup orders
        # instead of the total finished order
        new_orders_flag = torch.where(
            (selected < num_pickup) & (selected >= num_agents), 1, 0
        )
        new_orders_flag &= ~stay_flag
        finish_orders_flag = torch.where(selected >= num_pickup, 1, 0)
        finish_orders_flag &= ~stay_flag

        num_orders = td["num_orders"] + new_orders_flag - finish_orders_flag

        # Update the delivery record
        delivery_record = torch.scatter(
            td["delivery_record"],
            -1,
            torch.where(selected < num_pickup, selected, selected - int(num_cities / 2))[
                ..., None
            ],
            0,
        )
        delivery_record.scatter_(-1, selected[..., None], 1)

        # We are done there are no unvisited locations except the depot
        done = torch.sum(available[..., num_agents:], dim=-1) == 0

        # The reward is calculated outside via get_reward for efficiency, so we set it to -inf here
        reward = torch.zeros_like(done)

        # Update current
        td.update(
            {
                "current_length": current_length,
                "arrivetime_record": arrivetime,
                "current_node": selected,
                "delivery_record": delivery_record,
                "last_outer_node": last_outer_node,
                "num_orders": num_orders,
                "available": available,
                "i": td["i"] + 1,
                "done": done,
                "reward": reward,
            }
        )
        # Close and open problem have the same action mask calculation
        # NOTE: in open problem, depot may be added in actions after first but not counted towards reward as it is
        # is actually a dummy action
        td.set("action_mask", self.get_action_mask(td))
        return td

    def _check_conflicts(self, selected: torch.Tensor, num_agents: int):
        """Note: may be slow. Better disable this"""
        # Check locations are visited only once. Each agents has its own depot,
        # so we just need to check if there are no duplicate values in the selected nodes
        unique, counts = torch.unique(selected, return_counts=True, dim=-1)
        if (counts > 1).any():
            raise ValueError(f"Duplicate values in selected nodes: {unique[counts > 1]}")

    def _get_reward(self, td: TensorDict, action: torch.Tensor) -> torch.Tensor:
        """Return the reward for the current state (negative cost)

        Modes:
            - minmax: the reward is the maximum length of all agents
            - minsum: the reward is the sum of all agents' length
            - lateness: the reward is the sum of all agents' length plus the lateness with a weight
            - lateness_square: same as lateness but the lateness is squared
        """
        if self.reward_mode == "minmax":
            cost = torch.max(td["current_length"], dim=-1)[0]
        elif self.reward_mode == "minsum":
            cost = torch.sum(td["current_length"], dim=(-1))
        elif self.reward_mode in ["lateness_square", "lateness"]:
            # SECTION: get the cost (route length)
            cost = torch.sum(td["current_length"], dim=(-1))
            # SECTION: get the lateness (delivery time)
            num_agents = td["current_node"].size(-1)
            num_cities = td["locs"].shape[-2] - num_agents
            num_pickup = num_agents + int(num_cities / 2)
            lateness = td["arrivetime_record"][..., num_pickup:]
            if self.reward_mode == "lateness_square":
                lateness = lateness**2
            lateness = torch.sum(lateness, dim=-1)
            # lateness weight - note that if this is 0, the reward is the same as the cost
            # if this is 1, the reward is the same as the lateness
            cost = (
                cost * (1 - td["lateness_weight"].squeeze())
                + lateness * td["lateness_weight"].squeeze()
            )
        else:
            raise NotImplementedError(
                f"Invalid reward mode: {self.reward_mode}. Available modes: minmax, minsum, lateness_square, lateness"
            )
        return -cost  # minus for reward

    def get_distance(self, prev_loc, cur_loc):
        # Use L1 norm to calculate the distance for Manhattan distance
        if self.dist_norm == "L1":
            return torch.abs(cur_loc - prev_loc).norm(p=1, dim=-1)
        elif self.dist_norm == "L2":
            return torch.abs(cur_loc - prev_loc).norm(p=2, dim=-1)
        else:
            raise ValueError(f"Invalid distance norm: {self.dist_norm}")

    # @profile
    def get_action_mask(self, td: TensorDict) -> torch.Tensor:
        device = td.device

        action_mask = torch.repeat_interleave(
            td["available"][..., None, :], dim=-2, repeats=td["current_node"].shape[-1]
        )
        num_agents = td["current_node"].size(-1)
        num_cities = td["locs"].shape[-2] - num_agents
        num_pickup = num_agents + int(num_cities / 2) - 1

        # Status flag for that if an agent picked up something: 1-picked up something, 0-free status
        # Shape: [B, num_agents]
        pickup_flag = (
            torch.sum(td["delivery_record"][..., num_agents : num_pickup + 1], dim=-1) > 0
        )

        # Dilivery node only available when the agent picked up the matched item
        action_mask[..., num_pickup + 1 :] &= td["delivery_record"][
            ..., num_agents : num_pickup + 1
        ].bool()

        # Mask for agents went out and came back to depot
        # 1-went back and back to the depot; 0-still outside or init conflit in the depot;
        is_back_agent_mask = torch.logical_xor(
            td["current_node"] >= num_agents, td["current_length"]
        )
        is_back_agent_mask &= ~(td["current_node"] >= num_agents)

        # Mask for agents reached the max order
        reach_max_order_flag = td["num_orders"] >= td["capacity"]
        action_mask[..., num_agents : num_pickup + 1] &= ~reach_max_order_flag[..., None]

        # If back_agent_mask is True, set all nodes to be unavailable except the depot
        action_mask &= ~is_back_agent_mask[..., None]

        # Depot is available for agents: 1. back and stay in the depot; 2. finished one pickup and delivery
        action_mask[..., 0] = is_back_agent_mask | torch.logical_and(
            td["current_node"] >= num_agents, td["current_length"]
        )

        # If an agent picked up something, it can not go back to the depot
        action_mask[..., 0] &= ~pickup_flag

        # If all items are picked up, make the depot available to avoid the bug of existing the extream case:
        # agents can not select any node
        all_picked_up_flag = (
            torch.sum(td["available"][..., num_agents : num_pickup + 1], dim=-1) == 0
        )
        action_mask[..., 0] |= all_picked_up_flag[..., None] & ~pickup_flag

        # Check if all agents came back to the depot before finishing all nodes
        all_back_flag = torch.sum(td["current_node"] >= num_agents, dim=-1) == 0
        has_finished_early = (all_back_flag != td["done"]) & all_back_flag

        # If all agents come back to the depot before finishing all nodes, make all unfinished nodes available again and make the depot unavailable
        available_pickup = td["available"].clone()
        available_pickup[..., num_pickup + 1 :] = 0
        action_mask |= (
            has_finished_early[..., None, None] & available_pickup[..., None, :]
        )
        action_mask[..., 0] &= ~has_finished_early[..., None]

        # If done, set depot to available to pad the action sequence
        action_mask[..., 0] |= td["done"][..., None]

        # Create an eye matrix to extract the num_agent'th value in the num_node dimension for each batch
        eye_matrix = torch.eye(num_agents, device=device)
        eye_matrix = eye_matrix[None, ...].repeat(td["locs"].size(0), 1, 1).bool()
        eye_matrix &= action_mask[..., 0][..., None]

        # Update the original tensor using the mask
        action_mask[..., :num_agents] = eye_matrix

        return action_mask

    def _make_spec(self, td_params: TensorDict = None):
        # Looks like this is needed somehow
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))
        self.done_spec = BoundedTensorSpec(shape=(1,), dtype=torch.bool, low=0, high=1)
        pass

    @staticmethod
    def render(*args, **kwargs):
        return render(*args, **kwargs)
