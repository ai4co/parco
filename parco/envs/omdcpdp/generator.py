import torch

from rl4co.envs.common.utils import Generator
from rl4co.utils.pylogger import get_pylogger
from tensordict.tensordict import TensorDict

log = get_pylogger(__name__)


class OMDCPDPGenerator(Generator):
    def __init__(
        self,
        num_loc: int = 200,
        num_agents: int = 40,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        capacity_min: int = 3,
        capacity_max: int = 3,
        min_lateness_weight: float = 1.0,
        max_lateness_weight: float = 1.0,
        use_different_depot_locations: bool = True,
    ):
        self.num_loc = num_loc
        self.num_agents = num_agents
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.capacity_min = capacity_min
        self.capacity_max = capacity_max
        self.min_lateness_weight = min_lateness_weight
        self.max_lateness_weight = max_lateness_weight
        self.use_different_depot_locations = use_different_depot_locations

    def _generate(self, batch_size) -> TensorDict:
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        num_orders = int(self.num_loc / 2)

        # Generate the pickup locations
        pickup_locs = torch.FloatTensor(*batch_size, num_orders, 2).uniform_(
            self.min_loc, self.max_loc
        )

        # Generate the delivery locations
        delivery_locs = torch.FloatTensor(*batch_size, num_orders, 2).uniform_(
            self.min_loc, self.max_loc
        )

        # Depots: if we use different depot locations, we have to generate them randomly. Otherwise, we just copy the first node
        n_diff_depots = self.num_agents if self.use_different_depot_locations else 1
        depots = torch.FloatTensor(*batch_size, n_diff_depots, 2).uniform_(
            self.min_loc, self.max_loc
        )

        # Initialize the num_agents: either fixed or random integer between min and max
        num_agents = torch.ones(*batch_size, dtype=torch.int64) * n_diff_depots

        if self.capacity_min == self.capacity_max:
            # homogeneous capacity
            capacity = (
                torch.zeros(
                    *batch_size,
                    num_agents.max().item(),
                    dtype=torch.int64,
                )
                + self.capacity_min
            )
        else:
            # heterogeneous capacity
            capacity = torch.randint(
                self.capacity_min,
                self.capacity_max + 1,
                (*batch_size, num_agents.max().item()),
            )

        cities = torch.cat([pickup_locs, delivery_locs], dim=-2)

        # Lateness weight - note that if this is 0, the reward is the same as the cost.
        # If this is 1, the reward is the same as the lateness
        lateness_weight = torch.FloatTensor(*batch_size, 1).uniform_(
            self.min_lateness_weight, self.max_lateness_weight
        )

        return TensorDict(
            {
                "depots": depots,
                "locs": cities,  # NOTE: here locs does NOT include depot
                "num_agents": num_agents,
                "lateness_weight": lateness_weight,
                "capacity": capacity,
            },
            batch_size=batch_size,
        )
