from typing import Callable, Union

import torch

from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.pylogger import get_pylogger
from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

log = get_pylogger(__name__)


class HCVRPGenerator(Generator):
    """Data generator for the Heterogeneous Capacitated Vehicle Routing Problem (HCVRP).

    Args:
        - num_loc: Number of customers.
        - min_loc: Minimum location of the customers.
        - max_loc: Maximum location of the customers.
        - loc_distribution: Distribution of the locations of the customers.
        - depot_distribution: Distribution of the location of the depot.
        - min_demand: Minimum demand of the customers.
        - max_demand: Maximum demand of the customers.
        - demand_distribution: Distribution of the demand of the customers.
        - min_capacity: Minimum capacity of the agents.
        - max_capacity: Maximum capacity of the agents.
        - capacity_distribution: Distribution of the capacity of the agents.
        - min_speed: Minimum speed of the agents.
        - max_speed: Maximum speed of the agents.
        - speed_distribution: Distribution of the speed of the agents.
        - num_agents: Number of agents.

    Returns:
        A TensorDict containing the following keys:
            - locs [batch_size, num_loc, 2]: locations of the customers
            - depot [batch_size, 2]: location of the depot
            - demand [batch_size, num_loc]: demand of the customers
            - capacity [batch_size, num_agents]: capacity of the agents, different for each agents
            - speed [batch_size, num_agents]: speed of the agents, different for each agents

    Notes:
        - The capacity setting from 2D-Ptr paper is hardcoded to 20~41. It should change
            based on the size of the problem.
        - ? Demand and capacity are initialized as integers and then converted to floats.
            To avoid zero demands, we first sample from [min_demand - 1, max_demand - 1]
            and then add 1 to the demand.
        - ! Note that here the demand is not normalized by the capacity by default.
    """

    def __init__(
        self,
        num_loc: int = 40,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: Union[int, float, str, type, Callable] = Uniform,
        depot_distribution: Union[int, float, str, type, Callable] = None,
        min_demand: int = 1,
        max_demand: int = 10,
        demand_distribution: Union[int, float, type, Callable] = Uniform,
        min_capacity: float = 20,
        max_capacity: float = 41,
        capacity_distribution: Union[int, float, type, Callable] = Uniform,
        min_speed: float = 0.5,
        max_speed: float = 1.0,
        speed_distribution: Union[int, float, type, Callable] = Uniform,
        num_agents: int = 3,
        # if False, we don't normalize by capacity and speed
        # note that we are doing this in environment side for convenience
        scale_data: bool = False,  # leave False!
        **kwargs,
    ):
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.min_demand = min_demand
        self.max_demand = max_demand
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.num_agents = num_agents
        self.scale_data = scale_data

        # Location distribution
        if kwargs.get("loc_sampler", None) is not None:
            self.loc_sampler = kwargs["loc_sampler"]
        else:
            self.loc_sampler = get_sampler(
                "loc", loc_distribution, min_loc, max_loc, **kwargs
            )

        # Depot distribution
        if kwargs.get("depot_sampler", None) is not None:
            self.depot_sampler = kwargs["depot_sampler"]
        else:
            self.depot_sampler = (
                get_sampler("depot", depot_distribution, min_loc, max_loc, **kwargs)
                if depot_distribution is not None
                else None
            )

        # Demand distribution
        if kwargs.get("demand_sampler", None) is not None:
            self.demand_sampler = kwargs["demand_sampler"]
        else:
            self.demand_sampler = get_sampler(
                "demand", demand_distribution, min_demand - 1, max_demand - 1, **kwargs
            )

        # Capacity
        if kwargs.get("capacity_sampler", None) is not None:
            self.capacity_sampler = kwargs["capacity_sampler"]
        else:
            self.capacity_sampler = get_sampler(
                "capacity",
                capacity_distribution,
                0,
                max_capacity - min_capacity,
                **kwargs,
            )

        # Speed
        if kwargs.get("speed_sampler", None) is not None:
            self.speed_sampler = kwargs["speed_sampler"]
        else:
            self.speed_sampler = get_sampler(
                "speed", speed_distribution, min_speed, max_speed, **kwargs
            )

    def _generate(self, batch_size) -> TensorDict:
        # Sample locations: depot and customers
        if self.depot_sampler is not None:
            depot = self.depot_sampler.sample((*batch_size, 2))
            locs = self.loc_sampler.sample((*batch_size, self.num_loc, 2))
        else:
            # If depot_sampler is None, sample the depot from the locations
            locs = self.loc_sampler.sample((*batch_size, self.num_loc + 1, 2))
            depot = locs[..., 0, :]
            locs = locs[..., 1:, :]

        # Sample demands
        demand = self.demand_sampler.sample((*batch_size, self.num_loc))
        demand = (demand.int() + 1).float()

        # Sample capacities
        capacity = self.capacity_sampler.sample((*batch_size, self.num_agents))
        capacity = (capacity.int() + self.min_capacity).float()

        # Sample speed
        speed = self.speed_sampler.sample((*batch_size, self.num_agents))

        return TensorDict(
            {
                "locs": locs,
                "depot": depot,
                "num_agents": torch.full(
                    (*batch_size,), self.num_agents
                ),  # for compatibility
                "demand": demand / self.max_capacity if self.scale_data else demand,
                "capacity": capacity / self.max_capacity if self.scale_data else capacity,
                "speed": speed / self.max_speed if self.scale_data else speed,
            },
            batch_size=batch_size,
        )
