import argparse
import logging
import os
import sys

import numpy as np

from rl4co.data.utils import check_extension
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def generate_env_data(env_type, *args, **kwargs):
    """Generate data for a given environment type in the form of a dictionary"""
    try:
        # breakpoint()
        # remove all None values from args
        args = [arg for arg in args if arg is not None]

        return getattr(sys.modules[__name__], f"generate_{env_type}_data")(
            *args, **kwargs
        )
    except AttributeError:
        raise NotImplementedError(f"Environment type {env_type} not implemented")


def generate_omdcpdp_data(
    batch_size=128,
    num_loc=200,
    min_loc=0.0,
    max_loc=1.0,
    num_agents=5,
    use_different_depot_locations=True,
    capacity_min=3,
    capacity_max=3,
    min_lateness_weight=1.0,
    max_lateness_weight=1.0,
):
    """
    Generate a batch of data for the omdcpdp problem.

    Parameters:
    batch_size (int): Number of samples in the batch. Default is 128.
    num_loc (int): Total number of locations (pickups and deliveries). Default is 200.
    min_loc (float): Minimum value for location coordinates. Default is 0.0.
    max_loc (float): Maximum value for location coordinates. Default is 1.0.
    num_agents (int): Number of agents involved. Default is 5.
    use_different_depot_locations (bool): Whether to use different depot locations for each agent. Default is True.
    capacity_min (int): Minimum capacity for each agent. Default is 1.
    capacity_max (int): Maximum capacity for each agent. Default is 3.
    min_lateness_weight (float): Minimum lateness weight. Default is 1.0.
    max_lateness_weight (float): Maximum lateness weight. Default is 1.0.

    Returns:
    dict: A dictionary containing generated data arrays for depots, locations, number of agents, lateness weight, and capacity.
    """
    batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
    num_orders = int(num_loc / 2)

    # Generate the pickup and delivery locations
    pickup_locs = np.random.uniform(min_loc, max_loc, (*batch_size, num_orders, 2))
    delivery_locs = np.random.uniform(min_loc, max_loc, (*batch_size, num_orders, 2))

    # Generate depots
    n_diff_depots = num_agents if use_different_depot_locations else 1
    depots = np.random.uniform(min_loc, max_loc, (*batch_size, n_diff_depots, 2))

    # Initialize num_agents and capacity
    num_agents_array = np.ones((*batch_size,), dtype=np.int64) * n_diff_depots
    capacity = np.random.randint(
        capacity_min, capacity_max + 1, (*batch_size, np.max(num_agents_array))
    )

    # Combine pickup and delivery locations
    cities = np.concatenate([pickup_locs, delivery_locs], axis=-2)

    # Generate lateness weight
    lateness_weight = np.random.uniform(
        min_lateness_weight, max_lateness_weight, (*batch_size, 1)
    )

    data_dict = {
        "depots": depots.astype(np.float32),
        "locs": cities.astype(np.float32),  # Note: 'locs' does not include depot
        "num_agents": num_agents_array,
        "lateness_weight": lateness_weight.astype(np.float32),
        "capacity": capacity,
    }

    return data_dict


def generate_hcvrp_data(dataset_size, graph_size, num_agents=3):
    """Same dataset as 2D-Ptr paper
    https://github.com/farkguidao/2D-Ptr
    Note that we set the seed outside of this function
    """

    loc = np.random.uniform(0, 1, size=(dataset_size, graph_size + 1, 2))
    depot = loc[:, -1]
    cust = loc[:, :-1]
    d = np.random.randint(1, 10, [dataset_size, graph_size + 1])
    d = d[:, :-1]  # the demand of depot is 0, which do not need to generate here

    # vehicle feature
    speed = np.random.uniform(0.5, 1, size=(dataset_size, num_agents))
    cap = np.random.randint(20, 41, size=(dataset_size, num_agents))

    data = {
        "depot": depot.astype(np.float32),
        "locs": cust.astype(np.float32),
        "demand": d.astype(np.float32),
        "capacity": cap.astype(np.float32),
        "speed": speed.astype(np.float32),
    }
    return data


def generate_dataset(
    filename=None,
    data_dir="data",
    name=None,
    problem="hcvrp",
    dataset_size=10000,
    graph_sizes=[20, 50, 100],
    overwrite=False,
    seed=1234,
    disable_warning=True,
    **kwargs,
):
    """We keep a similar structure as in Kool et al. 2019 but save and load the data as npz
    This is way faster and more memory efficient than pickle and also allows for easy transfer to TensorDict
    """

    fname = filename
    if isinstance(graph_sizes, int):
        graph_sizes = [graph_sizes]
    for graph_size in graph_sizes:
        datadir = os.path.join(data_dir, problem)
        os.makedirs(datadir, exist_ok=True)

        if filename is None:
            fname = os.path.join(
                datadir,
                "{}{}_seed{}.npz".format(
                    graph_size,
                    "_{}".format(name) if name is not None else "",
                    seed,
                ),
            )
        else:
            fname = check_extension(filename, extension=".npz")

        if not overwrite and os.path.isfile(check_extension(fname, extension=".npz")):
            if not disable_warning:
                log.info(
                    "File {} already exists! Run with -f option to overwrite. Skipping...".format(
                        fname
                    )
                )
            continue

        # Set seed
        np.random.seed(seed)

        # Automatically generate dataset
        dataset = generate_env_data(problem, dataset_size, graph_size, **kwargs)

        # A function can return None in case of an error or a skip
        if dataset is not None:
            # Save to disk as dict
            log.info("Saving {} dataset to {}".format(problem, fname))
            np.savez(fname, **dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename", help="Filename of the dataset to create (ignores datadir)"
    )
    parser.add_argument(
        "--data_dir",
        default="data",
        help="Create datasets in data_dir/problem (default 'data')",
    )
    parser.add_argument(
        "--name", type=str, required=True, help="Name to identify dataset"
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="all",
        help="Problem" " or 'all' to generate all",
    )
    parser.add_argument(
        "--dataset_size", type=int, default=10000, help="Size of the dataset"
    )
    parser.add_argument(
        "--graph_sizes",
        type=int,
        nargs="+",
        default=[50, 100],
        help="Sizes of problem instances (default 20, 50, 100)",
    )
    parser.add_argument("-f", action="store_true", help="Set true to overwrite")
    parser.add_argument("--seed", type=int, default=3333, help="Random seed")
    parser.add_argument("--disable_warning", action="store_true", help="Disable warning")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    args.overwrite = args.f
    delattr(args, "f")
    generate_dataset(**vars(args))
