import os

from parco.data.generate import generate_dataset


def generate_with_agents(problem, size_agents_dict, **kwargs):
    """Helper function to generate data for a problem with different number of agents."""
    for graph_size, num_agents_ in size_agents_dict.items():
        for num_agents in num_agents_:
            kwargs["num_agents"] = num_agents

            print(
                "Generating instances: N {}, m {}...".format(
                    problem.upper(),
                    graph_size,
                )
            )

            kwargs["graph_sizes"] = graph_size
            fname = os.path.join(
                kwargs["data_dir"],
                problem,
                "n{}_m{}_seed{}.npz".format(
                    kwargs["graph_sizes"],
                    kwargs["num_agents"],
                    kwargs["seed"],
                ),
            )

            generate_dataset(problem=problem, filename=fname, **kwargs)


if __name__ == "__main__":
    data_dir = "data"

    kwargs = {
        "data_dir": data_dir,
        "seed": 3333,
        "dataset_size": 100,
        "graph_sizes": 100,
        "num_agents": 100,  # NOTE: dummy, generate more for mixed graph sizes and agents training
    }

    problem = "hcvrp"
    print(50 * "=" + f"\nGenerating instances for {problem.upper()}...\n" + 50 * "=")
    kwargs.update({"seed": 24610, "dataset_size": 1280})  # same as 2D-Ptr paper
    size_agents_dict = {60: [3, 5, 7], 80: [3, 5, 7], 100: [3, 5, 7]}
    generate_with_agents(problem, size_agents_dict, **kwargs)

    problem = "omdcpdp"
    print(50 * "=" + f"\nGenerating instances for {problem.upper()}...\n" + 50 * "=")
    size_agents_dict = {50: [5, 10, 15], 100: [10, 20, 30], 200: [20, 40, 60]}
    generate_with_agents(problem, size_agents_dict, **kwargs)
