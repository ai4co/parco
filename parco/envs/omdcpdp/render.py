import torch

from matplotlib.axes import Axes
from tensordict import TensorDict


def render(
    td: TensorDict,
    actions: torch.Tensor = None,
    ax: Axes = None,
    batch_idx: int = None,
    plot_number: bool = False,
    add_depot_to_actions: bool = True,
    print_subtours: bool = False,
    problem_mode: str = "open",
):
    """Visualize the solution of the problem
    Args:
        actions <torch.Tensor> [batch_size, num_agents * steps]: from i*steps to (i+1)*steps-1
            are the actions of agent i
    """
    import matplotlib.pyplot as plt

    num_agents = int(td["num_agents"].max().item())
    num_cities = td["locs"].shape[-2] - num_agents
    num_pickup = num_agents + int(num_cities / 2)

    def draw_line(src, dst, ax):
        ax.plot([src[0], dst[0]], [src[1], dst[1]], ls="--", c="gray")

    td = td.detach().cpu()

    if actions is None:
        actions = td.get("action", None)

    if td.batch_size != torch.Size([]):
        batch_idx = 0 if batch_idx is None else batch_idx
        td = td[0]
        actions = actions[0]

    if ax is None:
        # Create a plot of the nodes
        _, ax = plt.subplots(1, 1, figsize=(4, 4))

    ax.axis("equal")

    # Plot cities
    loc = td["locs"]

    # Plot the pickup cities
    ax.scatter(
        loc[num_agents:num_pickup, 0],
        loc[num_agents:num_pickup, 1],
        c="black",
        s=30,
        marker="^",
    )

    # Plot the delivery cities
    ax.scatter(loc[num_pickup:, 0], loc[num_pickup:, 1], c="black", s=30, marker="x")

    # Plot the depot
    ax.scatter(loc[:num_agents, 0], loc[:num_agents, 1], c="red", s=50, marker="s")

    # Plot number
    if plot_number:
        # Annotate the pickup cities
        for i, xy in enumerate(loc[num_agents:num_pickup]):
            ax.annotate(
                f"p{i}", xy=xy, textcoords="offset points", xytext=(0, 5), ha="center"
            )
        # Annotate the delivery cities
        for i, xy in enumerate(loc[num_pickup:]):
            ax.annotate(
                f"d{i}", xy=xy, textcoords="offset points", xytext=(0, 5), ha="center"
            )

    # Plot line connecting pickup and delivery
    for i in range(num_agents, num_pickup):
        draw_line(loc[i], loc[i + num_pickup - num_agents], ax)

    if actions is not None:  # draw solution if available.
        sub_tours = actions.reshape(num_agents, -1)
        loc = td["locs"]
        for v_i, sub_tour in enumerate(sub_tours):
            if add_depot_to_actions:
                d_ = torch.zeros(1, dtype=torch.int64) + v_i
                sub_tour = torch.cat([d_, sub_tour])
            if problem_mode == "open":
                # If the agent goes back to depot, do not plot the line
                init = sub_tour[0]
                sub_tour = sub_tour[sub_tour >= num_agents]
                sub_tour = torch.cat([init.unsqueeze(0), sub_tour])
            if print_subtours:
                print(f"Agent {v_i}: {sub_tour.numpy()}")
            ax.plot(loc[sub_tour][:, 0], loc[sub_tour][:, 1], color=f"C{v_i}")

    _ = ax.set_xlim(-0.05, 1.05)
    _ = ax.set_ylim(-0.05, 1.05)
