import matplotlib.pyplot as plt
import torch

from matplotlib import cm
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def render(td, actions=None, ax=None, plot_depot_transition=True, **kwargs):
    # Process the data
    td = td.detach().cpu()

    if actions is None:
        actions = td.get("action", None)

    # if batch_size greater than 0 , we need to select the first batch element
    if td.batch_size != torch.Size([]):
        td = td[0]
        actions = actions[0]

    num_agents = td["current_node"].size(-1)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    # Plot Depot
    ax.scatter(
        td["locs"][0, 0], td["locs"][0, 1], marker="s", color="r", s=100, label="Depot"
    )

    # Plot Customers
    ax.scatter(
        td["locs"][num_agents:, 0],
        td["locs"][num_agents:, 1],
        marker="o",
        color="gray",
        s=30,
        label="Customers",
    )

    # Plot Actions
    # add as first action of all the agents the depot (which is agent_idx)
    actions_first = torch.arange(num_agents).unsqueeze(0).expand(actions.size(0), -1)
    actions = torch.cat([actions_first, actions, actions_first], dim=-1)

    for agent_idx in range(num_agents):
        agent_action = actions[agent_idx]
        for action_idx in range(agent_action.size(0) - 1):
            from_loc = td["locs"][agent_action[action_idx]]
            to_loc = td["locs"][agent_action[action_idx + 1]]

            # if it is to or from depot, raise flag
            if (
                agent_action[action_idx] == agent_idx
                or agent_action[action_idx + 1] == agent_idx
            ):
                depot_transition = True
            else:
                depot_transition = False

            if depot_transition:
                if plot_depot_transition:
                    ax.plot(
                        [from_loc[0], to_loc[0]],
                        [from_loc[1], to_loc[1]],
                        color=cm.Set2(agent_idx),
                        lw=0.3,
                        linestyle="--",
                    )

            else:
                ax.plot(
                    [from_loc[0], to_loc[0]],
                    [from_loc[1], to_loc[1]],
                    color=cm.Set2(agent_idx),
                    lw=1,
                )

    # Plot Configs
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # remove axs labels
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
