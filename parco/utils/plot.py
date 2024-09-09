import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm


def actions_table(actions, num_agent, num_city):
    """Visualize the actions in a table
    Args:
        actions: (num_agent, num_step)
        num_agent: int
        num_city: int
    """
    actions_reshape = actions.reshape(num_agent, -1)
    _, num_step = actions_reshape.shape
    action_table = np.zeros((num_city, num_step))

    # Plot
    _, ax = plt.subplots(1, 1, figsize=(int(num_step / 2), int(num_city / 2)))
    back_to_depot_list = np.zeros(num_agent)
    for step_idx in range(num_step):
        for agent_idx in range(num_agent):
            pd_flag = 0  # Flag for pickup or delivery
            if actions_reshape[agent_idx, step_idx] > 10:
                item_idx = actions_reshape[agent_idx, step_idx] - 10
                pd_flag = 2
            else:
                item_idx = actions_reshape[agent_idx, step_idx]
                pd_flag = 1
            action_table[item_idx, step_idx:] = pd_flag

            # Not text in depot
            if item_idx == 0:
                back_to_depot_list[agent_idx] = 1
            else:
                ax.text(
                    step_idx,
                    item_idx,
                    f"A{agent_idx}",
                    ha="center",
                    va="center",
                    color=cm.Set3(agent_idx),
                )

        # Text the number of agents back to depot in the depot
        ax.text(
            step_idx,
            0,
            f"{int(sum(back_to_depot_list))}",
            ha="center",
            va="center",
            color="white",
        )

    action_table[0, :] = 2
    ax.matshow(action_table, cmap=plt.cm.Greys)

    ax.set_xticks(range(num_step))
    ax.set_xticklabels(np.array(range(num_step)) + 1)
    ax.xaxis.tick_bottom()

    ytick_lable_list = np.array(range(num_city))
    ytick_lable_list = ["D"] + list(ytick_lable_list[1:])
    ax.set_yticks(range(num_city))
    ax.set_yticklabels(ytick_lable_list)

    ax.set_xlabel("Step")
    ax.set_ylabel("Item")

    plt.tight_layout()
