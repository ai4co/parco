import torch

from tensordict import TensorDict


def scatter_at_index(src, idx):
    """Scatter elements from parco at index idx along specified dim

    Now this function is specific for the multi agent masking, you may
    want to create a general function.

    Example:
    >>> src: shape [64, 3, 20] # [batch_size, num_agents, num_nodes]
    >>> idx: shape [64, 3] # [batch_size, num_agents]
    >>> Returns: [64, 3, 20]
    """
    idx_ = torch.repeat_interleave(idx.unsqueeze(-2), dim=-2, repeats=src.shape[-2])
    return src.scatter(-1, idx_, 0)


def pad_tours(data):
    # Determine the maximum length from all the lists
    max_len = max(len(lst) for lst in data.values())

    # Pad each list to match the maximum length
    for key, value in data.items():
        data[key] = value + [0] * (max_len - len(value))

    # Make tensor
    tours = torch.tensor(list(data.values()))
    return tours


def pad_actions(tensors):
    # Determine the maximum length from all tensors
    max_len = max(t.size(1) for t in tensors)
    # Pad each tensor to match the maximum length
    padded_tensors = []
    for t in tensors:
        if t.size(1) < max_len:
            pad_size = max_len - t.size(1)
            pad = torch.zeros(t.size(0), pad_size).long()
            padded_t = torch.cat([t, pad], dim=-1)
        else:
            padded_t = t
        padded_tensors.append(padded_t)
    return torch.stack(padded_tensors)


def rollout(instances, actions, env, num_agents, preprocess_actions=True, verbose=True):
    assert env is not None, "Environment must be provided"

    if env.name == "mpdp":
        depots = instances["depots"]
        locs = instances["locs"]
        diff = num_agents - 1
    else:
        depots = instances[:, 0:1]
        locs = instances[:, 1:]
        diff = num_agents - 2

    td = TensorDict(
        {
            "depots": depots,
            "locs": locs,
            "num_agents": torch.tensor([num_agents] * locs.shape[0]),
        },
        batch_size=[locs.shape[0]],
    )

    td_init = env.reset(td)

    if preprocess_actions:
        # Make actions as follows: add to all numbers except 0 the number of agents
        actions_md = torch.where(actions > 0, actions + diff, actions)
    else:
        actions_md = actions

    # Rollout through the environment
    td_init_test = td_init.clone()
    next_td = td_init_test
    with torch.no_grad():
        # take actions from the last dimension
        for i in range(actions_md.shape[-1]):
            cur_a = actions_md[:, :, i]
            next_td.set("action", cur_a)
            next_td = env.step(next_td)["next"]

    # Plotting
    # env.render(td_init_test, actions_md, plot_number=True)
    reward = env.get_reward(next_td, actions_md)
    if verbose:
        print(f"Average reward: {reward.mean()}")
    # return instances, actions, actions_md, reward, next_td
    return {
        "instances": instances,
        "actions": actions,
        "actions_md": actions_md,
        "reward": reward,
        "td": next_td,
        "td_init": td_init,
    }
