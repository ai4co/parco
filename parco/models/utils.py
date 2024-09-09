import torch

from rl4co.utils.ops import gather_by_index


def replace_key_td(td, key, replacement):
    # TODO: check if best way in TensorDict?
    td.pop(key)
    td[key] = replacement
    return td


def resample_batch(td, num_agents, num_locs):
    # Remove depots until num_agents
    td.set_("num_agents", torch.full((*td.batch_size,), num_agents, device=td.device))
    if "depots" in td.keys():
        # note that if we have "depot" instead, this will automatically
        # be repeated inside the environment
        td = replace_key_td(td, "depots", td["depots"][..., :num_agents, :])

    if "pickup_et" in td.keys():
        # Ensure num_locs is even for omdcpdp
        num_locs = num_locs - 1 if num_locs % 2 == 0 else num_locs
        # also, set the "num_agents" key to the new number of agents
        td.set_("num_agents", torch.full((*td.batch_size,), num_agents, device=td.device))

    td = replace_key_td(td, "locs", td["locs"][..., :num_locs, :])

    # For early time windows
    if "pickup_et" in td.keys():
        td = replace_key_td(td, "pickup_et", td["pickup_et"][..., : num_locs // 2])
    if "delivery_et" in td.keys():
        td = replace_key_td(td, "delivery_et", td["delivery_et"][..., : num_locs // 2])

    # Capacities
    if "capacity" in td.keys():
        td = replace_key_td(td, "capacity", td["capacity"][..., :num_agents])

    if "speed" in td.keys():
        td = replace_key_td(td, "speed", td["speed"][..., :num_agents])

    if "demand" in td.keys():
        td = replace_key_td(td, "demand", td["demand"][..., :num_locs])

    return td


def get_log_likelihood(log_p, actions=None, mask=None, return_sum: bool = False):
    """Get log likelihood of selected actions

    Args:
        log_p: [batch, n_agents, (decode_len), n_nodes]
        actions: [batch, n_agents, (decode_len)]
        mask: [batch, n_agents, (decode_len)]
    """

    # NOTE: we do not use this since it is more inefficient, we do it in the decoder
    if actions is not None:
        if log_p.dim() > 3:
            log_p = gather_by_index(log_p, actions, dim=-1)

    # Optional: mask out actions irrelevant to objective so they do not get reinforced
    if mask is not None:
        log_p[mask] = 0

    assert (
        log_p > -1000
    ).data.all(), "Logprobs should not be -inf, check sampling procedure!"

    # Calculate log_likelihood
    # TODO: check the return sum argument.
    # TODO: Also, should we sum over agents too?
    if return_sum:
        return log_p.sum(-1)  # [batch, num_agents]
    else:
        return log_p  # [batch, num_agents, (decode_len)]
