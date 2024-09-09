import torch

from rl4co.utils.ops import gather_by_index

from parco.models.agent_handlers import RandomAgentHandler


class Heuristic:
    """Heuristic base class for multi-agent decision making in parallel"""

    def __init__(
        self,
        norm_p=2,
    ):
        self.norm_p = norm_p

    def set_dist(self, td):
        locs = td["locs"]
        self.dmat = torch.cdist(locs, locs, p=self.norm_p)

    def get_action(self, td):
        raise NotImplementedError("Implement in subclass")

    def __call__(self, td, env):
        actions = []
        while not td["done"].all():
            action = self.get_action(td)
            td.set("action", action)
            td = env.step(td)["next"]
            actions.append(action)
        actions = torch.stack(actions, dim=1)  # [batch, num decoding steps]
        rewards = env.get_reward(td, actions)
        return {"reward": rewards, "actions": actions, "td": td}


class ParallelRandomInsertionHeuristic(Heuristic):
    """Random insertion heuristic for multi-agent decision making in parallel"""

    def __init__(self, *args, agent_handler=RandomAgentHandler(), **kwargs):
        self.agent_handler = agent_handler

    def get_action(self, td):
        actions = torch.distributions.Categorical(td["action_mask"]).sample()
        current_loc_idx = td["current_node"].clone()
        actions = self.agent_handler(actions, current_loc_idx, td)[0]  # handle conflicts
        return actions


class ParallelNearestInsertionHeuristic(Heuristic):
    """Nearest neighbour heuristic for multi-agent decision making in parallel"""

    def __init__(self, norm_p=2, mode="open"):
        self.norm_p = norm_p
        assert mode in ["open", "close"], "mode must be either 'open' or 'close'"
        self.mode = mode

    def get_action(self, td):
        if td["i"][0].item() == 0:
            actions = td["current_node"].clone()
            return actions
        else:
            if not hasattr(self, "dmat"):
                self.set_dist(td)
            if not hasattr(self, "num_agents"):
                self.num_agents = td["current_node"].shape[-1]
                self.num_locs = td["locs"].shape[-2]
            actions = []
            action_mask = td["action_mask"].clone()

            if "available" not in td:
                available = td["visited"].clone()
            else:
                available = td["available"].clone()

            # For loop over agents to avoid collisions
            for i in range(self.num_agents):
                cur_dist = gather_by_index(self.dmat, td["current_node"][:, i])
                # if available has more dims than cur_dist, then we need to expand cur_dist
                if len(available.shape) > len(cur_dist.shape):
                    cur_dist = cur_dist.unsqueeze(0)
                cur_dist[~available] = float("inf")  # [batch, num_nodes, num_agents]
                if self.mode == "open":
                    # make sure that the depot is not selected if problem is open
                    cur_dist[action_mask[:, i, :] is False] = float("inf")
                cur_dist[
                    torch.arange(cur_dist.shape[0]), td["current_node"][:, i]
                ] = float(100000)
                action = cur_dist.argmin(dim=-1)
                available.scatter_(-1, action.unsqueeze(-1), False)  # update action mask
                actions.append(action)

        return torch.stack(actions, dim=-1)
