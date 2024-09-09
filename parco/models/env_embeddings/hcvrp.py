import torch
import torch.nn as nn

from rl4co.utils.ops import gather_by_index

from parco.models.nn.positional_encoder import PositionalEncoder

from .communication import BaseMultiAgentContextEmbedding


class HCVRPInitEmbedding(nn.Module):
    """TODO: description
    Note that in HCVRP capacities are not the same for all agents and
    they need to be rescaled.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        linear_bias: bool = False,
        demand_scaler: float = 40.0,
        speed_scaler: float = 1.0,
        use_polar_feats: bool = True,
    ):
        super(HCVRPInitEmbedding, self).__init__()
        # depot feats: [x0, y0]
        self.init_embed_depot = nn.Linear(2, embed_dim, linear_bias)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.pos_embedding_proj = nn.Linear(embed_dim, embed_dim, linear_bias)
        self.alpha = nn.Parameter(torch.Tensor([1]))
        # agent feats: [x0, y0, capacity, speed]
        self.init_embed_agents = nn.Linear(4, embed_dim, linear_bias)
        # combine depot and agent embeddings
        self.init_embed_depot_agents = nn.Linear(2 * embed_dim, embed_dim, linear_bias)
        # client feats: [x, y, demand]
        client_feats_dim = 5 if use_polar_feats else 3
        self.init_embed_clients = nn.Linear(client_feats_dim, embed_dim, linear_bias)

        self.demand_scaler = demand_scaler
        self.speed_scaler = speed_scaler
        self.use_polar_feats = use_polar_feats

    def forward(self, td):
        num_agents = td["action_mask"].shape[-2]  # [B, m, m+N]
        depot_locs = td["locs"][..., :num_agents, :]
        agents_locs = td["locs"][..., :num_agents, :]
        clients_locs = td["locs"][..., num_agents:, :]

        # Depots embedding with positional encoding
        depots_embedding = self.init_embed_depot(depot_locs)
        pos_embedding = self.pos_encoder(depots_embedding, add=False)
        pos_embedding = self.alpha * self.pos_embedding_proj(pos_embedding)
        depot_embedding = depots_embedding + pos_embedding

        # Agents embedding
        agents_feats = torch.cat(
            [
                agents_locs,
                td["capacity"][..., None] / self.demand_scaler,
                td["speed"][..., None] / self.speed_scaler,
            ],
            dim=-1,
        )
        agents_embedding = self.init_embed_agents(agents_feats)

        # Combine depot and agents embeddings
        depot_agents_feats = torch.cat([depot_embedding, agents_embedding], dim=-1)
        depot_agents_embedding = self.init_embed_depot_agents(depot_agents_feats)

        # Clients embedding
        demands = td["demand"][
            ..., 0, num_agents:
        ]  # [B, N] , note that demands is repeated but the same in the beginning
        clients_feats = torch.cat(
            [clients_locs, demands[..., None] / self.demand_scaler], dim=-1
        )

        if self.use_polar_feats:
            # Convert to polar coordinates
            depot = depot_locs[..., 0:1, :]
            client_locs_centered = clients_locs - depot  # centering
            dist_to_depot = torch.norm(client_locs_centered, p=2, dim=-1, keepdim=True)
            angle_to_depot = torch.atan2(
                client_locs_centered[..., 1:], client_locs_centered[..., :1]
            )
            clients_feats = torch.cat(
                [clients_feats, dist_to_depot, angle_to_depot], dim=-1
            )

        clients_embedding = self.init_embed_clients(clients_feats)

        return torch.cat(
            [depot_agents_embedding, clients_embedding], -2
        )  # [B, m+N, hdim]


class HCVRPContextEmbedding(BaseMultiAgentContextEmbedding):

    """TODO"""

    def __init__(
        self,
        embed_dim,
        agent_feat_dim=2,
        global_feat_dim=1,
        demand_scaler=40.0,
        speed_scaler=1.0,
        use_time_to_depot=True,
        **kwargs,
    ):
        if use_time_to_depot:
            agent_feat_dim += 1
        super(HCVRPContextEmbedding, self).__init__(
            embed_dim, agent_feat_dim, global_feat_dim, **kwargs
        )
        self.demand_scaler = demand_scaler
        self.speed_scaler = speed_scaler
        self.use_time_to_depot = use_time_to_depot

    def _agent_state_embedding(self, embeddings, td, num_agents, num_cities):
        context_feats = torch.stack(
            [
                td["current_length"]
                / (td["agents_speed"] / self.speed_scaler),  # current time
                (td["agents_capacity"] - td["used_capacity"])
                / self.demand_scaler,  # remaining capacity
            ],
            dim=-1,
        )
        if self.use_time_to_depot:
            depot = td["locs"][..., 0:1, :]
            cur_loc = gather_by_index(td["locs"], td["current_node"])
            dist_to_depot = torch.norm(cur_loc - depot, p=2, dim=-1, keepdim=True)
            time_to_depot = dist_to_depot / (
                td["agents_speed"][..., None] / self.speed_scaler
            )
            context_feats = torch.cat([context_feats, time_to_depot], dim=-1)
        return self.proj_agent_feats(context_feats)

    def _global_state_embedding(self, embeddings, td, num_agents, num_cities):
        global_feats = torch.cat(
            [
                td["visited"][..., num_agents:].sum(-1)[..., None]
                / num_cities,  # number of visited cities / total
            ],
            dim=-1,
        )
        return self.proj_global_feats(global_feats)[..., None, :].repeat(1, num_agents, 1)
