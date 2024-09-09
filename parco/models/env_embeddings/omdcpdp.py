import torch
import torch.nn as nn

from .communication import BaseMultiAgentContextEmbedding


class OMDCPDPInitEmbedding(nn.Module):
    """Encoder for the initial state of the OMDCPDP environment
    Encode the initial state of the environment into a fixed-size vector
    Features:
    - vehicle: initial position, capacity
    - pickup: location, corresponding delivery location
    - delivery: location, corresponding pickup location
    Note that we do not encode the distance from depot (i.e. initial position) to the pickup/delivery nodes since
    the problem is open.
    """

    def __init__(self, embed_dim: int):
        super(OMDCPDPInitEmbedding, self).__init__()
        # extra_feat: for vehicle: capacity, for pickup: early time, for delivery: ordered time
        node_dim = 2
        self.init_embed_vehicle = nn.Linear(
            node_dim + 1, embed_dim
        )  # vehicle has initial position and capacity
        self.init_embed_pick = nn.Linear(
            node_dim * 2, embed_dim
        )  # concatenate pickup and delivery
        self.init_embed_delivery = nn.Linear(
            node_dim * 2, embed_dim
        )  # concatenate delivery and pickup

    def forward(self, td):
        # [B, M, 2] , where M = num_agents + num_pickup + num_delivery (num_pickup = num_delivery)
        # num_agents = int(td["num_agents"].max().item())
        num_agents = td["current_node"].size(-1)
        num_cities = td["locs"].shape[-2] - num_agents
        num_pickup = num_agents + int(
            num_cities / 2
        )  # this is the _index_ of the first delivery node
        capacities = td[
            "capacity"
        ]  # in case there are different capacities (heteregenous)
        locs = td["locs"]
        depot_locs, pickup_locs, delivery_locs = (
            locs[..., :num_agents, :],
            locs[..., num_agents:num_pickup, :],
            locs[..., num_pickup:, :],
        )

        vehicle_feats = torch.cat([depot_locs, capacities[..., None]], dim=-1)
        vehicle_embed = self.init_embed_vehicle(vehicle_feats)
        pickup_embed = self.init_embed_pick(
            torch.cat(
                [pickup_locs, delivery_locs], dim=-1
            )  # merge feats of pickup and delivery
        )
        delivery_embed = self.init_embed_delivery(
            torch.cat(
                [delivery_locs, pickup_locs], dim=-1
            )  # merge feats of delivery and pickup
        )
        return torch.cat(
            [vehicle_embed, pickup_embed, delivery_embed], dim=-2
        )  # [B, N, hdim]


class OMDCPDPContextEmbedding(BaseMultiAgentContextEmbedding):
    """Context embedding for OMDCPDP
    Encode the following features:
    - Agent-wise state features: current length, number of orders
    - Global state features: number of visited cities
    Note that pickup-delivery pairs and more are embedded in the initial node embeddings
    """

    def __init__(self, embed_dim, agent_feat_dim=2, global_feat_dim=1, **kwargs):
        super(OMDCPDPContextEmbedding, self).__init__(
            embed_dim, agent_feat_dim, global_feat_dim, **kwargs
        )

    def _agent_state_embedding(self, embeddings, td, num_agents, num_cities):
        context_feats = torch.cat(
            [
                td["current_length"][..., None],  # cost
                td["num_orders"][..., None].float(),  # capacity
            ],
            dim=-1,
        )
        return self.proj_agent_feats(context_feats)

    def _global_state_embedding(self, embeddings, td, num_agents, num_cities):
        global_feats = torch.cat(
            [
                td["available"][..., num_agents:].sum(-1)[..., None]
                / num_cities,  # number of visited cities / total
            ],
            dim=-1,
        )
        return self.proj_global_feats(global_feats)[..., None, :].repeat(1, num_agents, 1)
