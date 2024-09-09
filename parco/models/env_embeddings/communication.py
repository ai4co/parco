import torch
import torch.nn as nn

from rl4co.utils.ops import gather_by_index

from parco.models.nn.transformer import (
    Normalization,
    TransformerBlock as CommunicationLayer,
)


class BaseMultiAgentContextEmbedding(nn.Module):
    """Base class for multi-agent context embedding

    Args:
        embed_dim: int, size of the input and output embeddings
        agent_feat_dim: int, size of the agent-wise state features
        global_feat_dim: int, size of the global state features
        linear_bias: bool, whether to use bias in linear layers
        use_communication: bool, whether to use communication layers
        num_heads: int, number of attention heads
        num_communication_layers: int, number of communication layers
        **communication_kwargs: dict, additional arguments for the communication layers
    """

    def __init__(
        self,
        embed_dim,
        agent_feat_dim=3,
        global_feat_dim=2,
        linear_bias=False,
        use_communication=True,
        use_final_norm=False,
        num_communication_layers=1,
        **communication_kwargs,  # note: see TransformerBlock
    ):
        super(BaseMultiAgentContextEmbedding, self).__init__()
        self.embed_dim = embed_dim

        # Feature projection
        self.proj_agent_feats = nn.Linear(agent_feat_dim, embed_dim, bias=linear_bias)
        self.proj_global_feats = nn.Linear(global_feat_dim, embed_dim, bias=linear_bias)
        self.project_context = nn.Linear(embed_dim * 4, embed_dim, bias=linear_bias)

        if use_communication:
            self.communication_layers = nn.Sequential(
                *(
                    CommunicationLayer(
                        embed_dim=embed_dim,
                        **communication_kwargs,
                    )
                    for _ in range(num_communication_layers)
                )
            )
        else:
            self.communication_layers = nn.Identity()

        self.norm = (
            Normalization(embed_dim, communication_kwargs.get("normalization", "rms"))
            if use_final_norm
            else None
        )

    def _agent_state_embedding(self, embeddings, td, num_agents, num_cities):
        """Embedding for agent-wise state features"""
        raise NotImplementedError("Implement in subclass")

    def _global_state_embedding(self, embeddings, td, num_agents, num_cities):
        """Embedding for global state features"""
        raise NotImplementedError("Implement in subclass")

    def forward(self, embeddings, td):
        # Collect embeddings
        num_agents = td["action_mask"].shape[-2]
        num_cities = td["locs"].shape[-2] - num_agents
        cur_node_embedding = gather_by_index(
            embeddings, td["current_node"]
        )  # [B, M, hdim]
        depot_embedding = gather_by_index(embeddings, td["depot_node"])  # [B, M, hdim]
        agent_state_embed = self._agent_state_embedding(
            embeddings, td, num_agents=num_agents, num_cities=num_cities
        )  # [B, M, hdim]
        global_embed = self._global_state_embedding(
            embeddings, td, num_agents=num_agents, num_cities=num_cities
        )  # [B, M, hdim]
        context_embed = torch.cat(
            [cur_node_embedding, depot_embedding, agent_state_embed, global_embed], dim=-1
        )
        # [B, M, hdim, 4] -> [B, M, hdim]
        context_embed = self.project_context(context_embed)
        h_comm = self.communication_layers(context_embed)
        if self.norm is not None:
            h_comm = self.norm(h_comm)
        return h_comm
