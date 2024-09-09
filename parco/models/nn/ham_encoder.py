import math

import numpy as np
import torch

from rl4co.models.nn.graph.attnnet import Normalization, SkipConnection
from torch import nn


class HeterogeneousMHA(nn.Module):
    def __init__(
        self,
        n_heads,
        input_dim,
        embed_dim=None,
        val_dim=None,
        key_dim=None,
        num_agents=None,  # to set as attribute
    ):
        super(HeterogeneousMHA, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim
        self.num_agents = num_agents

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        # pickup
        self.W1_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W2_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W3_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))

        # delivery
        self.W4_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W5_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W6_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    @property
    def set_agent_num(self, agent_num):
        self.num_agents = agent_num

    def forward(self, q, h=None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        assert self.num_agents is not None, "self.num_agents is not set"

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)  # [batch_size * graph_size, embed_dim]
        qflat = q.contiguous().view(-1, input_dim)  # [batch_size * n_query, embed_dim]

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # pickup -> its delivery attention
        # print(self.num_agents)
        # print(graph_size)
        # n_pick = (graph_size - self.num_agents - 1) // 2
        n_pick = (graph_size - self.num_agents) // 2  # !!!
        shp_delivery = (self.n_heads, batch_size, n_pick, -1)
        shp_q_pick = (self.n_heads, batch_size, n_pick, -1)

        # print("n_pick", n_pick)
        # print("shp_delivery", shp_delivery)
        # print("shp_q_pick", shp_q_pick)

        # pickup -> all pickups attention
        shp_allpick = (self.n_heads, batch_size, n_pick, -1)
        shp_q_allpick = (self.n_heads, batch_size, n_pick, -1)

        # pickup -> all pickups attention
        shp_alldelivery = (self.n_heads, batch_size, n_pick, -1)
        shp_q_alldelivery = (self.n_heads, batch_size, n_pick, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # NOTE: we make the agent number etc compatible with our implementation
        # pickup -> its delivery
        pick_flat = (
            h[:, self.num_agents : n_pick + self.num_agents + 0, :]
            .contiguous()
            .view(-1, input_dim)
        )  # [batch_size * n_pick, embed_dim]
        delivery_flat = (
            h[:, n_pick + self.num_agents + 0 :, :].contiguous().view(-1, input_dim)
        )  # [batch_size * n_pick, embed_dim]

        # pickup -> its delivery attention
        Q_pick = torch.matmul(pick_flat, self.W1_query).view(
            shp_q_pick
        )  # (self.n_heads, batch_size, n_pick, key_size)
        K_delivery = torch.matmul(delivery_flat, self.W_key).view(
            shp_delivery
        )  # (self.n_heads, batch_size, n_pick, -1)
        V_delivery = torch.matmul(delivery_flat, self.W_val).view(
            shp_delivery
        )  # (n_heads, batch_size, n_pick, key/val_size)

        # pickup -> all pickups attention
        Q_pick_allpick = torch.matmul(pick_flat, self.W2_query).view(
            shp_q_allpick
        )  # (self.n_heads, batch_size, n_pick, -1)
        K_allpick = torch.matmul(pick_flat, self.W_key).view(
            shp_allpick
        )  # [self.n_heads, batch_size, n_pick, key_size]
        V_allpick = torch.matmul(pick_flat, self.W_val).view(
            shp_allpick
        )  # [self.n_heads, batch_size, n_pick, key_size]

        # pickup -> all delivery
        Q_pick_alldelivery = torch.matmul(pick_flat, self.W3_query).view(
            shp_q_alldelivery
        )  # (self.n_heads, batch_size, n_pick, key_size)
        K_alldelivery = torch.matmul(delivery_flat, self.W_key).view(
            shp_alldelivery
        )  # (self.n_heads, batch_size, n_pick, -1)
        V_alldelivery = torch.matmul(delivery_flat, self.W_val).view(
            shp_alldelivery
        )  # (n_heads, batch_size, n_pick, key/val_size)

        # pickup -> its delivery
        V_additional_delivery = torch.cat(
            [  # [n_heads, batch_size, graph_size, key_size]
                torch.zeros(
                    self.n_heads,
                    batch_size,
                    self.num_agents + 0,
                    self.input_dim // self.n_heads,
                    dtype=V.dtype,
                    device=V.device,
                ),
                V_delivery,  # [n_heads, batch_size, n_pick, key/val_size]
                torch.zeros(
                    self.n_heads,
                    batch_size,
                    n_pick,
                    self.input_dim // self.n_heads,
                    dtype=V.dtype,
                    device=V.device,
                ),
            ],
            2,
        )

        # delivery -> its pickup attention
        Q_delivery = torch.matmul(delivery_flat, self.W4_query).view(
            shp_delivery
        )  # (self.n_heads, batch_size, n_pick, key_size)
        K_pick = torch.matmul(pick_flat, self.W_key).view(
            shp_q_pick
        )  # (self.n_heads, batch_size, n_pick, -1)
        V_pick = torch.matmul(pick_flat, self.W_val).view(
            shp_q_pick
        )  # (n_heads, batch_size, n_pick, key/val_size)

        # delivery -> all delivery attention
        Q_delivery_alldelivery = torch.matmul(delivery_flat, self.W5_query).view(
            shp_alldelivery
        )  # (self.n_heads, batch_size, n_pick, -1)
        K_alldelivery2 = torch.matmul(delivery_flat, self.W_key).view(
            shp_alldelivery
        )  # [self.n_heads, batch_size, n_pick, key_size]
        V_alldelivery2 = torch.matmul(delivery_flat, self.W_val).view(
            shp_alldelivery
        )  # [self.n_heads, batch_size, n_pick, key_size]

        # delivery -> all pickup
        Q_delivery_allpickup = torch.matmul(delivery_flat, self.W6_query).view(
            shp_alldelivery
        )  # (self.n_heads, batch_size, n_pick, key_size)
        K_allpickup2 = torch.matmul(pick_flat, self.W_key).view(
            shp_q_alldelivery
        )  # (self.n_heads, batch_size, n_pick, -1)
        V_allpickup2 = torch.matmul(pick_flat, self.W_val).view(
            shp_q_alldelivery
        )  # (n_heads, batch_size, n_pick, key/val_size)

        # delivery -> its pick up
        #        V_additional_pick = torch.cat([  # [n_heads, batch_size, graph_size, key_size]
        #            torch.zeros(self.n_heads, batch_size, 1, self.input_dim // self.n_heads, dtype=V.dtype, device=V.device),
        #            V_delivery2,  # [n_heads, batch_size, n_pick, key/val_size]
        #            torch.zeros(self.n_heads, batch_size, n_pick, self.input_dim // self.n_heads, dtype=V.dtype, device=V.device)
        #            ], 2)
        V_additional_pick = torch.cat(
            [  # [n_heads, batch_size, graph_size, key_size]
                torch.zeros(
                    self.n_heads,
                    batch_size,
                    self.num_agents + 0,
                    self.input_dim // self.n_heads,
                    dtype=V.dtype,
                    device=V.device,
                ),
                torch.zeros(
                    self.n_heads,
                    batch_size,
                    n_pick,
                    self.input_dim // self.n_heads,
                    dtype=V.dtype,
                    device=V.device,
                ),
                V_pick,  # [n_heads, batch_size, n_pick, key/val_size]
            ],
            2,
        )

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        ##Pick up
        # ??pair???attention??
        compatibility_pick_delivery = self.norm_factor * torch.sum(
            Q_pick * K_delivery, -1
        )  # element_wise, [n_heads, batch_size, n_pick]
        # [n_heads, batch_size, n_pick, n_pick]
        compatibility_pick_allpick = self.norm_factor * torch.matmul(
            Q_pick_allpick, K_allpick.transpose(2, 3)
        )  # [n_heads, batch_size, n_pick, n_pick]

        compatibility_pick_alldelivery = self.norm_factor * torch.matmul(
            Q_pick_alldelivery, K_alldelivery.transpose(2, 3)
        )  # [n_heads, batch_size, n_pick, n_pick]

        ##Delivery
        compatibility_delivery_pick = self.norm_factor * torch.sum(
            Q_delivery * K_pick, -1
        )  # element_wise, [n_heads, batch_size, n_pick]

        compatibility_delivery_alldelivery = self.norm_factor * torch.matmul(
            Q_delivery_alldelivery, K_alldelivery2.transpose(2, 3)
        )  # [n_heads, batch_size, n_pick, n_pick]

        compatibility_delivery_allpick = self.norm_factor * torch.matmul(
            Q_delivery_allpickup, K_allpickup2.transpose(2, 3)
        )  # [n_heads, batch_size, n_pick, n_pick]

        ##Pick up->
        # compatibility_additional?pickup????delivery????attention(size 1),1:n_pick+1??attention,depot?delivery??
        compatibility_additional_delivery = torch.cat(
            [  # [n_heads, batch_size, graph_size, 1]
                -np.inf
                * torch.ones(
                    self.n_heads,
                    batch_size,
                    self.num_agents + 0,
                    dtype=compatibility.dtype,
                    device=compatibility.device,
                ),
                compatibility_pick_delivery,  # [n_heads, batch_size, n_pick]
                -np.inf
                * torch.ones(
                    self.n_heads,
                    batch_size,
                    n_pick,
                    dtype=compatibility.dtype,
                    device=compatibility.device,
                ),
            ],
            -1,
        ).view(self.n_heads, batch_size, graph_size, 1)

        compatibility_additional_allpick = torch.cat(
            [  # [n_heads, batch_size, graph_size, n_pick]
                -np.inf
                * torch.ones(
                    self.n_heads,
                    batch_size,
                    self.num_agents + 0,
                    n_pick,
                    dtype=compatibility.dtype,
                    device=compatibility.device,
                ),
                compatibility_pick_allpick,  # [n_heads, batch_size, n_pick, n_pick]
                -np.inf
                * torch.ones(
                    self.n_heads,
                    batch_size,
                    n_pick,
                    n_pick,
                    dtype=compatibility.dtype,
                    device=compatibility.device,
                ),
            ],
            2,
        ).view(self.n_heads, batch_size, graph_size, n_pick)

        compatibility_additional_alldelivery = torch.cat(
            [  # [n_heads, batch_size, graph_size, n_pick]
                -np.inf
                * torch.ones(
                    self.n_heads,
                    batch_size,
                    self.num_agents + 0,
                    n_pick,
                    dtype=compatibility.dtype,
                    device=compatibility.device,
                ),
                compatibility_pick_alldelivery,  # [n_heads, batch_size, n_pick, n_pick]
                -np.inf
                * torch.ones(
                    self.n_heads,
                    batch_size,
                    n_pick,
                    n_pick,
                    dtype=compatibility.dtype,
                    device=compatibility.device,
                ),
            ],
            2,
        ).view(self.n_heads, batch_size, graph_size, n_pick)
        # [n_heads, batch_size, n_query, graph_size+1+n_pick+n_pick]

        ##Delivery->
        compatibility_additional_pick = torch.cat(
            [  # [n_heads, batch_size, graph_size, 1]
                -np.inf
                * torch.ones(
                    self.n_heads,
                    batch_size,
                    self.num_agents + 0,
                    dtype=compatibility.dtype,
                    device=compatibility.device,
                ),
                -np.inf
                * torch.ones(
                    self.n_heads,
                    batch_size,
                    n_pick,
                    dtype=compatibility.dtype,
                    device=compatibility.device,
                ),
                compatibility_delivery_pick,  # [n_heads, batch_size, n_pick]
            ],
            -1,
        ).view(self.n_heads, batch_size, graph_size, 1)

        compatibility_additional_alldelivery2 = torch.cat(
            [  # [n_heads, batch_size, graph_size, n_pick]
                -np.inf
                * torch.ones(
                    self.n_heads,
                    batch_size,
                    self.num_agents + 0,
                    n_pick,
                    dtype=compatibility.dtype,
                    device=compatibility.device,
                ),
                -np.inf
                * torch.ones(
                    self.n_heads,
                    batch_size,
                    n_pick,
                    n_pick,
                    dtype=compatibility.dtype,
                    device=compatibility.device,
                ),
                compatibility_delivery_alldelivery,  # [n_heads, batch_size, n_pick, n_pick]
            ],
            2,
        ).view(self.n_heads, batch_size, graph_size, n_pick)

        compatibility_additional_allpick2 = torch.cat(
            [  # [n_heads, batch_size, graph_size, n_pick]
                -np.inf
                * torch.ones(
                    self.n_heads,
                    batch_size,
                    self.num_agents + 0,
                    n_pick,
                    dtype=compatibility.dtype,
                    device=compatibility.device,
                ),
                -np.inf
                * torch.ones(
                    self.n_heads,
                    batch_size,
                    n_pick,
                    n_pick,
                    dtype=compatibility.dtype,
                    device=compatibility.device,
                ),
                compatibility_delivery_allpick,  # [n_heads, batch_size, n_pick, n_pick]
            ],
            2,
        ).view(self.n_heads, batch_size, graph_size, n_pick)

        compatibility = torch.cat(
            [
                compatibility,
                compatibility_additional_delivery,
                compatibility_additional_allpick,
                compatibility_additional_alldelivery,
                compatibility_additional_pick,
                compatibility_additional_alldelivery2,
                compatibility_additional_allpick2,
            ],
            dim=-1,
        )

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = torch.softmax(
            compatibility, dim=-1
        )  # [n_heads, batch_size, n_query, graph_size+1+n_pick*2] (graph_size include depot)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc
        # heads: [n_heads, batrch_size, n_query, val_size], attn????pick?deliver?attn
        heads = torch.matmul(
            attn[:, :, :, :graph_size], V
        )  # V: (self.n_heads, batch_size, graph_size, val_size)

        # heads??pick -> its delivery
        heads = (
            heads
            + attn[:, :, :, graph_size].view(self.n_heads, batch_size, graph_size, 1)
            * V_additional_delivery
        )  # V_addi:[n_heads, batch_size, graph_size, key_size]

        # heads??pick -> otherpick, V_allpick: # [n_heads, batch_size, n_pick, key_size]
        # heads: [n_heads, batch_size, graph_size, key_size]
        heads = heads + torch.matmul(
            attn[:, :, :, graph_size + 1 : graph_size + 1 + n_pick].view(
                self.n_heads, batch_size, graph_size, n_pick
            ),
            V_allpick,
        )

        # V_alldelivery: # (n_heads, batch_size, n_pick, key/val_size)
        heads = heads + torch.matmul(
            attn[:, :, :, graph_size + 1 + n_pick : graph_size + 1 + 2 * n_pick].view(
                self.n_heads, batch_size, graph_size, n_pick
            ),
            V_alldelivery,
        )

        # delivery
        heads = (
            heads
            + attn[:, :, :, graph_size + 1 + 2 * n_pick].view(
                self.n_heads, batch_size, graph_size, 1
            )
            * V_additional_pick
        )

        heads = heads + torch.matmul(
            attn[
                :, :, :, graph_size + 1 + 2 * n_pick + 1 : graph_size + 1 + 3 * n_pick + 1
            ].view(self.n_heads, batch_size, graph_size, n_pick),
            V_alldelivery2,
        )

        heads = heads + torch.matmul(
            attn[:, :, :, graph_size + 1 + 3 * n_pick + 1 :].view(
                self.n_heads, batch_size, graph_size, n_pick
            ),
            V_allpickup2,
        )

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim),
        ).view(batch_size, n_query, self.embed_dim)

        return out

    def set_num_agents(self, num_agents):
        self.num_agents = num_agents


class HeterogeneuousMHALayer(nn.Sequential):
    def __init__(
        self,
        num_heads=8,
        embed_dim=128,
        feed_forward_hidden=512,
        normalization="batch",
    ):
        super(HeterogeneuousMHALayer, self).__init__(
            SkipConnection(HeterogeneousMHA(num_heads, embed_dim, embed_dim)),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim),
                )
                if feed_forward_hidden > 0
                else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization),
        )

    def set_num_agents(self, num_agents):
        # for all HeterogeneousMHA inside, set num_agents
        for layer in self:
            if isinstance(layer, SkipConnection):
                if isinstance(layer.module, HeterogeneousMHA):
                    layer.module.set_num_agents(num_agents)


class GraphHeterogeneousAttentionEncoder(nn.Module):
    def __init__(
        self,
        init_embedding: nn.Module,
        num_heads=8,
        embed_dim=128,
        num_encoder_layers=3,
        env_name=None,
        normalization="batch",
        feed_forward_hidden=512,
        sdpa_fn=None,
    ):
        super(GraphHeterogeneousAttentionEncoder, self).__init__()

        self.init_embedding = init_embedding

        self.layers = nn.Sequential(
            *(
                HeterogeneuousMHALayer(
                    num_heads,
                    embed_dim,
                    feed_forward_hidden,
                    normalization,
                )
                for _ in range(num_encoder_layers)
            )
        )

    def forward(self, x, mask=None):
        assert mask is None, "Mask not yet supported!"
        # initial Embedding from features
        init_embeds = self.init_embedding(x)  # (batch_size, graph_size, embed_dim)
        # layers  (batch_size, graph_size, embed_dim)
        embeds = self.layers(init_embeds)
        return embeds, init_embeds

    def set_num_agents(self, num_agents):
        # for all HeterogeneousMHA inside, set num_agents
        for layer in self.layers:
            if isinstance(layer, HeterogeneuousMHALayer):
                layer.set_num_agents(num_agents)
