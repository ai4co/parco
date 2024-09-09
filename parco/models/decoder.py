from typing import Tuple
import torch
import torch.nn as nn

from rl4co.envs import RL4COEnvBase
from rl4co.models.zoo.am.decoder import AttentionModelDecoder, PrecomputedCache
from rl4co.utils.ops import unbatchify
from rl4co.utils.pylogger import get_pylogger
from tensordict import TensorDict
from torch import Tensor

from .env_embeddings import env_context_embedding, env_dynamic_embedding

log = get_pylogger(__name__)


class PARCODecoder(AttentionModelDecoder):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        env_name: str = "hcvrp",
        context_embedding: nn.Module = None,
        context_embedding_kwargs: dict = {},
        dynamic_embedding: nn.Module = None,
        dynamic_embedding_kwargs: dict = {},
        use_graph_context: bool = False,
        **kwargs,
    ):
        context_embedding_kwargs["embed_dim"] = embed_dim  # replace
        if context_embedding is None:
            context_embedding = env_context_embedding(
                env_name, context_embedding_kwargs)

        if dynamic_embedding is None:
            dynamic_embedding = env_dynamic_embedding(
                env_name, dynamic_embedding_kwargs)

        if use_graph_context:
            raise ValueError("PARCO does not use graph context")

        super(PARCODecoder, self).__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            env_name=env_name,
            context_embedding=context_embedding,
            dynamic_embedding=dynamic_embedding,
            use_graph_context=use_graph_context,
            **kwargs,
        )

    def forward(
        self,
        td: TensorDict,
        cached,
        num_starts: int = 0,
        do_unbatchify: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Compute the logits of the next actions given the current state

        Args:
            cache: Precomputed embeddings
            td: TensorDict with the current environment state
            num_starts: Number of starts for the multi-start decoding
        """

        # i.e. during sampling, operate only once during all steps
        if num_starts > 1 and do_unbatchify:
            td = unbatchify(td, num_starts)
            td = td.contiguous().view(-1)
        # agent embedding (glimpse_q): [B*S, m, N] B: batch size, S: sampling size, m: num_agents, N: embed_dim
        glimpse_q = self._compute_q(cached, td)
        glimpse_k, glimpse_v, logit_k = self._compute_kvl(cached, td)

        # Masking: 1 means available, 0 means not available
        mask = td["action_mask"]

        # After pass communication layer reshape glimpse_q [B*S, m, N] -> [B, S*m, N] for efficient pointer attiention
        if num_starts > 1:
            batch_size = glimpse_k.shape[0]
            glimpse_q = glimpse_q.reshape(batch_size, -1, self.embed_dim)
            mask = mask.reshape(batch_size, glimpse_q.shape[1], -1)

        # Compute logits
        logits = self.pointer(glimpse_q, glimpse_k, glimpse_v, logit_k, mask)

        # For passing to the next step commnuication layer, reshape logits and mask to [B*S, m, N] if num_starts > 1
        if num_starts > 1:
            logits = logits.reshape(
                batch_size * num_starts, -1, logits.shape[-1])
            mask = mask.reshape(batch_size * num_starts, -1, mask.shape[-1])
            glimpse_q = glimpse_q.reshape(
                batch_size * num_starts, -1, glimpse_q.shape[-1]
            )

        return logits, mask

    def pre_decoder_hook(
        self, td, env, embeddings, num_starts: int = 0
    ) -> Tuple[TensorDict, RL4COEnvBase, PrecomputedCache]:
        """Precompute the embeddings cache before the decoder is called"""
        cached = self._precompute_cache(embeddings, num_starts=num_starts)

        # when we do multi-sampling, only node embeddings are repeated
        if num_starts > 1:
            cached.node_embeddings = cached.node_embeddings.repeat_interleave(
                num_starts, dim=0
            )

        return td, env, cached


class MatNetDecoder(PARCODecoder):
    def __init__(
        self,
        stage_idx: int,
        stage_cnt: int, 
        embed_dim: int = 256,
        num_heads: int = 16, 
        scale_factor: int = 10,
        env_name: str = "ffsp", 
        context_embedding: nn.Module = None,
        context_embedding_kwargs: dict = {}, 
        dynamic_embedding: nn.Module = None, 
        dynamic_embedding_kwargs: dict = {},
        **kwargs
    ):
        
        context_embedding_kwargs.update({
            "stage_idx": stage_idx,
            "stage_cnt": stage_cnt,
            "embed_dim": embed_dim,
            "scale_factor": scale_factor,
        })

        dynamic_embedding_kwargs.update({
            "embed_dim": embed_dim
        })

        super(MatNetDecoder, self).__init__(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            env_name=env_name, 
            context_embedding=context_embedding,
            context_embedding_kwargs=context_embedding_kwargs, 
            dynamic_embedding=dynamic_embedding, 
            dynamic_embedding_kwargs=dynamic_embedding_kwargs,
            use_graph_context=False, 
            **kwargs
        )

        self.stage_idx = stage_idx
        self.project_agent_embeddings = nn.Linear(embed_dim, embed_dim, bias=False)
        self.no_job_emb = nn.Parameter(torch.rand(1, 1, embed_dim), requires_grad=True)

    def _precompute_cache(self, embeddings: Tuple[Tensor, Tensor], num_starts: int = 0):
        job_emb, ma_emb = embeddings

        queries = self.project_agent_embeddings(ma_emb)

        (
            glimpse_key_fixed,
            glimpse_val_fixed,
            logit_key,
        ) = self.project_node_embeddings(job_emb).chunk(3, dim=-1)

        # Organize in a dataclass for easy access
        return PrecomputedCache(
            node_embeddings=queries,
            graph_context=0,
            glimpse_key=glimpse_key_fixed,
            glimpse_val=glimpse_val_fixed,
            logit_key=logit_key,
        )
    
    def _compute_kvl(self, cached: PrecomputedCache, td: TensorDict):
        bs = td.batch_size
        glimpse_k, glimpse_v, logit_k = super()._compute_kvl(cached, td)
        encoded_no_job = self.no_job_emb.expand(*bs, 1, -1)
        # shape: (batch, pomo, jobs+1, embedding)
        logit_k_w_dummy = torch.cat((logit_k, encoded_no_job), dim=1)
        return glimpse_k, glimpse_v, logit_k_w_dummy

    def pre_decoder_hook(
        self, td, env, embeddings, num_starts: int = 0
    ) -> Tuple[TensorDict, RL4COEnvBase, PrecomputedCache]:
        """Precompute the embeddings cache before the decoder is called"""
        cached = self._precompute_cache(embeddings, num_starts=num_starts)
        
        has_dyn_emb_multi_start = self.is_dynamic_embedding and num_starts > 1

        # Handle efficient multi-start decoding
        if has_dyn_emb_multi_start:
            # if num_starts > 0 and we have some dynamic embeddings, we need to reshape them to [B*S, ...]
            # since keys and values are not shared across starts (i.e. the episodes modify these embeddings at each step)
            cached = cached.batchify(num_starts=num_starts)

        elif num_starts > 1:
            td = unbatchify(td, num_starts)

        return td, env, cached