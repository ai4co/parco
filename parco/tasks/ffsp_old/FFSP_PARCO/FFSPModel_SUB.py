
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from typing import Optional


class Normalization(nn.Module):
    def __init__(self, **model_params):
        super(Normalization, self).__init__()
        embedding_dim = model_params['embedding_dim']
        normalization = model_params['normalization']

        normalizer_class = {
            "batch": nn.BatchNorm1d,
            "instance": nn.InstanceNorm1d,
            "layer": nn.LayerNorm,
        }.get(normalization, None)

        if normalizer_class == nn.LayerNorm:
            self.normalizer = normalizer_class(embedding_dim, elementwise_affine=True)
        else:
            self.normalizer = normalizer_class(embedding_dim, affine=True)


    def forward(self, x):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(x.view(-1, x.size(-1))).view(*x.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(x.permute(0, 2, 1)).permute(0, 2, 1)
        elif isinstance(self.normalizer, nn.LayerNorm):
            return self.normalizer(x.view(-1, x.size(-1))).view(*x.size())
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return x


class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))


class MultiHeadAttention(nn.Module):

    def __init__(self, model_params) -> None:

        super().__init__()
        embed_dim = model_params['embedding_dim']
        self.num_heads = model_params['head_num']
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x, attn_mask=None):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        attn_mask: bool tensor of shape (batch, seqlen)
        """

        # Project query, key, value
        q, k, v = rearrange(
            self.Wqkv(x), "b s (three h d) -> three b h s d", three=3, h=self.num_heads
        ).unbind(dim=0)

        if attn_mask is not None:
            attn_mask = (
                attn_mask.unsqueeze(1)
                if attn_mask.ndim == 3
                else attn_mask.unsqueeze(1).unsqueeze(2)
            )

        # Scaled dot product attention
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
        )
        h = self.out_proj(rearrange(out, "b h s d -> b s (h d)"))
        return h


class MixedScore_MultiHeadAttention(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        ms_hidden_dim = model_params['ms_hidden_dim']
        mix1_init = (1/2)**(1/2)
        mix2_init = (1/16)**(1/2)

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        mix1_weight = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample((head_num, 2, ms_hidden_dim))
        mix1_bias = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample((head_num, ms_hidden_dim))
        self.mix1_weight = nn.Parameter(mix1_weight)
        # shape: (head, 2, ms_hidden)
        self.mix1_bias = nn.Parameter(mix1_bias)
        # shape: (head, ms_hidden)

        mix2_weight = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample((head_num, ms_hidden_dim, 1))
        mix2_bias = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample((head_num, 1))
        self.mix2_weight = nn.Parameter(mix2_weight)
        # shape: (head, ms_hidden, 1)
        self.mix2_bias = nn.Parameter(mix2_bias)
        # shape: (head, 1)

    def forward(self, row_emb, col_emb, cost_mat):
        # q shape: (batch, head_num, row_cnt, qkv_dim)
        # k,v shape: (batch, head_num, col_cnt, qkv_dim)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(row_emb), head_num=head_num)
        # q shape: (batch, head_num, row_cnt, qkv_dim)
        k = reshape_by_heads(self.Wk(col_emb), head_num=head_num)
        v = reshape_by_heads(self.Wv(col_emb), head_num=head_num)

        batch_size = q.size(0)
        row_cnt = q.size(2)
        col_cnt = k.size(2)

        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        dot_product = torch.matmul(q, k.transpose(2, 3))
        # shape: (batch, head_num, row_cnt, col_cnt)

        dot_product_score = dot_product / math.sqrt(qkv_dim)
        # shape: (batch, head_num, row_cnt, col_cnt)

        cost_mat_score = cost_mat[:, None, :, :].expand(batch_size, head_num, row_cnt, col_cnt)
        # shape: (batch, head_num, row_cnt, col_cnt)

        two_scores = torch.stack((dot_product_score, cost_mat_score), dim=4)
        # shape: (batch, head_num, row_cnt, col_cnt, 2)

        two_scores_transposed = two_scores.transpose(1,2)
        # shape: (batch, row_cnt, head_num, col_cnt, 2)

        ms1 = torch.matmul(two_scores_transposed, self.mix1_weight)
        # shape: (batch, row_cnt, head_num, col_cnt, ms_hidden_dim)

        ms1 = ms1 + self.mix1_bias[None, None, :, None, :]
        # shape: (batch, row_cnt, head_num, col_cnt, ms_hidden_dim)

        ms1_activated = F.relu(ms1)

        ms2 = torch.matmul(ms1_activated, self.mix2_weight)
        # shape: (batch, row_cnt, head_num, col_cnt, 1)

        ms2 = ms2 + self.mix2_bias[None, None, :, None, :]
        # shape: (batch, row_cnt, head_num, col_cnt, 1)

        mixed_scores = ms2.transpose(1,2)
        # shape: (batch, head_num, row_cnt, col_cnt, 1)

        mixed_scores = mixed_scores.squeeze(4)
        # shape: (batch, head_num, row_cnt, col_cnt)

        weights = nn.Softmax(dim=3)(mixed_scores)
        # shape: (batch, head_num, row_cnt, col_cnt)

        out = torch.matmul(weights, v)
        # shape: (batch, head_num, row_cnt, qkv_dim)

        out_transposed = out.transpose(1, 2)
        # shape: (batch, row_cnt, head_num, qkv_dim)

        out_concat = out_transposed.reshape(batch_size, row_cnt, head_num * qkv_dim)
        # shape: (batch, row_cnt, head_num*qkv_dim)

        return out_concat


class InitEmbeddings(nn.Module):
    def __init__(self, model_params) -> None:
        super().__init__()
        self.model_params = model_params

    def forward(self, problems):
        # problems.shape: (batch, job_cnt, machine_cnt)
        batch_size = problems.size(0)
        job_cnt = problems.size(1)
        machine_cnt = problems.size(2)
        embedding_dim = self.model_params['embedding_dim']

        row_emb = torch.zeros(size=(batch_size, job_cnt, embedding_dim))
        
        # shape: (batch, job_cnt, embedding)
        col_emb = torch.zeros(size=(batch_size, machine_cnt, embedding_dim))
        # shape: (batch, machine_cnt, embedding)

        seed_cnt = max(machine_cnt, self.model_params['one_hot_seed_cnt'])
        rand = torch.rand(batch_size, seed_cnt)
        batch_rand_perm = rand.argsort(dim=1)
        rand_idx = batch_rand_perm[:, :machine_cnt]

        b_idx = torch.arange(batch_size)[:, None].expand(batch_size, machine_cnt)
        m_idx = torch.arange(machine_cnt)[None, :].expand(batch_size, machine_cnt)
        col_emb[b_idx, m_idx, rand_idx] = 1
        # shape: (batch, machine_cnt, embedding)
        return row_emb, col_emb


class CommunicationLayer(nn.Module):

    def __init__(self, model_params):
        super().__init__()
        self.mha = MultiHeadAttention(model_params)
        self.feed_forward = TransformerFFN(**model_params)

    def forward(self, x):
        bs, pomo = x.shape[:2]
        x = rearrange(x, "b p ... -> (b p) ...")
        # mha with residual connection and normalization
        x = self.feed_forward(self.mha(x), x)
        return rearrange(x, "(b p) ... -> b p ...", b=bs, p=pomo)
    


class TransformerFFN(nn.Module):
    def __init__(self, **model_params) -> None:
        super().__init__()

        if model_params.get("parallel_gated_mlp", None) is not None:
            ffn = ParallelGatedMLP(**model_params["parallel_gated_mlp"])
        else:
            ffn = FeedForward(**model_params)

        self.ops = nn.ModuleDict(
            {
                "norm1": Normalization(**model_params),
                "ffn": ffn,
                "norm2": Normalization(**model_params),
            }
        )

    def forward(self, x, x_old):

        x = self.ops["norm1"](x_old + x)
        x = self.ops["norm2"](x + self.ops["ffn"](x))

        return x
    

class ParallelGatedMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float]=None,
        out_dim: int = None
    ):

        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        out_dim = out_dim or hidden_dim


        self.l1 = nn.Linear(
            in_features=dim,
            out_features=hidden_dim,
            bias=False,
        )
        self.l2 = nn.Linear(
            in_features=dim,
            out_features=hidden_dim,
            bias=False,
        )
        self.l3 = nn.Linear(
            in_features=hidden_dim,
            out_features=out_dim,
            bias=False,
        )

    def forward(self, x):
        return self.l3(F.silu(self.l1(x)) * self.l2(x))

    

class MatNetBlock(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.mixed_score_MHA = MixedScore_MultiHeadAttention(**model_params)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.feed_forward = TransformerFFN(**model_params)

    def forward(self, row_emb, col_emb, cost_mat):
        # NOTE: row and col can be exchanged, if cost_mat.transpose(1,2) is used
        # input1.shape: (batch, row_cnt, embedding)
        # input2.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)

        out_concat = self.mixed_score_MHA(row_emb, col_emb, cost_mat)
        # shape: (batch, row_cnt, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, row_cnt, embedding)
        ffn_out = self.feed_forward(multi_head_out, row_emb)

        return ffn_out
        # shape: (batch, row_cnt, embedding)
    
########################################
def reshape_by_heads(qkv, head_num):
    return rearrange(qkv, "... g (h s) -> ... h g s", h=head_num)