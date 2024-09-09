import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from FFSPModel_SUB import MultiHeadAttention, TransformerFFN, Normalization


class MixedScoreFF(nn.Module):
    def __init__(self, **model_params) -> None:
        super().__init__()
        ms_hidden_dim = model_params['ms_hidden_dim']
        num_heads = model_params['head_num']

        self.lin1 = nn.Linear(2 * num_heads, num_heads * ms_hidden_dim, bias=False)
        self.lin2 = nn.Linear(num_heads * ms_hidden_dim, 2 * num_heads, bias=False)

    def forward(self, dot_product_score, cost_mat_score):
        # dot_product_score shape: (batch, head_num, row_cnt, col_cnt)
        # cost_mat_score shape: (batch, head_num, row_cnt, col_cnt)
        # shape: (batch, head_num, row_cnt, col_cnt, 2)
        two_scores = torch.stack((dot_product_score, cost_mat_score), dim=-1)
        two_scores = rearrange(two_scores, "b h r c s -> b r c (h s)")
        # shape: (batch, row_cnt, col_cnt, 2 * num_heads)
        ms = self.lin2(F.relu(self.lin1(two_scores)))
        # shape: (batch, row_cnt, head_num, col_cnt)
        mixed_scores = rearrange(ms, "b r c (h two) -> b h r c two", two=2)
        ms1, ms2 = mixed_scores.chunk(2, dim=-1)

        return ms1.squeeze(-1), ms2.squeeze(-1)


class EfficientMixedScoreMultiHeadAttention(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.num_heads = model_params['head_num']
        qkv_dim = model_params["qkv_dim"]
        self.scale_dots = model_params["scale_dots"]

        self.qkv_dim = qkv_dim
        self.norm_factor = 1 / math.sqrt(qkv_dim)

        self.Wqv1 = nn.Linear(embedding_dim, 2 * embedding_dim, bias=False)
        self.Wkv2 = nn.Linear(embedding_dim, 2 * embedding_dim, bias=False)

        # self.init_parameters()
        self.mixed_scores_layer = MixedScoreFF(**model_params)

        self.out_proj1 = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.out_proj2 = nn.Linear(embedding_dim, embedding_dim, bias=False)


    def forward(self, x1, x2, attn_mask = None, cost_mat = None):
        batch_size = x1.size(0)
        row_cnt = x1.size(-2)
        col_cnt = x2.size(-2)

        # Project query, key, value
        q, v1 = rearrange(
            self.Wqv1(x1), "b s (two h d) -> two b h s d", two=2, h=self.num_heads
        ).unbind(dim=0)

        # Project query, key, value
        k, v2 = rearrange(
            self.Wqv1(x2), "b s (two h d) -> two b h s d", two=2, h=self.num_heads
        ).unbind(dim=0)

        # shape: (batch, num_heads, row_cnt, col_cnt)
        dot = self.norm_factor * torch.matmul(q, k.transpose(-2, -1))
        
        if cost_mat is not None:
            # shape: (batch, num_heads, row_cnt, col_cnt)
            cost_mat_score = cost_mat[:, None, :, :].expand_as(dot)
            ms1, ms2 = self.mixed_scores_layer(dot, cost_mat_score)

        if attn_mask is not None:
            attn_mask = attn_mask.view(batch_size, 1, row_cnt, col_cnt).expand_as(dot)
            dot.masked_fill_(~attn_mask, float("-inf"))

        h1 = self.out_proj1(
            apply_weights_and_combine(ms1, v2, scale=self.scale_dots)
        )
        h2 = self.out_proj2(
            apply_weights_and_combine(ms2.transpose(-2, -1), v1, scale=self.scale_dots)
        )

        return h1, h2
    

class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()

        self.op_attn = MultiHeadAttention(model_params)
        self.ma_attn = MultiHeadAttention(model_params)
        self.cross_attn = EfficientMixedScoreMultiHeadAttention(**model_params)

        self.op_ffn = TransformerFFN(**model_params)
        self.ma_ffn = TransformerFFN(**model_params)

        self.op_norm = Normalization(**model_params)
        self.ma_norm = Normalization(**model_params)


    def forward(
        self, 
        op_in, 
        ma_in, 
        cost_mat, 
        op_mask=None, 
        ma_mask=None, 
        cross_mask=None
    ):
        
        op_cross_out, ma_cross_out = self.cross_attn(op_in, ma_in, attn_mask=cross_mask, cost_mat=cost_mat)
        op_cross_out = self.op_norm(op_cross_out + op_in)
        ma_cross_out = self.ma_norm(ma_cross_out + ma_in)

        # (bs, num_jobs, ops_per_job, d)
        op_self_out = self.op_attn(op_cross_out, attn_mask=op_mask)
        # (bs, num_ma, d)
        ma_self_out = self.ma_attn(ma_cross_out, attn_mask=ma_mask)

        op_out = self.op_ffn(op_cross_out, op_self_out)
        ma_out = self.ma_ffn(ma_cross_out, ma_self_out)

        return op_out, ma_out



def apply_weights_and_combine(logits, v, tanh_clipping=10, scale=True):
    if scale:
        # scale to avoid numerical underflow
        logits = logits / logits.std()
    if tanh_clipping > 0:
        # tanh clipping to avoid explosions
        logits = torch.tanh(logits) * tanh_clipping
    # shape: (batch, num_heads, row_cnt, col_cnt)
    weights = nn.Softmax(dim=-1)(logits)
    weights = weights.nan_to_num(0)
    # shape: (batch, num_heads, row_cnt, qkv_dim)
    out = torch.matmul(weights, v)
    # shape: (batch, row_cnt, num_heads, qkv_dim)
    out = rearrange(out, "b h s d -> b s (h d)")
    return out
