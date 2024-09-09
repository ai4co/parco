
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from typing import Optional

from parco.models.nn.transformer import ParallelGatedMLP, Normalization
from rl4co.models.nn.attention import MultiHeadAttention

class FeedForward(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        feedforward_hidden: int = 512,
    ):
        super().__init__()
        self.W1 = nn.Linear(embed_dim, feedforward_hidden)
        self.W2 = nn.Linear(feedforward_hidden, embed_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))


class MixedScoreMultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        head_num: int = 16,
        ms_hidden_dim: int = 32,
    ):
        super().__init__()

        self.head_num = head_num
        qkv_dim = embed_dim // head_num
        self.qkv_dim = qkv_dim

        mix1_init = (1/2)**(1/2)
        mix2_init = (1/16)**(1/2)

        self.Wq = nn.Linear(embed_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, head_num * qkv_dim, bias=False)

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
        head_num = self.head_num
        qkv_dim = self.qkv_dim

        q = reshape_by_heads(self.Wq(row_emb), head_num=head_num)
        # q shape: (batch, head_num, row_cnt, qkv_dim)
        k = reshape_by_heads(self.Wk(col_emb), head_num=head_num)
        v = reshape_by_heads(self.Wv(col_emb), head_num=head_num)

        batch_size = q.size(0)
        row_cnt = q.size(2)
        col_cnt = k.size(2)

        # shape: (batch, head_num, row_cnt, col_cnt)
        dot_product = torch.matmul(q, k.transpose(2, 3))

        # shape: (batch, head_num, row_cnt, col_cnt)
        dot_product_score = dot_product / math.sqrt(qkv_dim)

        # shape: (batch, head_num, row_cnt, col_cnt)
        cost_mat_score = cost_mat[:, None, :, :].expand(batch_size, head_num, row_cnt, col_cnt)

        # shape: (batch, head_num, row_cnt, col_cnt, 2)
        two_scores = torch.stack((dot_product_score, cost_mat_score), dim=4)

        # shape: (batch, row_cnt, head_num, col_cnt, 2)
        two_scores_transposed = two_scores.transpose(1,2)

        # shape: (batch, row_cnt, head_num, col_cnt, ms_hidden_dim)
        ms1 = torch.matmul(two_scores_transposed, self.mix1_weight)

        # shape: (batch, row_cnt, head_num, col_cnt, ms_hidden_dim)
        ms1 = ms1 + self.mix1_bias[None, None, :, None, :]

        ms1_activated = F.relu(ms1)

        # shape: (batch, row_cnt, head_num, col_cnt, 1)
        ms2 = torch.matmul(ms1_activated, self.mix2_weight)

        # shape: (batch, row_cnt, head_num, col_cnt, 1)
        ms2 = ms2 + self.mix2_bias[None, None, :, None, :]

        # shape: (batch, head_num, row_cnt, col_cnt, 1)
        mixed_scores = ms2.transpose(1,2)

        # shape: (batch, head_num, row_cnt, col_cnt)
        mixed_scores = mixed_scores.squeeze(4)

        # shape: (batch, head_num, row_cnt, col_cnt)
        weights = nn.Softmax(dim=3)(mixed_scores)

        # shape: (batch, head_num, row_cnt, qkv_dim)
        out = torch.matmul(weights, v)

        # shape: (batch, row_cnt, head_num, qkv_dim)
        out_transposed = out.transpose(1, 2)

        # shape: (batch, row_cnt, head_num*qkv_dim)
        out_concat = out_transposed.reshape(batch_size, row_cnt, head_num * qkv_dim)

        return out_concat


class TransformerFFN(nn.Module):
    def __init__(
        self, 
        embed_dim: int = 256,
        feedforward_hidden: int = 512,
        normalization: Optional[str] = "instance",
        parallel_gated_kwargs: Optional[dict] = None,
    ):
        
        super().__init__()

        if parallel_gated_kwargs is not None:
            ffn = ParallelGatedMLP(**parallel_gated_kwargs)
        else:
            ffn = FeedForward(embed_dim=embed_dim, feedforward_hidden=feedforward_hidden)

        self.ops = nn.ModuleDict(
            {
                "norm1": Normalization(embed_dim=embed_dim, normalization=normalization),
                "ffn": ffn,
                "norm2": Normalization(embed_dim=embed_dim, normalization=normalization),
            }
        )

    def forward(self, x, x_old):

        x = self.ops["norm1"](x_old + x)
        x = self.ops["norm2"](x + self.ops["ffn"](x))

        return x
    
    

class MatNetBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        head_num: int = 16,
        ms_hidden_dim: int = 32,
        feedforward_hidden: int = 512,
        normalization: Optional[str] = "instance",
        parallel_gated_kwargs: Optional[dict] = None,
    ):
        
        super().__init__()

        self.mixed_score_MHA = MixedScoreMultiHeadAttention(
            embed_dim=embed_dim,
            head_num=head_num,
            ms_hidden_dim=ms_hidden_dim
        )

        self.multi_head_combine = nn.Linear(embed_dim, embed_dim)

        self.feed_forward = TransformerFFN(
            embed_dim=embed_dim,
            feedforward_hidden=feedforward_hidden,
            normalization=normalization,
            parallel_gated_kwargs=parallel_gated_kwargs
        )

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


class MatNetLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        head_num: int = 16,
        ms_hidden_dim: int = 32,
        feedforward_hidden: int = 512,
        normalization: Optional[str] = "instance",
        parallel_gated_kwargs: Optional[dict] = None,
        **kwargs
    ):
        super().__init__()
        self.row_encoding_block = MatNetBlock(
            embed_dim=embed_dim,
            head_num=head_num,
            ms_hidden_dim=ms_hidden_dim,
            feedforward_hidden=feedforward_hidden,
            normalization=normalization,
            parallel_gated_kwargs=parallel_gated_kwargs
        )
        self.col_encoding_block = MatNetBlock(
            embed_dim=embed_dim,
            head_num=head_num,
            ms_hidden_dim=ms_hidden_dim,
            feedforward_hidden=feedforward_hidden,
            normalization=normalization,
            parallel_gated_kwargs=parallel_gated_kwargs
        )

    def forward(self, row_emb, col_emb, cost_mat):
        # row_emb.shape: (batch, row_cnt, embedding)
        # col_emb.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        row_emb_out = self.row_encoding_block(row_emb, col_emb, cost_mat)
        col_emb_out = self.col_encoding_block(col_emb, row_emb, cost_mat.transpose(1, 2))

        return row_emb_out, col_emb_out


class MixedScoreFF(nn.Module):
    def __init__(self, num_heads, ms_hidden_dim) -> None:
        super().__init__()

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
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 16,
        ms_hidden_dim: int = 32,
    ):
        super().__init__()

        self.num_heads = num_heads
        qkv_dim = embed_dim // num_heads
        self.scale_dots = True

        self.qkv_dim = qkv_dim
        self.norm_factor = 1 / math.sqrt(qkv_dim)

        self.Wqv1 = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.Wkv2 = nn.Linear(embed_dim, 2 * embed_dim, bias=False)

        # self.init_parameters()
        self.mixed_scores_layer = MixedScoreFF(num_heads, ms_hidden_dim)

        self.out_proj1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj2 = nn.Linear(embed_dim, embed_dim, bias=False)


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
    

class HAMEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 16,
        ms_hidden_dim: int = 32,
        feedforward_hidden: int = 512,
        normalization: Optional[str] = "instance",
        parallel_gated_kwargs: Optional[dict] = None,
        **kwargs
    ):
        super().__init__()

        self.op_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            # bias=False,
        )
        self.ma_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=False,
        )
        
        self.cross_attn = EfficientMixedScoreMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ms_hidden_dim=ms_hidden_dim
        )

        self.op_ffn = TransformerFFN(
            embed_dim=embed_dim,
            feedforward_hidden=feedforward_hidden,
            normalization=normalization,
            parallel_gated_kwargs=parallel_gated_kwargs
        )
        self.ma_ffn = TransformerFFN(
            embed_dim=embed_dim,
            feedforward_hidden=feedforward_hidden,
            normalization=normalization,
            parallel_gated_kwargs=parallel_gated_kwargs
        )

        self.op_norm = Normalization(embed_dim=embed_dim, normalization=normalization)
        self.ma_norm = Normalization(embed_dim=embed_dim, normalization=normalization)


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



#######################################
def reshape_by_heads(qkv, head_num):
    return rearrange(qkv, "... g (h s) -> ... h g s", h=head_num)

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