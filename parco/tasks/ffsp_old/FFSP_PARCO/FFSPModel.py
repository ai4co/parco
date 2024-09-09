
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from FFSPModel_SUB import (
    MatNetBlock,
    InitEmbeddings,
    CommunicationLayer,
    reshape_by_heads
)
from FFSPEnv import Reset_State, Step_State
from HAMLayer import EncoderLayer as HamEncoderLayer


class FFSPModel(nn.Module):
    """Apply a OneStageModel for each stage"""
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.stage_cnt = len(self.model_params['machine_cnt_list'])
        self.stage_models = nn.ModuleList([OneStageModel(stage_idx, **model_params) for stage_idx in range(self.stage_cnt)])

    def pre_forward(self, reset_state: Reset_State):
        for stage_idx in range(self.stage_cnt):
            problems = reset_state.problems_list[stage_idx]
            model = self.stage_models[stage_idx]
            model.pre_forward(problems)

    def soft_reset(self):
        # Nothing to reset
        pass

    def forward(self, state: Step_State):

        jobs_stack = []
        ma_stack = []
        prob_stack = []

        for stage_idx in range(self.stage_cnt):
            model = self.stage_models[stage_idx]
            jobs, mas, probs = model(state)

            if jobs is not None:
                jobs_stack.append(jobs)
                ma_stack.append(mas)
                prob_stack.append(probs)

        jobs_stack = torch.cat(jobs_stack, dim=-1)
        ma_stack = torch.cat(ma_stack, dim=-1)
        prob_stack = torch.cat(prob_stack, dim=-1)
        return jobs_stack, ma_stack, prob_stack
    

class OneStageModel(nn.Module):
    def __init__(self, stage_idx, **model_params):
        super().__init__()
        self.stage_idx = stage_idx
        self.model_params = model_params

        self.encoder = FFSP_Encoder(stage_idx=stage_idx, **model_params)
        self.decoder = FFSP_Decoder(stage_idx=stage_idx, **model_params)

        self.encoded_col = None
        # shape: (batch, machine_cnt, embedding)
        self.encoded_row = None
        # shape: (batch, job_cnt, embedding)
        self.num_ma = None

        self.use_pos_token = model_params["use_pos_token"]

    def pre_forward(self, problems):
        self.encoded_row, self.encoded_col = self.encoder(problems)
        # encoded_row.shape: (batch, job_cnt, embedding)
        # encoded_col.shape: (batch, machine_cnt, embedding)
        self.decoder.set_qkv(self.encoded_row, self.encoded_col)

        self.num_job = self.encoded_row.size(1)
        self.num_ma = self.encoded_col.size(1)

    def forward(self, state: Step_State):

        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)
        # shape: (batch, pomo, num_agents, job_cnt+1)
        logits = self.decoder(state)
        # shape: (batch, pomo, num_agents, job_cnt+1)
        mask = state.get_stage_mask(self.stage_idx).clone()

        jobs_selected, mas_selected, actions_probs = [], [], []
        # shape: (batch * pomo, num_agents)
        idle_machines = torch.arange(0, self.num_ma)[None,:].expand(batch_size*pomo_size, -1)

        temp = 1.0
        while not mask[...,:-1].all():
            # get the probabilities of all actions given the current mask
            logits_masked = logits.masked_fill(mask, -torch.inf)
            # shape: (batch * pomo, num_agents * job_cnt+1)
            logits_reshaped = rearrange(logits_masked, "b p m j -> (b p) (j m)") / temp
            probs = F.softmax(logits_reshaped, dim=-1)
            # perform decoding
            if self.training or self.model_params['eval_type'] == 'softmax':
                # shape: (batch * pomo)
                selected_action = probs.multinomial(1).squeeze(1)
                action_prob = probs.gather(1, selected_action.unsqueeze(1)).squeeze(1)
            else:
                # shape: (batch * pomo)
                selected_action = probs.argmax(dim=-1)
                action_prob = torch.zeros(size=(batch_size*pomo_size,))
            # translate the action 
            # shape: (batch * pomo)
            job_selected = selected_action // self.num_ma
            selected_stage_machine = selected_action % self.num_ma
            selected_machine = selected_stage_machine + self.num_ma * self.stage_idx
            # determine which machines still have to select an action
            idle_machines = (
                idle_machines[idle_machines!=selected_stage_machine[:, None]]
                .view(batch_size*pomo_size, -1)
            )
            # add action to the buffer
            jobs_selected.append(job_selected)
            mas_selected.append(selected_machine)
            actions_probs.append(action_prob)
            # mask job that has been selected in the current step so it cannot be selected by other agents
            mask = mask.scatter(-1, job_selected.view(batch_size, pomo_size, 1, 1).expand(-1, -1, self.num_ma, 1), True)
            if self.use_pos_token:
                # allow machines that are still idle to wait (for jobs to become available for example)
                mask[..., -1] = mask[..., -1].scatter(-1, idle_machines.view(batch_size, pomo_size, -1), False)
            else:
                mask[..., -1] = mask[..., -1].scatter(-1, idle_machines.view(batch_size, pomo_size, -1), ~(mask[..., :-1].all(-1)))
            # lastly, mask all actions for the selected agent
            mask = mask.scatter(-2, selected_stage_machine.view(batch_size, pomo_size, 1, 1).expand(-1, -1, 1, self.num_job+1), True)

        if len(jobs_selected) > 0:
            jobs_selected = torch.stack(jobs_selected, dim=-1).view(batch_size, pomo_size, -1)
            mas_selected = torch.stack(mas_selected, dim=-1).view(batch_size, pomo_size, -1)
            actions_probs = torch.stack(actions_probs, dim=-1).view(batch_size, pomo_size, -1)

            return jobs_selected, mas_selected, actions_probs
    
        else:
            return None, None, None



########################################
# ENCODER
########################################
class FFSP_Encoder(nn.Module):
    def __init__(self, stage_idx, **model_params):
        super().__init__()
        self.stage_idx = stage_idx
        encoder_layer_num = model_params['encoder_layer_num']
        self.init_embed = InitEmbeddings(model_params)
        if model_params['use_ham']:
            self.layers = nn.ModuleList([HamEncoderLayer(**model_params) for _ in range(encoder_layer_num)])
        else:
            self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])
        self.scale_factor = model_params["scale_factor"]

    def forward(self, cost_mat):
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        row_emb, col_emb = self.init_embed(cost_mat)
        cost_mat = cost_mat / self.scale_factor
        for layer in self.layers:
            row_emb, col_emb = layer(
                row_emb, 
                col_emb, 
                cost_mat
            )
        return row_emb, col_emb


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.row_encoding_block = MatNetBlock(**model_params)
        self.col_encoding_block = MatNetBlock(**model_params)

    def forward(self, row_emb, col_emb, cost_mat):
        # row_emb.shape: (batch, row_cnt, embedding)
        # col_emb.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        row_emb_out = self.row_encoding_block(row_emb, col_emb, cost_mat)
        col_emb_out = self.col_encoding_block(col_emb, row_emb, cost_mat.transpose(1, 2))

        return row_emb_out, col_emb_out


########################################
# Decoder
########################################

class FFSP_Decoder(nn.Module):
    def __init__(self, stage_idx, **model_params):
        super().__init__()
        self.stage_idx = stage_idx
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        self.scale_factor = model_params["scale_factor"]
        self.use_graph_proj = model_params["use_graph_proj"]
        self.use_comm_layer = model_params["use_comm_layer"]
        self.use_decoder_mha_mask = model_params["use_decoder_mha_mask"]
        self.sqrt_embedding_dim = math.sqrt(embedding_dim)
        self.sqrt_qkv_dim = math.sqrt(qkv_dim)
        # dummy embedding
        self.encoded_NO_JOB = nn.Parameter(torch.rand(1, 1, 1, embedding_dim))
        # qkv
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wl = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        # dyn embeddings
        self.dyn_context = nn.Linear(2, embedding_dim)
        self.dyn_kv = nn.Linear(2, 3 * embedding_dim)
        # optional layers
        if self.use_comm_layer:
            self.communication_layer = CommunicationLayer(model_params)
        if self.use_graph_proj:
            self.graph_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)


    def set_qkv(self, encoded_jobs, encoded_machine):
        # shape: (batch, job, embedding)
        self.encoded_jobs = encoded_jobs
        # shape: (batch, ma, embedding)
        self.q = self.Wq(encoded_machine)
        # shape: (batch, job, embedding)
        self.k = self.Wk(encoded_jobs)
        # shape: (batch, job, embedding)
        self.v = self.Wv(encoded_jobs)
        # shape: (batch, job, embedding)
        self.single_head_key = self.Wl(encoded_jobs)

    def forward(self, state: Step_State):
        head_num = self.model_params['head_num']

        # dynamic embeddings
        # shape: (batch, pomo, ma)
        ma_wait = state.machine_state[self.stage_idx].to(torch.float32) / self.scale_factor
        # shape: (batch, pomo, job)
        job_in_stage = (state.job_state.curr_stage == self.stage_idx)
        # shape: (batch, pomo)
        num_in_stage = (job_in_stage.sum(-1) / job_in_stage.size(-1))
        # shape: (batch, pomo, ma, embedding)
        ma_wait_proj = self.dyn_context(
            torch.stack((ma_wait, num_in_stage.unsqueeze(-1).expand_as(ma_wait)), dim=-1)
        )
        # shape: (batch, pomo, ma, embedding)
        q = self.q.unsqueeze(1) + ma_wait_proj
        # shape: (batch, pomo, ma, embedding)
        if self.use_comm_layer:
            q = self.communication_layer(q)

        if self.use_graph_proj:
            # shape: (batch, pomo, embedding)
            graph_emb = (
                self.encoded_jobs
                .unsqueeze(1)
                .masked_fill(~job_in_stage.unsqueeze(-1), 0)
                .sum(-2)
            ) / (num_in_stage.unsqueeze(-1) + 1e-9)
            # shape: (batch, pomo, embedding)
            graph_emb_proj = self.graph_projection(graph_emb)
            # shape: (batch, pomo, ma, embedding)
            q = q + graph_emb_proj.unsqueeze(-2)

        # shape: (batch, pomo, head_num, ma, qkv_dim)
        q = reshape_by_heads(q, head_num=head_num)
        # shape: (batch, pomo, jobs, 2)
        job_dyn = torch.stack(
            (state.job_state.curr_stage, state.job_state.wait_step / self.scale_factor), 
            dim=-1
        ).to(torch.float32)
        # shape: (batch, pomo, jobs, 3*embedding)
        dyn_job_proj = self.dyn_kv(job_dyn)
        # shape: 3 * (batch, pomo, jobs, embedding)
        dyn_k, dyn_v, dyn_l = dyn_job_proj.chunk(3, dim=-1)
        # shape: 2 * (batch, pomo, head_num, jobs, qkv_dim); (batch, pomo, jobs, embedding)
        k, v, l = (
            reshape_by_heads(self.k.unsqueeze(1) + dyn_k, head_num=head_num), 
            reshape_by_heads(self.v.unsqueeze(1) + dyn_v, head_num=head_num), 
            self.single_head_key.unsqueeze(1) + dyn_l
        )

        bs, pomo = l.shape[:2]
        encoded_no_job = self.encoded_NO_JOB.expand(bs, pomo, 1, -1)
        # shape: (batch, pomo, jobs+1, embedding)
        l_plus_one = torch.cat((l, encoded_no_job), dim=2)
        # MHA
        dec_mha_mask = state.get_stage_mask(self.stage_idx)[..., :-1] if self.use_decoder_mha_mask else None
        # shape: (batch, pomo, num_agents, head_num*qkv_dim)
        out_concat = self._multi_head_attention_for_decoder(
            q, k, v, rank3_mask=dec_mha_mask
        )

        # shape: (batch, pomo, num_agents, embedding)
        mh_atten_out = self.multi_head_combine(out_concat)

        #  Single-Head Attention, for probability calculation
        #######################################################
        # shape: (batch, pomo, num_agents, job_cnt+1)
        score = torch.matmul(mh_atten_out, l_plus_one.transpose(-1, -2))
        logit_clipping = self.model_params['logit_clipping']
        score_scaled = score / self.sqrt_embedding_dim
        score_clipped = logit_clipping * torch.tanh(score_scaled)
        return score_clipped


    def _multi_head_attention_for_decoder(self, q, k, v, rank3_mask=None):
        # q shape: (batch, pomo, head_num, ma, qkv_dim)
        # k,v shape: (batch, pomo, head_num, job_cnt, qkv_dim)
        # rank3_ninf_mask.shape: (batch, pomo, ma, job_cnt)
        head_num = self.model_params['head_num']

        # shape: (batch, pomo, head_num, ma, job_cnt)
        score = torch.matmul(q, k.transpose(-2, -1))
        score_scaled = score / self.sqrt_qkv_dim

        if rank3_mask is not None:
            mask = rank3_mask[:, :, None, :].expand_as(score_scaled)
            score_scaled[mask] = -torch.inf

        # shape: (batch, pomo, head_num, ma, job_cnt)
        weights = nn.Softmax(dim=-1)(score_scaled)

        # shape: (batch, pomo, head_num, ma, qkv_dim)
        out = torch.matmul(weights, v)
        # shape: (batch, pomo, ma, embedding)
        return rearrange(out, "b p h n d -> b p n (h d)", h=head_num)