import torch
import torch.nn as nn

from parco.models.nn.transformer import TransformerBlock as CommunicationLayer

class FFSPInitEmbeddings(nn.Module):
    def __init__(self, one_hot_seed_cnt: int, embed_dim: int = 256) -> None:
        super().__init__()
        self.one_hot_seed_cnt = one_hot_seed_cnt
        self.embed_dim = embed_dim

    def forward(self, problems: torch.Tensor):
        # problems.shape: (batch, job_cnt, machine_cnt)
        batch_size = problems.size(0)
        job_cnt = problems.size(1)
        machine_cnt = problems.size(2)
        device = problems.device
        row_emb = torch.zeros(size=(batch_size, job_cnt, self.embed_dim), device=device)
        
        # shape: (batch, job_cnt, embedding)
        col_emb = torch.zeros(size=(batch_size, machine_cnt, self.embed_dim), device=device)
        # shape: (batch, machine_cnt, embedding)

        seed_cnt = max(machine_cnt, self.one_hot_seed_cnt)
        rand = torch.rand(batch_size, seed_cnt, device=device)
        batch_rand_perm = rand.argsort(dim=1)
        rand_idx = batch_rand_perm[:, :machine_cnt]

        b_idx = torch.arange(batch_size, device=device)[:, None].expand(batch_size, machine_cnt)
        m_idx = torch.arange(machine_cnt, device=device)[None, :].expand(batch_size, machine_cnt)
        col_emb[b_idx, m_idx, rand_idx] = 1
        # shape: (batch, machine_cnt, embedding)
        return row_emb, col_emb
    

class FFSPContextEmbedding(nn.Module):
    def __init__(
        self, 
        stage_idx: int = None,
        stage_cnt: int = None,
        embed_dim: int = 256,
        scale_factor: int = 10,
        use_comm_layer: bool = True,
        **communication_layer_kwargs
    ) -> None:
        
        super().__init__()
        self.stage_idx = stage_idx
        self.stage_cnt = stage_cnt
        self.dyn_context = nn.Linear(2, embed_dim)
        self.scale_factor = scale_factor
        self.use_comm_layer = use_comm_layer
        # optional layers
        if self.use_comm_layer:
            self.communication_layer = CommunicationLayer(
                embed_dim=embed_dim,
                **communication_layer_kwargs
            )

    def forward(self, ma_emb_proj, td):
        # (b, ma)
        t_ma_idle = td["t_ma_idle"].chunk(self.stage_cnt, dim=-1)[self.stage_idx]
        t_ma_idle = t_ma_idle.to(torch.float32) / self.scale_factor
        # shape: (batch, job)
        job_in_stage = td["job_location"][:, :-1] == self.stage_idx
        # shape: (batch, pomo)
        num_in_stage = (job_in_stage.sum(-1) / job_in_stage.size(-1))
        # shape: (batch, pomo, ma, embedding)
        ma_wait_proj = self.dyn_context(
            torch.stack((t_ma_idle, num_in_stage.unsqueeze(-1).expand_as(t_ma_idle)), dim=-1)
        )
        context = ma_emb_proj + ma_wait_proj
        if self.use_comm_layer:
            context = self.communication_layer(context)

        return context
    
class FFSPDynamicEmbedding(nn.Module):
    def __init__(
        self, 
        embed_dim: int = 256,
        scale_factor: int = 10,
    ):
        super(FFSPDynamicEmbedding, self).__init__()
        self.dyn_kv = nn.Linear(2, 3 * embed_dim)
        self.scale_factor = scale_factor


    def forward(self, td):
        job_dyn = torch.stack(
            (td["job_location"][:, :-1], td["t_job_ready"][:, :-1] / self.scale_factor), 
            dim=-1
        ).to(torch.float32)
        # shape: (batch, pomo, jobs, 3*embedding)
        dyn_job_proj = self.dyn_kv(job_dyn)
        # shape: 3 * (batch, pomo, jobs, embedding)
        dyn_k, dyn_v, dyn_l = dyn_job_proj.chunk(3, dim=-1)
        return dyn_k, dyn_v, dyn_l
