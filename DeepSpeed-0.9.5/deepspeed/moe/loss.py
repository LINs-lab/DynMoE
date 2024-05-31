import torch
import torch.nn.functional as F

def diverse_and_simple_gate_loss(gates, expert_mask):
    sims = torch.matmul(F.normalize(gates, dim=0).T, F.normalize(gates, dim=0))

    targets = torch.eye(sims.shape[0]).to(sims.device)

    sim_mask = torch.matmul(expert_mask.unsqueeze(0).T, expert_mask.unsqueeze(0))

    sim_loss = torch.norm(sims * sim_mask - targets * sim_mask)

    simple_loss = torch.mean(torch.norm(gates, dim=0))

    return sim_loss + simple_loss
