
import torch
import torch.nn.functional as F

from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast
from torch import Tensor

class GAMoEGateBackward(torch.autograd.Function):
    # jump the sign operation as the sign operation does not have gradients

    @staticmethod
    def forward(ctx: Any, scores: Tensor):
        signed_scores = torch.sign(scores)
        return signed_scores

    @staticmethod
    def backward(ctx:Any, grad_output: Tensor):
        return grad_output

class GAMoEGateT(torch.nn.Module):
    def __init__(self, model_dim, num_global_experts, fp32_gate=False, max_expert_num=64, adaptive_experts=False, init_t=0.1, **options):
        super().__init__()
        torch.manual_seed(1)
        self.expert_num = num_global_experts
        self.register_parameter('sim_matrix', torch.nn.Parameter(torch.nn.init.orthogonal_(torch.empty(max_expert_num, model_dim)).T.contiguous(), requires_grad=True))
        self.register_parameter('gates', torch.nn.Parameter(torch.zeros(size=(max_expert_num,)), requires_grad=True))
        self.register_parameter('experts_mask', torch.nn.Parameter(torch.zeros(size=(max_expert_num,)), requires_grad=False))
        self.register_parameter('temperature', torch.nn.Parameter(torch.log(torch.full([1], 1.0 / init_t)), requires_grad=True))
        self.clamp_max = torch.log(torch.tensor(1. / 0.01)).item()

        # self.gates.data[0] = 0.0
        self.experts_mask[:num_global_experts] = 1.0
        
        self.fp32_gate = fp32_gate
        self.max_expert_num = max_expert_num
        self.adaptive_top_k = True
        self.adaptive_experts = adaptive_experts
        # print('adaptive:'+str(self.adaptive_experts))
        self.top_k = 0

        self.normalize_gate = options.get('normalize_one_score_gate', False)
        self.capacity_factor = 0.0 # always use the adaptive capacity to avoid drop tokens
        self.gate_noise = 0.0 # always do not allow gate noise
        self.enable_softmax_logits = False

        for opt in options:
            if opt not in ('capacity_factor', 'gate_noise', 'normalize_one_score_gate'):
                raise Exception('Unrecognized argument provided to Gating module: %s' % opt)

    def forward(self, x):
        if self.fp32_gate:
            x = x.float()
            sim_matrix = self.sim_matrix.float()
            gates = self.gates.float()
        else:
            sim_matrix = self.sim_matrix
            gates = self.gates
        
        # logits = torch.sigmoid(torch.matmul(F.normalize(x, dim=1),
        #                       F.normalize(sim_matrix[:, self.activate_mask], dim=0)))
        # gates = torch.sigmoid(self.gates[self.activate_mask])

        # 现在的实现方式与已有的相关方法相同，即单纯通过mask一些expert实现adaptive（pruning）。后续可能需要把我们之前实现的真的adaptive实现好。
        logit_scale = torch.clamp(self.temperature, max=self.clamp_max).exp()
        logits = torch.sigmoid(torch.matmul(F.normalize(x, dim=1),
                              F.normalize(sim_matrix, dim=0)) * logit_scale)
        logits = logits * self.experts_mask
        gates = torch.sigmoid(self.gates * logit_scale)

        if self.training:
            logits = F.relu(logits - gates)
            logits = GAMoEGateBackward.apply(logits)
            top_k = torch.sum(logits > 0, dim=1).to(torch.int)
        else:
            # logits = F.relu(logits - gates)
            # logits = GAMoEGateBackward.apply(logits)
            # top_k = torch.sum(logits > 0, dim=1).to(torch.int)

            new_logits = F.relu(logits - gates)
            new_logits = GAMoEGateBackward.apply(new_logits)
            top_k = torch.sum(new_logits > 0, dim=1).to(torch.int)
            logits = ((torch.sum(new_logits, dim=1) == 0).to(torch.int).repeat(logits.shape[1]).reshape(logits.shape[1], -1).T) * logits + new_logits
            top_k = torch.max(top_k, torch.ones(top_k.shape).to(top_k.device)).to(torch.int)
        
        # print(gates)
        print('Average Top K is {}, max is {}, logic scale is {}'.format(sum(top_k) / len(top_k), max(top_k), logit_scale.item()))
        return logits, top_k
    
    def add_experts(self, global_rank, expert_local_rank, world_size):
        assert world_size % self.max_expert_num == 0, "the capacity of the expert pool should be the integral multiply of world size"
        num_experts_per_rank = self.max_expert_num // world_size
        expert_global_rank = global_rank * num_experts_per_rank + expert_local_rank
        self.experts_mask[expert_global_rank] = 1.0

    def remove_experts(self, global_rank, expert_local_rank, world_size):
        assert world_size % self.max_expert_num == 0, "the capacity of the expert pool should be the integral multiply of world size"
        num_experts_per_rank = self.max_expert_num // world_size
        expert_global_rank = global_rank * num_experts_per_rank + expert_local_rank
        self.experts_mask[expert_global_rank] = 0.0



    
Gate = GAMoEGateT