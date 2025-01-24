# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
The file has been adapted from two fairscale files:
 (1) https://github.com/facebookresearch/fairscale/blob/master/fairscale/nn/moe/moe_layer.py
 (2) https://github.com/facebookresearch/fairscale/blob/master/fairscale/nn/moe/top2gate.py
 Git commit hash: 34df606902a240567a0d898037ece55c2f1336cf
 We retain the following license from the original files:
"""

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from deepspeed.utils.timer import SynchronizedWallClockTimer
from deepspeed.utils import logger
from typing import Callable, Dict, TYPE_CHECKING, Any, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
import numpy as np
from deepspeed.utils import groups
from .mappings import drop_tokens, gather_tokens
from .loss import diverse_and_simple_gate_loss

if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module

uniform_map: Dict[torch.device, Callable] = {}
gumbel_map: Dict[torch.device, Callable] = {}
exp_selection_uniform_map: Dict[torch.device, Callable] = {}

try:
    # To enable Tutel MoE optimizations:
    #   python3 -m pip install --user --upgrade git+https://github.com/microsoft/tutel@v0.1.x
    from tutel import moe as tutel_moe
    TUTEL_INSTALLED = True
except:
    # Fail silently so we don't spam logs unnecessarily if user isn't using tutel
    TUTEL_INSTALLED = False
    pass

# === OURS ===
def get_world_size(group=None):
    try:
        return dist.get_world_size(group)
    except:
        return 1

def simple_all_reduce(input, group=None, op=torch.distributed.ReduceOp.SUM):
    world_size = get_world_size(group)
    if world_size == 1:
        return input
    output = torch.clone(input, memory_format=torch.contiguous_format)
    dist.all_reduce(output, op=op, group=group)
    return output
# === OURS ===

def multiplicative_jitter(x, device: torch.device, epsilon=1e-2):
    """
    Modified from switch transformer paper. mesh transformers
    Multiply values by a random number between 1-epsilon and 1+epsilon.
    Makes models more resilient to rounding errors introduced by bfloat16.
    This seems particularly important for logits.
    Args:
        x: a torch.tensor
        device: torch.device
        epsilon: a floating point value
    Returns:
        a jittered x.
    """
    if epsilon == 0:
        return x
    uniform = uniform_map.get(device)
    if uniform is None:
        uniform = torch.distributions.uniform.Uniform(low=torch.tensor(1.0 - epsilon, device=device),
                                                      high=torch.tensor(1.0 + epsilon,
                                                                        device=device)).rsample  # type: ignore
        uniform_map[device] = uniform
    return x * uniform(x.shape)


def gumbel_rsample(shape: Tuple, device: torch.device) -> Tensor:
    gumbel = gumbel_map.get(device)
    if gumbel is None:
        one = torch.tensor(1.0, device=device)
        zero = torch.tensor(0.0, device=device)
        gumbel = torch.distributions.gumbel.Gumbel(zero, one).rsample  # type: ignore
        gumbel_map[device] = gumbel
    return gumbel(shape)


from deepspeed import comm as dist

# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.


# Based on https://github.com/pytorch/pytorch/pull/40762
class _AllToAll(torch.autograd.Function):

    @staticmethod
    def forward(
            ctx: Any,
            # TODO: replace with DS process group
            group: torch.distributed.ProcessGroup,
            input: Tensor) -> Tensor:  # type: ignore
        ctx.group = group
        input = input.contiguous()
        output = torch.empty_like(input)
        dist.all_to_all_single(output, input, group=group)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return (None, _AllToAll.apply(ctx.group, *grad_output))


# einsum rewrites are on par or more performant
# switch can be bubbled up in future
USE_EINSUM = True


# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.
def einsum(rule, a, b):
    if USE_EINSUM:
        return torch.einsum(rule, a, b)
    elif rule == 's,se->se':
        return a.reshape(a.shape[0], -1) * b
    elif rule == 'se,sc->sec':
        return a.unsqueeze(2) * b.unsqueeze(1)
    elif rule == 'se,se->s':
        return torch.bmm(a.unsqueeze(1), b.unsqueeze(2)).reshape(-1)
    elif rule == 'sec,sm->ecm':
        s = a.shape[0]
        e = a.shape[1]
        c = a.shape[2]
        m = b.shape[1]
        return torch.matmul(a.reshape(s, -1).t(), b).reshape(e, c, m)
    elif rule == 'sec,ecm->sm':
        return torch.matmul(a.reshape(a.shape[0], -1), b.reshape(-1, b.shape[-1]))
    elif rule == 'ks,ksm->sm':
        k = b.shape[0]
        s = b.shape[1]
        m = b.shape[2]
        # [k, s] -> [s, k] -> [s, 1, k]
        a = a.t().unsqueeze(1)
        # [k,s,m] -> [k, sm] -> [sm, k] -> [s, m, k]
        b = b.reshape(k, -1).t().reshape(s, m, k)
        # bmm([s, 1, k], [s, m, k]^t) -> [s, m, 1]
        return torch.bmm(a, b.transpose(1, 2)).squeeze(2)
    else:
        return torch.einsum(rule, a, b)


# The following functions are extracted and scripted
# because otherwise during a torch.jit.trace, the non-Tensor
# values used in the calculations get recorded as constants.
# torch.jit.script coerces them into Tensors and preserves
# their dynamic shapes. This enables ONNX export.
# We can't script the entire top1gating function because it
# includes stateful caching logic which is incompatible with ONNX.


@torch.jit.script
def _capacity(gates: Tensor, capacity_factor: Tensor, min_capacity: Tensor) -> Tensor:
    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    # to(torch.int64) works around a bug in torch.onnx.export:
    # it should cast k to int64 when converting torch.topk but it doesn't.
    capacity = torch.ceil((num_tokens / num_experts) * capacity_factor).to(torch.int64)
    if capacity < min_capacity:
        capacity = min_capacity.to(torch.int64)
    return capacity


@torch.jit.script
def _top_idx(source, k):
    return torch.topk(source, k=k, dim=0)[1]


@torch.jit.script
def _one_hot_to_float(x, num_classes):
    return F.one_hot(x, num_classes=num_classes).float()


def top1gating(logits: Tensor,
               capacity_factor: float,
               min_capacity: int,
               used_token: Tensor = None,
               noisy_gate_policy: Optional[str] = None,
               drop_tokens: bool = True,
               use_rts: bool = True,
               use_tutel: bool = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements Top1Gating on logits."""
    if noisy_gate_policy == 'RSample':
        logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    # everything is in fp32 in this function
    gates = F.softmax(logits, dim=1)

    capacity = _capacity(gates, torch.tensor(capacity_factor), torch.tensor(min_capacity))

    # Create a mask for 1st's expert per token
    # noisy gating
    indices1_s = torch.argmax(logits_w_noise if noisy_gate_policy == 'RSample' else gates, dim=1)
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    # mask only used tokens
    if used_token is not None:
        mask1 = einsum("s,se->se", used_token, mask1)

    # gating decisions
    exp_counts = torch.sum(mask1, dim=0).detach().to('cpu')

    # if we don't want to drop any tokens
    if not drop_tokens:
        new_capacity = torch.max(exp_counts).to(logits.device)
        dist.all_reduce(new_capacity, op=dist.ReduceOp.MAX, group=dist.get_world_group())
        capacity = new_capacity

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.sum(me * ce) * num_experts

    # Random Token Selection
    if use_rts:
        uniform = exp_selection_uniform_map.get(logits.device)
        if uniform is None:
            uniform = torch.distributions.uniform.Uniform(low=torch.tensor(0.0, device=logits.device),
                                                          high=torch.tensor(1.0, device=logits.device)).rsample
            exp_selection_uniform_map[logits.device] = uniform

        mask1_rand = mask1 * uniform(mask1.shape)
    else:
        mask1_rand = mask1

    assert logits.shape[
        0] >= min_capacity, "No. of tokens (batch-size) should be greater than min_capacity. Either set min_capacity to 0 or increase your batch size."

    top_idx = _top_idx(mask1_rand, capacity)

    new_mask1 = mask1 * torch.zeros_like(mask1).scatter_(0, top_idx, 1)
    mask1 = new_mask1

    if use_tutel:
        # Tutel doesn't support index values masked with zero
        # so we need to replace masked indices with -1
        indices_mask = mask1.sum(dim=1) * num_experts - 1
        indices1_s = torch.min(indices1_s, indices_mask)

    # Compute locations in capacity buffer
    if use_tutel:
        locations1 = tutel_moe.fast_cumsum_sub_one(mask1)
    else:
        locations1 = torch.cumsum(mask1, dim=0) - 1

    if use_tutel:
        gates1_s = (gates * mask1).sum(dim=1)
        locations1_s = torch.sum(locations1 * mask1, dim=1)
        return l_aux, capacity, num_experts, [
            indices1_s,
        ], [
            locations1_s,
        ], [
            gates1_s,
        ], exp_counts

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)

    # Normalize gate probabilities
    mask1_float = mask1.float()
    gates = gates * mask1_float

    locations1_sc = _one_hot_to_float(locations1_s, capacity)
    combine_weights = einsum("se,sc->sec", gates, locations1_sc)

    dispatch_mask = combine_weights.bool()

    return l_aux, combine_weights, dispatch_mask, exp_counts


def top2gating(logits: Tensor, capacity_factor: float, min_capacity: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements Top2Gating on logits."""
    # everything is in fp32 in this function
    gates = F.softmax(logits, dim=1)

    capacity = _capacity(gates, torch.tensor(capacity_factor * 2), torch.tensor(min_capacity))

    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1)
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    # Create a mask for 2nd's expert per token using Gumbel-max trick
    # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
    logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    # Replace top-expert with min value
    logits_except1 = logits_w_noise.masked_fill(mask1.bool(), float("-inf"))
    indices2_s = torch.argmax(logits_except1, dim=1)
    mask2 = F.one_hot(indices2_s, num_classes=num_experts)

    # Compute locations in capacity buffer
    locations1 = torch.cumsum(mask1, dim=0) - 1
    locations2 = torch.cumsum(mask2, dim=0) - 1
    # Update 2nd's location by accounting for locations of 1st
    locations2 += torch.sum(mask1, dim=0, keepdim=True)

    # gating decisions
    exp_counts = torch.sum(mask1, dim=0).detach().to('cpu')

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.mean(me * ce) * num_experts * num_experts

    # Remove locations outside capacity from mask
    mask1 *= torch.lt(locations1, capacity)
    mask2 *= torch.lt(locations2, capacity)

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)

    # Normalize gate probabilities
    mask1_float = mask1.float()
    mask2_float = mask2.float()
    gates1_s = einsum("se,se->s", gates, mask1_float)
    gates2_s = einsum("se,se->s", gates, mask2_float)
    denom_s = gates1_s + gates2_s
    # Avoid divide-by-zero
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    gates1_s /= denom_s
    gates2_s /= denom_s

    # Calculate combine_weights and dispatch_mask
    gates1 = einsum("s,se->se", gates1_s, mask1_float)
    gates2 = einsum("s,se->se", gates2_s, mask2_float)
    locations1_sc = _one_hot_to_float(locations1_s, capacity)
    locations2_sc = _one_hot_to_float(locations2_s, capacity)
    combine1_sec = einsum("se,sc->sec", gates1, locations1_sc)
    combine2_sec = einsum("se,sc->sec", gates2, locations2_sc)
    combine_weights = combine1_sec + combine2_sec
    dispatch_mask = combine_weights.bool()

    return l_aux, combine_weights, dispatch_mask, exp_counts

# === OURS ===
def topanygating(logits: Tensor, capacity_factor: float, min_capacity: int, K: Tensor, gate_tensor=None, expert_mask=None, ep_group=None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements TopanyGating on logits."""
    # everything is in fp32 in this function
    gates = logits
    max_K = max(K)
    max_K = max(max_K, 1)
    average_K = sum(K) / len(K)

    # Create a mask for k-st's expert per token
    logits_except_pre = logits + 0.0
    masks = []
    num_experts = int(gates.shape[1])

    for k in range(max_K):
        indicesk_s = torch.argmax(logits_except_pre, dim=1)
        mask_k = F.one_hot(indicesk_s, num_classes=num_experts)
        mask_k = (mask_k.T * (K > k)).T
        masks.append(mask_k)
        logits_except_pre = logits_except_pre.masked_fill(mask_k.bool(), float("-inf"))

    # Compute locations in capacity buffer
    locations = []
    pre_locations = torch.zeros(num_experts).to(masks[0].device)

    for mask in masks:
        locationsk = torch.cumsum(mask, dim=0) - 1
        locationsk += pre_locations.int()
        pre_locations += torch.sum(mask, dim=0)
        locations.append(locationsk)

    new_capacity = torch.max(pre_locations).to(logits.device).int() 
    dist.all_reduce(new_capacity, op=dist.ReduceOp.MAX, group=dist.get_world_group())
    capacity = new_capacity

    # gating decisions -- not sure the meaning now
    exp_counts = torch.sum(masks[0], dim=0).detach().to('cpu')

    # Compute l_aux
    if gate_tensor is None or expert_mask is None:
        me = torch.mean(gates, dim=0)
        ce = torch.mean(masks[0].float(), dim=0)
        l_aux = torch.mean(me * ce) * num_experts * num_experts
    else:
        l_aux = diverse_and_simple_gate_loss(gate_tensor, expert_mask)

    # Remove locations outside capacity from mask <--- Dropless, can optimize
    for i in range(len(masks)):
        masks[i] *= torch.lt(locations[i], capacity)

    # Store the capacity location for each token
    locations_s = [torch.sum(locations[i] * masks[i], dim=1) for i in range(len(masks))]

    # Normalize gate probabilities
    masks_float = [mask.float() for mask in masks]
    gates_s = [einsum("se,se->s", gates, mask_float) for mask_float in masks_float]
    denom_s = sum(gates_s)

    # Avoid divide-by-zero
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    gates_s = [gate_s / denom_s for gate_s in gates_s]

    combine_weights = torch.zeros_like(gates).unsqueeze(-1).expand(-1, -1, capacity.item())

    # Calculate combine_weights within the loop
    for gatesk_s, maskk_float, locationsk_s in zip(gates_s, masks_float, locations_s):
        gatessk = torch.einsum("s,se->se", gatesk_s, maskk_float)
        locationsk_sc = _one_hot_to_float(locationsk_s, capacity)
        combinesk_sec = torch.einsum("se,sc->sec", gatessk, locationsk_sc)
        combine_weights = combine_weights.add(combinesk_sec)

    dispatch_mask = combine_weights.bool()

    return l_aux, combine_weights, dispatch_mask, exp_counts

def topanygating_opt(logits: Tensor, capacity_factor: float, min_capacity: int, K: Tensor, gate_tensor=None, expert_mask=None, ep_group=None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    
    """Implements TopanyGating on logits."""
    gates = logits
    mask = gates.int()
    exp_counts = torch.sum(mask, dim=0).detach().to('cpu')

    new_capacity = torch.max(exp_counts).to(logits.device)
    dist.all_reduce(new_capacity, op=dist.ReduceOp.MAX, group=dist.get_world_group())
    capacity = new_capacity

    num_experts = int(gates.shape[1])

    # Compute l_aux
    if gate_tensor is None or expert_mask is None:
        me = torch.mean(gates, dim=0)
        ce = torch.mean(mask.float(), dim=0)
        l_aux = torch.mean(me * ce) * num_experts * num_experts
    else:
        # load balance + efficiency loss
        non_zero_mask = torch.sum(gates, dim=1) > 0 # mask tokens donnot actiavte any expert
        if non_zero_mask.sum() != 0:
            normalized_gates = gates[non_zero_mask]
            normalized_gates = normalized_gates / (torch.sum(normalized_gates, dim=1, keepdim=True))
            me = torch.mean(normalized_gates, dim=0)
            ce = torch.mean(mask[non_zero_mask].float(), dim=0)
            load_balance_loss = torch.mean(me * ce) * num_experts * num_experts
        else:
            load_balance_loss = 0
        efficiency_loss = torch.mean(gates)
        
        l_aux = diverse_and_simple_gate_loss(gate_tensor, expert_mask) \
                                            + load_balance_loss + efficiency_loss

    # Store the capacity location for each token
    locations1 = torch.cumsum(mask, dim=0) # sample * expert
    locations1_s = ((locations1 * mask) - 1 ) % (capacity + 1) # sample * expert, mod `capacity + 1` to keep indices positive

    # mask_k = K > 0
    # gates[mask_k] = gates[mask_k] / K[mask_k].unsqueeze(1)
    gates /= torch.clamp(K, min=1).unsqueeze(1)

    locations1_sc = _one_hot_to_float(locations1_s, capacity + 1) # (sample, expert, capacity + 1) 
    combine_weights = einsum("se,sec->sec", gates, locations1_sc[:,:,:-1]) # (sample, expert, capacity)

    dispatch_mask = combine_weights.bool()

    return l_aux, combine_weights, dispatch_mask, exp_counts
# === OURS ===

# === OURS ===
class GAMoEGateBackward(torch.autograd.Function):
    # jump the sign operation as the sign operation does not have gradients

    @staticmethod
    def forward(ctx: Any, scores: Tensor):
        signed_scores = torch.sign(scores)
        return signed_scores

    @staticmethod
    def backward(ctx:Any, grad_output: Tensor):
        return grad_output
# === OURS ===

# === OURS ===

class GAMoEGateT(torch.nn.Module):
    def __init__(self, model_dim, num_global_experts, fp32_gate=False, max_expert_num=64, adaptive_experts=False, init_t=1.0):
        super().__init__()
        self.expert_num = num_global_experts
        self.register_parameter('sim_matrix', torch.nn.Parameter(torch.nn.init.orthogonal_(torch.empty(max_expert_num, model_dim, dtype=torch.float32)).T.contiguous(), requires_grad=True))
        # self.register_parameter('sim_matrix', torch.nn.Parameter(torch.empty(max_expert_num, model_dim).T.contiguous(), requires_grad=True))
        self.register_parameter('gates', torch.nn.Parameter(torch.zeros(size=(max_expert_num,)), requires_grad=True))
        self.register_parameter('experts_mask', torch.nn.Parameter(torch.zeros(size=(max_expert_num,)), requires_grad=False))
        self.register_parameter('temperature', torch.nn.Parameter(torch.log(torch.full([1], 1.0 / init_t, dtype=torch.float32)), requires_grad=False))
        self.clamp_max = torch.log(torch.tensor(1. / 0.01, dtype=torch.float32)).item()
        # self.register_parameter('experts_mask', torch.nn.Parameter(torch.zeros(size=(max_expert_num,)), requires_grad=False))

        self.experts_mask.requires_grad_(False) # FIX ME
        self.experts_mask[:num_global_experts] = 1.0
        
        self.fp32_gate = fp32_gate
        self.max_expert_num = max_expert_num
        self.adaptive_experts = adaptive_experts

    def forward(self, x):
        if self.fp32_gate:
            x = x.float()
            sim_matrix = self.sim_matrix.float()
            gates = self.gates.float()
        else:
            sim_matrix = self.sim_matrix
            gates = self.gates
        
        logit_scale = torch.clamp(self.temperature, max=self.clamp_max).exp()
        logits = torch.sigmoid(torch.matmul(F.normalize(x, dim=1),
                              F.normalize(sim_matrix, dim=0)) * logit_scale)
        logits = logits * self.experts_mask
        gates = torch.sigmoid(self.gates * logit_scale)
        
        if self.training:
            # print('gate forward in train')
            logits = F.relu(logits - gates)
            logits = GAMoEGateBackward.apply(logits)
            top_k = torch.sum(logits > 0, dim=1).to(torch.int)
        else:

            new_logits = F.relu(logits - gates)
            # If remove this, gating scores range will changed from {0, 1} to [0, x]
            new_logits = GAMoEGateBackward.apply(new_logits)
            
            top_k = torch.sum(new_logits > 0, dim=1).to(torch.int)

            mask = (torch.sum(new_logits, dim=1) == 0).to(torch.int).repeat(logits.shape[1]).reshape(logits.shape[1], -1).T # s * e
            max_index = torch.argmax(logits, dim=1)
            one_hot = F.one_hot(max_index, num_classes=logits.shape[1])
            logits = mask * one_hot + new_logits
            
            top_k = torch.max(top_k, torch.ones(top_k.shape).to(top_k.device)).to(torch.int)

        return logits, top_k

# === OURS ===

class TopKGate(Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = TopKGate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    wg: torch.nn.Linear

    def __init__(self,
                 model_dim: int,
                 num_experts: int,
                 k: int = 1,
                 capacity_factor: float = 1.0,
                 eval_capacity_factor: float = 1.0,
                 min_capacity: int = 8,
                 noisy_gate_policy: Optional[str] = None,
                 drop_tokens: bool = True,
                 use_rts: bool = True,
                 max_expert_num: int = 4, # OURS
                 ep_group = None # OURS
                 ) -> None:
        super().__init__()

        # Only top-1 and top-2 are supported at the moment.
        if k != 1 and k != 2 and k != -1:
            raise ValueError('Only top-1, top-2, and top-Any gatings are supported.')
        
        # OURS
        self.k = k
        if k == 1 or k == 2:
            self.wg = torch.nn.Linear(model_dim, num_experts, bias=False).float()
        else:
            assert max_expert_num >= num_experts, f"Number of experts ({num_experts}) should be less than max number of experts ({max_expert_num})"
    
            self.wg = GAMoEGateT(model_dim, num_experts, max_expert_num=max_expert_num,  fp32_gate=True, adaptive_experts=True, init_t=1.0)
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.min_capacity = min_capacity
        self.noisy_gate_policy = noisy_gate_policy
        self.timers = SynchronizedWallClockTimer()
        self.wall_clock_breakdown = False
        self.gate_time = 0.0
        self.drop_tokens = drop_tokens
        self.use_rts = use_rts
        
        # OURS
        self.max_expert_num = max_expert_num
        self.record_routing = False
        self.ep_group = ep_group
        self.total_tokens = None
        
    # === OURS === 
    def begin_record_routing(self):
        self.reset_record_routing()
        self.record_routing = True

    def end_record_routing(self):
        self.record_routing = False
    
    def reset_record_routing(self):
        if not isinstance(self.wg, GAMoEGateT):
            raise ValueError('Only support GAMoE for record routing!')
        self.routing_records = None
        self.sample_records = None
    # === OURS ===

    def forward(self,
                input: torch.Tensor,
                used_token: torch.Tensor = None,
                use_tutel: bool = False) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore

        if self.wall_clock_breakdown:
            self.timers('TopKGate').start()
        
        # OURS
        if isinstance(self.wg, torch.nn.Linear) and self.wg.weight.dtype != torch.float32:
            self.wg = self.wg.float()
        input_fp32 = input.float()
        # input jittering
        if self.noisy_gate_policy == 'Jitter' and self.training:
            input_fp32 = multiplicative_jitter(input_fp32, device=input.device)
        
        # OURS
        if isinstance(self.wg, torch.nn.Linear):
            logits = self.wg(input_fp32)
            top_K = self.k
        else:
            logits, top_K = self.wg(input_fp32)
            
        # Only support our method currently (OURS)
        if self.record_routing:
            assert isinstance(self.wg, GAMoEGateT)
            
            # OURS
            current_routing_records = torch.sum(torch.sign(logits), dim=0)
            if self.routing_records is None:
                self.routing_records = current_routing_records
            else:
                self.routing_records += current_routing_records

            if self.total_tokens is None:
                self.total_tokens = logits.shape[0]
            else:
                self.total_tokens += logits.shape[0]
            
            sample_routing = torch.sum(torch.sign(logits), dim=1)
            samples_not_routing = input_fp32[sample_routing == 0]
            if len(samples_not_routing) > 0:
                current_sample_records = torch.sum(samples_not_routing, dim=0)
                if self.sample_records is None:
                    self.sample_records = current_sample_records
                else:
                    self.sample_records += current_sample_records
            elif self.sample_records is None:
                self.sample_records = torch.zeros(input_fp32[0].shape).to(input_fp32[0].device)
        
        if self.k == 1:
            gate_output = top1gating(logits, self.capacity_factor if self.training else self.eval_capacity_factor,
                                     self.min_capacity, used_token, self.noisy_gate_policy if self.training else None,
                                     self.drop_tokens, self.use_rts, use_tutel)

        elif self.k == 2:
            gate_output = top2gating(logits, self.capacity_factor if self.training else self.eval_capacity_factor,
                                     self.min_capacity)
        # OURS
        elif isinstance(self.wg, GAMoEGateT):
            gate_output = topanygating_opt(logits, self.capacity_factor if self.training else self.eval_capacity_factor,
                                     self.min_capacity, top_K, self.wg.sim_matrix, self.wg.experts_mask, self.ep_group)
        else:
            raise ValueError('invaild k')

        if self.wall_clock_breakdown:
            self.timers('TopKGate').stop()
            self.gate_time = self.timers('TopKGate').elapsed(reset=False)

        return gate_output


class MOELayer(Base):
    """MOELayer module which implements MixtureOfExperts as described in Gshard_.
    ::

        gate = TopKGate(model_dim, num_experts)
        moe = MOELayer(gate, expert)
        output = moe(input)
        l_aux = moe.l_aux

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        gate (torch.nn.Module):
            gate network
        expert (torch.nn.Module):
            expert network
    """

    def __init__(self,
                 gate: Module,
                 experts: Module,
                 ep_group_name,
                 ep_size,
                 num_local_experts: int,
                 use_tutel: bool = False) -> None:
        super().__init__()
        self.gate = gate
        self.experts = experts
        self.ep_group = None
        self.ep_size = ep_size
        self.ep_group_name = ep_group_name
        self.num_local_experts = num_local_experts
        self.time_falltoall = 0.0
        self.time_salltoall = 0.0
        self.time_moe = 0.0
        self.timers = SynchronizedWallClockTimer()
        self.wall_clock_breakdown = False

        self.use_tutel = use_tutel and TUTEL_INSTALLED and gate.k == 1

        if self.use_tutel:
            logger.info('Using Tutel optimizations.')
        elif use_tutel and not TUTEL_INSTALLED:
            logger.warning("Tutel optimization requested but not installed. "
                           "Proceeding without Tutel.")
        elif use_tutel and TUTEL_INSTALLED and gate.k != 1:
            logger.warning("To enable Tutel optimization, use top-1 instead of top-2 gate. "
                           "Proceeding without Tutel.")

    def _set_ep_group(self, ep_group):
        self.ep_group = ep_group
        self.gate.ep_group = ep_group # OURS

    # === OURS ===
    def begin_record_routing(self):
        self.gate.begin_record_routing()

    def end_record_routing(self):
        self.gate.end_record_routing()

    def remove_experts(self):
        assert self.gate.record_routing, "must record routing before removing experts"
        assert isinstance(self.gate.wg, GAMoEGateT), "gate network must have experts mask to allow adaptive process"

        if self.gate.routing_records is None:
            return
        
        routing_records = simple_all_reduce(self.gate.routing_records, self.ep_group)

        signed_routing_records = torch.sign(routing_records)
        updated_experts_mask = self.gate.wg.experts_mask * signed_routing_records
        self.gate.wg.experts_mask.data = updated_experts_mask.to(self.gate.wg.experts_mask.device)
        self.gate.wg.experts_mask.data = updated_experts_mask

    def add_experts(self):
        assert self.gate.record_routing, "must record routing before adding experts"
        assert isinstance(self.gate.wg, GAMoEGateT), "gate network must have experts mask to allow adaptive process"
        
        
        self.gate.sample_records = simple_all_reduce(self.gate.sample_records, self.ep_group)
        

        if sum(self.gate.sample_records) == 0:
            return
        
        normalized_sample_records = self.gate.sample_records / torch.norm(self.gate.sample_records)

        # choose one expert that is not active
        non_active_experts = np.argwhere(self.gate.wg.experts_mask.cpu().numpy() == 0)
        if len(non_active_experts) > 0:
            new_expert_index = non_active_experts[0][0]
            self.gate.wg.experts_mask.data[new_expert_index] = 1.0
            self.gate.wg.sim_matrix.data[:, new_expert_index] = normalized_sample_records.data
            self.gate.wg.gates.data[new_expert_index] = torch.tensor(0.0)
    
    def adaptive_update_experts(self):
        # print(self.gates[gate_index].gates)
        before_num = self.gate.wg.expert_num
        self.remove_experts()
        self.add_experts()
        self.gate.wg.expert_num = int(sum(self.gate.wg.experts_mask))
        end_num = self.gate.wg.expert_num
        # print('Adaptive update: From {} experts -> {} experts'.format(before_num, end_num))
        
    # === OURS ===

    def forward(self, *input: Tensor, **kwargs: Any) -> Tensor:

        if self.wall_clock_breakdown:
            self.timers('moe').start()

        # Implement Algorithm 2 from GShard paper.
        d_model = input[0].shape[-1]

        # Initial implementation -> Reshape into S tokens by dropping sequence dimension.
        # Reshape into G groups so that each group can distribute tokens equally
        # group_size = kwargs['group_size'] if 'group_size' in kwargs.keys() else 1
        reshaped_input = input[0].reshape(-1, d_model)

        if self.use_tutel:
            self.l_aux, C, E, indices_, locations_, gates_, self.exp_counts = self.gate(reshaped_input, input[1], True)
            S, M = reshaped_input.size(0), reshaped_input.size(1)

            if not hasattr(self, '_tutel_dispatcher'):
                self._tutel_dispatcher = tutel_moe.fast_dispatcher(E, C, M, dispatch_dtype=reshaped_input.dtype)
            self._tutel_dispatcher.update(indices_, locations_, gates_, capacity=C)
            dispatched_input = self._tutel_dispatcher.encode(reshaped_input)
        else:
            self.l_aux, combine_weights, dispatch_mask, self.exp_counts = self.gate(reshaped_input, input[1])
            dispatched_input = einsum("sec,sm->ecm", dispatch_mask.type_as(input[0]), reshaped_input)

        if self.wall_clock_breakdown:
            self.timers('falltoall').start()

        if groups._get_expert_model_parallel_world_size() == 1:
            # If the non-expert is tensor-parallel, it will create
            # duplicate tokens on the tensor-parallel ranks.
            # Since our experts are not tensor-parallel, these duplicates
            # need to be dropped to ensure correctness.
            # this also doubles up as a communication optimization as we are
            # reducing the all-to-all communication volume.
            dispatched_input = drop_tokens(dispatched_input, dim=1)

        dispatched_input = _AllToAll.apply(self.ep_group, dispatched_input)

        if self.wall_clock_breakdown:
            self.timers('falltoall').stop()
            self.time_falltoall = self.timers('falltoall').elapsed(reset=False)

        # Re-shape after all-to-all: ecm -> gecm
        dispatched_input = dispatched_input.reshape(self.ep_size, self.num_local_experts, -1, d_model)

        expert_output = self.experts(dispatched_input)

        if self.wall_clock_breakdown:
            self.timers('salltoall').start()

        expert_output = _AllToAll.apply(self.ep_group, expert_output)

        if self.wall_clock_breakdown:
            self.timers('salltoall').stop()
            self.time_salltoall = self.timers('salltoall').elapsed(reset=False)

        # Re-shape back: gecm -> ecm
        expert_output = expert_output.reshape(self.ep_size * self.num_local_experts, -1, d_model)

        if groups._get_expert_model_parallel_world_size() == 1:
            # the dropped duplicate tokens need to be gathered on each
            # tensor parallel rank again for the tensor-parallel
            # non-expert of the next layer.
            expert_output = gather_tokens(expert_output, dim=1)

        if self.use_tutel:
            combined_output = self._tutel_dispatcher.decode(expert_output.view(E * C, M))
        else:
            combined_output = einsum("sec,ecm->sm", combine_weights.type_as(input[0]), expert_output)

        a = combined_output.reshape(input[0].shape)

        if self.wall_clock_breakdown:
            self.timers('moe').stop()
            self.time_moe = self.timers('moe').elapsed(reset=False)

        return a
