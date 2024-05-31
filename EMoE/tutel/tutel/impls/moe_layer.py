# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import copy
import os
import re
import time
import logging 
import collections
import importlib

import torch
from torch import Tensor
import torch.distributed as dist
from torch.nn import ModuleList
import torch.nn.functional as F
import numpy as np

from ..impls import communicate as C
from ..impls.generalized_fast_dispatch import fast_encode, fast_decode, extract_critical
from ..impls.overlap import a2a_ffn_overlap_forward
from . import losses

from ..gates.top import LinearTopKGate

class MOELayer(torch.nn.Module):
    """Tutel optimized MOELayer
    """
    @staticmethod
    def global_expert_count(num_local_experts, group=None):
        if not isinstance(num_local_experts, int):
            num_local_experts = -int(1 / (num_local_experts + 1e-5))
        world_size = C.get_world_size(group)
        if num_local_experts == 0:
            raise Exception("Invalid value of num_local_experts: %d" % num_local_experts)
        if num_local_experts > 0:
            return num_local_experts * world_size
        assert world_size % -num_local_experts == 0, f"Excepting {-num_local_experts} devices to share an expert param, while global device count is {world_size}."
        return world_size // -num_local_experts
    
    @staticmethod
    def local_expert_count(num_global_experts, group=None):       
        world_size = C.get_world_size(group)
        assert num_global_experts % world_size == 0, f"Excepting {num_global_experts} devices to share an expert param, while global device count is {world_size}."
        return num_global_experts // world_size

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        buff_name = prefix + '_num_global_experts'
        if buff_name not in state_dict:
            logging.warning(f"\033[31mYou are loading a legacy format of checkpoint with at least one Tutel MoE layer inside, which wouldn't support new Tutel feature allowing the number of experts per checkpoint file to mutate.\033[0m")
            logging.warning(f"\033[31m  The next time you overwrite it with new checkpoint, the recording format will be updated automatically.\033[0m")
            logging.warning(f"\033[31m  However, the new format won't be compatible with early Tutel versions, unless you force loading it with `model.load_state_dict(.., strict=False)`.\033[0m")
            state_dict[buff_name] = self._num_global_experts
        else:
            state_experts, expect_experts = int(state_dict[buff_name]), self.num_global_experts
            # assert state_experts == expect_experts, "Failed to load state from checkpoint: the number of global experts mismatch (%s <- %s)" % (expect_experts, state_experts)

        buff_name = prefix + '_max_num_global_experts'
        if buff_name not in state_dict:
            logging.warning(f"\033[31mYou are loading a legacy format of checkpoint with at least one Tutel MoE layer inside, which wouldn't support new Tutel feature allowing the number of experts per checkpoint file to mutate.\033[0m")
            logging.warning(f"\033[31m  The next time you overwrite it with new checkpoint, the recording format will be updated automatically.\033[0m")
            logging.warning(f"\033[31m  However, the new format won't be compatible with early Tutel versions, unless you force loading it with `model.load_state_dict(.., strict=False)`.\033[0m")
            state_dict[buff_name] = self._max_num_global_experts
        else:
            state_experts, expect_experts = int(state_dict[buff_name]), self.max_num_global_experts
            assert state_experts == expect_experts, "Failed to load state from checkpoint: the max number of global experts mismatch (%s <- %s)" % (expect_experts, state_experts)

        for name, param in self.experts.named_parameters():
            buff_name = prefix + 'experts.' + name
            assert buff_name in state_dict, "Could not find parameter `%s` in state_dict." % buff_name
            if state_dict[buff_name].numel() == param.numel():
                state_dict[buff_name] = state_dict[buff_name].view(param.shape)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return super().state_dict(destination, prefix, keep_vars)

    @property
    def num_global_experts(self):
        return int(self._num_global_experts)
    
    @property
    def max_num_global_experts(self):
        return int(self._max_num_global_experts)

    def __init__(
        self,
        gate_type,
        model_dim: int,
        experts=None,
        scan_expert_func=None,
        result_func=None,
        group=None,
        seeds=None,
        a2a_ffn_overlap_degree=1,
        is_postscore=True,
        batch_prioritized_routing=False,
        normalize_gate=True,
        is_gshard_loss=True,
        parallel_type='adaptive:1',
        use_2dh=False,
        one_score_gate=False,
        normalize_one_score_gate=False,
        value_norm_weighted=False,
        update_momentum=0.0,
        # share_value=False,
        **kwargs
    ):
        super().__init__()
        assert model_dim % 2 == 0, "Model_dim (%s) must be even value, while this Model_dim mod 2 > 0." % model_dim
        group = group or dist.group.WORLD

        if 'pad_samples' in kwargs:
            logging.warning(f"`pad_samples` option in Tutel Moe-layer has been deprecated, as Tutel always assumes `pad_samples=False` for better efficiency.")
            kwargs.pop('pad_samples')
        for k in kwargs:
            raise Exception('Unrecognized argument provided to Tutel Moe-layer: %s' % k)

        self.group = group
        self.result_func = result_func
        self.skip_moe = (int(os.environ.get('SKIP_MOE', '0')) != 0)

        # initially set max number equal to local number
        self.num_local_experts = experts.pop('count_per_node', 1)
        self.max_num_local_experts = self.num_local_experts
        self.register_buffer('_num_global_experts', torch.tensor(MOELayer.global_expert_count(self.num_local_experts, self.group)))
        self.register_buffer('_max_num_global_experts', torch.tensor(MOELayer.global_expert_count(self.max_num_local_experts, self.group)))

        self.world_size = C.get_world_size(self.group)
        # in fact this will not happen
        if self.num_global_experts < self.world_size:
            self.sharded_count = self.world_size // self.num_global_experts
            self.num_local_experts = 1
        else:
            self.sharded_count = 1

        self.auto_parallel, self.adaptive_degree, self.use_model_parallel = False, self.sharded_count, True
        self.valid_rs = [0] + [i for i in range(1, self.sharded_count + 1) if self.sharded_count % i == 0]
        # 实际上valid_rs总是[0]

        if parallel_type.startswith('adaptive:'):
            self.adaptive_degree = int(parallel_type[parallel_type.index(':') + 1:])
            self.adaptive_degree = min(max(self.adaptive_degree, 0), self.sharded_count)
            if self.adaptive_degree not in self.valid_rs:
                raise Exception("Unexpected value of adaptive_degree: %d, expecting a candidate within %s." % (self.adaptive_degree, self.valid_rs))
        elif self.sharded_count == 1:
            pass
        elif parallel_type in ('data', 'model'):
            self.adaptive_degree = 1 if parallel_type == 'data' else self.sharded_count
        elif parallel_type == 'auto':
            self.adaptive_degree = 1
        else:
            raise Exception('Unrecognized parallel type specified: %s' % parallel_type)

        self.model_dim = model_dim

        self.is_postscore = is_postscore
        self.batch_prioritized_routing = batch_prioritized_routing
        if int(os.environ.get('BATCH_PRIO', 0)) != 0:
            self.batch_prioritized_routing = True
        self.normalize_gate = normalize_gate
        self.is_gshard_loss = is_gshard_loss

        self.a2a_ffn_overlap_degree = a2a_ffn_overlap_degree
        self.use_2dh = use_2dh

        if seeds is not None and seeds[1] is not None:
            torch.manual_seed(seeds[1])
            
        # self.share_value = share_value

        # GAMoE requires define the gates first.
        if isinstance(gate_type, str) and (not gate_type.startswith("GMGate")):
            
            assert re.match(r'^Top[0-9]+Gate$', gate_type), "Unrecognized gate_type: %s" % gate_type
            top_k = int(gate_type[3:-4])
            logging.warning(f"gate_type value `{gate_type}` in Tutel Moe-layer has been deprecated, please use gate_type = {{'type': 'top', 'k': {top_k}}} instead.")
            gate_type = {'type': 'top', 'k': top_k}

        elif isinstance(gate_type, str) and gate_type.startswith("GMGate"):
            max_K = int(gate_type[6:])
            gate_type = {'type': 'gated_multi_gate', 'max_expert_num': max_K}
            # if use GAMoE, set the max number of global experts to the parameters in config
            
        if 'max_expert_num' in gate_type:
            self._max_num_global_experts.data = torch.tensor(int(gate_type['max_expert_num']))
            self.max_num_local_experts = self.local_expert_count(self.max_num_global_experts, self.group)

        if not isinstance(gate_type, list):
            gate_type = [gate_type]

        self.gates = []
        for gi, single_gate_type in enumerate(gate_type):
            gate_type = single_gate_type['type']
            single_gate_type.pop('type')
            assert re.match(r'[a-zA-Z0-9\_]+', gate_type), "Gate type must only include digits, letters and underline characters."

            if seeds is not None and seeds[0] is not None:
                torch.manual_seed(seeds[0] + gi)
            try:
                single_gate = importlib.import_module(f'...gates.{gate_type}', __name__)
            except ModuleNotFoundError:
                raise Exception("Unrecognized gate_type: %s" % gate_type)

            gate_module = single_gate.Gate(model_dim=self.model_dim, num_global_experts=self.num_global_experts, normalize_one_score_gate=normalize_one_score_gate, **single_gate_type)
            if not hasattr(gate_module, 'gate_noise'):
                gate_module.gate_noise = single_gate_type.get('gate_noise', 0.0)
            if not hasattr(gate_module, 'capacity_factor'):
                gate_module.capacity_factor = single_gate_type.get('capacity_factor', float(os.environ.get('CAP_FACTOR', 1.0)))

            self.gates += [gate_module]
        print("Gate types: ", [x.__class__.__name__ for x in self.gates])
        self.gates = ModuleList(self.gates)

        experts_type = experts.pop('type')
        if experts_type == 'custom':
            self.experts = cast(ModuleList, experts['module'])
        else:
            assert re.match(r'[a-zA-Z0-9\_]+', experts_type), "Expert type must only include digits, letters and underline characters."
            try:
                fused_experts = importlib.import_module(f'...experts.{experts_type}', __name__)
            except ModuleNotFoundError:
                raise Exception('Builtin expert type is not recognized: %s' % experts_type)

            if experts_type == 'ffn':
                assert 'fused_custom_fn' not in experts, "`fused_custom_fn` option for Tutel Moe-layer has been deprecated, please follows helloworld_from_scratch.py for custom construction instead."
                assert 'implicit_dropout_p' not in experts, "`implicit_dropout_p` option for Tutel Moe-layer has been deprecated, please use torch.nn.Dropout(p=implicit_dropout_p) on custom activation_fn (for fc1_dropout) and after Tutel Moe-layer (for fc2_dropout) instead."

            self.experts = fused_experts.ExpertModule(**experts)

        self.experts.update(self)

        if scan_expert_func is not None:
            for n, p in self.experts.named_parameters():
                scan_expert_func(n, p)
        for n, p in self.experts.named_parameters():
            setattr(p, '_tutel_expert', True)

        
        #专门用于EMoE
        self.one_score_gate = one_score_gate
        if self.one_score_gate:
            self.update_momentum = update_momentum
            assert isinstance(self.gates[0], LinearTopKGate), "only simple gate is supported"
            print("Using one score gate with momentum {}, freeze gate weight!".format(self.update_momentum))
            self.normalize_one_score_gate = normalize_one_score_gate
            self.gates[0].wg.weight.require_grad = False
            self.value_norm_weighted = value_norm_weighted
            if self.value_norm_weighted:
                print("### using value norm weighted key-gate ###")
        
        self.record_routing = False    
        
        if seeds is not None and len(seeds) > 2 and seeds[2] is not None:
            torch.manual_seed(seeds[2])

    def extra_repr(self):
        return 'Top-K(s) = %s, Total-Experts = %d [managed by %d device(s)],' % (
            [f'k={x.top_k}, noise={x.gate_noise}' for x in self.gates],
            self.num_global_experts,
            self.world_size,
        )

    def get_parameter_iterator(self, param_type):
        if param_type == 'gate':
            return self.gates.named_parameters()
        elif param_type == 'local_experts':
            return self.experts.named_parameters()
        else:
            raise Exception("Specified parameter type is not recognized: %s. Valid `param_type` includes: gate, local_experts." % param_type)

    def expert_local(self, x, reserve_shape):
        y = self.experts(x.view(x.size(0), x.size(1), *reserve_shape), self)
        self.protected_shape = y.shape
        return y.reshape(y.size(0), y.size(1), -1)

    def begin_record_routing(self):
        self.reset_record_routing()
        self.record_routing = True

    def end_record_routing(self):
        self.record_routing = False
    
    def reset_record_routing(self):
        # self.routing_records = torch.zeros(self.num_global_experts, dtype=torch.long, device=self.experts.batched_fc1_w.device)
        self.routing_records = torch.zeros(self.max_num_global_experts + 1, dtype=torch.long, device=self.experts.batched_fc1_w.device)
        self.sample_records = None

    def get_routing_records(self):
        return self.routing_records[:self.max_num_global_experts]
    
    def get_sample_records(self):
        return self.sample_records
    
    def remove_experts(self, gate_index):
        assert self.record_routing, "must record routing before removing experts"
        assert hasattr(self.gates[gate_index], "experts_mask"), "gate network must have experts mask to allow adaptive process"

        self.routing_records = C.simple_all_reduce(self.routing_records, self.group)

        # print(self.routing_records)

        signed_rounting_records = torch.sign(self.routing_records)
        # print(self.gates[gate_index].experts_mask * signed_rounting_records[:self.max_num_global_experts])
        self.gates[gate_index].experts_mask.data = (self.gates[gate_index].experts_mask * signed_rounting_records[:self.max_num_global_experts])

    def add_experts(self, gate_index):
        assert self.record_routing, "must record routing before adding experts"
        assert hasattr(self.gates[gate_index], "experts_mask"), "gate network must have experts mask to allow adaptive process"

        if self.sample_records is None:
            return
        
        self.sample_records = C.simple_all_reduce(self.sample_records, self.group)
        
        # print(self.sample_records.shape)
        normalized_sample_records = self.sample_records / torch.norm(self.sample_records)
        # choose one expert that is not active
        non_active_experts = np.argwhere(self.gates[gate_index].experts_mask.cpu().numpy() == 0)
        if len(non_active_experts) > 0:
            new_expert_index = non_active_experts[0][0]
            self.gates[gate_index].experts_mask.data[new_expert_index] = 1.0
            self.gates[gate_index].sim_matrix.data[:, new_expert_index] = normalized_sample_records.data
            self.gates[gate_index].gates.data[new_expert_index] = torch.tensor(0.0)

    def adaptive_update_experts(self, gate_index=0):
        # print(self.gates[gate_index].gates)
        before_num = int(self._num_global_experts.data)
        self.remove_experts(gate_index)
        self.add_experts(gate_index)
        self._num_global_experts.data = torch.tensor(int(sum(self.gates[gate_index].experts_mask)))
        end_num = int(self._num_global_experts.data)
        print('Adaptive update: From {} experts -> {} experts'.format(before_num, end_num))

        # print(self.num_global_experts, int(sum(self.gates[gate_index].experts_mask)))
        # print(self.gates[gate_index].experts_mask)
        # print(self.gates[gate_index].gates)




    def forward(self, input: Tensor, gate_index=0, capacity_factor=None, top_k=None, a2a_ffn_overlap_degree=None, reserve_dims=1, inequivalent_tokens=False, adaptive_r=None):

        # if  hasattr(self.gates[gate_index], 'adaptive_experts') and self.gates[gate_index].adaptive_experts:
        #     self.begin_record_routing()


        if self.skip_moe:
            result_output = input
            result_output.l_aux = None
            return self.result_func(result_output) if self.result_func is not None else result_output

        original_shape, original_dtype  = input.shape, input.dtype
        assert len(original_shape) >= 2, "Input data must be at least 2D tensor: (s)amples, .., (m)odel_dim"

        x = input.reshape(-1, original_shape[-reserve_dims:].numel())
        for p in self.experts.parameters():
            x = x.to(p.dtype)
            break
        gctx = self.gates[gate_index]
        if a2a_ffn_overlap_degree is not None:
            self.a2a_ffn_overlap_degree = a2a_ffn_overlap_degree
        a2a_ffn_overlap_degree = self.a2a_ffn_overlap_degree

        def routing(top_k):
            logits, gate_top_k = gctx(x)

            if self.training and gctx.gate_noise > 0:
                logits_w_noise = logits + gctx.gate_noise * torch.randn_like(logits) / self.num_global_experts
            else:
                logits_w_noise = logits

            if hasattr(gctx, "adaptive_top_k") and gctx.adaptive_top_k:
                top_k = gate_top_k
                scores = logits_w_noise
            else:
                scores = F.softmax(logits_w_noise, dim=1)


            if self.is_gshard_loss:
                _loss_fn = lambda gates, topk_ids: losses.gshard_loss(gates, topk_ids, self.num_global_experts)
            elif gctx.enable_softmax_logits:
                _loss_fn = lambda gates, topk_ids: losses.load_importance_loss(
                    F.softmax(logits, dim=1), logits_w_noise.gather(index=topk_ids, dim=1),
                    self.num_global_experts, gctx.gate_noise)
            else:
                _loss_fn = lambda gates, topk_ids: losses.diverse_and_simple_gate_loss(gates, topk_ids, gctx.sim_matrix, gctx.experts_mask)
            return logits.dtype, extract_critical(scores,
                top_k = gctx.top_k if top_k is None else top_k,
                loss_fn = _loss_fn,
                capacity_factor = gctx.capacity_factor if capacity_factor is None else capacity_factor,
                batch_prioritized_routing = self.batch_prioritized_routing,
                normalize_gate = self.normalize_gate,
                group = self.group,
                alignment = self.sharded_count * a2a_ffn_overlap_degree,
                inequivalent_tokens = inequivalent_tokens,
                one_score_gate = self.one_score_gate
            )


        if x.is_cuda:
            with torch.cuda.amp.autocast(enabled=False):
                logits_dtype, (crit, l_aux) = routing(top_k)
        else:
            logits_dtype, (crit, l_aux) = routing(top_k)

        y = fast_encode(x.to(logits_dtype), crit, self.is_postscore).to(x.dtype)

        if adaptive_r is not None:
            self.adaptive_degree = adaptive_r

        if self.adaptive_degree == 0:
            y = self.expert_local(y, original_shape[-reserve_dims:])
        else:
            if self.auto_parallel:
                self.use_model_parallel = (y.numel() * (self.sharded_count - 1) * 2 < sum([x.numel() for x in self.experts.parameters()]))

            if self.num_global_experts < self.world_size: # need to promise this thing won't happen now
                if self.use_model_parallel:
                    y = y.repeat(1, self.adaptive_degree, 1).view(self.world_size, -1, y.size(2))
                else:
                    y = y.view(self.world_size, -1, y.size(2))

            # Currently, GAMoE does not support this
            if a2a_ffn_overlap_degree > 1 and y.is_cuda:
                def expert_fn(expert_input):
                    return self.expert_local(expert_input, original_shape[-reserve_dims:])
                y = a2a_ffn_overlap_forward(y, expert_fn=expert_fn, a2a_ffn_overlap_degree=a2a_ffn_overlap_degree, use_2dh=self.use_2dh, group=self.group)
            else:
                # Currently, GAMoE only support this
                y = C.all_to_all(y, 1, 0, use_2dh=self.use_2dh, group=self.group)
                y = self.expert_local(y, original_shape[-reserve_dims:])
                y = C.all_to_all(y, 0, 1, use_2dh=self.use_2dh, group=self.group)

            if self.num_global_experts < self.world_size:
                if self.use_model_parallel:
                    y = torch.sum(y.view(self.num_global_experts, self.adaptive_degree, -1, y.size(2)), dim=1)
                else:
                    y = y.view(self.num_global_experts, -1, y.size(2))

        y = fast_decode(y.to(logits_dtype), crit, self.is_postscore)
        
        if self.record_routing:
            top_k_gates = torch.stack(crit[1]).to(torch.long).reshape(-1)
            ones = torch.ones_like(top_k_gates)
            self.routing_records.scatter_add_(0, top_k_gates, ones)

            # get the samples that do not activate any experts
            sample_experts_records = crit[1][0].to(torch.long)
            # print(sample_experts_records)
            # sample_experts_records = torch.sum(sample_experts_records < self.max_num_global_experts, dim=1)
            # print(sample_experts_records)
            sample_embeddings = x[sample_experts_records == self.max_num_global_experts]
            # print(sample_embeddings.shape[0])
            if sample_embeddings.shape[0] > 0:
                if self.sample_records is None:
                    self.sample_records = torch.zeros(self.model_dim, dtype=self.experts.batched_fc1_w.dtype, device=self.experts.batched_fc1_w.device)
                self.sample_records += torch.sum(sample_embeddings, dim=0)

            

        y = y.view(list(original_shape[:-reserve_dims]) + list(self.protected_shape[-reserve_dims:])).to(original_dtype)
        self.l_aux = y.l_aux = l_aux

        # if  hasattr(self.gates[gate_index], 'adaptive_experts') and self.gates[gate_index].adaptive_experts:
        #     self.adaptive_update_experts(gate_index)
        #     self.end_record_routing()

        return self.result_func(y) if self.result_func is not None else y

    def update_one_score_gate(self):
        assert self.one_score_gate, "one_score_gate must be True"

        if self.value_norm_weighted:
            value_norm = torch.norm(self.experts.batched_fc2_w, p=2, dim=-1, keepdim=True)
            value_weight = F.normalize(value_norm, dim=-2, p=1)
            weighted_keys = (self.experts.batched_fc1_w * value_weight).sum(dim=1)
        else:
            weighted_keys = self.experts.batched_fc1_w.data.mean(dim=1)

        if self.normalize_one_score_gate:
            if self.update_momentum:
                self.gates[0].wg.weight.data.copy_(self.update_momentum * self.gates[0].wg.weight.data + (1 - self.update_momentum) * F.normalize(weighted_keys, dim=-1))
            else:
                self.gates[0].wg.weight.data.copy_(F.normalize(weighted_keys, dim=-1))
        else:
            if self.update_momentum:
                self.gates[0].wg.weight.data.copy_(self.update_momentum * self.gates[0].wg.weight.data + (1 - self.update_momentum) * weighted_keys)
            else:
                if self.value_norm_weighted:
                    value_norm
                self.gates[0].wg.weight.data.copy_(weighted_keys)

moe_layer = MOELayer
