# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from torch import nn
from torch.nn import functional as F

from deepspeed.utils import log_dist

from deepspeed.utils import groups
from .sharded_moe import MOELayer, TopKGate
from .experts import Experts
import typing


class MoE(torch.nn.Module):
    """Initialize an MoE layer.

    Arguments:
        hidden_size (int): the hidden dimension of the model, importantly this is also the input and output dimension.
        expert (torch.nn.Module): the torch module that defines the expert (e.g., MLP, torch.linear).
        num_experts (int, optional): default=1, the total number of experts per layer.
        ep_size (int, optional): default=1, number of ranks in the expert parallel world or group.
        k (int, optional): default=1, top-k gating value, only supports k=1 or k=2.
        capacity_factor (float, optional): default=1.0, the capacity of the expert at training time.
        eval_capacity_factor (float, optional): default=1.0, the capacity of the expert at eval time.
        min_capacity (int, optional): default=4, the minimum capacity per expert regardless of the capacity_factor.
        use_residual (bool, optional): default=False, make this MoE layer a Residual MoE (https://arxiv.org/abs/2201.05596) layer.
        noisy_gate_policy (str, optional): default=None, noisy gate policy, valid options are 'Jitter', 'RSample' or 'None'.
        drop_tokens (bool, optional): default=True, whether to drop tokens - (setting to False is equivalent to infinite capacity).
        use_rts (bool, optional): default=True, whether to use Random Token Selection.
        use_tutel (bool, optional): default=False, whether to use Tutel optimizations (if installed).
        enable_expert_tensor_parallelism (bool, optional): default=False, whether to use tensor parallelism for experts
    """

    def __init__(self,
                 hidden_size,
                 expert,
                 num_experts=1,
                 ep_size=1,
                 k=1,
                 capacity_factor=1.,
                 eval_capacity_factor=1.,
                 min_capacity=4,
                 use_residual=False,
                 noisy_gate_policy: typing.Optional[str] = None,
                 drop_tokens: bool = True,
                 use_rts=True,
                 use_tutel: bool = False,
                 enable_expert_tensor_parallelism: bool = False,
                 max_expert_num: int = 4):

        super(MoE, self).__init__()

        self.use_residual = use_residual
        self.enable_expert_tensor_parallelism = enable_expert_tensor_parallelism
        #OURS
        assert num_experts % ep_size == 0 or (max_expert_num > 0 and max_expert_num % ep_size == 0), f"Number of experts ({num_experts}) should be divisible by expert parallel size ({ep_size})"
        self.ep_size = ep_size
        self.expert_group_name = f"ep_size_{self.ep_size}"
        self.num_experts = num_experts
        self.num_local_experts = num_experts // self.ep_size
        
        # OURS
        if max_expert_num > 0:
            self.num_local_experts = max_expert_num // self.ep_size
            self.num_experts = max_expert_num
        else:
            self.num_local_experts = num_experts // self.ep_size

        log_dist(
            f'Creating MoE layer with num_experts: {num_experts} | num_local_experts: {self.num_local_experts} | expert_parallel_size: {self.ep_size}',
            [0])

        assert noisy_gate_policy is None or noisy_gate_policy in ['None', 'Jitter', 'RSample'], \
            'Unsupported noisy_gate_policy: ' + noisy_gate_policy

        experts = Experts(expert, self.num_local_experts, self.expert_group_name)
        self.deepspeed_moe = MOELayer(TopKGate(hidden_size, num_experts, k, capacity_factor, eval_capacity_factor,
                                               min_capacity, noisy_gate_policy, drop_tokens, use_rts, max_expert_num), # OURS
                                      experts,
                                      self.expert_group_name,
                                      self.ep_size,
                                      self.num_local_experts,
                                      use_tutel=use_tutel)
        if self.use_residual:
            self.mlp = expert
            # coefficient is used for weighted sum of the output of expert and mlp
            self.coefficient = torch.nn.Linear(hidden_size, 2)
        
        self.forward_count = 0
        self.test_time_record = False
        self.switch = True if k == -1 else False

    def set_deepspeed_parallelism(self):
        self._create_process_groups()

    def _create_process_groups(self):
        # Create process group for a layer if needed
        if self.expert_group_name not in groups._get_expert_parallel_group_dict():
            print(f"No existing process group found, creating a new group named: {self.expert_group_name}")
            if (groups.mpu is None) or (not self.enable_expert_tensor_parallelism):
                # Condition 1 - no groups.mpu means no tensor parallelism
                # Condition 2 - disabling expert tensor parallelism on purpose
                groups._create_expert_and_data_parallel(self.ep_size)
            else:
                # expert tensor parallelism is enabled
                groups._create_expert_data_and_model_parallel(self.ep_size, mpu=groups.mpu)
        # Set the group handle for the MOELayer (deepspeed_moe) object
        self.deepspeed_moe._set_ep_group(groups._get_expert_parallel_group(self.expert_group_name))
    
    # === OURS ===
    def begin_record_routing(self):
        self.deepspeed_moe.begin_record_routing()

    def end_record_routing(self):
        self.deepspeed_moe.end_record_routing()

    def adaptive_update_experts(self):
        self.deepspeed_moe.adaptive_update_experts()
    # === OURS ===

    def forward(self, hidden_states, used_token=None,
                if_begin_record_routing: bool = False, # OURS
                if_end_record_routing: bool = False):
        """ MoE forward

        Arguments:
            hidden_states (Tensor): input to the layer
            used_token (Tensor, optional): default: None, mask only used tokens

        Returns:
            A tuple including output, gate loss, and expert count.

            * output (Tensor): output of the model

            * l_aux (Tensor): gate loss value

            * exp_counts (int): expert count
        """
        # === OURS ===
        # only support dynmoe
        if self.switch:
            if self.forward_count % 300 == 0 and (self.forward_count // 300) % 3 == 1:
                if_begin_record_routing = True
                if_end_record_routing = False
                print('Start record routing at {}'.format(self.forward_count))
            elif self.forward_count % 300 == 0 and (self.forward_count // 300) % 3 == 2:
                if_begin_record_routing = False
                if_end_record_routing = True
                print('Update at {}'.format(self.forward_count))
            else:
                if_begin_record_routing = False
                if_end_record_routing = False

            # OURS
            if self.test_time_record and self.switch and not self.training:
                self.begin_record_routing()
                self.test_time_record = False
            
            if if_begin_record_routing:
                self.begin_record_routing()
    
        # === OURS ===
        
        output = self.deepspeed_moe(hidden_states, used_token)
                  
        if self.use_residual:
            # Residual MoE
            output_mlp = self.mlp(hidden_states)
            if type(output_mlp) is tuple:
                output_mlp = output_mlp[0]  # Ignore the bias term for now
            coef = self.coefficient(hidden_states)
            coef = torch.nn.functional.softmax(coef, dim=-1)
            output = output * coef[..., 0:1] + output_mlp * coef[..., 1:]
        
        # === OURS ===

        if self.switch:
            if if_end_record_routing:
                self.adaptive_update_experts()
                self.end_record_routing()

            if self.training:
                self.forward_count += 1

            # === OURS ===
        
        return output, self.deepspeed_moe.l_aux, self.deepspeed_moe.exp_counts
