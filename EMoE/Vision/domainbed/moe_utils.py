import os
import types
import tqdm
import torch
import torch.nn as nn
from datasets import load_dataset
import numpy as np
import time
import turtle
from tutel import moe as tutel_moe
import math

from k_means_constrained import KMeansConstrained
import torch.nn.functional as F
from transformers import ViTForImageClassification

import logging
_logger = logging.getLogger(__name__)

from typing import Dict, List, Optional, Set, Tuple, Union



def vit_to_MoE(args, model:ViTForImageClassification):
    print("all layers: ", len(model.vit.encoder.layer))
    if not os.path.exists('./cluster_ids'):
        os.mkdir('./cluster_ids')
    for i, layer_module in enumerate(model.vit.encoder.layer):
        if i in args.moe_layers:
            weight_0 = layer_module.intermediate.dense.weight.data # [4h, h]
            bias_0 = layer_module.intermediate.dense.bias.data # [4h]
            weight_1 = layer_module.output.dense.weight.data # [h, 4h]
            bias_1 = layer_module.output.dense.bias.data # [h]
            print("layer {}, weight_0 {}, weight_1 {}, bias_1 {}".format(i, weight_0.shape, weight_1.shape, bias_1.shape))
            print("begin constructing MoEs...")
            begin_time = time.time()
            if not args.MoE_from_ffn:
                expert_size = weight_0.shape[0]
                if args.router == 'gated_multi_gate':
                    layer_module.mlp = tutel_moe.moe_layer(
                        gate_type={'type': 'gated_multi_gate', 'max_expert_num': args.max_expert_num, 'fp32_gate': True, 'adaptive_experts': args.adaptive_experts},
                        experts={'type': 'ffn', 'count_per_node': args.num_experts,
                                'hidden_size_per_expert': expert_size,
                                'activation_fn': nn.GELU(),
                                },
                        model_dim=weight_0.shape[1],
                        batch_prioritized_routing=True,
                        is_gshard_loss=args.is_gshard_loss,
                        one_score_gate=args.one_score_gate,
                        normalize_one_score_gate=args.normalize_one_score_gate,
                        update_momentum=args.one_score_gate_update_momentum)
                else:
                    layer_module.mlp = tutel_moe.moe_layer(
                        gate_type={'type': args.router, 'k': args.topk, 'fp32_gate': True, 
                                'gate_noise': args.gate_noise,  #kwargs.get('gate_noise', 1.0), 
                                'capacity_factor': args.capacity_factor}, # kwargs.get('capacity_factor', 1.5)},
                        experts={'type': 'ffn', 'count_per_node': args.num_experts,
                                'hidden_size_per_expert': expert_size,
                                'activation_fn': nn.GELU(),
                                },
                        model_dim=weight_0.shape[1],
                        batch_prioritized_routing=True,
                        is_gshard_loss=False,
                        one_score_gate=args.one_score_gate,
                        normalize_one_score_gate=args.normalize_one_score_gate,
                        update_momentum=args.one_score_gate_update_momentum)
                
                expert_w0 = weight_0.unsqueeze(0).repeat(args.num_experts, 1, 1)
                bias_0 = bias_0.unsqueeze(0).repeat(args.num_experts, 1)
                expert_w1 = weight_1.unsqueeze(0).repeat(args.num_experts, 1, 1).transpose(1, 2)
                bias_1 = bias_1.unsqueeze(0).repeat(args.num_experts, 1)
                
                assert layer_module.mlp.experts.batched_fc1_w.data.shape == expert_w0.shape, f"{layer_module.mlp.experts.batched_fc1_w.data.shape} != {expert_w0.shape}"
                assert layer_module.mlp.experts.batched_fc1_bias.data.shape == bias_0.shape,  f"{layer_module.mlp.experts.htoh4.bias.data.shape} != {bias_0.shape}"
                assert layer_module.mlp.experts.batched_fc2_w.data.shape == expert_w1.shape, f"{layer_module.mlp.experts.batched_fc2_w.data.shape} != {expert_w1.shape}"
                assert layer_module.mlp.experts.batched_fc2_bias.data.shape == bias_1.shape, f"{layer_module.mlp.experts.batched_fc1_bias.data.shape} != {bias_1.shape}"
                
                layer_module.mlp.experts.batched_fc1_w.data.copy_(expert_w0)
                layer_module.mlp.experts.batched_fc1_bias.data.copy_(bias_0)
                layer_module.mlp.experts.batched_fc2_w.data.copy_(expert_w1)
                layer_module.mlp.experts.batched_fc2_bias.data.copy_(bias_1)
            else:
                assert weight_0.shape[0] % args.num_experts == 0, "number of experts must be divisible by number of keys"
                expert_size = weight_0.shape[0] // args.num_experts
                w_shape = weight_0.shape
                id_path = './cluster_ids/expert{}/vit_layer{}_{}_{}.npy'.format(args.num_experts, i, w_shape[0], w_shape[1])
                if os.path.exists(id_path):
                    _logger.info("Loading cluster ids from {}".format(id_path))
                    expert_ids = np.load(id_path)
                    expert_ids = torch.from_numpy(expert_ids).to(weight_0.device)
                else:
                    begin_time = time.time()
                    _logger.info("Clustering with KMeansConstrained...")
                    cluster_weight = torch.nn.functional.normalize(weight_0, dim=1, p=2).cpu().numpy()
                    kmeans = KMeansConstrained(n_clusters=args.num_experts, size_min=expert_size,\
                                                size_max=expert_size, random_state=0, n_jobs=16,
                                                max_iter=1000).fit(cluster_weight)
                    if not os.path.exists('./cluster_ids/expert{}'.format(args.num_experts)):
                        os.makedirs('./cluster_ids/expert{}'.format(args.num_experts))
                    np.save(id_path, kmeans.labels_)
                    _logger.info("Saving cluster ids to {}".format(id_path))
                    expert_ids = torch.from_numpy(kmeans.labels_).to(weight_0.device)
                    _logger.info("Clustering done in {:.2f} seconds, return key shape {}, centers {}.".format(time.time() - begin_time, kmeans.labels_.shape, kmeans.cluster_centers_.shape))
                    print("Clustering done in {:.2f} seconds, return key shape {}, centers {}.".format(time.time() - begin_time, kmeans.labels_.shape, kmeans.cluster_centers_.shape))
                
                if args.expert_repeat > 1:
                    print("Repeat experts {} times, each expert size {}".format(args.expert_repeat, expert_size*args.expert_repeat))
                    expert_size *= args.expert_repeat

                # select weight from weight_0 corresponding to expert_ids
                if args.expert_repeat > 1:
                    expert_w0 = []
                    new_bias_0 = []
                    for i in range(args.num_experts):
                        used_ids = expert_ids == i
                        for j in range(args.expert_repeat):
                            used_ids = torch.bitwise_or(used_ids, expert_ids == ((i + j) % args.num_experts))
                        expert_w0.append(weight_0[used_ids])
                        new_bias_0.append(bias_0[used_ids])
                    bias_0 = new_bias_0
                else:
                    expert_w0 = [weight_0[expert_ids == i] for i in range(args.num_experts)]
                    bias_0 = [bias_0[expert_ids == i] for i in range(args.num_experts)]
                expert_w0 = torch.stack(expert_w0) # [num_experts, expert_size, h]
                bias_0 = torch.stack(bias_0) # [num_experts, expert_size]

                if not args.random_init_gate:
                    expert_keys = [expert_w0[i].mean(dim=0) for i in range(args.num_experts)]
                    expert_keys = torch.stack(expert_keys)
                
                # select weight from weight_1 corresponding to expert_ids
                weight_1 = weight_1.t()
                if args.expert_repeat > 1:
                    expert_w1 = []
                    for i in range(args.num_experts):
                        used_ids = expert_ids == i
                        for j in range(args.expert_repeat):
                            used_ids = torch.bitwise_or(used_ids, expert_ids == ((i + j) % args.num_experts))
                        expert_w1.append(weight_1[used_ids])
                else:
                    expert_w1 = [weight_1[expert_ids == i] for i in range(args.num_experts)]
                expert_w1 = torch.stack(expert_w1) #.transpose(1, 2) # [num_experts, h, expert_size]
                
                bias_1 = bias_1.unsqueeze(0).repeat(args.num_experts, 1).div(args.topk)

                if args.router == 'gated_multi_gate':
                    layer_module.mlp = tutel_moe.moe_layer(
                        gate_type={'type': 'gated_multi_gate', 'max_expert_num': args.max_expert_num, 'fp32_gate': True, 'adaptive_experts': args.adaptive_experts},
                        experts={'type': 'ffn', 'count_per_node': args.num_experts,
                                'hidden_size_per_expert': expert_size,
                                'activation_fn': nn.GELU(),
                                },
                        model_dim=weight_0.shape[1],
                        batch_prioritized_routing=True,
                        is_gshard_loss=args.is_gshard_loss,
                        one_score_gate=args.one_score_gate,
                        normalize_one_score_gate=args.normalize_one_score_gate,
                        update_momentum=args.one_score_gate_update_momentum)
                else:
                    layer_module.mlp = tutel_moe.moe_layer(
                        gate_type={'type': args.router, 'k': args.topk, 'fp32_gate': True, 
                                'gate_noise': args.gate_noise,  #kwargs.get('gate_noise', 1.0), 
                                'capacity_factor': args.capacity_factor}, # kwargs.get('capacity_factor', 1.5)},
                        experts={'type': 'ffn', 'count_per_node': args.num_experts,
                                'hidden_size_per_expert': expert_size,
                                'activation_fn': nn.GELU(),
                                },
                        model_dim=weight_0.shape[1],
                        batch_prioritized_routing=True,
                        is_gshard_loss=False,
                        one_score_gate=args.one_score_gate,
                        normalize_one_score_gate=args.normalize_one_score_gate,
                        update_momentum=args.one_score_gate_update_momentum,
                    )
                

                assert layer_module.mlp.experts.batched_fc1_w.data.shape == expert_w0.shape, f"{layer_module.mlp.experts.batched_fc1_w.data.shape} != {expert_w0.shape}"
                assert layer_module.mlp.experts.batched_fc1_bias.data.shape == bias_0.shape,  f"{layer_module.mlp.experts.htoh4.bias.data.shape} != {bias_0.shape}"
                assert layer_module.mlp.experts.batched_fc2_w.data.shape == expert_w1.shape, f"{layer_module.mlp.experts.batched_fc2_w.data.shape} != {expert_w1.shape}"
                assert layer_module.mlp.experts.batched_fc2_bias.data.shape == bias_1.shape, f"{layer_module.mlp.experts.batched_fc1_bias.data.shape} != {bias_1.shape}"
                
                layer_module.mlp.experts.batched_fc1_w.data.copy_(expert_w0)  # correct
                layer_module.mlp.experts.batched_fc1_bias.data.copy_(bias_0)     # correct
                layer_module.mlp.experts.batched_fc2_w.data.copy_(expert_w1) # correct
                layer_module.mlp.experts.batched_fc2_bias.data.copy_(bias_1)     # correct

                
                if not args.random_init_gate and args.router=="top":
                    assert layer_module.mlp.gates[0].wg.weight.data.shape == expert_keys.shape, f"gate shape {layer_module.mlp.gates[0].wg.weight.data.shape} != keys shape {expert_keys.shape}"
                    layer_module.mlp.gates[0].wg.weight.data.copy_(expert_keys)
                    if layer_module.mlp.gates[0].wg.bias is not None:
                        layer_module.mlp.gates[0].wg.bias.data.zero_()
                else:
                    assert not args.one_score_gate, "Random init gate only support one_score=False"
                    print("Random init gate")
                    
            def _forward(self, 
                        hidden_states: torch.Tensor,
                        head_mask: Optional[torch.Tensor] = None,
                        output_attentions: bool = False,
                         ):
                self_attention_outputs = self.attention(
                    self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
                    head_mask,
                    output_attentions=output_attentions,)
                attention_output = self_attention_outputs[0]
                outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

                hidden_states = attention_output + hidden_states
                layer_output = self.layernorm_after(hidden_states)
                
                layer_output = self.mlp(layer_output)
                layer_output = self.output.dropout(layer_output)
                layer_output = layer_output + hidden_states
                
                outputs = (layer_output,) + outputs
                return outputs
            
            layer_module.forward = types.MethodType(_forward, layer_module)
            
            print("turn FFN to MoE")

