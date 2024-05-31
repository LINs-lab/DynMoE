# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import sys
from itertools import chain

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import torch.autograd as autograd
from domainbed.lib.misc import (
    random_pairs_of_minibatches, ParamDict, MovingAverage, l2_between_dicts
)
from copy import deepcopy
import copy

import domainbed.vision_transformer as vision_transformer #, vision_transformer_hybrid
from collections import defaultdict, OrderedDict

try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
except:
    backpack = None

from domainbed import networks
# from domainbed import resnet_variants
import torchvision.models as models

ALGORITHMS = [
    'ERM',
    'Fish',
    'IRM',
    'GroupDRO',
    'Mixup',
    'MLDG',
    'CORAL',
    'MMD',
    'DANN',
    'CDANN',
    'MTL',
    'SagNet',
    'ARM',
    'VREx',
    'RSC',
    'SD',
    'ANDMask',
    'SANDMask',
    'IGA',
    'SelfReg',
    "Fishr",
    'TRM',
    'IB_ERM',
    'IB_IRM',
    'CAD',
    'CondCAD',
    'GMOE'
]

class Model_args:
    def __init__(self, hparams):
        self.hparams = hparams
        for key in hparams:
            setattr(self, key, hparams[key])

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    transforms = {}

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class MovingAvg:
    def __init__(self, network):
        self.network = network
        self.network_sma = copy.deepcopy(network)
        self.network_sma.eval()
        self.sma_start_iter = 100
        self.global_iter = 0
        self.sma_count = 0

    def update_sma(self):
        self.global_iter += 1
        if self.global_iter >= self.sma_start_iter:
            self.sma_count += 1
            for param_q, param_k in zip(self.network.parameters(), self.network_sma.parameters()):
                param_k.data = (param_k.data * self.sma_count + param_q.data) / (1. + self.sma_count)
        else:
            for param_q, param_k in zip(self.network.parameters(), self.network_sma.parameters()):
                param_k.data = param_q.data


class ERM_SMA(Algorithm, MovingAvg):
    """
    Empirical Risk Minimization (ERM) with Simple Moving Average (SMA) prediction model
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        Algorithm.__init__(self, input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        MovingAvg.__init__(self, self.network)

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.network(all_x), all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_sma()
        return {'loss': loss.item()}

    def predict(self, x):
        self.network_sma.eval()
        return self.network_sma(x)


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier).cuda()
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)

from torch.cuda.amp import autocast, GradScaler

class GMOE(Algorithm):
    """
    SFMOE
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(GMOE, self).__init__(input_shape, num_classes, num_domains, hparams)
        if hparams.get('vit_type', 'small') == 'small':
            print("using small ViT")
            create_model = vision_transformer.deit_small_patch16_224
            layer_num = 12
        elif hparams.get('vit_type', 'small') == 'base':
            print("using base ViT")
            create_model = vision_transformer.deit_base_patch16_224
            layer_num = 12
        elif hparams.get('vit_type', 'small') == 'large':
            print("using large ViT")
            from transformers import ViTForImageClassification
            self.model = ViTForImageClassification.from_pretrained('/mnt/share/agiuser/qiuzihan/models/vit-large-patch16-224')
            args = Model_args(hparams)
            args.tune_moe_layers_only = hparams.get('tune_moe_layers_only', True)
            args.tune_gates_only = hparams.get('tune_gates_only', False)
            args.tune_layernorm = hparams.get('tune_layernorm', False)
            args.tune_embeddings = hparams.get('tune_embeddings', True)
            args.tune_cls = hparams.get('tune_cls', True)
            args.expert_repeat = hparams.get('expert_repeat', 1)
            args.gate_noise = hparams.get('gate_noise', 1.0)
            args.capacity_factor = hparams.get('capacity_factor', 1.5)
            args.normalize_one_score_gate = hparams.get('normalize_one_score_gate', False)
            args.MoE_from_ffn = hparams.get('MoE_from_ffn', False)
            args.one_score_gate = hparams.get('one_score_gate', False)
            args.random_init_gate = hparams.get('random_init_gate', False)
            args.one_score_gate_update_momentum = hparams.get('one_score_gate_update_momentum', 0.0)
            args.adaptive_experts = hparams.get('adaptive_experts', False)
            args.is_gshard_loss = hparams.get('is_gshard_loss', True)
            args.max_expert_num = hparams.get('max_expert_num', 8)
            args.use_sparse = hparams.get('use_sparse', False)
            layer_num = 24
            if hparams.get('every_moe', True):
                moe_layers = list(range(layer_num-hparams.get('num_inter', 2)*2, layer_num, 2))
            else:
                moe_layers = list(range(layer_num-hparams.get('num_inter', 2), layer_num))
            args.moe_layers = moe_layers
            self.args = args
            if hparams.get('peft', True):
                print("using PEFT")
                from peft import (get_peft_config, get_peft_model, get_peft_model_state_dict,
                set_peft_model_state_dict, LoraConfig, PeftType, PrefixTuningConfig, PromptEncoderConfig, )
                peft_config = LoraConfig(
                    r=hparams.get('lora_config', 32),
                    lora_alpha=hparams.get('lora_config', 32),
                    target_modules=["query", "value", "key"],
                    lora_dropout=0.1,
                    bias="none",
                )
                self.model = get_peft_model(self.model, peft_config, adapter_name="lora")
            
            if hparams.get('vanilla_ViT', False):
                print("using vanilla ViT")
                for n, p in self.model.named_parameters():
                    if 'embeddings' in n and args.tune_embeddings:
                        p.requires_grad = True
                    if 'classifier' in n:
                        p.requires_grad = True
                    print(n, p.requires_grad)
            else:
                print("using MOE")
                from domainbed.moe_utils import vit_to_MoE
                vit_to_MoE(args, self.model)

                for n, p in self.model.named_parameters():
                    n_list = n.split('.')
                    freeze_flag = False
                    if 'layer' in n_list:
                        temp_layer = int(n_list[n_list.index('layer') + 1])
                        if args.tune_moe_layers_only:
                            if temp_layer not in args.moe_layers:
                                freeze_flag = True
                            else:
                                if args.tune_gates_only:
                                    if args.tune_layernorm and 'layernorm' in n and temp_layer in args.moe_layers:
                                        freeze_flag = False
                                    else:
                                        freeze_flag = True
                    if args.tune_moe_layers_only and (('embeddings' in n_list and not args.tune_embeddings) or 'pooler' in n_list or 'layernorm' in n_list):
                        freeze_flag = True
                    
                    if args.tune_gates_only and 'gate' in n and not args.one_score_gate:
                        freeze_flag = False
                    if 'lora' in n:
                        freeze_flag = False
                    if freeze_flag:
                        p.requires_grad = False
                        print(f'freeze: {n}')
                    else:
                        p.requires_grad = True
                        print(f'****train: {n}')
                
            self.scaler = GradScaler()

                
        if hparams.get('vit_type', 'small') != 'large':
            if hparams.get('vanilla_ViT', False):
                print("using vanilla ViT")
                self.model = create_model(pretrained=True, num_classes=num_classes,
                                                                        mlp_ratio=4., drop_path_rate=0.1)
            else:
                if hparams.get('every_moe', True):
                    moe_layers = ['F'] * (layer_num-2*hparams.get('num_inter', 2)-hparams.get('skip_last', 0)) + ['S', 'F'] * hparams.get('num_inter', 2) + ['F']*hparams.get('skip_last', 0)
                else:
                    moe_layers = ['F'] * (layer_num-hparams.get('num_inter', 2)-hparams.get('skip_last', 0)) + ['S'] * hparams.get('num_inter', 2) + ['F']*hparams.get('skip_last', 0)
                print("MoE layers: ", moe_layers)
                self.model = create_model(pretrained=True, 
                                            num_classes=num_classes, 
                                            moe_layers=moe_layers,
                                            mlp_ratio=4.,
                                            num_experts=hparams.get('num_experts', 6),
                                            topk=hparams.get('topk', 1),
                                            is_tutel=True,
                                            drop_path_rate=0.1,
                                            router=hparams.get('router', 'cosine_top'),
                                            max_expert_num=hparams.get('max_expert_num', 8),
                                            use_sparse = hparams.get('use_sparse', False),
                                            is_gshard_loss = hparams.get('is_gshard_loss', True),
                                            one_score_gate=hparams.get('one_score_gate', False),
                                            normalize_one_score_gate=hparams.get('normalize_one_score_gate', False),
                                            one_score_gate_update_momentum=hparams.get('one_score_gate_update_momentum', 0.0),
                                            gate_noise=hparams.get('gate_noise', 1.0),
                                            capacity_factor=hparams.get('capacity_factor', 1.5),
                                            MoE_from_ffn=hparams.get('MoE_from_ffn', False),
                                            expert_repeat=hparams.get('expert_repeat', 1),
                                            )
                print("using GMOE with {} inter layers, {} experts, topk {}".format(hparams.get('num_inter', 2), hparams.get('num_experts', 6), hparams.get('topk', 1)))
        
        self.model = self.model.cuda()
        
        if hparams.get('vit_type', 'small') != 'large':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams['weight_decay'])
        else:
            fast_lr = ['lora', 'score', 'classifier']
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in fast_lr)],
                    "lr": self.hparams["lr"]*10,
                    "weight_decay": self.hparams['weight_decay']
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in fast_lr)],
                    "lr": self.hparams["lr"],
                    "weight_decay": self.hparams['weight_decay']
                }
            ]
            self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        
        self.adaptive_experts = hparams.get('adaptive_experts', False)
        self.one_score_gate=hparams.get('one_score_gate', False)
        self.aux_loss_weight = hparams.get('aux_loss_weight', 1.0)
        self.one_score_gate_update_momentum = hparams.get('one_score_gate_update_momentum', 0.0)
        if self.one_score_gate:
            print("using one score update momentume:", self.one_score_gate_update_momentum)

        if self.adaptive_experts:
            if self.hparams.get('vit_type', 'small') == 'large':
                for i, layer in enumerate(self.model.vit.encoder.layer):
                    if i in self.args.moe_layers:
                        layer.mlp.begin_record_routing()
            else:
                for block in self.model.blocks:
                    if block.cur_layer == 'S':
                        block.mlp.begin_record_routing()

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        if self.hparams.get('vit_type', 'small') == 'large':
            self.optimizer.zero_grad()
            with autocast():
                loss = F.cross_entropy(self.predict(all_x), all_y)
                loss_aux = 0
                if not self.hparams.get('vanilla_ViT', False):
                    for i, layer in enumerate(self.model.vit.encoder.layer):
                        if i in self.args.moe_layers:
                            loss_aux += layer.mlp.l_aux * 0.01
                    loss += loss_aux * self.aux_loss_weight
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss = F.cross_entropy(self.predict(all_x), all_y)
            loss_aux_list = []
            for block in self.model.blocks:
                if getattr(block, 'aux_loss') is not None:
                    loss_aux_list.append(block.aux_loss)
            loss_aux = 0
            for layer_loss in loss_aux_list:
                loss_aux += layer_loss
            loss += loss_aux * self.aux_loss_weight
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        if self.one_score_gate:
            if self.hparams.get('vit_type', 'small') == 'large':
                for i, layer in enumerate(self.model.vit.encoder.layer):
                    if i in self.args.moe_layers:
                        layer.mlp.update_one_score_gate()
            else:
                for block in self.model.blocks:
                    if block.cur_layer == 'S':
                        block.mlp.update_one_score_gate()

        if self.adaptive_experts:
            if self.hparams.get('vit_type', 'small') == 'large':
                for i, layer in enumerate(self.model.vit.encoder.layer):
                    if i in self.args.moe_layers:
                        layer.mlp.adaptive_update_experts()
                        # layer.mlp.end_record_routing()
                        layer.mlp.begin_record_routing()
            else:
                for block in self.model.blocks:
                    if block.cur_layer == 'S':
                        block.mlp.adaptive_update_experts()
                        # block.mlp.end_record_routing()
                        block.mlp.begin_record_routing()
        
        return {'loss': loss.item(), 'loss_aux': loss_aux.item() if loss_aux != 0 else 0}

    def predict(self, x, forward_feature=False):
        if forward_feature:
            return self.model.forward_features(x)
        else:    
            prediction = self.model(x)
            if self.hparams.get('vit_type', 'small') == 'large':
                prediction = prediction.logits
            if type(prediction) is tuple:
                return (prediction[0] + prediction[1]) / 2
            else:
                return prediction


class Fish(Algorithm):
    """
    Implementation of Fish, as seen in Gradient Matching for Domain
    Generalization, Shi et al. 2021.
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Fish, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.network = networks.WholeFish(input_shape, num_classes, hparams)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.optimizer_inner_state = None

    def create_clone(self, device):
        self.network_inner = networks.WholeFish(self.input_shape, self.num_classes, self.hparams,
                                                weights=self.network.state_dict()).to(device)
        self.optimizer_inner = torch.optim.Adam(
            self.network_inner.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        if self.optimizer_inner_state is not None:
            self.optimizer_inner.load_state_dict(self.optimizer_inner_state)

    def fish(self, meta_weights, inner_weights, lr_meta):
        meta_weights = ParamDict(meta_weights)
        inner_weights = ParamDict(inner_weights)
        meta_weights += lr_meta * (inner_weights - meta_weights)
        return meta_weights

    def update(self, minibatches, unlabeled=None):
        self.create_clone(minibatches[0][0].device)

        for x, y in minibatches:
            loss = F.cross_entropy(self.network_inner(x), y)
            self.optimizer_inner.zero_grad()
            loss.backward()
            self.optimizer_inner.step()

        self.optimizer_inner_state = self.optimizer_inner.state_dict()
        meta_weights = self.fish(
            meta_weights=self.network.state_dict(),
            inner_weights=self.network_inner.state_dict(),
            lr_meta=self.hparams["meta_lr"]
        )
        self.network.reset_weights(meta_weights)

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)


class AbstractDANN(Algorithm):
    """Domain-Adversarial Neural Networks (abstract class)"""

    def __init__(self, input_shape, num_classes, num_domains,
                 hparams, conditional, class_balance):

        super(AbstractDANN, self).__init__(input_shape, num_classes, num_domains,
                                           hparams)

        self.register_buffer('update_count', torch.tensor([0]))
        self.conditional = conditional
        self.class_balance = class_balance

        # Algorithms
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.discriminator = networks.MLP(self.featurizer.n_outputs,
                                          num_domains, self.hparams)
        self.class_embeddings = nn.Embedding(num_classes,
                                             self.featurizer.n_outputs)

        # Optimizers
        self.disc_opt = torch.optim.Adam(
            (list(self.discriminator.parameters()) +
             list(self.class_embeddings.parameters())),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams['weight_decay_d'],
            betas=(self.hparams['beta1'], 0.9))

        self.gen_opt = torch.optim.Adam(
            (list(self.featurizer.parameters()) +
             list(self.classifier.parameters())),
            lr=self.hparams["lr_g"],
            weight_decay=self.hparams['weight_decay_g'],
            betas=(self.hparams['beta1'], 0.9))

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        self.update_count += 1
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        if self.conditional:
            disc_input = all_z + self.class_embeddings(all_y)
        else:
            disc_input = all_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat([
            torch.full((x.shape[0],), i, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(minibatches)
        ])

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1. / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction='none')
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        disc_softmax = F.softmax(disc_out, dim=1)
        input_grad = autograd.grad(disc_softmax[:, disc_labels].sum(),
                                   [disc_input], create_graph=True)[0]
        grad_penalty = (input_grad ** 2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams['grad_penalty'] * grad_penalty

        d_steps_per_g = self.hparams['d_steps_per_g_step']
        if (self.update_count.item() % (1 + d_steps_per_g) < d_steps_per_g):

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {'disc_loss': disc_loss.item()}
        else:
            all_preds = self.classifier(all_z)
            classifier_loss = F.cross_entropy(all_preds, all_y)
            gen_loss = (classifier_loss +
                        (self.hparams['lambda'] * -disc_loss))
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {'gen_loss': gen_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))


class DANN(AbstractDANN):
    """Unconditional DANN"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DANN, self).__init__(input_shape, num_classes, num_domains,
                                   hparams, conditional=False, class_balance=False)


#
#
# class CDANN(AbstractDANN):
#     """Conditional DANN"""
#
#     def __init__(self, input_shape, num_classes, num_domains, hparams):
#         super(CDANN, self).__init__(input_shape, num_classes, num_domains,
#                                     hparams, conditional=True, class_balance=True)
#
#
class IRM(ERM):
    """Invariant Risk Minimization"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IRM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        penalty_weight = (self.hparams['irm_lambda'] if self.update_count
                                                        >= self.hparams['irm_penalty_anneal_iters'] else
                          1.0)
        nll = 0.
        penalty = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.hparams['irm_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
                'penalty': penalty.item()}


class Fishr(Algorithm):
    "Invariant Gradients variances for Out-of-distribution Generalization"

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        assert backpack is not None, "Install backpack with: 'pip install backpack-for-pytorch==1.3.0'"
        super(Fishr, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.num_domains = num_domains

        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = extend(
            networks.Classifier(
                self.featurizer.n_outputs,
                num_classes,
                self.hparams['nonlinear_classifier'],
            )
        )
        self.network = nn.Sequential(self.featurizer, self.classifier)

        self.register_buffer("update_count", torch.tensor([0]))
        self.bce_extended = extend(nn.CrossEntropyLoss(reduction='none'))
        self.ema_per_domain = [
            MovingAverage(ema=self.hparams["ema"], oneminusema_correction=True)
            for _ in range(self.num_domains)
        ]
        self._init_optimizer()

    def _init_optimizer(self):
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, minibatches, unlabeled=False):
        assert len(minibatches) == self.num_domains
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        len_minibatches = [x.shape[0] for x, y in minibatches]

        all_z = self.featurizer(all_x)
        all_logits = self.classifier(all_z)

        penalty = self.compute_fishr_penalty(all_logits, all_y, len_minibatches)
        all_nll = F.cross_entropy(all_logits, all_y)

        penalty_weight = 0
        if self.update_count >= self.hparams["penalty_anneal_iters"]:
            penalty_weight = self.hparams["lambda"]
            if self.update_count == self.hparams["penalty_anneal_iters"] != 0:
                # Reset Adam as in IRM or V-REx, because it may not like the sharp jump in
                # gradient magnitudes that happens at this step.
                self._init_optimizer()
        self.update_count += 1

        objective = all_nll + penalty_weight * penalty
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item(), 'nll': all_nll.item(), 'penalty': penalty.item()}

    def compute_fishr_penalty(self, all_logits, all_y, len_minibatches):
        dict_grads = self._get_grads(all_logits, all_y)
        grads_var_per_domain = self._get_grads_var_per_domain(dict_grads, len_minibatches)
        return self._compute_distance_grads_var(grads_var_per_domain)

    def _get_grads(self, logits, y):
        self.optimizer.zero_grad()
        loss = self.bce_extended(logits, y).sum()
        with backpack(BatchGrad()):
            loss.backward(
                inputs=list(self.classifier.parameters()), retain_graph=True, create_graph=True
            )

        # compute individual grads for all samples across all domains simultaneously
        dict_grads = OrderedDict(
            [
                (name, weights.grad_batch.clone().view(weights.grad_batch.size(0), -1))
                for name, weights in self.classifier.named_parameters()
            ]
        )
        return dict_grads

    def _get_grads_var_per_domain(self, dict_grads, len_minibatches):
        # grads var per domain
        grads_var_per_domain = [{} for _ in range(self.num_domains)]
        for name, _grads in dict_grads.items():
            all_idx = 0
            for domain_id, bsize in enumerate(len_minibatches):
                env_grads = _grads[all_idx:all_idx + bsize]
                all_idx += bsize
                env_mean = env_grads.mean(dim=0, keepdim=True)
                env_grads_centered = env_grads - env_mean
                grads_var_per_domain[domain_id][name] = (env_grads_centered).pow(2).mean(dim=0)

        # moving average
        for domain_id in range(self.num_domains):
            grads_var_per_domain[domain_id] = self.ema_per_domain[domain_id].update(
                grads_var_per_domain[domain_id]
            )

        return grads_var_per_domain

    def _compute_distance_grads_var(self, grads_var_per_domain):

        # compute gradient variances averaged across domains
        grads_var = OrderedDict(
            [
                (
                    name,
                    torch.stack(
                        [
                            grads_var_per_domain[domain_id][name]
                            for domain_id in range(self.num_domains)
                        ],
                        dim=0
                    ).mean(dim=0)
                )
                for name in grads_var_per_domain[0].keys()
            ]
        )

        penalty = 0
        for domain_id in range(self.num_domains):
            penalty += l2_between_dicts(grads_var_per_domain[domain_id], grads_var)
        return penalty / self.num_domains

    def predict(self, x):
        return self.network(x)

