import torch
import torch.nn.functional as F

from tutel import system
from tutel import moe
from tutel import net

from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast
from torch import Tensor

if torch.cuda.is_available():
  dist = system.init_data_model_parallel(backend='nccl')
else:
  dist = system.init_data_model_parallel(backend='gloo')

num_samples = 1280
model_dim, hidden_size = 128, 128
num_local_experts = [4,4]
num_global_experts = 8

def get_param_group_index(optim, param):
    for i, optim_param_group in enumerate(optim.param_groups):
        optim_param_group_list = optim_param_group["params"]
        assert len(optim_param_group_list) == 1
        print(optim_param_group_list)
        optim_param = optim_param_group_list[0]
        if param.shape == optim_param.shape and (param==optim_param).all():
            return i
    # raise Exception("Could not find param in optim.param_groups")

def remove_param_from_optimizer(optim, pg_index):
    # Remove corresponding state
    for param in optim.param_groups[pg_index]['params']:
        if param in optim.state:
            del optim.state[param]
    del optim.param_groups[pg_index]

class GMoEGateBackward(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, scores: Tensor):
        signed_scores = torch.sign(scores) # 有点不理解为啥不能normalize
        return signed_scores

    @staticmethod
    def backward(ctx:Any, grad_output: Tensor):
        return grad_output

    

class GMoEGate(torch.nn.Module):
    def __init__(self, expert_num, model_dim, max_expert_num=64):
        super().__init__()
        torch.manual_seed(1)
        self.expert_num = expert_num
        self.register_parameter('sim_matrix', torch.nn.Parameter(torch.randn(size=(model_dim, max_expert_num)), requires_grad=True))
        self.register_parameter('gates', torch.nn.Parameter(torch.zeros(size=(max_expert_num,)), requires_grad=True))

        self.activate_mask = [i for i in range(expert_num)]

    def forward(self, x):
        logits = torch.sigmoid(torch.matmul(F.normalize(x, dim=1),
                              F.normalize(self.sim_matrix[:, self.activate_mask], dim=0)))
        gates = torch.sigmoid(self.gates[self.activate_mask])
        
        # print(gates)
        return F.relu(logits - gates)
    
class GMoEExpert(torch.nn.Module):
    def __init__(self, expert_num, model_dim, hidden_size):
        super().__init__()
        torch.manual_seed(dist.global_rank + 1)
        self.expert_num = expert_num
        self.register_parameter(name='batched_fc1_w', param=torch.nn.Parameter(torch.randn([expert_num, model_dim, hidden_size]) * 1e-3))
        self.register_parameter(name='batched_fc2_w', param=torch.nn.Parameter(torch.randn([expert_num, hidden_size, model_dim]) * 1e-3))
        self.register_parameter(name='batched_fc1_bias', param=torch.nn.Parameter(torch.zeros([expert_num, 1, hidden_size])))
        self.register_parameter(name='batched_fc2_bias', param=torch.nn.Parameter(torch.zeros([expert_num, 1, model_dim])))
        for x in self.parameters(): setattr(x, 'skip_allreduce', True)

    def forward(self, x):
        y = torch.add(torch.matmul(x, self.batched_fc1_w), self.batched_fc1_bias)
        y = F.relu(y)
        y = torch.add(torch.matmul(y, self.batched_fc2_w), self.batched_fc2_bias)
        return y
    
class GMoE(torch.nn.Module):
    def __init__(self, expert_num, model_dim, hidden_size, max_expert_num=64):
        super().__init__()
        self.gates = GMoEGate(num_global_experts, model_dim, max_expert_num=max_expert_num)
        self.experts = torch.nn.ModuleList([GMoEExpert(expert_num[dist.global_rank], model_dim, hidden_size)])
        
        # current experts information
        self.expert_splits = [expert_num[dist.global_rank]]
        self.num_global_experts = torch.tensor(num_global_experts).to(dist.local_device)
        self.num_local_experts = torch.tensor(sum(self.expert_splits)).to(dist.local_device)
        self.removed_experts = torch.zeros(max_expert_num).to(dist.local_device)

        # number of experts of each process
        self.local_experts_splits = [torch.tensor(0).to(dist.local_device) for _ in range(torch.distributed.get_world_size())]

        # save the params for Experts
        self.model_dim = model_dim
        self.hidden_size = hidden_size
        self.max_expert_num = max_expert_num


    def add_experts(self, num_new_experts, optimizer):

        print('process-{}-add-{}'.format(dist.global_rank, num_new_experts))

        if num_new_experts <= 0:
            return optimizer

        # add the experts
        new_experts = GMoEExpert(num_new_experts, model_dim, hidden_size).to(dist.local_device)
        self.experts.append(new_experts)
        self.expert_splits.append(num_new_experts)
        self.num_local_experts += num_new_experts

        # update the local optimizer information
        optimizer.add_param_group({'params': new_experts.parameters()})

        return optimizer
    
    def remove_experts(self, local_experts_rank, optimizer):
        print('process-{}-remove-{}-current-{}'.format(dist.global_rank, self.experts[0].expert_num, self.num_global_experts))
        if len(self.experts) == 1:
            return optimizer
        # remove the expert from optimizer and experts list
        for local_expert_rank in local_experts_rank:
            expert = self.experts.pop(local_expert_rank)
            remove_param_from_optimizer(optimizer, local_expert_rank)

            # prepare communication to other processes
            if dist.global_rank > 0 or local_expert_rank > 0:
                pre_experts_num = (sum(self.local_experts_splits[:dist.global_rank]) + sum(self.expert_splits[:local_expert_rank])).item()
            else:
                pre_experts_num = 0
            self.removed_experts[[i + pre_experts_num for i in range(expert.expert_num)]] = 1

            # remove the expert from the experts list
            self.expert_splits.pop(local_expert_rank)
            self.num_local_experts -= expert.expert_num

        return optimizer




    
    def update_gates(self, optimizer):

        # gather removed experts from all processes
        torch.distributed.all_reduce(self.removed_experts, op=torch.distributed.ReduceOp.SUM)
        num_removed_experts = sum(self.removed_experts)

        # update gates for removed experts
        removed_experts_indicies = [self.gates.activate_mask[i] for i in range(self.max_expert_num) if self.removed_experts[i] > 0]
        for index in removed_experts_indicies:
            self.gates.activate_mask.remove(index)

        self.removed_experts = torch.zeros(self.max_expert_num).to(dist.local_device)


        # gather current number of experts from all processes
        torch.distributed.all_gather(self.local_experts_splits, self.num_local_experts)
        new_num_global_experts = sum(self.local_experts_splits)

        # update gates if the collected number of global experts excels current number of global experts
        residule_num_global_experts = new_num_global_experts - self.num_global_experts + num_removed_experts
        # print(new_num_global_experts, self.num_global_experts, self.num_local_experts)
        # print(residule_num_global_experts, self.gates.activate_mask)
        if residule_num_global_experts <= 0:
            self.num_global_experts = new_num_global_experts
            return optimizer
        
        for i in range(self.max_expert_num):
            if i not in self.gates.activate_mask:
                self.gates.activate_mask.append(i)
                self.gates.gates.data[i] = 0.0
                self.gates.sim_matrix.data[:, i] = torch.randn(self.model_dim)
                residule_num_global_experts -= 1
            if residule_num_global_experts == 0:
                break

        self.num_global_experts = new_num_global_experts

        

        return optimizer





        



    def forward(self, x):
        logits = self.gates(x)
        scores = logits
        k = torch.sum(scores > 0, dim=1).to(torch.int)
        scores = GMoEGateBackward.apply(scores)
        # scores = torch.softmax(scores, dim=1)
        
        crit, l_aux = moe.top_k_routing(scores, top_k=k)
        y = moe.fast_encode(x, crit)
        
        y, capacity_list = net.all_to_all(y, 1, 0, input_splits=self.local_experts_splits)

        y = y.split(self.expert_splits, dim=0)
        y = [self.experts[i](y[i]) for i in range(len(self.experts))]
        y = torch.cat(y, dim=0)
        
        # y = self.expert(y)
        y = net.all_to_all(y, 0, 1, input_splits=capacity_list)
        output = moe.fast_decode(y, crit)
        return output, l_aux

model = GMoE(num_local_experts, model_dim, hidden_size).to(dist.local_device)

torch.manual_seed(dist.global_rank + 1)
data = torch.randn([num_samples, model_dim], device=dist.local_device)
label = torch.LongTensor(num_samples).random_(1).to(dist.local_device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

for i in range(300):
    t_start = system.record_time()

    optimizer = model.update_gates(optimizer)

    optimizer.zero_grad()
    result, l_aux = model(data)
    result = F.log_softmax(result, dim=1)
    loss = F.nll_loss(result, label) + 0.0001 * l_aux
    loss.backward()

    for p in model.parameters():
        if not hasattr(p, 'skip_allreduce'):
            p.grad = net.simple_all_reduce(p.grad)
    optimizer.step()

    if i % ((dist.global_rank + 1) * 50) == 0:
        optimizer = model.add_experts(dist.global_rank + 1, optimizer)

    if (i+1) % ((dist.global_rank + 1) * 75) == 0:
        optimizer = model.remove_experts([0], optimizer)
        # optimizer = model.add_experts(dist.global_rank + 1, optimizer)

    t_stop = system.record_time()

    dist.dist_print('STEP-%d: loss = %.5f, step_time = %.3f s' % (i, loss, t_stop - t_start))

