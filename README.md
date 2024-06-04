<h2 align="center"> <a href="https://arxiv.org/abs/2405.14297">Dynamic Mixture of Experts: An Auto-Tuning Approach for Efficient Transformer Models</a></h2>
<h5 align="center"> If our project helps you, please give us a star ⭐ and cite our <a href="#citation">paper</a>!</h2>
<h5 align="center">

[![hf_space](https://img.shields.io/badge/🤗-Paper%20In%20HF-red.svg)](https://huggingface.co/papers/2405.14297)
[![arxiv](https://img.shields.io/badge/Arxiv-2405.14297-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2405.14297)
[![visitor](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FLINs-lab%2FDynMoE&count_bg=%2379C83D&title_bg=%23454343&icon=&icon_color=%23E7E7E7&title=visitor&edge_flat=false)](https://hits.seeyoufarm.com)

## News
- **[2024.05.25]** 🔥 Our [checkpoints](https://huggingface.co/collections/LINs-lab/dynmoe-family-665ed5a331a7e84463cab01a) are available now!
- **[2024.05.23]** 🔥 Our [paper](https://arxiv.org/abs/2405.14297) is released!

## Why Do We Need DynMoE?

Sparse MoE (SMoE) has an unavoidable drawback: *the performance of SMoE heavily relies on the choice of hyper-parameters, such as the number of activated experts per token (top-k) and the number of experts.*

Also, *identifying the optimal hyper-parameter without a sufficient number of ablation studies is challenging.* As the size of the models continues to grow, this limitation could result in a significant waste of computational resources, and in turn, could hinder the efficiency of training MoE-based models in practice.

Now, our **DynMoE** addresses these challenges through the two components introduced in [Dynamic Mixture of Experts (DynMoE)](#dynamic-mixture-of-experts-dynmoe).

## Dynamic Mixture of Experts (DynMoE)

## Top-Any Gating

![hh](./assets/moe-overview.gif)

We first introduce a novel gating method that enables each token to **automatically determine the number of experts to activate**.

## Adaptive Training Process

![adaptive-training](https://cdn.jsdelivr.net/gh/QAQdev/Pics@master/uPic/adaptive.png)

Our method also includes an adaptive process **automatically adjusts the number of experts** during training.

## Can We Trust DynMoE? Yes!

- On language tasks, **DynMoE surpasses the average performance among various MoE settings.**
- **Effectiveness of DynMoE remains consistent** in both Vision and Vision-Language tasks.
- Although sparsity is not enforced in DynMoE, it **maintains efficiency by activating even less parameters!**

## Model Zoo

| Model | Activated Params / Total Params| Transformers(HF) |
| ----- | --------------- | ---------------- |
| DynMoE-StableLM-1.6B | 1.8B / 2.9B | [LINs-lab/DynMoE-StableLM-1.6B](https://huggingface.co/LINs-lab/DynMoE-StableLM-1.6B)
| DynMoE-Qwen-1.8B | 2.2B / 3.1B | [LINs-lab/DynMoE-Qwen-1.8B](https://huggingface.co/LINs-lab/DynMoE-Qwen-1.8B)
| DynMoE-Phi-2-2.7B | 3.4B / 5.3B| [LINs-lab/DynMoE-Phi-2-2.7B](https://huggingface.co/LINs-lab/DynMoE-Phi-2-2.7B)

##  Directory Specification

### Experiment Code

- `EMoE/` contains experiments on language and vision tasks, which uses tutel-based DynMoE.
- `MoE-LLaVA/` contains experiments on language-vision tasks, which uses deepspeed-0.9.5-based DynMoE.

### DynMoE Implementations

- `Deepspeed/` provides DynMoE-Deepspeed implementation.
- `EMoE/tutel/` provides DynMoE-Tutel implementation.

## Environment Setup

Please refer to instructions under `EMoE/` and `MoE-LLaVA`.

## Usage

### Tutel Examples

Please refer to `EMoE/Language/README.md` and `EMoE/Language/Vision.md`.

### DeepSpeed Examples

Network Configuration

```python
deepspeed.moe.layer.MoE(
  hidden_size=84,
  expert=fc3,
  num_experts=n_e // 2,
  ep_size=args.ep_world_size,
  use_residual=args.mlp_type == "residual",
  k=-1, # -1 means using DynMoE
  min_capacity=args.min_capacity,
  noisy_gate_policy=args.noisy_gate_policy,
  max_expert_num=n_e
)
```

Training model forward, you can control the adaptive process by using `if_begin_record_routing`, `if_end_record_routing`.

```python
outputs = model_engine(inputs, if_begin_record_routing=True, if_end_record_routing=True)
```

## Acknowledgement

We are grateful for the following awesome projects:

- [tutel](https://github.com/microsoft/tutel)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)
- [GMoE](https://github.com/Luodian/Generalizable-Mixture-of-Experts)
- [EMoE](https://github.com/qiuzh20/EMoE)
- [MoE-LLaVA](https://github.com/PKU-YuanGroup/MoE-LLaVA)
- [GLUE-X](https://github.com/YangLinyi/GLUE-X)

## Citation

If you find this project helpful, please consider citing our work:

```bibtex
@misc{guo2024dynamic,
      title={Dynamic Mixture of Experts: An Auto-Tuning Approach for Efficient Transformer Models}, 
      author={Yongxin Guo and Zhenglin Cheng and Xiaoying Tang and Tao Lin},
      year={2024},
      eprint={2405.14297},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
