# DynMoE for Language ID and OOD tasks

## Preparation

### Environment

```sh
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

cd ./EMoE/tutel
pip3 install ./

pip3 install -r requirements.txt
```

### OOD data

Most of the data can be loaded directly trough `datasets`. For some addtional self-collected OOD tasks, please download them from [GLUE-X](https://github.com/YangLinyi/GLUE-X) and put them in `./dataset/datasets_self_collected`.

## DynMoE Usage

- Change the config of moe layer in `moe_utils.py`.
- If you are using DynMoE's new load balance loss, please set `is_gshard_loss` to `False`.

```python
layer_module.mlp = tutel_moe.moe_layer(
    gate_type={
        'type': args.gate_type, 
        'max_expert_num': args.max_expert_num,
        'fp32_gate': True, 
        'adaptive_experts': True,
        'gate_noise': 0.0,  
        'capacity_factor': 0.0
    },
    experts={
        'type': 'ffn', 
        'count_per_node': args.num_experts,
        'hidden_size_per_expert': expert_size,
        'activation_fn': dropout_gelu(args.moe_drop)
    },
    model_dim=weight_0.shape[1],
    batch_prioritized_routing=True,
    is_gshard_loss=args.is_gshard_loss,
    one_score_gate=args.one_score,
    normalize_one_score_gate=args.normalize_one_score_gate,
    update_momentum=args.one_score_gate_update_momentum,
)
```

- Enable adaptive during the training in `search_glue_no_trainer.py`. If the current training epoch is selected to update the number of experts, then we need to turn on the rounting records by calling `begin_record_routing` method of each moe layer, and call `adaptive_update_experts` at the end of the training epoch. You can also turn off the rounting records by calling `end_record_routing()`. One example in `search_glue_no_trainer.py`:

```python
# at the beginning of each training epoch
if args.to_MoE and args.adaptive_experts and step == len(train_dataloader) // 2:
    if 'bert' in args.model_name_or_path:
        for i, layer in enumerate(model.bert.encoder.layer):
            if i in args.moe_layers:
                layer.mlp.begin_record_routing()
    elif 'gpt' in args.model_name_or_path:
        for i, layer in enumerate(model.transformer.h):
            if i in args.moe_layers:
                layer.mlp.begin_record_routing()

# other training code...

# at the end of each training epoch
if args.to_MoE and args.adaptive_experts and step == (len(train_dataloader) * 3) // 4:
    if 'bert' in args.model_name_or_path:
        for i, layer in enumerate(model.bert.encoder.layer):
            if i in args.moe_layers:
                layer.mlp.adaptive_update_experts()
    elif 'gpt' in args.model_name_or_path:
        for i, layer in enumerate(model.transformer.h):
            if i in args.moe_layers:
                layer.mlp.adaptive_update_experts()
```

## Start Experiments

### Full Fine-tuning

Full Fine-tuning BERT-Large on GLUE benchmark, default `search_glue_no_trainer.py` would search learning rates in `[2e-5, 3e-5, 5e-5]` and repeat seeds `[0, 1, 2]`. We provide off-the-shelf scripts to run all experiments under `scripts/`.

```sh
# COLA, as an example

# GMOE with 8 experts and topk = 2
python search_glue_no_trainer.py --model_name_or_path ~/data/bert-large-cased --task_name cola --to_MoE --gate_type cosine_top --num_experts 8 --top_k 2 --moe_layers $moe_layers --expert_repeat 8 --random_cluster --save_model;

# DynMoE (Ours)
python search_glue_no_trainer.py --model_name_or_path bert-large-cased --task_name cola --to_MoE --num_experts 8 --moe_layers 10 --expert_repeat 16 --random_cluster --save_model --max_expert_num 16 --adaptive_experts --gate_type gated_multi_gate --save_model;
```

### OOD testing

#### OOD Settings

| Train Set | OOD Test Set |
| --- | --- |
| COLA | COLA_OOD |
| QNLI | QNLI_OOD |
| RTE | SciTail, HANS |
| MNLI | SNLI, SICK |
| MRPC | QQP, Twitter |

After finish ID training, do OOD testing with:

```sh
python test_glue_no_trainer.py --task_name ${OOD_task_name} --use_fp16  --model_name_or_path gpt2-xl --source_dir ${experiment_dir_with_all_checkpoints}

python test_glue_no_trainer.py --task_name cola --use_fp16 --model_name_or_path ~/bert-large-cased --source_dir ~/bert-large-cased_save/cola
```

the OOD results will be recorded in the corresponding subfiles.
