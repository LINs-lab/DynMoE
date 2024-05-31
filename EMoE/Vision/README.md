# Domainbed Experiments

## Preparation

```sh
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

cd ./EMoE/tutel
pip3 install ./

pip3 install -r requirements.txt
```

## Datasets

```sh
python3 -m domainbed.scripts.download \
       --data_dir=./domainbed/data
```

## DynMoE Usage

- Change the config of moe layer in `moe_utils.py`.
- If you are using DynMoE's new load balance loss, please set `is_gshard_loss` to `False`.

```python
layer_module.mlp = tutel_moe.moe_layer(
      gate_type={'type': 'gated_multi_gate', 
                  'max_expert_num': args.max_expert_num, 
                  'fp32_gate': True, 
                  'adaptive_experts': args.adaptive_experts},
      experts={'type': 'ffn', 
               'count_per_node': args.num_experts,
               'hidden_size_per_expert': expert_size,
                'activation_fn': nn.GELU(),},
      model_dim=weight_0.shape[1],
      batch_prioritized_routing=True,
      is_gshard_loss=args.is_gshard_loss,
      one_score_gate=args.one_score_gate,
      normalize_one_score_gate=args.normalize_one_score_gate,
      update_momentum=args.one_score_gate_update_momentum
)
```

- Enable adaptive during the training. We need to turn on the rounting records by calling `begin_record_routing` method of each moe layer, and call `adaptive_update_experts` at the end of the training epoch. You can also turn off the rounting records by calling `end_record_routing()`. One example in `domainbed/algorithms.py`:

```python
class GMOE(Algorithm):
  # in GMOE.__init__()
  def __init__():
    if self.adaptive_experts:
        if self.hparams.get('vit_type', 'small') == 'large':
            for i, layer in enumerate(self.model.vit.encoder.layer):
                if i in self.args.moe_layers:
                    layer.mlp.begin_record_routing()
        else:
            for block in self.model.blocks:
                if block.cur_layer == 'S':
                    block.mlp.begin_record_routing()

  # other code...

  # in GMOE.update()
  def update():
    if self.adaptive_experts:
        if self.hparams.get('vit_type', 'small') == 'large':
            for i, layer in enumerate(self.model.vit.encoder.layer):
                if i in self.args.moe_layers:
                    layer.mlp.adaptive_update_experts()
                    layer.mlp.begin_record_routing()
        else:
            for block in self.model.blocks:
                if block.cur_layer == 'S':
                    block.mlp.adaptive_update_experts()
                    block.mlp.begin_record_routing()
```

## Start Experiments

### Training

We provide off-the-shelf scripts to run all experiments on DomainBed under `run_scripts/`, `run_scripts_domainnet/`, `run_scripts_office_home/` and `run_scripts_vlcs/`.

### Evaluation

After training, go to the corresponding `output_dir` and run

```sh
python3 -m domainbed.scripts.collect_results --input_dir=${output_dir}
```

### Hyper-params

We put hparams for each dataset into
```sh
./domainbed/hparams_registry.py
```

Basically, you just need to choose `--algorithm` and `--dataset`. The optimal hparams will be loaded accordingly. 

