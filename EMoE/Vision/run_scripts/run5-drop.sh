python3 -m domainbed.scripts.train --data_dir=./domainbed/data --algorithm GMOE --dataset PACS --hparams '{"vanilla_ViT":false, "vit_type":"small", "router": "gated_multi_gate", "adaptive_experts":true, "max_expert_num": 12}' --output_dir outputs-ours-adapt-drop-12 --test_envs 0 --device 0;

python3 -m domainbed.scripts.train --data_dir=./domainbed/data --algorithm GMOE --dataset PACS --hparams '{"vanilla_ViT":false, "vit_type":"small", "router": "gated_multi_gate", "adaptive_experts":true, "max_expert_num": 12}' --output_dir outputs-ours-adapt-drop-12 --test_envs 1 --device 0;

python3 -m domainbed.scripts.train --data_dir=./domainbed/data --algorithm GMOE --dataset PACS --hparams '{"vanilla_ViT":false, "vit_type":"small", "router": "gated_multi_gate", "adaptive_experts":true, "max_expert_num": 12}' --output_dir outputs-ours-adapt-drop-12 --test_envs 2 --device 0;

python3 -m domainbed.scripts.train --data_dir=./domainbed/data --algorithm GMOE --dataset PACS --hparams '{"vanilla_ViT":false, "vit_type":"small", "router": "gated_multi_gate", "adaptive_experts":true, "max_expert_num": 12}' --output_dir outputs-ours-adapt-drop-12 --test_envs 3 --device 0;

python3 -m domainbed.scripts.train --data_dir=./domainbed/data --algorithm GMOE --dataset PACS --hparams '{"vanilla_ViT":false, "vit_type":"small", "router": "gated_multi_gate", "adaptive_experts":true, "max_expert_num": 12}' --output_dir outputs-ours-adapt-drop-12 --test_envs 0 --device 0 --seed 1;

python3 -m domainbed.scripts.train --data_dir=./domainbed/data --algorithm GMOE --dataset PACS --hparams '{"vanilla_ViT":false, "vit_type":"small", "router": "gated_multi_gate", "adaptive_experts":true, "max_expert_num": 12}' --output_dir outputs-ours-adapt-drop-12 --test_envs 1 --device 0 --seed 1;

python3 -m domainbed.scripts.train --data_dir=./domainbed/data --algorithm GMOE --dataset PACS --hparams '{"vanilla_ViT":false, "vit_type":"small", "router": "gated_multi_gate", "adaptive_experts":true, "max_expert_num": 12}' --output_dir outputs-ours-adapt-drop-12 --test_envs 2 --device 0 --seed 1;

python3 -m domainbed.scripts.train --data_dir=./domainbed/data --algorithm GMOE --dataset PACS --hparams '{"vanilla_ViT":false, "vit_type":"small", "router": "gated_multi_gate", "adaptive_experts":true, "max_expert_num": 12}' --output_dir outputs-ours-adapt-drop-12 --test_envs 3 --device 0 --seed 1;

python3 -m domainbed.scripts.train --data_dir=./domainbed/data --algorithm GMOE --dataset PACS --hparams '{"vanilla_ViT":false, "vit_type":"small", "router": "gated_multi_gate", "adaptive_experts":true, "max_expert_num": 12}' --output_dir outputs-ours-adapt-drop-12 --test_envs 0 --device 0 --seed 2;

python3 -m domainbed.scripts.train --data_dir=./domainbed/data --algorithm GMOE --dataset PACS --hparams '{"vanilla_ViT":false, "vit_type":"small", "router": "gated_multi_gate", "adaptive_experts":true, "max_expert_num": 12}' --output_dir outputs-ours-adapt-drop-12 --test_envs 1 --device 0 --seed 2;

python3 -m domainbed.scripts.train --data_dir=./domainbed/data --algorithm GMOE --dataset PACS --hparams '{"vanilla_ViT":false, "vit_type":"small", "router": "gated_multi_gate", "adaptive_experts":true, "max_expert_num": 12}' --output_dir outputs-ours-adapt-drop-12 --test_envs 2 --device 0 --seed 2;

python3 -m domainbed.scripts.train --data_dir=./domainbed/data --algorithm GMOE --dataset PACS --hparams '{"vanilla_ViT":false, "vit_type":"small", "router": "gated_multi_gate", "adaptive_experts":true, "max_expert_num": 12}' --output_dir outputs-ours-adapt-drop-12 --test_envs 3 --device 0 --seed 2;