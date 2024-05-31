python3 -m domainbed.scripts.train --data_dir=./domainbed/data --algorithm GMOE --dataset PACS --hparams '{"vanilla_ViT":false, "vit_type":"small", "topk": 4, "num_experts": 6, "router": "top"}' --output_dir outputs-top-4 --test_envs 0 --device 0;

python3 -m domainbed.scripts.train --data_dir=./domainbed/data --algorithm GMOE --dataset PACS --hparams '{"vanilla_ViT":false, "vit_type":"small", "topk": 4, "num_experts": 6, "router": "top"}' --output_dir outputs-top-4 --test_envs 1 --device 0;

python3 -m domainbed.scripts.train --data_dir=./domainbed/data --algorithm GMOE --dataset PACS --hparams '{"vanilla_ViT":false, "vit_type":"small", "topk": 4, "num_experts": 6, "router": "top"}' --output_dir outputs-top-4 --test_envs 2 --device 0;

python3 -m domainbed.scripts.train --data_dir=./domainbed/data --algorithm GMOE --dataset PACS --hparams '{"vanilla_ViT":false, "vit_type":"small", "topk": 4, "num_experts": 6, "router": "top"}' --output_dir outputs-top-4 --test_envs 3 --device 0;

python3 -m domainbed.scripts.train --data_dir=./domainbed/data --algorithm GMOE --dataset PACS --hparams '{"vanilla_ViT":false, "vit_type":"small", "topk": 4, "num_experts": 6, "router": "top"}' --output_dir outputs-top-4 --test_envs 0 --device 0 --trial_seed 1;

python3 -m domainbed.scripts.train --data_dir=./domainbed/data --algorithm GMOE --dataset PACS --hparams '{"vanilla_ViT":false, "vit_type":"small", "topk": 4, "num_experts": 6, "router": "top"}' --output_dir outputs-top-4 --test_envs 1 --device 0 --trial_seed 1;

python3 -m domainbed.scripts.train --data_dir=./domainbed/data --algorithm GMOE --dataset PACS --hparams '{"vanilla_ViT":false, "vit_type":"small", "topk": 4, "num_experts": 6, "router": "top"}' --output_dir outputs-top-4 --test_envs 2 --device 0 --trial_seed 1;

python3 -m domainbed.scripts.train --data_dir=./domainbed/data --algorithm GMOE --dataset PACS --hparams '{"vanilla_ViT":false, "vit_type":"small", "topk": 4, "num_experts": 6, "router": "top"}' --output_dir outputs-top-4 --test_envs 3 --device 0 --trial_seed 1;

python3 -m domainbed.scripts.train --data_dir=./domainbed/data --algorithm GMOE --dataset PACS --hparams '{"vanilla_ViT":false, "vit_type":"small", "topk": 4, "num_experts": 6, "router": "top"}' --output_dir outputs-top-4 --test_envs 0 --device 0 --trial_seed 2;

python3 -m domainbed.scripts.train --data_dir=./domainbed/data --algorithm GMOE --dataset PACS --hparams '{"vanilla_ViT":false, "vit_type":"small", "topk": 4, "num_experts": 6, "router": "top"}' --output_dir outputs-top-4 --test_envs 1 --device 0 --trial_seed 2;

python3 -m domainbed.scripts.train --data_dir=./domainbed/data --algorithm GMOE --dataset PACS --hparams '{"vanilla_ViT":false, "vit_type":"small", "topk": 4, "num_experts": 6, "router": "top"}' --output_dir outputs-top-4 --test_envs 2 --device 0 --trial_seed 2;

python3 -m domainbed.scripts.train --data_dir=./domainbed/data --algorithm GMOE --dataset PACS --hparams '{"vanilla_ViT":false, "vit_type":"small", "topk": 4, "num_experts": 6, "router": "top"}' --output_dir outputs-top-4 --test_envs 3 --device 0 --trial_seed 2;