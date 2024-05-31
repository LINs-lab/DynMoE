moe_layers="10"

# GMoE with different number of experts and top-k experts settings
python search_glue_no_trainer.py --model_name_or_path ~/data/bert-large-cased --task_name qnli --to_MoE --gate_type cosine_top --num_experts 8 --top_k 1 --moe_layers $moe_layers --expert_repeat 8 --random_cluster --save_model;

python search_glue_no_trainer.py --model_name_or_path ~/data/bert-large-cased --task_name qnli --to_MoE --gate_type cosine_top --num_experts 8 --top_k 2 --moe_layers $moe_layers --expert_repeat 8 --random_cluster --save_model;

python search_glue_no_trainer.py --model_name_or_path ~/data/bert-large-cased --task_name qnli --to_MoE --gate_type cosine_top --num_experts 8 --top_k 4 --moe_layers $moe_layers --expert_repeat 8 --random_cluster --save_model;

python search_glue_no_trainer.py --model_name_or_path ~/data/bert-large-cased --task_name qnli --to_MoE --gate_type cosine_top --num_experts 8 --top_k 8 --moe_layers $moe_layers --expert_repeat 8 --random_cluster --save_model;

python search_glue_no_trainer.py --model_name_or_path ~/data/bert-large-cased --task_name qnli --to_MoE --gate_type cosine_top --num_experts 16 --top_k 1 --moe_layers $moe_layers --expert_repeat 16 --random_cluster --save_model;

python search_glue_no_trainer.py --model_name_or_path ~/data/bert-large-cased --task_name qnli --to_MoE --gate_type cosine_top --num_experts 16 --top_k 2 --moe_layers $moe_layers --expert_repeat 16 --random_cluster --save_model;

python search_glue_no_trainer.py --model_name_or_path ~/data/bert-large-cased --task_name qnli --to_MoE --gate_type cosine_top --num_experts 16 --top_k 4 --moe_layers $moe_layers --expert_repeat 16 --random_cluster --save_model;

python search_glue_no_trainer.py --model_name_or_path ~/data/bert-large-cased --task_name qnli --to_MoE --gate_type cosine_top --num_experts 16 --top_k 8 --moe_layers $moe_layers --expert_repeat 16 --random_cluster --save_model;

# DynMoE (Ours)
python search_glue_no_trainer.py --model_name_or_path ~/data/bert-large-cased --task_name qnli --to_MoE --num_experts 8 --moe_layers $moe_layers --expert_repeat 16 --random_cluster --save_model --max_expert_num 10 --adaptive_experts --gate_type gated_multi_gate --save_model;