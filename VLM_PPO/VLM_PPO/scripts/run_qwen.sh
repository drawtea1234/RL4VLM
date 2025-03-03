TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="0,1" accelerate launch --config_file config_zero2.yaml --main_process_port 29488 ../main.py \
    --env-name gym_cards/NumberLine-v0 \
    --init-lr 1e-5 \
    --end-lr 1e-9 \
    --lr_max_steps 25 \
    --eval-num-per-episode 200 \
    --num-env-steps 15000 \
    --num-steps 512 \
    --grad-accum-steps 128 \
    --max-new-tokens 256 \
    --thought-prob-coef 0.5 \
    --use-gae \
    --seed 1 \
    --temperature 0.2 \
    --ppo-epoch 4 \
    --mini-batch-size 1 \
    --model-path /home/franka/workspace/policy/Qwen2-VL/finetune/Qwen2-VL-finetune-LatexOCR/Qwen/Qwen2-VL-2B-Instruct \
    --use-lora \
    --train-vision all \
    # --wandb-project you_wandb_proj \
    # --wandb-run you_wandb_run \
    # --use-wandb \
    # --q4
