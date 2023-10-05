#!/bin/bash

# Fine-tuning the 64k, 13b model on the kye/all-lucidrain-code-python-tokenized-65536-1 dataset.
# The model name is set as lucidrains.
accelerate launch finetune.py \
    --batch-size 32 \
    --gradient-accumulate-every 8 \
    --output-dir output/atom-65k-pytorch \
    --wandb yarn \
    --seed 42 \
    --max-train-steps 400 \
    --warmup-steps 20 \
    --learning-rate 2e-5 \
    --grad-norm \
    --lora \
    --model kye/atom-65k-pytorc \
    --yarn-factor 16.0 \
    --dataset kye/all-lucidrain-code-python-tokenized-65536-1Z