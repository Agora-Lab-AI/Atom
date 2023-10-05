#!/bin/bash

# Fine-tuning the 64k, 13b model on the kye/all-lucidrain-code-python-tokenized-65536-1 dataset.
# The model name is set as lucidrains.
accelerate launch finetune.py \
    --wandb yarn \
    --output-dir output/lucidrains-13b-64k \
    --model NousResearch/Llama-2-13b-hf \
    --dataset kye/all-lucidrain-code-python-tokenized-65536-1 \
    --max-train-steps 200 \
    --seed 31337 \
    --batch-size 32 \
    --learning-rate 2e-5 \
    --warmup-steps 100 \
    --gradient-accumulate-every 4 \
    