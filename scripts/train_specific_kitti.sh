#!/bin/bash
# Training Reloc3r on KITTI dataset using a single GPU (3090)

python train.py \
     --train_dataset "400 @ KITTI_specific_seq(split='train', resolution=[(512, 384), (512, 336), (512, 288)], transform=ColorJitter)" \
     --test_dataset "40 @ KITTI_specific_seq(resolution=(512, 384), seed=777)" \
     --model "Reloc3rRelpose(img_size=512)" \
     --lr 1e-5 --warmup_epochs 0 --epochs 100 --batch_size 8 --accum_iter 1 \
     --save_freq 10 --keep_freq 10 --eval_freq 1 \
     --freeze_encoder \
     --output_dir "checkpoints/_kitti-only_" \
     --pretrained "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" \


