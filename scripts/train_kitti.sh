# #!/bin/bash
# # Training Reloc3r on KITTI dataset using a single GPU (3090)

# python train.py \
#     --train_dataset "10_000 @ KITTI(split='train', resolution=[(512, 384), (512, 336), (512, 288)], transform=ColorJitter)" \
#     --test_dataset "1_000 @ KITTI(resolution=(512, 384), seed=777)" \
#     --model "Reloc3rRelpose(img_size=512)" \
#     --lr 1e-5 --warmup_epochs 0 --epochs 100 --batch_size 8 --accum_iter 1 \
#     --save_freq 10 --keep_freq 10 --eval_freq 1 \
#     --freeze_encoder \
#     --output_dir "checkpoints/_kitti-only_"


#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train.py \
    --train_dataset "10_000 @ KITTI(split='train', resolution=[(384,288)], transform=ColorJitter)" \
    --test_dataset "1_000 @ KITTI(resolution=(384,288), seed=777)" \
    --model "Reloc3rRelpose(img_size=384)" \
    # --pretrained "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" \
    --lr 1e-5 --epochs 100 \
    --batch_size 1 --accum_iter 1 \
    --freeze_encoder \
    --output_dir "checkpoints/_kitti-only_" 
