#!/bin/bash

python infer_kitti_relpose.py \
    --v1_path "/home/aniruth/Desktop/Courses/Independent - Study/reloc3r-KITTI/data/KITTI_10/image_left/000000.png" \
    --v2_path "/home/aniruth/Desktop/Courses/Independent - Study/reloc3r-KITTI/data/KITTI_10/image_left/000001.png" \
    --checkpoint "/home/aniruth/Desktop/Courses/Independent - Study/reloc3r-KITTI/checkpoints/kitti-ckp/checkpoint-best.pth" \
    --img_reso 512 \
    --output_folder results/kitti_relpose

python infer_two_images.py \
    --v1_path "/home/aniruth/Desktop/Courses/Independent - Study/reloc3r-KITTI/data/KITTI_10/image_left/000000.png" \
    --v2_path "/home/aniruth/Desktop/Courses/Independent - Study/reloc3r-KITTI/data/KITTI_10/image_left/000001.png" \
    --checkpoint "/home/aniruth/Desktop/Courses/Independent - Study/reloc3r-KITTI/checkpoints/kitti-ckp/checkpoint-best.pth" \
  --img_reso 512 \
  --output_path results/pose2to1.txt

