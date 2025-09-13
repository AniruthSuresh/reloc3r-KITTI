import os
import numpy as np
import torch
from reloc3r.utils.image import load_images, check_images_shape_format
from reloc3r.reloc3r_relpose import Reloc3rRelpose, inference_relpose
from reloc3r.utils.device import to_numpy


def setup_reloc3r_relpose_model(model_args, device, ckpt_path=None):
    """Load Reloc3r model, optionally from local checkpoint"""
    if '224' in model_args:
        base_repo = 'siyan824/reloc3r-224'
    else:
        base_repo = 'siyan824/reloc3r-512'

    print(f"Initializing Reloc3r base model {base_repo}")
    model = Reloc3rRelpose.from_pretrained(base_repo)

    if ckpt_path is not None and os.path.exists(ckpt_path):
        print(f"Loading local checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        if "model" in ckpt:
            ckpt = ckpt["model"]
        model.load_state_dict(ckpt, strict=False)

    model.to(device)
    model.eval()
    return model


def wild_relpose_sequence(img_reso, img_folder, ckpt_path=None, output_folder=None, start_frame=0, end_frame=100):
    """Run relative pose estimation for consecutive frames in a folder"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reloc3r_relpose = setup_reloc3r_relpose_model(model_args=img_reso, device=device, ckpt_path=ckpt_path)

    if output_folder is None:
        output_folder = img_folder
    os.makedirs(output_folder, exist_ok=True)

    # Load all frames as sorted paths
    img_files = sorted([os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith(('.png', '.jpg'))])
    img_files = img_files[start_frame:end_frame+1]

    print(img_files[:5])

    print(f"Running relative pose estimation for frames {start_frame} → {end_frame}...")

    for i in range(1, len(img_files)):
        v1_path = img_files[i-1]
        v2_path = img_files[i]

        
        images = load_images([v1_path, v2_path], size=int(img_reso))
        images = check_images_shape_format(images, device)

        batch = [images[0], images[1]]
        pose2to1 = to_numpy(inference_relpose(batch, reloc3r_relpose, device)[0])
        pose2to1[0:3, 3] = pose2to1[0:3, 3] / np.linalg.norm(pose2to1[0:3, 3])

        out_path = os.path.join(output_folder, f'pose_{i-1}_{i}.txt')
        np.savetxt(out_path, pose2to1)
        print(f"Saved relative pose {i-1} → {i} to {out_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Infer relative poses for a sequence of frames')
    parser.add_argument('--img_reso', type=str, default='512')
    parser.add_argument('--img_folder', type=str, required=True, help='Folder containing image sequence')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--output_folder', type=str, default=None)
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--end_frame', type=int, default=100)

    args = parser.parse_args()

    wild_relpose_sequence(
        img_reso=args.img_reso,
        img_folder=args.img_folder,
        ckpt_path=args.checkpoint,
        output_folder=args.output_folder,
        start_frame=args.start_frame,
        end_frame=args.end_frame
    )
