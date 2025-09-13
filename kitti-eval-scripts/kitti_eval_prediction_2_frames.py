import os
import numpy as np
import torch
from reloc3r.utils.image import load_images, check_images_shape_format
from reloc3r.reloc3r_relpose import Reloc3rRelpose, inference_relpose
from reloc3r.utils.device import to_numpy


def setup_reloc3r_relpose_model(model_args, device, ckpt_path=None):
    """
    Load Reloc3r model, optionally from a local checkpoint
    """
    if '224' in model_args:
        base_repo = 'siyan824/reloc3r-224'
    else:
        base_repo = 'siyan824/reloc3r-512'

    print(f"Initializing Reloc3r base model {base_repo}")
    model = Reloc3rRelpose.from_pretrained(base_repo)

    # Load local checkpoint if provided
    if ckpt_path is not None and os.path.exists(ckpt_path):
        print(f"Loading local checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        if "model" in ckpt:  # if saved as dict with key "model"
            ckpt = ckpt["model"]
        model.load_state_dict(ckpt, strict=False)

    model.to(device)
    model.eval()
    return model


def wild_relpose(img_reso, v1_path, v2_path, ckpt_path=None, output_folder=None):
    """
    Run relative pose estimation between two images
    """
    if output_folder is None:
        output_folder = os.path.dirname(v1_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Reloc3r
    print('Loading Reloc3r...')
    reloc3r_relpose = setup_reloc3r_relpose_model(model_args=img_reso, device=device, ckpt_path=ckpt_path)

    # Load images
    print('Loading images...')
    images = load_images([v1_path, v2_path], size=int(img_reso))
    images = check_images_shape_format(images, device)

    # Run relative pose estimation
    print('Running relative pose estimation...')
    batch = [images[0], images[1]]
    pose2to1 = to_numpy(inference_relpose(batch, reloc3r_relpose, device)[0])
    pose2to1[0:3, 3] = pose2to1[0:3, 3] / np.linalg.norm(pose2to1[0:3, 3])  # normalize scale to 1 meter

    # Save pose to file
    out_path = os.path.join(output_folder, 'pose2to1.txt')
    np.savetxt(out_path, pose2to1)
    print(f'Pose saved to {out_path}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Infer relative pose from two images')
    parser.add_argument('--img_reso', type=str, default='512', help='Input image resolution for model')
    parser.add_argument('--v1_path', type=str, required=True, help='Path to first image')
    parser.add_argument('--v2_path', type=str, required=True, help='Path to second image')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to local checkpoint (.pth)')
    parser.add_argument('--output_folder', type=str, default=None, help='Folder to save output pose')

    args = parser.parse_args()

    wild_relpose(
        img_reso=args.img_reso,
        v1_path=args.v1_path,
        v2_path=args.v2_path,
        ckpt_path=args.checkpoint,
        output_folder=args.output_folder
    )
