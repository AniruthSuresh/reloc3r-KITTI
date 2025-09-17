import numpy as np
import os
from scipy.spatial.transform import Rotation as R

# Folders
pred_folder = "./output/"
gt_file = "/home/aniruth/Desktop/Courses/Independent - Study/reloc3r-KITTI/data/KITTI_10/gt_relpose.txt"

def load_gt_poses(file_path):
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            values = np.fromstring(line, sep=' ')
            if len(values) != 12:
                raise ValueError(f"Expected 12 values per line, got {len(values)}")
            mat = values.reshape(3, 4)
            mat = np.vstack([mat, [0, 0, 0, 1]])  # make 4x4 homogeneous
            poses.append(mat)
    return poses

gt_poses = load_gt_poses(gt_file)
print(f"Loaded {len(gt_poses)} ground truth relative poses from {gt_file}")



def load_pred_pose(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        if len(lines) != 4:
            raise ValueError(f"Expected 4 lines for 4x4 pose, got {len(lines)}")
        values = []
        for line in lines:
            # Split line by whitespace and convert to float
            numbers = [float(x) for x in line.strip().split()]
            values.append(numbers)
        mat = np.array(values)
        if mat.shape != (4, 4):
            raise ValueError(f"Pose shape mismatch: {mat.shape}")
    return mat



trans_errors = []
rot_errors = []

# Ensure files are sorted by frame number
pred_files = sorted([f for f in os.listdir(pred_folder) if f.startswith("pose_") and f.endswith(".txt")],
                    key=lambda x: int(x.split("_")[1]))

print(f"Evaluating {len(pred_files)} predicted poses from {pred_folder}...")

import numpy as np
import math

def pose_error(pose_pred, pose_gt):
    """
    Args:
        pose_pred: [4,4] predicted pose matrix (numpy)
        pose_gt:   [4,4] ground-truth pose matrix (numpy)
    Returns:
        trans_err: translation error (normalized L2)
        rot_err:   rotation error (angle in degrees)
    """
    # Extract translation vectors
    t_pred = pose_pred[:3, 3]
    t_gt = pose_gt[:3, 3]

    # Normalized translation vectors
    t_pred_norm = t_pred / (np.linalg.norm(t_pred) + 1e-8)
    t_gt_norm = t_gt / (np.linalg.norm(t_gt) + 1e-8)

    # Translation error: L2 norm of difference
    trans_err = np.linalg.norm(t_pred_norm - t_gt_norm)

    # Extract rotation matrices
    R_pred = pose_pred[:3, :3]
    R_gt = pose_gt[:3, :3]

    # R1 = R_pred
    # R2 = R_gt
    # eps=1e-6
    # R_rel = R1.T @ R2
    # trace = np.trace(R_rel)
    # cosine = np.clip((trace - 1) / 2, -1.0+eps, 1.0-eps)

    # return trans_err, np.arccos(cosine)  

    R_pred = pred_pose[:3, :3]
    R_gt = gt_pose[:3, :3]
    R_diff = R_pred @ R_gt.T
    rot_err = R.from_matrix(R_diff).magnitude() * (180 / np.pi)
    
    return trans_err, rot_err



for i, fname in enumerate(pred_files):

    
    pred_pose = load_pred_pose(os.path.join(pred_folder, fname))
    gt_pose = gt_poses[i]

    # print(f"Frame {i} → {i+1}")
    # print(pred_pose , gt_pose)
    trans_err, rot_err = pose_error(pred_pose, gt_pose)

    trans_errors.append(trans_err)
    rot_errors.append(rot_err)

    # rot_deg = np.degrees(rot_err)
#     print(f"Frame {i} → {i+1}: Trans Error = {trans_err:.6f}, Rot Error = {rot_deg:.6f} deg")

    print(f"Frame {i} → {i+1}: Trans Error = {trans_err:.6f}, Rot Error = {rot_err:.6f} deg")

