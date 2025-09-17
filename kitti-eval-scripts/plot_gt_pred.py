import numpy as np
import os
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# -------------------------------
# Paths
# -------------------------------
pred_folder = "../output/"
gt_file = "/home/aniruth/Desktop/Courses/Independent - Study/reloc3r-KITTI/data/KITTI_10/gt_relpose.txt"

# -------------------------------
# Load Ground Truth Relative Poses
# -------------------------------
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

# -------------------------------
# Load Predicted Pose
# -------------------------------
def load_pred_pose(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        if len(lines) != 4:
            raise ValueError(f"Expected 4 lines for 4x4 pose, got {len(lines)}")
        values = []
        for line in lines:
            numbers = [float(x) for x in line.strip().split()]
            values.append(numbers)
        mat = np.array(values)
        if mat.shape != (4, 4):
            raise ValueError(f"Pose shape mismatch: {mat.shape}")
    return mat

# -------------------------------
# Pose Error Function
# -------------------------------
def pose_error(pose_pred, pose_gt):
    """
    Args:
        pose_pred: [4,4] predicted pose matrix
        pose_gt:   [4,4] ground-truth pose matrix
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

    # Translation error
    trans_err = np.linalg.norm(t_pred_norm - t_gt_norm)

    # Rotation error
    R_pred = pose_pred[:3, :3]
    R_gt = pose_gt[:3, :3]
    R_diff = R_pred @ R_gt.T
    rot_err = R.from_matrix(R_diff).magnitude() * (180 / np.pi)
    
    return trans_err, rot_err

# -------------------------------
# Accumulate Relative Poses into Global Trajectory
# -------------------------------
def accumulate_poses(rel_poses, scale_translation=True, gt_poses=None):
    traj = []
    current_pose = np.eye(4)
    traj.append(current_pose[:3, 3])  # start at origin

    for i, rel in enumerate(rel_poses):
        rel_copy = rel.copy()

        if scale_translation:
            t = rel_copy[:3, 3]

            if gt_poses is not None and i < len(gt_poses):  
                # Scale to match GT step length
                gt_step = np.linalg.norm(gt_poses[i][:3, 3])
                pred_step = np.linalg.norm(t) + 1e-8
                scale = gt_step / pred_step
            else:
                # Default: normalize to unit step
                scale = 1.0 / (np.linalg.norm(t) + 1e-8)

            rel_copy[:3, 3] = t * scale

        current_pose = current_pose @ rel_copy
        traj.append(current_pose[:3, 3])

    return np.array(traj)


# -------------------------------
# Load Prediction Files
# -------------------------------
pred_files = sorted([f for f in os.listdir(pred_folder) if f.startswith("pose_") and f.endswith(".txt")],
                    key=lambda x: int(x.split("_")[1]))

print(f"Evaluating {len(pred_files)} predicted poses from {pred_folder}...")

# Load predicted relative poses
pred_rel_poses = [load_pred_pose(os.path.join(pred_folder, f)) for f in pred_files]

# -------------------------------
# Compute Errors
# -------------------------------
trans_errors = []
rot_errors = []

for i, (pred_pose, gt_pose) in enumerate(zip(pred_rel_poses, gt_poses)):
    trans_err, rot_err = pose_error(pred_pose, gt_pose)
    trans_errors.append(trans_err)
    rot_errors.append(rot_err)
    print(f"Frame {i} → {i+1}: Trans Error = {trans_err:.6f}, Rot Error = {rot_err:.6f} deg")

# -------------------------------
# Build Trajectories
# -------------------------------
# Build trajectories
gt_traj = accumulate_poses(gt_poses, scale_translation=False)
pred_traj = accumulate_poses(pred_rel_poses, scale_translation=True, gt_poses=gt_poses)


# -------------------------------
# Plot Trajectories
# -------------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

ax.plot(gt_traj[:, 0], gt_traj[:, 1], gt_traj[:, 2],
        label="Ground Truth", color="blue", linewidth=2, marker="o", markersize=3)
ax.plot(pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2],
        label="Predicted", color="red", linewidth=2, marker="^", markersize=3)

ax.set_title("Camera Trajectory from Relative Poses")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
ax.grid(True)

plt.show()
