import numpy as np

# Path to input and output files
input_file = "/home/aniruth/Desktop/Courses/Independent - Study/reloc3r-KITTI/data/KITTI_10/10.txt"
output_file = "/home/aniruth/Desktop/Courses/Independent - Study/reloc3r-KITTI/data/KITTI_10/gt_relpose.txt"

# Load poses
lines = open(input_file).read().strip().split("\n")
poses = []
for line in lines:
    nums = [float(x) for x in line.split()]
    poses.append(np.array(nums).reshape(3, 4))

# Convert to 4x4 homogeneous matrices
poses_h = []
for P in poses:
    P_h = np.eye(4)
    P_h[:3, :4] = P
    poses_h.append(P_h)

# Compute relative poses
rel_poses = []
for i in range(1, len(poses_h)):
    # Relative pose: T_i_to_i-1 = inv(T_i-1) @ T_i
    T_rel = np.linalg.inv(poses_h[i-1]) @ poses_h[i]
    rel_poses.append(T_rel)

# Save relative poses (flatten 3x4 for each)
with open(output_file, "w") as f:
    for T in rel_poses:
        f.write(" ".join([f"{x:.12e}" for x in T[:3, :].flatten()]) + "\n")

print(f"Saved {len(rel_poses)} relative poses to {output_file}")
