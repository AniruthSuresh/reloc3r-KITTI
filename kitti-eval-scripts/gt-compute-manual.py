import numpy as np

# GT poses: 3x4, camera-to-world
pose1 = np.array([
    [1.000000e+00, 1.197625e-11, 1.704638e-10, 1.665335e-16],
    [1.197625e-11, 1.000000e+00, 3.562503e-10, -1.110223e-16],
    [1.704638e-10, 3.562503e-10, 1.000000e+00, 2.220446e-16]
])

pose2 = np.array([
    [9.998804e-01, 1.381571e-03, 1.540756e-02, 1.210187e-02],
    [-1.365955e-03, 9.999985e-01, -1.023970e-03, 4.468736e-04],
    [-1.540895e-02, 1.002801e-03, 9.998808e-01, 1.267281e-01]
])

# Convert to 4x4 homogeneous matrices
def to_homogeneous(pose):
    mat = np.eye(4)
    mat[:3, :4] = pose
    mat[:3, 3] = pose[:, 3]
    return mat

T1 = to_homogeneous(pose1)
T2 = to_homogeneous(pose2)

# Compute relative pose T_2_to_1
T2to1 = np.linalg.inv(T1) @ T2

# Optional: normalize translation to unit length
T2to1[:3, 3] /= np.linalg.norm(T2to1[:3, 3])

print("Relative Pose 2->1:")
np.set_printoptions(suppress=True, precision=6)
print(T2to1)
