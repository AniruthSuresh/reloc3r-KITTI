import numpy as np
import os
from reloc3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from reloc3r.utils.image import imread_cv2, cv2


DATA_ROOT = "/home/aniruth/Desktop/Courses/Independent - Study/reloc3r-KITTI/data/KITTI_10"


class KITTI(BaseStereoViewDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_root = DATA_ROOT

        # paths
        self.images_dir = os.path.join(self.data_root, "image_left")
        self.calib_path = os.path.join(self.data_root, "calib_left_rgb.txt")
        self.pose_path = os.path.join(self.data_root, "10.txt")

        # list of image files sorted numerically
        self.image_names = sorted(
            [f for f in os.listdir(self.images_dir) if f.endswith(".png")],
            key=lambda x: int(os.path.splitext(x)[0])
        )
        # print(f"[INFO] Found {len(self.image_names)} images in {self.images_dir}")

        # load intrinsics correctly from calib file
        self.intrinsics = self._load_intrinsics(self.calib_path)
        # print(f"[INFO] Camera intrinsics loaded:\n{self.intrinsics}")

        # load all poses (each line has 12 numbers = 3x4 matrix)
        self.poses = self._load_poses(self.pose_path)
        # print(f"[INFO] Loaded {len(self.poses)} camera poses from {self.pose_path}")

    def _load_intrinsics(self, calib_file):
        """
        Parses KITTI single-line calibration and returns 3x3 intrinsic matrix K.
        """
        vals = np.loadtxt(calib_file, dtype=np.float32)  # shape: (12,)
        if vals.shape[0] != 12:
            raise ValueError(f"Expected 12 numbers in calib file, got {vals.shape[0]}")

        mat = vals.reshape(3, 4)  # reshape into 3x4
        K = mat[:, :3].copy()      # take 3x3 intrinsic part
        # print(f"[DEBUG] Parsed intrinsics K:\n{K}")
        return K


    def _load_poses(self, pose_file):
        """
        Loads all poses from a file, each line is 12 numbers forming a 3x4 matrix.
        Converts to 4x4 homogeneous matrices.
        """
        poses = []
        with open(pose_file, "r") as f:
            for i, line in enumerate(f):
                vals = list(map(float, line.strip().split()))
                mat = np.eye(4, dtype=np.float32)
                mat[0:3, :] = np.array(vals, dtype=np.float32).reshape(3, 4)
                poses.append(mat)
                # if i < 3:  # print first 3 poses for sanity check
                    # print(f"[DEBUG] Pose {i}:\n{mat}")
        return poses

    def __len__(self):
        # return number of *pairs* available
        return len(self.image_names) - 1

    def _get_views(self, idx, resolution, rng):
        """
        Return a pair of consecutive views (idx, idx+1).
        """
        views = []
        for offset in [0, 1]:
            img_name = self.image_names[idx + offset]
            img_path = os.path.join(self.images_dir, img_name)

            # load image
            color_image = imread_cv2(img_path)
            color_image = cv2.resize(color_image, (1242, 375))

            # intrinsics
            intrinsics = self.intrinsics.copy()

            # pose
            camera_pose = self.poses[idx + offset]

            # crop/resize
            color_image, intrinsics = self._crop_resize_if_necessary(
                color_image,
                intrinsics,
                resolution,
                rng=rng
            )

            views.append(dict(
                img=color_image,
                camera_intrinsics=intrinsics,
                camera_pose=camera_pose,
                dataset='KITTI',
                label="KITTI_sequence10",
                instance=img_name
            ))
        return views
