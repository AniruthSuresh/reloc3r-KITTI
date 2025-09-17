import numpy as np
import os
from reloc3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from reloc3r.utils.image import imread_cv2, cv2


DATA_ROOT = "/home/aniruth/Desktop/Courses/Independent - Study/reloc3r-KITTI/data/2011-09-26-KITTI/"

class KITTI_specific_seq(BaseStereoViewDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_root = DATA_ROOT

        # fixed paths
        self.images_dir = os.path.join(
            self.data_root, "2011_09_26_drive_0093_sync/image_02/data"
        )
        self.calib_path = os.path.join(self.data_root, "calib_left_rgb.txt")
        self.pose_path = os.path.join(self.data_root, "10.txt")

        # list of image files sorted numerically
        self.image_names = sorted(
            [f for f in os.listdir(self.images_dir) if f.endswith(".png")],
            key=lambda x: int(os.path.splitext(x)[0])
        )

        # load intrinsics (already 3x3)
        self.intrinsics = self._load_intrinsics(self.calib_path)

        # load all poses (already 4x4)
        self.poses = self._load_poses(self.pose_path)

    def _load_intrinsics(self, calib_file):
        """
        Loads KITTI intrinsics file with 9 values (3x3 matrix).
        """
        vals = np.loadtxt(calib_file, dtype=np.float32)
        if vals.size != 9:
            raise ValueError(f"Expected 9 numbers in calib file, got {vals.size}")
        K = vals.reshape(3, 3)
        return K

    def _load_poses(self, pose_file):
        """
        Loads all poses from file.
        Each line has 16 numbers forming a 4x4 matrix.
        """
        poses = []
        with open(pose_file, "r") as f:
            for line in f:
                vals = list(map(float, line.strip().split()))
                if len(vals) != 16:
                    raise ValueError(f"Expected 16 numbers per line in pose file, got {len(vals)}")
                mat = np.array(vals, dtype=np.float32).reshape(4, 4)
                poses.append(mat)
        return poses

    def __len__(self):
        return len(self.image_names) - 1

    def _get_views(self, idx, resolution, rng):
        """
        Return a pair of consecutive views (idx, idx+1).
        """
        views = []
        for offset in [0, 1]:
            img_name = self.image_names[idx + offset]
            img_path = os.path.join(self.images_dir, img_name)

            color_image = imread_cv2(img_path)
            color_image = cv2.resize(color_image, (1242, 375))

            intrinsics = self.intrinsics.copy()
            camera_pose = self.poses[idx + offset]

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
                label="KITTI_specific_seq",
                instance=img_name
            ))
            
        return views
