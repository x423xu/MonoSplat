import unittest

import torch

from src.model.da3_pose_utils import (
    da3_pixel_intrinsics_to_normalized,
    da3_w2c_to_c2w,
    replace_batch_poses_from_da3,
)


class DA3PoseUtilsTest(unittest.TestCase):
    def test_da3_pixel_intrinsics_to_normalized_divides_image_axes_only(self):
        intrinsics = torch.tensor(
            [[[126.0, 0.0, 63.0], [0.0, 84.0, 42.0], [0.0, 0.0, 1.0]]]
        )

        normalized = da3_pixel_intrinsics_to_normalized(intrinsics, height=252, width=126)

        self.assertTrue(torch.allclose(normalized[0, 0, 0], torch.tensor(1.0)))
        self.assertTrue(torch.allclose(normalized[0, 1, 1], torch.tensor(84.0 / 252.0)))
        self.assertTrue(torch.allclose(normalized[0, 0, 2], torch.tensor(0.5)))
        self.assertTrue(torch.allclose(normalized[0, 1, 2], torch.tensor(42.0 / 252.0)))

    def test_da3_w2c_to_c2w_does_not_scale_translation_by_image_size(self):
        w2c = torch.eye(4).repeat(1, 1, 1)
        w2c[0, :3, 3] = torch.tensor([4.0, 8.0, 12.0])

        c2w = da3_w2c_to_c2w(w2c, height=252, width=252)

        self.assertTrue(torch.allclose(c2w[0, :3, 3], torch.tensor([-4.0, -8.0, -12.0])))

    def test_replace_batch_poses_from_da3_splits_context_and_target_views(self):
        batch = {
            "context": {
                "image": torch.zeros(1, 2, 3, 4, 4),
                "intrinsics": torch.zeros(1, 2, 3, 3),
                "extrinsics": torch.zeros(1, 2, 4, 4),
            },
            "target": {
                "image": torch.zeros(1, 1, 3, 4, 4),
                "intrinsics": torch.zeros(1, 1, 3, 3),
                "extrinsics": torch.zeros(1, 1, 4, 4),
            },
        }
        da3_w2c = torch.eye(4).repeat(1, 3, 1, 1)
        da3_w2c[0, :, :3, 3] = torch.tensor(
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        )
        da3_k = torch.eye(3).repeat(1, 3, 1, 1)
        da3_k[0, :, 0, 0] = torch.tensor([252.0, 126.0, 63.0])
        da3_k[0, :, 1, 1] = torch.tensor([252.0, 126.0, 63.0])

        replaced = replace_batch_poses_from_da3(batch, da3_w2c, da3_k, height=252, width=252)

        self.assertTrue(
            torch.allclose(
                replaced["context"]["extrinsics"][0, :, :3, 3],
                torch.tensor([[-1.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]),
            )
        )
        self.assertTrue(
            torch.allclose(
                replaced["target"]["extrinsics"][0, :, :3, 3],
                torch.tensor([[-3.0, 0.0, 0.0]]),
            )
        )
        self.assertTrue(
            torch.allclose(replaced["context"]["intrinsics"][0, :, 0, 0], torch.tensor([1.0, 0.5]))
        )
        self.assertTrue(
            torch.allclose(replaced["target"]["intrinsics"][0, :, 0, 0], torch.tensor([0.25]))
        )


if __name__ == "__main__":
    unittest.main()