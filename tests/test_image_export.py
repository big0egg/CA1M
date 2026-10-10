import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cubifyanything.boxes import GeneralInstance3DBoxes
from cubifyanything.image_export import project_box_edge, save_prediction_image
from cubifyanything.instances import Instances3D


class ImageExportTests(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[100., 0., 100.], [0., 100., 100.], [0., 0., 1.]])

    def test_known_pinhole_projection(self):
        segment = project_box_edge([-1, -1, 2], [1, -1, 2], self.K, 200, 200)
        np.testing.assert_allclose(segment, [50, 50, 150, 50])

    def test_behind_camera_is_not_drawn(self):
        self.assertIsNone(project_box_edge([0, 0, -2], [1, 1, -1], self.K, 200, 200))
        self.assertIsNone(project_box_edge([np.nan, 0, 2], [1, 1, 2], self.K, 200, 200))

    def test_near_plane_and_image_boundary_clipping(self):
        # This edge crosses the camera plane; clip in 3D before projecting.
        segment = project_box_edge([0, 0, -1], [1, 0, 2], self.K, 200, 200)
        self.assertIsNotNone(segment)
        self.assertTrue(np.isfinite(segment).all())
        self.assertTrue(all(0 <= coordinate <= 199 for coordinate in segment))
        np.testing.assert_allclose(
            project_box_edge([-10, 0, 2], [10, 0, 2], self.K, 200, 200),
            [0, 100, 199, 100])

    def test_boxes_outside_image_are_not_drawn(self):
        self.assertIsNone(project_box_edge([10, 0, 1], [20, 0, 1], self.K, 200, 200))

    def test_png_and_jpg_keep_image_size_and_draw_predictions(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        instances = Instances3D(
            scores=torch.tensor([0.9]),
            pred_boxes_3d=GeneralInstance3DBoxes(
                torch.tensor([[0., 0., 3., 2., 2., 2.]]), torch.eye(3)[None]))
        with tempfile.TemporaryDirectory() as directory:
            for image_format in ("png", "jpg"):
                path = save_prediction_image(image, instances, self.K, directory,
                                             42, 1.25, 0, image_format)
                self.assertEqual(path.parent, Path(directory) / "42")
                self.assertEqual(path.name, f"frame_000000_1250000000.{image_format}")
                with Image.open(path) as saved:
                    self.assertEqual(saved.size, (200, 200))
                    pixels = np.array(saved)
                    self.assertGreater(int(pixels.sum()), 0)
                    # Visible front-face edge from the known box projection.
                    self.assertGreater(int(pixels[50, 100].sum()), 0)

    def test_empty_and_invisible_predictions_still_save_frame(self):
        image = np.full((200, 200, 3), 30, dtype=np.uint8)
        cases = [
            Instances3D(scores=torch.empty(0), pred_boxes_3d=GeneralInstance3DBoxes.empty()),
            Instances3D(scores=torch.tensor([0.9]), pred_boxes_3d=GeneralInstance3DBoxes(
                torch.tensor([[0., 0., -3., 2., 2., 2.]]), torch.eye(3)[None])),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for index, instances in enumerate(cases):
                path = save_prediction_image(image, instances, self.K, directory,
                                             42, float(index), index)
                with Image.open(path) as saved:
                    np.testing.assert_array_equal(np.array(saved), image)


if __name__ == "__main__":
    unittest.main()
