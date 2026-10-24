#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "preprocessing"))
sys.path.insert(0, str(ROOT / "scripts"))

from beam_components_to_log_ra import components_to_log_ra
from ra_log_to_png import rasterize
from evaluate_image_checkpoints import point_metrics, polar_to_points


class PreprocessingTests(unittest.TestCase):
    def test_component_reduction_and_near_field_suppression(self):
        components = np.zeros((37, 2, 192), dtype=np.float32)
        components[:, 0, :] = 1.0
        components[:, 1, :] = -2.0
        matrix = components_to_log_ra(components, near_bins=19)
        self.assertEqual(matrix.shape, (37, 192))
        np.testing.assert_array_equal(matrix[:, :19], 0.0)
        np.testing.assert_allclose(matrix[:, 19:], np.log1p(2.0))

    def test_raster_shape_and_range(self):
        matrix = np.linspace(0.0, 3.0, 37 * 192, dtype=np.float32).reshape(37, 192)
        image = rasterize(matrix)
        self.assertEqual(image.shape, (128, 128))
        self.assertEqual(image.dtype, np.uint8)
        self.assertGreater(int(image.max()), int(image.min()))


class MetricTests(unittest.TestCase):
    def test_identical_scans_have_unit_fscore(self):
        image = np.zeros((128, 128), dtype=np.uint8)
        image[20, 10] = 255
        image[80, 100] = 255
        metrics = point_metrics(image, image, 150, 4.0, 0.10)
        self.assertEqual(metrics["fscore"], 1.0)
        self.assertEqual(metrics["cd_m"], 0.0)

    def test_range_mapping(self):
        image = np.zeros((128, 128), dtype=np.uint8)
        image[0, 0] = 255
        points = polar_to_points(image, 150, 4.0)
        self.assertAlmostEqual(float(np.linalg.norm(points[0])), 127.0 / 32.0)


if __name__ == "__main__":
    unittest.main()

