from pathlib import Path

import cv2
import numpy as np
import pytest

from processing_3d import generate_depth_map, generate_point_cloud, process_image


def test_generate_depth_map_validates_input_and_output():
    with pytest.raises(ValueError):
        generate_depth_map(None)

    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[:, 15:] = 255
    depth = generate_depth_map(image)

    assert depth.shape == image.shape[:2]
    assert depth.dtype == np.uint8
    assert depth[:, 0].mean() < depth[:, -1].mean()


def test_generate_point_cloud_coordinates():
    depth = np.array([[0, 255], [128, 64]], dtype=np.uint8)
    points = generate_point_cloud(depth)

    assert points.shape == (2, 2, 3)
    assert points.dtype == np.float32
    np.testing.assert_allclose(points[0, 1], [1, 0, 1])


def test_process_image_creates_all_results(tmp_path: Path):
    image_path = tmp_path / "sample.jpg"
    cv2.imwrite(str(image_path), np.full((24, 32, 3), 127, dtype=np.uint8))

    paths = process_image(image_path, tmp_path / "results")

    assert all(path.exists() and path.stat().st_size > 0 for path in paths.values())


def test_process_image_rejects_missing_file(tmp_path: Path):
    with pytest.raises(ValueError, match="불러올 수 없습니다"):
        process_image(tmp_path / "missing.jpg", tmp_path / "results")
