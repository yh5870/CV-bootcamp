from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def generate_depth_map(image: np.ndarray) -> np.ndarray:
    """Create a normalized pseudo-depth map from image brightness."""
    if image is None:
        raise ValueError("입력된 이미지가 없습니다.")
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("BGR 컬러 이미지를 입력해야 합니다.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # ponytail: brightness is only pseudo-depth; replace with MiDaS when metric depth matters.
    return cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def generate_point_cloud(depth_map: np.ndarray) -> np.ndarray:
    """Convert a 2D depth map into an (height, width, XYZ) point cloud."""
    if not isinstance(depth_map, np.ndarray) or depth_map.ndim != 2:
        raise ValueError("2차원 Depth Map을 입력해야 합니다.")

    height, width = depth_map.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    z = depth_map.astype(np.float32) / 255.0
    return np.dstack((x, y, z)).astype(np.float32)


def save_point_cloud_plot(points: np.ndarray, output_path: Path, step: int = 4) -> None:
    sampled = points[::step, ::step].reshape(-1, 3)
    figure = plt.figure(figsize=(8, 6))
    axis = figure.add_subplot(111, projection="3d")
    scatter = axis.scatter(
        sampled[:, 0],
        sampled[:, 1],
        sampled[:, 2],
        c=sampled[:, 2],
        cmap="turbo",
        s=5,
    )
    axis.set(xlabel="X (pixel)", ylabel="Y (pixel)", zlabel="Pseudo depth")
    axis.invert_yaxis()
    figure.colorbar(scatter, ax=axis, pad=0.1, label="Normalized depth")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def process_image(input_path: Path, output_dir: Path) -> dict[str, Path]:
    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError(f"이미지를 불러올 수 없습니다: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    depth = generate_depth_map(image)
    points = generate_point_cloud(depth)
    paths = {
        "original": output_dir / "original.png",
        "depth_gray": output_dir / "depth_map_gray.png",
        "depth_color": output_dir / "depth_map_color.png",
        "point_cloud_data": output_dir / "point_cloud.npy",
        "point_cloud_plot": output_dir / "point_cloud.png",
    }

    cv2.imwrite(str(paths["original"]), image)
    cv2.imwrite(str(paths["depth_gray"]), depth)
    cv2.imwrite(str(paths["depth_color"]), cv2.applyColorMap(depth, cv2.COLORMAP_JET))
    np.save(paths["point_cloud_data"], points)
    save_point_cloud_plot(points, paths["point_cloud_plot"])
    return paths


if __name__ == "__main__":
    root = Path(__file__).parent
    for name, path in process_image(root / "sample.jpg", root / "results").items():
        print(f"{name}: {path}")
