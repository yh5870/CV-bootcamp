"""3주차 과제: YOLOv8 학습, 평가, OpenCV 시각화를 한 번에 실행한다."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / ".ultralytics"
CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(CONFIG_DIR))
MPL_CONFIG_DIR = ROOT / ".matplotlib"
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO, settings

RUNS_DIR = ROOT / "runs"
ARTIFACTS_DIR = ROOT / "artifacts"
DATASET = "coco8.yaml"

settings.update(
    {
        "datasets_dir": str(ROOT / "datasets"),
        "runs_dir": str(RUNS_DIR),
        "weights_dir": str(ROOT),
    }
)


def train(name: str, epochs: int, augmented: bool) -> Path:
    """같은 초기 가중치에서 기준/증강 모델을 학습한다."""
    augmentation = (
        {"degrees": 10, "translate": 0.1, "scale": 0.2, "fliplr": 0.5, "hsv_h": 0.015, "hsv_s": 0.5, "hsv_v": 0.3}
        if augmented
        else {"degrees": 0, "translate": 0, "scale": 0, "fliplr": 0, "hsv_h": 0, "hsv_s": 0, "hsv_v": 0, "mosaic": 0}
    )
    YOLO("yolov8n.pt").train(
        data=DATASET,
        epochs=epochs,
        imgsz=640,
        batch=8,
        device="cpu",
        workers=0,
        project=str(RUNS_DIR),
        name=name,
        exist_ok=True,
        seed=42,
        deterministic=True,
        plots=True,
        verbose=False,
        **augmentation,
    )
    return RUNS_DIR / name / "weights" / "best.pt"


def evaluate(name: str, weights: Path) -> dict[str, float]:
    metrics = YOLO(str(weights)).val(
        data=DATASET,
        imgsz=640,
        device="cpu",
        workers=0,
        project=str(RUNS_DIR),
        name=f"{name}_val",
        exist_ok=True,
        plots=True,
        verbose=False,
    )
    return {
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50-95": round(float(metrics.box.map), 4),
    }


def draw_detections(image, detections: list[tuple[int, int, int, int, str, float]]):
    """OpenCV로 탐지 박스, 클래스, 신뢰도를 그린다."""
    output = image.copy()
    for x1, y1, x2, y2, label, confidence in detections:
        cv2.rectangle(output, (x1, y1), (x2, y2), (35, 210, 120), 2)
        cv2.putText(
            output,
            f"{label} {confidence:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (35, 210, 120),
            2,
            cv2.LINE_AA,
        )
    return output


def predict(weights: Path) -> dict:
    image_path = next((ROOT / "datasets" / "coco8" / "images" / "val").glob("*.jpg"))
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")

    result = YOLO(str(weights)).predict(source=str(image_path), imgsz=640, conf=0.25, verbose=False)[0]
    detections = []
    class_counts: dict[str, int] = {}
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        label = result.names[int(box.cls[0])]
        confidence = float(box.conf[0])
        detections.append((x1, y1, x2, y2, label, confidence))
        class_counts[label] = class_counts.get(label, 0) + 1

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(ARTIFACTS_DIR / "detection_result.jpg"), draw_detections(image, detections))
    return {
        "source_image": image_path.name,
        "detection_count": len(detections),
        "class_counts": class_counts,
        "average_confidence": round(sum(d[-1] for d in detections) / len(detections), 4) if detections else 0,
    }


def save_comparison(metrics: dict[str, dict[str, float]]) -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    labels = list(next(iter(metrics.values())))
    x = range(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - width / 2 for i in x], [metrics["baseline"][k] for k in labels], width, label="Baseline")
    ax.bar([i + width / 2 for i in x], [metrics["augmented"][k] for k in labels], width, label="Augmented")
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("YOLOv8n validation performance on COCO8")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "performance_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    baseline = train("baseline", 10, augmented=False)
    augmented = train("augmented", 20, augmented=True)
    metrics = {"baseline": evaluate("baseline", baseline), "augmented": evaluate("augmented", augmented)}
    save_comparison(metrics)

    best_name = max(metrics, key=lambda name: metrics[name]["mAP50-95"])
    best_weights = baseline if best_name == "baseline" else augmented
    analysis = {"selected_model": best_name, **predict(best_weights)}
    (ARTIFACTS_DIR / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    for name in metrics:
        result_plot = RUNS_DIR / name / "results.png"
        if result_plot.exists():
            shutil.copy2(result_plot, ARTIFACTS_DIR / f"{name}_training.png")

    print(json.dumps({"metrics": metrics, "analysis": analysis}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
