# CV Bootcamp

- [1주차: Computer Vision 이미지 전처리](image_preprocessing.ipynb)
- [2주차: Unit Test 및 2D → 3D 변환](week2_2d_to_3d/README.md)
- [3주차: YOLOv8n 객체 탐지](#3주차-yolov8n-객체-탐지)

## 3주차: YOLOv8n 객체 탐지

YOLOv8n으로 COCO8 객체 탐지 모델을 학습하고, 데이터 증강 전후 성능을 비교한 뒤 OpenCV로 예측 결과를 시각화한 프로젝트입니다.

### 실행

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe week3_yolo.py
.\.venv\Scripts\python.exe -m unittest -v
```

첫 실행 시 YOLOv8n 가중치와 COCO8 데이터셋이 자동으로 내려받아집니다. 학습 데이터, 실행 로그, 가중치는 각각 `datasets/`, `runs/`, `*.pt`에 생성되며 Git에는 포함하지 않습니다.

### 실험 설정

| 항목 | 기준 모델 | 증강 모델 |
|---|---:|---:|
| 모델 | YOLOv8n 사전학습 가중치 | YOLOv8n 사전학습 가중치 |
| 데이터셋 | COCO8 (학습 4장, 검증 4장) | COCO8 (학습 4장, 검증 4장) |
| Epoch | 10 | 20 |
| 이미지 크기 | 640 | 640 |
| 증강 | 없음 | 회전, 이동, 스케일, 좌우 반전, HSV |
| 장치 / 시드 | CPU / 42 | CPU / 42 |

### 평가 결과

| 지표 | 기준 모델 | 증강 모델 | 변화 |
|---|---:|---:|---:|
| Precision | 0.6402 | **0.7716** | +0.1314 |
| Recall | 0.7500 | **0.8333** | +0.0833 |
| mAP50 | **0.8893** | 0.8885 | -0.0008 |
| mAP50-95 | 0.6265 | **0.6506** | +0.0241 |

증강 모델은 mAP50이 사실상 유지되는 동안 Precision, Recall, mAP50-95가 개선되어 최종 모델로 선택했습니다. 전체 수치는 [`artifacts/metrics.json`](artifacts/metrics.json), 비교 그래프는 [`artifacts/performance_comparison.png`](artifacts/performance_comparison.png)에서 확인할 수 있습니다.

### OpenCV 시각화 및 패턴 분석

검증 이미지 `000000000036.jpg`에서 `person` 1개와 `umbrella` 1개를 탐지했으며 평균 신뢰도는 **0.8388**입니다. OpenCV로 경계 상자, 클래스명, 신뢰도를 그린 결과는 [`artifacts/detection_result.jpg`](artifacts/detection_result.jpg), 요약은 [`artifacts/analysis.json`](artifacts/analysis.json)에 저장됩니다.

### 산출물

- `week3_yolo.py`: 학습, 평가, 비교 그래프, OpenCV 시각화
- `test_week3_yolo.py`: 시각화 함수 최소 단위 테스트
- `artifacts/`: 실제 실행 지표와 이미지
- `report/week3_yolo_report.pptx`: 4페이지 결과 보고서

> COCO8은 파이프라인 검증용 초소형 표본이므로 일반화 성능을 주장할 수 없습니다. 실제 적용 전 더 큰 검증셋과 클래스별 오탐 분석이 필요합니다.
