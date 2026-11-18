# Sen1Floods11 SAR 홍수 분할 (U-Net) 실습

Sentinel-1 이중편파(VV, VH) 위성 레이더 영상에서 홍수로 잠긴 물 영역을 픽셀 단위로 찾는 U-Net(ResNet34 인코더) 학습·평가 노트북입니다.
공개 데이터셋 Sen1Floods11의 사람이 직접 라벨링한(hand-labeled) 446장을 사용했고, 어떤 편파가 도움이 되는지 보는 간단한 ablation과 오류 분석을 포함합니다.

> 학습용 실습 프로젝트입니다. 새로운 방법을 제안하는 연구가 아닙니다.

- 노트북: `Sen1Floods11.ipynb` (Google Colab, T4 GPU 기준)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<아이디>/<저장소>/blob/main/Sen1Floods11.ipynb](https://colab.research.google.com/github/AnHyunJun-michael/sen1floods11-unet-sar/blob/main/Sen1Floods11.ipynb#scrollTo=sJaWsWQNTdv3)

## 데이터
Sen1Floods11 (Bonafilia et al., 2020). 공식 저장소에 별도 라이선스가 명시되어 있지 않아 데이터는 이 저장소에 포함하지 않고, 공식 GCS 버킷에서 내려받습니다.

인용: Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. (2020). Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1. CVPR Workshops, pp. 210-211.

## 설정

| 항목 | 내용 |
|---|---|
| 입력 | GRD 이중편파 2채널(VV, VH), 512×512 칩. 학습 시 256×256 랜덤 크롭과 상하좌우 뒤집기 |
| 전처리 | train 통계(VV -10.39 / VH -17.24 dB)로 z-score, ±5σ 클리핑, NaN은 0으로 대체 |
| 모델 | `segmentation_models_pytorch` U-Net, ResNet34 인코더, 사전학습 없음(`encoder_weights=None`) |
| 손실 | Cross-Entropy(물 클래스 가중치 3) + Dice. 라벨 -1(판정 불가, 약 13%)은 손실과 평가에서 제외 |
| 학습 | AdamW(lr 1e-3, weight decay 1e-4), Cosine 스케줄, 50 epoch, batch 8. valid IoU가 가장 높은 epoch 저장 |
| 평가 | 물 픽셀이 약 9.5%뿐이라 accuracy 대신 IoU, F1 사용. 칩별 평균이 아니라 분할 전체 픽셀을 합산해 계산 |
| 분할 | train 252 / valid 89 / test 90 / Bolivia 15 (Bolivia 사건은 학습에서 완전히 제외해 별도 시험용으로 사용) |

## 결과 (IoU)

| 입력 | 실행 횟수 | test | Bolivia |
|---|---|---|---|
| VV+VH | 3 seeds | 0.661 (0.655–0.665) | 0.645 (0.627–0.674) |
| VV only | 1 seed | 0.563 | 0.681 |
| VH only | 1 seed | 0.634 | 0.687 |

VV+VH의 test F1은 0.79~0.80입니다. 괄호는 시드 3개의 최소–최대입니다.

### 해석할 때 주의

- test에서는 VV only(0.563)가 VV+VH(0.655~0.665)보다 분명히 낮습니다. 시드 간 편차(약 0.01)보다 차이가 훨씬 큽니다.
- Bolivia는 15장뿐이고 시드 간 편차가 0.047입니다. VV only, VH only, VV+VH의 Bolivia 차이는 이 편차 안에 있어서 우열을 말할 수 없습니다.
- VV only와 VH only는 각각 1회 실행이라 시드 변동을 모릅니다.
- 같은 시드를 다시 돌려도 결과가 달랐습니다(seed1 Bolivia IoU 0.617 vs 0.633). GPU 연산의 비결정성 때문으로 보이며, 이 노트북의 소수점 셋째 자리 차이는 의미가 없습니다.

### NaN이 많은 칩의 영향

NaN 비율이 20%를 넘는 칩이 test에 4장, Bolivia에 3장 있었습니다(Paraguay 칩 1장은 전체가 NaN). 그중 test 2장, Bolivia 1장을 빼고 다시 평가했습니다(VV+VH 3 seeds 평균).

| split | 전체 | NaN 큰 칩 제외 |
|---|---|---|
| test | 0.661 | 0.673 |
| Bolivia | 0.645 | 0.630 |

제외 결과의 방향이 split마다 달라서, NaN이 점수를 일관되게 깎는다고 결론낼 수 없습니다.

## 한계

- 입력은 GRD(세기만 있는 SAR)라서 위상 정보가 필요한 PolSAR 분해(H/α)와 coherence는 쓰지 않았습니다.
- 질감 특징(GLCM), 지형 정보(DEM/HAND) 등 추가 입력을 쓰지 않았습니다.
- 사전학습 가중치와 다른 모델(CPNet 등)과의 비교는 하지 않았습니다.
- 하이퍼파라미터(클래스 가중치 3, lr 등)는 튜닝하지 않고 정한 값입니다.
- 라벨은 사람이 만든 hand-labeled 데이터이고, 약라벨(weak label)은 쓰지 않았습니다.

## 실행

1. Colab에서 노트북을 열고 런타임을 GPU로 설정합니다.
2. 처음부터 순서대로 실행합니다. 데이터 다운로드와 패키지 설치(`segmentation-models-pytorch`, `rasterio`)가 포함되어 있습니다.
3. 50 epoch 학습은 설정당 약 9분(T4)이 걸립니다. 첫 학습 전 `epochs=2` 점검 셀로 먼저 확인할 수 있습니다.
