# VadCLIP에 시공간 특징을 결합한 영상 이상행위 탐지

UCF-Crime에서 **영상 속 이상행위의 유형과 발생 구간을 함께 찾는 성능**을 비교한 개인 연구다. 공식 VadCLIP은 CLIP 영상 특징을 사용한다. 이 연구에서는 같은 구간의 움직임을 나타내는 I3D-R50 또는 Video Swin-B 특징을 추가했다.

Video Swin-B의 융합층만 학습했을 때 평균 구간 mAP는 공식 모델의 **6.68%에서 8.08%**로 올랐다. 같은 조건에서 프레임 단위 C-Branch AUC는 **88.02%에서 87.79%**로 낮아졌다. 다섯 실험 조건의 지표와 체크포인트 선택 방법은 [결과표](docs/results.md)에 정리했다.

## 모델과 실험

1. 공식 CLIP 특징의 시간 길이 `T`를 기준으로 같은 영상에서 I3D-R50 또는 Video Swin-B 특징을 추출한다.
2. 매 구간의 3D 특징(I3D 2,048차원, Swin 1,024차원)을 512차원으로 투영한다. CLIP 특징과 투영 특징을 이어 붙여 게이트를 계산하고, 게이트가 선택한 만큼 원래 CLIP 특징에 더한다.
3. 결합한 특징을 기존 VadCLIP의 LGT-Adapter와 두 예측 분기에 전달한다. 각 백본에서 **융합층만 학습**하는 조건과 **초반 융합층 학습 후 VadCLIP 모듈도 함께 미세조정**하는 조건을 비교한다. 입력 특징을 추출하는 백본은 두 조건 모두 고정한다.

## 실행 환경

연구 환경에서 기록한 주요 버전은 Python 3.13, PyTorch `2.12.0.dev20260408+cu128`, torchvision `0.27.0.dev20260407+cu128`이다. CUDA 12.8용 nightly 조합으로 실행했다. 같은 환경 또는 사용할 장치에 맞는 PyTorch·torchvision 조합을 먼저 설치한 다음 아래 명령을 저장소 루트에서 실행한다.

```sh
python -m pip install -r requirements.txt
git clone https://github.com/nwpu-zxr/VadCLIP.git vendor/VadCLIP
git -C vendor/VadCLIP checkout --detach c41067f07d252efcda18008bea367886070c33b0
git -C vendor/VadCLIP apply ../../patches/clip-packaging.patch ../../patches/graph-device.patch
```

I3D-R50 특징을 직접 추출할 때는 `python -m pip install pytorchvideo==0.1.5`도 실행한다. 모델을 처음 실행하면 VadCLIP이 사용하는 CLIP ViT-B/16 가중치를 다운로드하므로 인터넷 연결이나 기존 모델 캐시가 필요하다.

공식 저장소에서 UCF-Crime용 **CLIP 특징과 공식 가중치**를 받아 가중치를 `weights/model_ucf.pth`에 둔다. 평가에는 위 커밋의 `list/`에 있는 분할과 정답을 사용한다. 원본 영상에서 3D 특징을 추출하는 방법, 이미 추출한 특징을 연결하는 방법과 CSV 형식은 [데이터 준비](docs/data.md)에 정리했다. [공식 특징·가중치 제공처](https://github.com/nwpu-zxr/VadCLIP/tree/c41067f07d252efcda18008bea367886070c33b0#training)

## 학습 및 평가

```sh
python run_experiment.py eval baseline --baseline-weights weights/model_ucf.pth
python run_experiment.py train swin-gate-only --baseline-weights weights/model_ucf.pth
python run_experiment.py eval swin-gate-only --weights outputs/swin-gate-only/best.pth
```

| 실험 이름 | 입력 특징 | 학습 범위 |
| --- | --- | --- |
| `baseline` | CLIP | 공식 가중치 평가 |
| `i3d-gate-only` | CLIP + I3D-R50 | 투영층·게이트 |
| `i3d-joint` | CLIP + I3D-R50 | 투영층·게이트 이후 기존 모듈까지 |
| `swin-gate-only` | CLIP + Video Swin-B | 투영층·게이트 |
| `swin-joint` | CLIP + Video Swin-B | 투영층·게이트 이후 기존 모듈까지 |

네 가지 융합 실험은 같은 명령에서 실험 이름을 바꿔 실행한다. 학습한 가중치가 있으면 `eval`의 `--weights`에 해당 파일을 지정한다. 학습과 평가에 필요한 입력 파일은 각자의 환경에 준비한다. [기반 코드와 연구 기여](THIRD_PARTY.md)

## 재현 확인

테스트 영상 290개로 다섯 모델의 체크포인트를 재평가했다. 학습 코드는 실제 정상·이상 영상 한 쌍으로 네 조건에서 각각 한 번씩 가중치를 갱신하고 원래 학습 코드와 손실값을 대조했다. 특징 추출은 영상 한 편의 9개 구간에서 두 백본의 출력 배열이 보관된 배열과 원소 단위로 일치하는지 확인했다. [체크포인트 재평가 수치](results/checkpoint-evaluation.csv)

## 코드를 읽는 순서

| 파일 | 확인할 내용 |
| --- | --- |
| [특징 추출](tools/extract_features.py) · [입력 목록 생성](tools/prepare_lists.py) | 같은 영상의 CLIP·3D 특징을 공식 데이터 분할에 맞춰 짝짓는 과정 |
| [특징 융합 모델](src/fusion.py) | 보조 특징의 투영, 게이트 초기화, CLIP 특징에 더하는 연산 |
| [데이터 로더](src/datasets.py) · [학습](src/training.py) | 정상·이상 영상의 배치 구성과 실험 조건별 학습 범위 |
| [평가](src/evaluate.py) | 두 분기의 프레임 점수와 이상행위 구간의 탐지 성능 계산 |
