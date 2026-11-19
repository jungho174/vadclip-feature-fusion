# UCF-Crime 입력 준비

공식 VadCLIP의 CLIP 특징 배열 한 행은 영상의 16프레임 구간 하나에 대응한다. 이 연구에서는 원본 영상에서 같은 구간의 I3D 또는 Video Swin 특징을 추출한 뒤 두 배열을 시간 순서대로 한 쌍으로 읽는다. 공식 CLIP 특징만 준비하면 기준 모델을 평가할 수 있다.

## 1. 공식 CLIP 특징과 분할

[저장소 설정](../README.md#실행-환경)을 마친 뒤 공식 UCF-Crime CLIP 특징을 내려받는다. `--clip-root`는 `Abuse/`, `Training_Normal_Videos_Anomaly/` 등의 카테고리 폴더를 바로 포함하는 경로다. 아래 예시의 `data/` 아래 경로는 실제 저장 위치로 바꿀 수 있다. [VadCLIP 공식 다운로드](https://github.com/nwpu-zxr/VadCLIP/tree/c41067f07d252efcda18008bea367886070c33b0#training)

```sh
python tools/prepare_lists.py clip --clip-root data/features/clip_official
```

이 명령은 공식 `list/ucf_CLIP_rgb.csv`와 `list/ucf_CLIP_rgbtest.csv`의 행 순서와 라벨을 보존하고, 각 행을 로컬 특징 파일에 연결한다. `data/lists/`에 생성되는 `ucf_official_CLIP_train_local.csv`와 `ucf_official_CLIP_test_local.csv`의 열은 `path,label`이다.

| 분할 | 고유 영상 | 영상당 CLIP 특징 | CSV 행 |
| --- | ---: | --- | ---: |
| 학습 | 1,610 | crop 0–9 | 16,100 |
| 테스트 | 290 | crop 5 | 290 |

공식 테스트 순서는 프레임 단위 정답과 구간 정답을 연결하는 기준이다. 평가는 고정한 VadCLIP 커밋의 `list/gt_ucf.npy`, `gt_segment_ucf.npy`, `gt_label_ucf.npy`를 사용한다. 특징 배열은 `(T, 512)`이며 `T`는 영상마다 다르다.

## 2. 원본 영상에서 3D 특징 추출

이미 아래 규격의 특징 파일이 있으면 3단계에서 바로 목록을 생성할 수 있다. 직접 추출하려면 UCF-Crime 원본 영상과 해당 영상의 공식 CLIP 특징을 준비한다. 예를 들어 `Abuse001_x264.mp4`는 `Abuse/Abuse001_x264__5.npy`와 짝을 이룬다. `--video-root` 아래의 하위 폴더는 재귀 검색하며 동일한 영상 ID가 둘 이상 발견되면 경로를 확인하도록 오류를 표시한다.

```sh
python -m pip install pytorchvideo==0.1.5
python tools/extract_features.py --backbone i3d --video-root data/videos --output-root data/features/i3d
python tools/extract_features.py --backbone videoswin --video-root data/videos --output-root data/features/videoswin_b
```

원본 영상은 데이터셋 제공처에서 준비한다. 추출기는 첫 실행 시 백본의 공식 사전학습 가중치가 로컬 캐시에 없으면 내려받는다. 특정 영상만 확인하려면 `--video-id RoadAccidents021_x264`처럼 확장자 없는 ID를 지정한다. 장치 기본값은 CUDA가 사용 가능하면 CUDA, 그 외에는 CPU다. CUDA에서는 기본으로 혼합 정밀도를 사용하며 `--no-amp`로 끌 수 있다. [UCF-Crime 데이터셋](https://www.crcv.ucf.edu/projects/real-world/)

| 기준 | I3D-R50 | Video Swin-B |
| --- | --- | --- |
| 입력 구간 (프레임 번호는 0부터) | 구간 `i`의 `16i … 16i+15`에서 균등 간격 8장 | 같은 구간 중심의 `16i−8 … 16i+23` 연속 32장 |
| 공간 전처리 | 짧은 변 256, 중앙 256×256 | 짧은 변 256, 중앙 224×224 |
| RGB 정규화 | 채널별 평균 0.45, 표준편차 0.225 | 평균 (0.485, 0.456, 0.406), 표준편차 (0.229, 0.224, 0.225) |
| 출력 | Kinetics-400 I3D-R50 마지막 잔차 블록의 전역 평균 풀링, `(T, 2048)` | `Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1` 분류기 직전 특징, `(T, 1024)` |

두 방식 모두 `__5.npy`에서 읽은 `T`를 사용한다. 경계 밖까지 이어지는 창은 첫 프레임 또는 마지막 프레임을 반복해 채운다. 영상 디코딩이 중간에 끊기거나 영상 길이가 CLIP 구간 수와 맞지 않으면 해당 파일의 추출을 중단한다.

## 3. CLIP과 3D 특징 연결

```sh
python tools/prepare_lists.py pairs --backbone i3d --clip-root data/features/clip_official --feature-root data/features/i3d
python tools/prepare_lists.py pairs --backbone videoswin --clip-root data/features/clip_official --feature-root data/features/videoswin_b
```

각 영상의 3D 특징은 카테고리별로 `<영상 ID>.npy` 한 개씩 저장한다. 따라서 같은 영상의 CLIP 학습 crop 10개가 하나의 3D 시퀀스를 공유한다. 예를 들어 `Abuse/Abuse001_x264__0.npy`부터 `__9.npy`까지 모두 `i3d/Abuse/Abuse001_x264.npy`에 연결된다. 생성기는 파일 존재 여부, 특징 차원, CLIP과 3D의 시간 길이, crop 중복과 누락을 검사한다.

생성 파일은 `ucf_full_CLIP_I3D_{train,test}.csv`와 `ucf_full_CLIP_VideoSwin_{train,test}.csv`다. 열은 `video_id`, `clip_path`, `i3d_path` 또는 `videoswin_path`, `label`, `crop_index`, `temporal_length`, `clip_dim`, `i3d_dim` 또는 `videoswin_dim`이다. CSV에는 생성한 컴퓨터의 절대 경로가 들어간다. 데이터 위치가 바뀌면 해당 목록을 다시 생성한다.
