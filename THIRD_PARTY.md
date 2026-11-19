# 원본 코드와 연구 기여

이 연구의 기준 모델은 Peng Wu 외의 **VadCLIP: Adapting Vision-Language Models for Weakly Supervised Video Anomaly Detection**(AAAI 2024)이다. CLIP 기반 시각·언어 모델, LGT-Adapter, 프롬프트, 두 예측 분기와 UCF-Crime의 평가 기준은 공식 구현을 사용한다. [VadCLIP 저장소와 논문 정보](https://github.com/nwpu-zxr/VadCLIP/tree/c41067f07d252efcda18008bea367886070c33b0)

연구 코드는 공식 CLIP 특징과 같은 시간축의 I3D-R50 및 Video Swin-B 특징을 추출하고, 이를 투영층과 게이트를 통해 VadCLIP 입력에 결합한다. 융합층만 학습하는 조건과 기존 VadCLIP 모듈도 함께 미세조정하는 조건을 비교한다. 공식 코드를 바탕으로 작성한 융합·학습·평가 파일은 이 저장소의 `src/`에 있다. 공식 저장소의 Apache License 2.0 본문은 [라이선스 파일](licenses/VadCLIP-Apache-2.0.txt)에 보존했다.

`patches/clip-packaging.patch`는 실험 환경에서 사용한 CLIP 로더의 버전 확인 import 수정이다. `patches/graph-device.patch`는 비교 실험을 정리하면서 추가했다. 거리 인접행렬을 모델과 같은 장치에 생성해 CPU 실행도 지원한다. [OpenAI CLIP](https://github.com/openai/CLIP)과 VadCLIP의 기반 모델 및 가중치는 각각의 제공처를 따른다. I3D-R50은 [PyTorchVideo](https://github.com/facebookresearch/pytorchvideo), Video Swin-B는 [torchvision](https://github.com/pytorch/vision)의 사전학습 모델을 사용했다.
