"""C/A-Branch 프레임 지표와 공식 구간 mAP를 같은 순서의 예측에서 계산한다."""

import csv

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from utils.tools import get_batch_mask, get_prompt_text
from utils.ucf_detectionMAP import getDetectionMAP

from datasets import LABEL_MAP, MAX_LENGTH


FEATURE_STRIDE = 16


def read_ground_truth(reference, video_count):
    frames = np.load(reference / "gt_ucf.npy", allow_pickle=False).reshape(-1)
    # 공식 저장소가 제공하는 두 가변 길이 배열만 pickle 사용을 허용한다.
    segments = np.load(reference / "gt_segment_ucf.npy", allow_pickle=True)
    labels = np.load(reference / "gt_label_ucf.npy", allow_pickle=True)
    if len(segments) != video_count or len(labels) != video_count:
        raise ValueError("Ground-truth video count differs from the official test manifest")
    if any(len(s) != len(l) for s, l in zip(segments, labels)) or not np.isin(frames, [0, 1]).all():
        raise ValueError("Invalid segment labels or frame ground truth")
    return frames, segments, labels


@torch.inference_mode()
def evaluate(model, loader, ground_truth, device):
    model.eval()
    prompts = get_prompt_text(LABEL_MAP)
    frame_c, frame_a, class_scores = [], [], []
    for *features, _, length in loader:
        n = int(length.item())
        lengths = torch.arange(0, n, MAX_LENGTH)
        lengths = (n - lengths).clamp(max=MAX_LENGTH)
        # process_split은 T가 정확히 256의 배수일 때 끝에 빈 청크를 하나 더 붙인다.
        chunks = [
            stream.reshape(-1, MAX_LENGTH, stream.shape[-1])[:len(lengths)].to(device)
            for stream in features
        ]
        mask = get_batch_mask(lengths, MAX_LENGTH).to(device)
        _, coarse, aligned = model(*chunks, mask, prompts, lengths)
        coarse = coarse.reshape(-1)[:n].sigmoid().cpu().numpy()
        aligned = aligned.reshape(-1, aligned.shape[-1])[:n].softmax(dim=-1)
        scores = aligned.cpu().numpy()
        frame_c.append(np.repeat(coarse, FEATURE_STRIDE))
        frame_a.append(np.repeat((1.0 - aligned[:, 0]).cpu().numpy(), FEATURE_STRIDE))
        class_scores.append(np.repeat(scores, FEATURE_STRIDE, axis=0))

    frames, segments, labels = ground_truth
    coarse, aligned = np.concatenate(frame_c), np.concatenate(frame_a)
    if len(frames) != len(coarse) or len(frames) != len(aligned):
        raise ValueError(f"Frame counts differ: GT={len(frames)}, C={len(coarse)}, A={len(aligned)}")

    thresholds_map, thresholds = getDetectionMAP(class_scores, segments, labels, excludeNormal=False)
    return {
        "C_Branch_AUC": roc_auc_score(frames, coarse),
        "C_Branch_AP": average_precision_score(frames, coarse),
        "A_Branch_AUC": roc_auc_score(frames, aligned),
        "A_Branch_AP": average_precision_score(frames, aligned),
        **{f"mAP@{threshold:.1f}": value for threshold, value in zip(thresholds, thresholds_map)},
        "average_mAP": float(np.mean(thresholds_map)),
    }


def save_metrics(path, experiment, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"model": experiment, **metrics}
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(record))
        writer.writeheader()
        writer.writerow(record)
    print(" | ".join(f"{key}={value:.6f}" for key, value in metrics.items()))
    print(path)
