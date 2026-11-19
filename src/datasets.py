"""UCF-Crime 공식 crop 순서와 2D·3D 특징의 시간축을 검사한다."""

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from utils.tools import process_feat, process_split


MAX_LENGTH = 256
LABEL_MAP = {
    "Normal": "normal", "Abuse": "abuse", "Arrest": "arrest", "Arson": "arson",
    "Assault": "assault", "Burglary": "burglary", "Explosion": "explosion",
    "Fighting": "fighting", "RoadAccidents": "roadAccidents", "Robbery": "robbery",
    "Shooting": "shooting", "Shoplifting": "shoplifting", "Stealing": "stealing",
    "Vandalism": "vandalism",
}


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def basename(path):
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def load_official_split(manifest, reference, backbone):
    """대상 영상, 순서, crop, 레이블이 공식 목록과 같은지 특징을 읽기 전에 검증한다."""
    rows, expected = read_rows(manifest), read_rows(reference)
    if not rows or len(rows) != len(expected):
        raise ValueError(f"{manifest}: expected {len(expected)} official rows, found {len(rows)}")
    clip_key = "path" if backbone == "baseline" else "clip_path"
    aux_key = {"i3d": "i3d_path", "swin": "videoswin_path"}.get(backbone)
    for number, (row, official) in enumerate(zip(rows, expected), start=2):
        name, label = basename(row[clip_key]), row["label"]
        if (name, label) != (basename(official["path"]), official["label"]) or label not in LABEL_MAP:
            raise ValueError(f"{manifest}:{number}: official feature order or label mismatch ({name}, {label})")
        if aux_key:
            video_id, crop = Path(name).stem.rsplit("__", 1)
            if (row["video_id"] != video_id or int(row["crop_index"]) != int(crop)
                    or Path(basename(row[aux_key])).stem != video_id):
                raise ValueError(f"{manifest}:{number}: auxiliary video or crop mismatch")
    return rows


class FeatureDataset(Dataset):
    def __init__(self, rows, backbone, *, test=False, normal=False):
        self.rows = rows if test else [r for r in rows if (r["label"] == "Normal") == normal]
        self.backbone = backbone
        self.test = test

    def __len__(self):
        return len(self.rows)

    @staticmethod
    def load_feature(path, dimension):
        feature = np.load(path, allow_pickle=False)
        if feature.ndim != 2 or feature.shape[1] != dimension or not len(feature):
            raise ValueError(f"{path}: expected nonempty [T,{dimension}], found {feature.shape}")
        return feature.astype(np.float32, copy=False)

    def __getitem__(self, index):
        row = self.rows[index]
        clip_key = "path" if self.backbone == "baseline" else "clip_path"
        features = [self.load_feature(row[clip_key], 512)]
        if self.backbone != "baseline":
            key, dim = ("i3d_path", 2048) if self.backbone == "i3d" else ("videoswin_path", 1024)
            features.append(self.load_feature(row[key], dim))
            if len(features[0]) != len(features[1]) or len(features[0]) != int(row["temporal_length"]):
                raise ValueError(f"Temporal alignment mismatch: {row[clip_key]} and {row[key]}")
        process = process_split if self.test else process_feat
        tensors = []
        for array in features:
            array, length = process(array, MAX_LENGTH)
            tensors.append(torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)))
        return (*tensors, row["label"], int(length))
