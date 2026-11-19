"""Rebuild the final UCF-Crime feature lists from the official VadCLIP split."""

import argparse
import csv
import os
import re
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np


OFFICIAL_FILES = {"train": "ucf_CLIP_rgb.csv", "test": "ucf_CLIP_rgbtest.csv"}
OFFICIAL_VIDEO_COUNTS = {"train": 1610, "test": 290}
BACKBONES = {"i3d": ("I3D", 2048), "videoswin": ("VideoSwin", 1024)}
LOCAL_LISTS = Path("data/lists")
FEATURE_NAME = re.compile(r"(.+)__(\d+)\.npy")


def clip_list_name(split):
    return f"ucf_official_CLIP_{split}_local.csv"


def paired_list_name(backbone, split):
    name, _ = BACKBONES[backbone]
    return f"ucf_full_CLIP_{name}_{split}.csv"


def read_rows(path, required):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(required).issubset(reader.fieldnames or ()):
            raise ValueError(f"{path} must contain columns {', '.join(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty split: {path}")
    return rows


def video_crop(path):
    match = FEATURE_NAME.fullmatch(path.name)
    if not match:
        raise ValueError(f"Invalid CLIP crop name: {path.name}")
    return match.group(1), int(match.group(2))


def feature_shape(path, width):
    data = np.load(path, mmap_mode="r", allow_pickle=False)
    if data.ndim != 2 or data.shape[0] < 1 or data.shape[1] != width:
        raise ValueError(f"{path}: expected (T, {width}) with T > 0, got {data.shape}")
    if not np.issubdtype(data.dtype, np.floating):
        raise ValueError(f"{path}: expected floating-point features, got {data.dtype}")
    return data.shape


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                     dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def map_official_path(raw, clip_root):
    # The upstream manifest contains the authors' absolute path. Keep category and filename.
    parts = PurePosixPath(raw.replace("\\", "/")).parts
    if len(parts) < 2 or any(p in ("", ".", "..", "/") or ":" in p for p in parts[-2:]):
        raise ValueError(f"Invalid official feature path: {raw!r}")
    return clip_root.joinpath(*parts[-2:])


def prepare(args):
    clip_root = args.clip_root.resolve()
    outputs = {}
    split_ids = {}
    for split in OFFICIAL_FILES:
        paired = args.action == "pairs"
        source = (args.list_dir / clip_list_name(split) if paired
                  else args.official_list_dir / OFFICIAL_FILES[split])
        records = []
        videos = {}
        motion_shapes = {}
        rows = read_rows(source, ("path", "label"))
        if paired:
            reference = read_rows(args.official_list_dir / OFFICIAL_FILES[split], ("path", "label"))
            official_order = [(map_official_path(r["path"], clip_root), r["label"]) for r in reference]
            if [(Path(r["path"]), r["label"]) for r in rows] != official_order:
                raise ValueError(f"{source} differs from the official {split} split; rebuild the CLIP lists")
        for row in rows:
            path = Path(row["path"]) if paired else map_official_path(row["path"], clip_root)
            relative = path.relative_to(clip_root)
            if len(relative.parts) != 2:
                raise ValueError(f"Expected <category>/<crop>.npy under {clip_root}: {path}")
            video_id, crop = video_crop(path)
            label = row["label"].strip()
            if not label:
                raise ValueError(f"Missing label: {path}")
            length, width = feature_shape(path, 512)
            key = (relative.parent, label, length)
            previous, crops = videos.setdefault(video_id, (key, set()))
            if previous != key or crop in crops:
                raise ValueError(f"Inconsistent category, label, length or duplicate crop: {path}")
            crops.add(crop)
            if paired:
                _, dim = BACKBONES[args.backbone]
                motion = args.feature_root / relative.parent / f"{video_id}.npy"
                if motion not in motion_shapes:
                    motion_shapes[motion] = feature_shape(motion, dim)
                if motion_shapes[motion] != (length, dim):
                    raise ValueError(f"Temporal mismatch: {path} vs {motion}: "
                                     f"{length} vs {motion_shapes[motion][0]}")
                records.append({"video_id": video_id, "clip_path": str(path),
                                f"{args.backbone}_path": str(motion), "label": label,
                                "crop_index": crop, "temporal_length": length,
                                "clip_dim": width, f"{args.backbone}_dim": dim})
            else:
                records.append({"path": str(path), "label": label})
        expected = set(range(10)) if split == "train" else {5}
        missing = next((vid for vid, (_, crops) in videos.items() if crops != expected), None)
        if missing:
            raise ValueError(f"{split}: unexpected or incomplete crops for {missing}: {videos[missing][1]}")
        if len(videos) != OFFICIAL_VIDEO_COUNTS[split]:
            raise ValueError(f"Expected {OFFICIAL_VIDEO_COUNTS[split]} videos in {split}, found {len(videos)}")
        split_ids[split] = set(videos)
        name = paired_list_name(args.backbone, split) if paired else clip_list_name(split)
        outputs[args.list_dir / name] = records
    if split_ids["train"] & split_ids["test"]:
        raise ValueError("A video occurs in both official splits")
    # Validate all inputs before changing either output file.
    for path, rows in outputs.items():
        write_csv(path, rows)
        print(f"{path}: {len(rows)} rows")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    clip = commands.add_parser("clip", help="Remap the official full-UCF CLIP split")
    pairs = commands.add_parser("pairs", help="Join 3D features to the verified CLIP split")
    pairs.add_argument("--backbone", choices=BACKBONES, required=True)
    pairs.add_argument("--feature-root", type=Path, required=True)
    for command in (clip, pairs):
        command.add_argument("--official-list-dir", type=Path, default=Path("vendor/VadCLIP/list"))
        command.add_argument("--clip-root", type=Path, required=True)
        command.add_argument("--list-dir", type=Path, default=LOCAL_LISTS)
    args = parser.parse_args()
    if args.action == "pairs":
        args.feature_root = args.feature_root.resolve()
    try:
        prepare(args)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Feature list error: {error}\n")


if __name__ == "__main__":
    main()
