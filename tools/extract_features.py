"""Extract temporally aligned I3D-R50 or Video Swin-B features for full UCF-Crime."""

import argparse
import os
import tempfile
from collections import deque
from pathlib import Path

import numpy as np

from prepare_lists import BACKBONES, LOCAL_LISTS, clip_list_name, feature_shape, read_rows, video_crop


def aligned_windows(frames, segments, *, swin):
    """Stream one window per 16-frame CLIP step, repeating real frames at video boundaries."""
    frames = iter(frames)
    first = next(frames, None)
    if first is None or segments < 1:
        raise ValueError("Empty video or CLIP sequence")
    width, left = (32, 8) if swin else (16, 0)
    window = deque([first] * (left + 1), maxlen=width)
    decoded = 1
    last = first
    for index in range(segments):
        for _ in range(width - len(window) if index == 0 else 16):
            frame = next(frames, None)
            if frame is not None:
                last = frame
                decoded += 1
            window.append(last)
        if decoded <= 16 * index:
            raise ValueError(f"Missing video frames for CLIP snippet {index} (decoded {decoded})")
        yield tuple(window)
    # Official CLIP arrays can use either floor(N/16) or ceil(N/16).
    for _ in frames:
        decoded += 1
        if decoded >= 16 * (segments + 1):
            break
    if segments not in (decoded // 16, (decoded + 15) // 16):
        raise ValueError(f"Video length {decoded} does not match {segments} CLIP snippets")


def decoded_frames(capture):
    import cv2

    declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            if declared > 0 and count < declared:
                raise ValueError(f"Video decoding stopped at frame {count} of {declared}")
            return
        count += 1
        yield frame


def preprocess(window, backbone):
    import cv2
    import torch

    if backbone == "i3d":
        window = [window[i] for i in np.linspace(0, 15, 8).round().astype(int)]
        crop_size, mean, std = 256, (0.45,) * 3, (0.225,) * 3
    else:
        crop_size, mean, std = 224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    mean, std = np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)
    images = []
    for bgr in window:
        height, width = bgr.shape[:2]
        short_side = min(height, width)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (round(width * 256 / short_side), round(height * 256 / short_side)),
                         interpolation=cv2.INTER_LINEAR)
        y, x = (rgb.shape[0] - crop_size) // 2, (rgb.shape[1] - crop_size) // 2
        center = rgb[y:y + crop_size, x:x + crop_size].astype(np.float32)
        images.append((center / 255.0 - mean) / std)
    return torch.from_numpy(np.ascontiguousarray(np.stack(images).transpose(3, 0, 1, 2)))


def load_backbone(name, device):
    import torch

    if name == "videoswin":
        from torchvision.models.video import Swin3D_B_Weights, swin3d_b

        model = swin3d_b(weights=Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1)
        model.head = torch.nn.Identity()
        return model.eval().to(device)

    from pytorchvideo.models.hub import i3d_r50

    model = i3d_r50(pretrained=True).eval().to(device)
    # Run the final residual stage and pool its spatiotemporal map before the classification head.
    trunk = torch.nn.Sequential(*list(model.blocks)[:-1])
    return torch.nn.Sequential(trunk, torch.nn.AdaptiveAvgPool3d(1), torch.nn.Flatten(1)).eval()


def extract(video, segments, backbone, model, device, batch_size, amp):
    import cv2
    import torch

    output = []
    batch = []
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video}")
        with torch.inference_mode():
            for index, window in enumerate(aligned_windows(decoded_frames(capture), segments,
                                                         swin=backbone == "videoswin")):
                batch.append(preprocess(window, backbone))
                if len(batch) == batch_size or index == segments - 1:
                    inputs = torch.stack(batch).to(device)
                    with torch.autocast(device_type=device.type, enabled=amp):
                        embedding = model(inputs)
                    output.append(embedding.float().cpu().numpy())
                    batch.clear()
    except ValueError as error:
        raise ValueError(f"{video}: {error}") from error
    finally:
        capture.release()
    result = np.concatenate(output)
    if result.shape != (segments, BACKBONES[backbone][1]) or not np.isfinite(result).all():
        raise ValueError(f"Invalid {backbone} output for {video}: {result.shape}")
    return result


def extraction_tasks(args):
    clips = {}
    for split in ("train", "test"):
        for row in read_rows(args.list_dir / clip_list_name(split), ("path", "label")):
            path = Path(row["path"])
            video_id, crop = video_crop(path)
            if crop == 5 and (not args.video_id or video_id in args.video_id):
                if video_id in clips:
                    raise ValueError(f"Duplicate center crop for {video_id}")
                clips[video_id] = path
    if args.video_id and (missing := set(args.video_id) - clips.keys()):
        raise ValueError(f"No CLIP feature for requested video: {sorted(missing)[0]}")
    if not clips:
        raise ValueError("No center crop found in CLIP manifests")
    videos = {}
    for path in args.video_root.rglob("*"):
        if path.suffix.lower() in (".mp4", ".avi", ".mkv", ".mov") and path.stem in clips:
            if path.stem in videos:
                raise ValueError(f"Ambiguous video ID: {path} and {videos[path.stem]}")
            videos[path.stem] = path
    if missing := clips.keys() - videos.keys():
        raise ValueError(f"{len(missing)} raw videos missing, including {sorted(missing)[0]}")
    for video_id, clip in clips.items():
        length, _ = feature_shape(clip, 512)
        yield videos[video_id], length, args.output_root / clip.parent.name / f"{video_id}.npy"


def save_feature(path, feature):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            np.save(handle, feature, allow_pickle=False)
        except BaseException:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=BACKBONES, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--list-dir", type=Path, default=LOCAL_LISTS)
    parser.add_argument("--video-id", action="append", help="Process only this video (repeatable)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    model = None
    tasks = list(extraction_tasks(args))
    for index, (video, length, path) in enumerate(tasks, start=1):
        if path.exists() and not args.overwrite:
            if feature_shape(path, BACKBONES[args.backbone][1])[0] != length:
                raise ValueError(f"{path} has the wrong temporal length; use --overwrite to replace it")
            continue
        if model is None:
            import torch

            device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                                  if args.device == "auto" else args.device)
            model = load_backbone(args.backbone, device)
        print(f"[{index}/{len(tasks)}] {video.name}: {length} snippets", flush=True)
        feature = extract(video, length, args.backbone, model, device, args.batch_size,
                          device.type == "cuda" and not args.no_amp)
        save_feature(path, feature)


if __name__ == "__main__":
    main()
