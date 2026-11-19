"""Run the five UCF-Crime comparisons from the final report."""

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ("baseline", "i3d-gate-only", "i3d-joint", "swin-gate-only", "swin-joint")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "eval"))
    parser.add_argument("experiment", choices=EXPERIMENTS)
    parser.add_argument("--baseline-weights", type=Path,
                        help="Official model_ucf.pth (required for baseline evaluation and fusion training)")
    parser.add_argument("--weights", type=Path, help="Fusion checkpoint for evaluation (default: <output>/best.pth)")
    parser.add_argument("--lists-dir", type=Path, default=ROOT / "data" / "lists")
    parser.add_argument("--upstream", type=Path, default=ROOT / "vendor" / "VadCLIP",
                        help="Pinned VadCLIP checkout containing src/ and list/")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for each of the normal and anomaly loaders")
    parser.add_argument("--eval-interval", type=int, default=10, help="Evaluate every N updates after step zero")
    parser.add_argument("--seed", type=int, default=234)
    args = parser.parse_args()
    args.backbone = args.experiment.split("-")[0]
    args.output = args.output_root / args.experiment
    if args.action == "train" and args.experiment == "baseline":
        parser.error("baseline uses the published weights; run 'eval baseline'")
    if args.action == "train" or args.experiment == "baseline":
        if args.weights:
            parser.error("--weights applies to fusion evaluation only")
        if not args.baseline_weights or not args.baseline_weights.is_file():
            parser.error("supply the official checkpoint using --baseline-weights")
    else:
        args.weights = args.weights or args.output / "best.pth"
        if not args.weights.is_file():
            parser.error(f"fusion checkpoint not found: {args.weights}")
    if args.action == "train" and any((args.output / name).exists() for name in ("checkpoint.pth", "best.pth")):
        parser.error(f"training outputs already exist in {args.output}; use a fresh --output-root")
    if min(args.epochs, args.batch_size, args.eval_interval) <= 0:
        parser.error("--epochs, --batch-size and --eval-interval must be positive")
    if not (args.upstream / "src" / "model.py").is_file():
        parser.error(f"VadCLIP source not found at {args.upstream}; prepare the pinned upstream checkout")
    return args


def main():
    args = parse_args()
    sys.path[:0] = [str(ROOT / "src"), str(args.upstream.resolve() / "src")]
    import torch
    from torch.utils.data import DataLoader
    from datasets import FeatureDataset, load_official_split
    from evaluate import evaluate, read_ground_truth, save_metrics
    from fusion import build_model
    from training import read_weights, setup_seed, train

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    setup_seed(args.seed)
    reference = args.upstream / "list"

    def split(mode):
        if args.backbone == "baseline":
            manifest = f"ucf_official_CLIP_{mode}_local.csv"
        else:
            stream = {"i3d": "I3D", "swin": "VideoSwin"}[args.backbone]
            manifest = f"ucf_full_CLIP_{stream}_{mode}.csv"
        official = "ucf_CLIP_rgb.csv" if mode == "train" else "ucf_CLIP_rgbtest.csv"
        return load_official_split(args.lists_dir / manifest, reference / official, args.backbone)

    test_rows = split("test")
    train_rows = split("train") if args.action == "train" else None
    loader = DataLoader(FeatureDataset(test_rows, args.backbone, test=True), batch_size=1,
                        shuffle=False, num_workers=0, pin_memory=device == "cuda")
    ground_truth = read_ground_truth(reference, len(test_rows))
    model = build_model(args.backbone, device)
    if args.action == "train":
        train(model, train_rows, loader, ground_truth, args, device)
    else:
        weights = args.baseline_weights if args.experiment == "baseline" else args.weights
        model.load_state_dict(read_weights(weights, device), strict=True)
        metrics = evaluate(model, loader, ground_truth, device)
        save_metrics(args.output / "evaluation.csv", args.experiment, metrics)


if __name__ == "__main__":
    main()
