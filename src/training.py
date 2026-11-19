"""공식 UCF 체크포인트에서 시작한 gate-only 및 joint 학습."""

import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from utils.tools import get_batch_label, get_prompt_text

from datasets import FeatureDataset, LABEL_MAP
from evaluate import evaluate


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_weights(path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "state_dict" in state:
        state = state["state_dict"]
    if state and all(name.startswith("module.") for name in state):
        state = {name.removeprefix("module."): value for name, value in state.items()}
    return state


def load_baseline(model, path, device):
    state = read_weights(path, device)
    expected = {name for name in model.state_dict() if model.is_fusion_parameter(name)}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or set(missing) != expected:
        raise ValueError(f"Expected official baseline weights; unexpected={unexpected}, missing={missing}")


def set_training_stage(model, gate_only, warmup=False):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(model.is_fusion_parameter(name) or
                                 (not gate_only and not warmup and not name.startswith("clipmodel.")))
    if gate_only:
        model.eval()
        model.projection.train()
        model.fusion_gate.train()
        model.fusion_dropout.train()
    else:
        # Joint의 첫 epoch에도 원래 실험과 같이 VadCLIP 전체의 dropout을 켠다.
        model.train()


def make_optimizer(model, gate_only):
    fusion = [p for name, p in model.named_parameters() if model.is_fusion_parameter(name)]
    if gate_only:
        return torch.optim.AdamW(fusion, lr=1e-4)
    base = [p for name, p in model.named_parameters()
            if not name.startswith("clipmodel.") and not model.is_fusion_parameter(name)]
    return torch.optim.AdamW([{"params": base, "lr": 2e-5}, {"params": fusion, "lr": 1e-4}])


def batch_loss(model, normal, anomaly, prompts, device):
    clip = torch.cat([normal[0], anomaly[0]]).to(device)
    aux = torch.cat([normal[1], anomaly[1]]).to(device)
    lengths = torch.cat([normal[3], anomaly[3]]).to(device)
    labels = get_batch_label(list(normal[2]) + list(anomaly[2]), prompts, LABEL_MAP).to(device)
    text, coarse, aligned = model(clip, aux, None, prompts, lengths)

    # 각 영상에서 유효 길이에 따라 상위 k개 점수를 선택해 영상 단위 약지도 손실을 구한다.
    binary_scores, class_logits = [], []
    for i, n in enumerate(lengths.tolist()):
        k = min(n, n // 16 + 1)
        c = coarse[i, :n, 0].sigmoid()
        binary_scores.append(c.topk(k, dim=0).values.mean())
        class_logits.append(aligned[i, :n].topk(k, dim=0).values.mean(dim=0))
    binary_loss = F.binary_cross_entropy(torch.stack(binary_scores), 1.0 - labels[:, 0])
    distribution = labels / labels.sum(dim=1, keepdim=True).clamp_min(1e-8)
    class_loss = -(distribution * torch.stack(class_logits).log_softmax(dim=1)).sum(dim=1).mean()
    text_alignment = (F.normalize(text[1:], dim=-1) @ F.normalize(text[0], dim=-1)).abs().mean()
    return binary_loss + class_loss + 0.1 * text_alignment


def train(model, train_rows, test_loader, ground_truth, args, device):
    gate_only = args.experiment.endswith("-gate-only")
    load_baseline(model, args.baseline_weights, device)
    optimizer = make_optimizer(model, gate_only)
    scheduler = MultiStepLR(optimizer, milestones=[4, 8], gamma=0.1)
    loaders = [
        DataLoader(FeatureDataset(train_rows, args.backbone, normal=normal),
                   batch_size=args.batch_size, shuffle=True, drop_last=True,
                   num_workers=0, pin_memory=device == "cuda")
        for normal in (True, False)
    ]
    if min(map(len, loaders)) <= args.eval_interval:
        raise ValueError("Insufficient balanced batches for the requested evaluation interval")
    prompts = get_prompt_text(LABEL_MAP)
    best_auc = 0.0
    best = args.output / "best.pth"
    checkpoint = args.output / "checkpoint.pth"
    args.output.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        set_training_stage(model, gate_only, warmup=epoch == 0)
        # zip stops when the smaller of the normal and abnormal loaders is exhausted.
        for step, (normal, anomaly) in enumerate(zip(*loaders)):
            loss = batch_loss(model, normal, anomaly, prompts, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step == 0 or step % args.eval_interval != 0:
                continue
            metrics = evaluate(model, test_loader, ground_truth, device)
            auc = float(metrics["C_Branch_AUC"])
            print(f"epoch={epoch+1} step={step} C-Branch AUC={auc:.6f}", flush=True)
            if auc > best_auc:
                best_auc = auc
                torch.save({
                    "epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(), "best_auc": best_auc, "ap": best_auc,
                }, checkpoint)
                torch.save(model.state_dict(), best)
            set_training_stage(model, gate_only, warmup=epoch == 0)
        scheduler.step()
        # 원래 실험은 epoch 경계에서 모델만 최고 C-Branch AUC 시점으로 복원한다.
        if checkpoint.is_file():
            model.load_state_dict(read_weights(checkpoint, device))
    if not best.is_file():
        raise RuntimeError("Training did not produce a best checkpoint")
    print(f"best C-Branch AUC={best_auc:.6f} | {best}")
