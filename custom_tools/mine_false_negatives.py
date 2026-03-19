"""
Pseudo-label mining script for LVIS false negative analysis.

Runs a trained OV-DQUO model on the LVIS *training* set,
finds high-confidence predictions that don't overlap with any GT annotation,
and reports statistics + saves visualizations.

Usage (single GPU):
    python custom_tools/mine_false_negatives.py \
        -c config/OV_LVIS/OVDQUO_ViTB16.py \
        --resume logs/r50_ovlvis/checkpoint0034.pth \
        --output_dir logs/false_neg_analysis \
        --max_images 2000 \
        --score_thresh 0.3 \
        --iou_thresh 0.1 \
        --num_vis 20
"""
import argparse
import json
import os
import sys
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader, SequentialSampler
from torchvision.ops import box_iou
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from util.slconfig import SLConfig
import util.misc as utils
import datasets
from datasets import build_dataset
from engine import convert_to_xywh


def get_args():
    parser = argparse.ArgumentParser("False-negative mining on LVIS train set")
    parser.add_argument("--config_file", "-c", type=str, required=True)
    parser.add_argument("--resume", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument("--output_dir", type=str, default="logs/false_neg_analysis")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_images", type=int, default=2000,
                        help="Max number of training images to scan (0=all)")
    parser.add_argument("--score_thresh", type=float, default=0.3,
                        help="Min score for a prediction to be considered a potential false negative")
    parser.add_argument("--iou_thresh", type=float, default=0.1,
                        help="Max IoU with any GT box for a prediction to be a false negative candidate")
    parser.add_argument("--num_vis", type=int, default=20,
                        help="Number of example images to visualize")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def build_model(args, cfg_args):
    from models.registry import MODULE_BUILD_FUNCS
    build_func = MODULE_BUILD_FUNCS.get(cfg_args.modelname)
    model, criterion, postprocessors = build_func(cfg_args)
    return model, criterion, postprocessors


def draw_boxes(image, boxes, labels, colors, category_names=None, scores=None):
    """Draw bounding boxes on a PIL image."""
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    for i, (box, label, color) in enumerate(zip(boxes, labels, colors)):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        if category_names is not None and label < len(category_names):
            name = category_names[label]
        else:
            name = str(label)
        if scores is not None:
            name = f"{name} {scores[i]:.2f}"
        draw.text((x1, y1 - 16), name, fill=color, font=font)
    return image


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "vis"), exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    # ── Load config ──
    cfg = SLConfig.fromfile(args.config_file)
    cfg_dict = cfg._cfg_dict.to_dict()
    cfg_args = argparse.Namespace(**cfg_dict)
    # Add required fields
    cfg_args.device = args.device
    cfg_args.distributed = False
    cfg_args.rank = 0
    cfg_args.world_size = 1
    cfg_args.debug = False
    cfg_args.eval = True
    cfg_args.amp = True
    cfg_args.num_workers = args.num_workers
    cfg_args.analysis = False

    # ── Build model ──
    print("Building model...")
    model, criterion, postprocessors = build_model(args, cfg_args)
    model.to(device)
    model.eval()

    # Load checkpoint
    print(f"Loading checkpoint from {args.resume}")
    ckpt = torch.load(args.resume, map_location="cpu")
    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "ema_model" in ckpt:
        state_dict = ckpt["ema_model"]
    else:
        state_dict = ckpt
    # Remove 'module.' prefix if present
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys: {missing[:5]}... ({len(missing)} total)")
    if unexpected:
        print(f"Unexpected keys: {unexpected[:5]}... ({len(unexpected)} total)")
    print("Checkpoint loaded.")

    # ── Build TRAINING dataset (not val!) ──
    print("Building training dataset...")
    dataset_train = build_dataset(image_set="train", args=cfg_args)
    category_list = dataset_train.category_list
    label2catid = dataset_train.label2catid
    num_categories = len(category_list)
    print(f"  {len(dataset_train)} images, {num_categories} categories")

    # Subsample if needed
    num_images = len(dataset_train)
    if args.max_images > 0 and args.max_images < num_images:
        indices = random.sample(range(num_images), args.max_images)
        num_images = args.max_images
    else:
        indices = list(range(num_images))

    # ── Scan for false negatives ──
    print(f"\nScanning {num_images} images (score_thresh={args.score_thresh}, iou_thresh={args.iou_thresh})...")

    # Statistics
    total_predictions = 0
    total_fn_candidates = 0
    fn_per_category = defaultdict(int)  # category_label -> count
    fn_images = []  # list of (image_idx, image_id, num_fn, fn_details)
    images_with_fn = 0

    collate_fn = utils.CollateFn(cfg_args.resolution) if "EVA" in cfg_args.backbone else utils.collate_fn

    t0 = time.time()
    with torch.no_grad():
        for scan_i, idx in enumerate(indices):
            if (scan_i + 1) % 200 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (scan_i + 1) * (num_images - scan_i - 1)
                print(f"  [{scan_i+1}/{num_images}] "
                      f"fn_candidates={total_fn_candidates} in {images_with_fn} images  "
                      f"ETA: {eta/60:.1f}min")

            # Load single image
            try:
                img, target = dataset_train[idx]
            except Exception:
                continue

            # Collate for single image
            samples, targets = collate_fn([(img, target)])
            samples = samples.to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in targets[0].items()}]

            # Forward
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(samples, categories=category_list, targets=targets)

            # Postprocess
            orig_size = targets[0]["orig_size"].unsqueeze(0)
            results = postprocessors["bbox"](outputs, orig_size)[0]

            pred_boxes = results["boxes"]    # [K, 4] xyxy in original coords
            pred_scores = results["scores"]  # [K]
            pred_labels = results["labels"]  # [K]

            # Filter by score threshold
            high_conf = pred_scores >= args.score_thresh
            pred_boxes = pred_boxes[high_conf]
            pred_scores = pred_scores[high_conf]
            pred_labels = pred_labels[high_conf]

            total_predictions += len(pred_boxes)

            if len(pred_boxes) == 0:
                continue

            # Get GT boxes (in original image coords)
            gt_boxes = targets[0]["boxes"]  # normalized cxcywh after dataset transforms
            h, w = targets[0]["orig_size"]
            # Convert cxcywh -> xyxy, then scale to original image coords
            cx, cy, bw, bh = gt_boxes.unbind(-1)
            gt_boxes_scaled = torch.stack([cx - bw / 2, cy - bh / 2,
                                           cx + bw / 2, cy + bh / 2], dim=-1)
            gt_boxes_scaled[:, 0::2] *= w
            gt_boxes_scaled[:, 1::2] *= h
            gt_labels = targets[0]["labels"]

            if len(gt_boxes_scaled) == 0:
                # No GT → all predictions are potential FN
                fn_mask = torch.ones(len(pred_boxes), dtype=torch.bool)
            else:
                # Compute IoU between predictions and GT
                ious = box_iou(pred_boxes.cpu(), gt_boxes_scaled.cpu())  # [num_pred, num_gt]
                max_iou, _ = ious.max(dim=1)  # [num_pred]
                fn_mask = max_iou < args.iou_thresh

            num_fn = fn_mask.sum().item()
            if num_fn > 0:
                images_with_fn += 1
                total_fn_candidates += num_fn

                fn_labels = pred_labels[fn_mask].cpu().tolist()
                fn_scores = pred_scores[fn_mask].cpu().tolist()
                fn_boxes_list = pred_boxes[fn_mask].cpu().tolist()

                for lab in fn_labels:
                    fn_per_category[lab] += 1

                image_id = targets[0]["image_id"].item()
                fn_images.append({
                    "image_idx": idx,
                    "image_id": image_id,
                    "num_fn": num_fn,
                    "num_gt": len(gt_boxes_scaled),
                    "num_pred_highconf": len(pred_boxes),
                    "fn_details": [
                        {"label": lab, "score": sc, "box": bx, "category": category_list[lab]}
                        for lab, sc, bx in zip(fn_labels, fn_scores, fn_boxes_list)
                    ],
                })

    elapsed = time.time() - t0

    # ── Report ──
    print("\n" + "=" * 60)
    print("FALSE NEGATIVE MINING RESULTS")
    print("=" * 60)
    print(f"Images scanned:        {num_images}")
    print(f"Images with FN:        {images_with_fn} ({100*images_with_fn/max(num_images,1):.1f}%)")
    print(f"Total high-conf preds: {total_predictions}")
    print(f"FN candidates:         {total_fn_candidates}")
    print(f"FN rate:               {100*total_fn_candidates/max(total_predictions,1):.2f}% of high-conf preds")
    print(f"Time:                  {elapsed/60:.1f} min")

    # Top categories with most false negatives
    print(f"\nTop-20 categories with most FN candidates:")
    sorted_cats = sorted(fn_per_category.items(), key=lambda x: -x[1])[:20]
    for rank, (lab, count) in enumerate(sorted_cats, 1):
        cat_name = category_list[lab] if lab < len(category_list) else f"label_{lab}"
        print(f"  {rank:2d}. {cat_name:30s}  count={count}")

    # ── Save statistics ──
    stats = {
        "config": args.config_file,
        "checkpoint": args.resume,
        "score_thresh": args.score_thresh,
        "iou_thresh": args.iou_thresh,
        "num_images_scanned": num_images,
        "num_images_with_fn": images_with_fn,
        "total_highconf_predictions": total_predictions,
        "total_fn_candidates": total_fn_candidates,
        "fn_rate_percent": 100 * total_fn_candidates / max(total_predictions, 1),
        "top_fn_categories": [
            {"label": lab, "category": category_list[lab], "count": cnt}
            for lab, cnt in sorted_cats
        ],
    }
    stats_path = os.path.join(args.output_dir, "fn_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to {stats_path}")

    # Save per-image FN details
    details_path = os.path.join(args.output_dir, "fn_images.json")
    with open(details_path, "w") as f:
        json.dump(fn_images, f, indent=2)
    print(f"Per-image details saved to {details_path}")

    # ── Visualize ──
    if args.num_vis > 0 and len(fn_images) > 0:
        print(f"\nGenerating {min(args.num_vis, len(fn_images))} visualizations...")
        # Pick images with most FN candidates
        fn_images_sorted = sorted(fn_images, key=lambda x: -x["num_fn"])
        vis_samples = fn_images_sorted[:args.num_vis]

        # We need to reload images from disk for visualization
        # Get image root from config
        lvis_path = cfg_args.lvis_path
        # LVIS images are in COCO train2017
        img_root = os.path.join(lvis_path, "Images")

        for vi, info in enumerate(vis_samples):
            image_id = info["image_id"]
            # LVIS image filename: 000000XXXXXX.jpg
            img_filename = f"{image_id:012d}.jpg"
            img_path = os.path.join(img_root, "train2017", img_filename)

            if not os.path.exists(img_path):
                print(f"  Warning: {img_path} not found, skipping")
                continue

            pil_img = Image.open(img_path).convert("RGB")

            # Reload GT for this image
            idx = info["image_idx"]
            try:
                _, target = dataset_train[idx]
            except Exception:
                continue

            # Draw GT boxes in green
            gt_img = pil_img.copy()
            # target boxes are normalized cxcywh [0,1], convert to xyxy and scale
            w_img, h_img = pil_img.size
            gt_boxes_vis = target["boxes"].clone()
            # cxcywh -> xyxy
            cx, cy, bw, bh = gt_boxes_vis.unbind(-1)
            gt_boxes_vis = torch.stack([cx - bw / 2, cy - bh / 2,
                                        cx + bw / 2, cy + bh / 2], dim=-1)
            gt_boxes_vis[:, 0::2] *= w_img
            gt_boxes_vis[:, 1::2] *= h_img
            gt_labels_vis = target["labels"].tolist()

            gt_img = draw_boxes(
                gt_img,
                gt_boxes_vis.tolist(),
                gt_labels_vis,
                ["green"] * len(gt_labels_vis),
                category_names=category_list,
            )

            # Draw FN candidates in red on the same image
            fn_details = info["fn_details"]
            fn_boxes_vis = [d["box"] for d in fn_details]
            fn_labels_vis = [d["label"] for d in fn_details]
            fn_scores_vis = [d["score"] for d in fn_details]

            combined_img = draw_boxes(
                gt_img,
                fn_boxes_vis,
                fn_labels_vis,
                ["red"] * len(fn_labels_vis),
                category_names=category_list,
                scores=fn_scores_vis,
            )

            save_path = os.path.join(args.output_dir, "vis",
                                     f"fn_{vi:03d}_img{image_id}_nfn{info['num_fn']}.jpg")
            combined_img.save(save_path)

        print(f"  Visualizations saved to {os.path.join(args.output_dir, 'vis/')}")
        print("  GREEN = GT annotations, RED = FN candidates (high-conf pred, no GT overlap)")

    print("\nDone!")


if __name__ == "__main__":
    main()
