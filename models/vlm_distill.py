"""VLM soft target distillation for OV-DQUO.

Loads pre-computed VLM soft targets and provides KL distillation loss
for open-vocabulary object detection.
"""

import logging
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

OVCOCO_65_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "kite", "skateboard", "surfboard", "bottle", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "pizza", "donut", "cake", "chair", "couch",
    "bed", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "microwave", "oven", "toaster",
    "sink", "refrigerator", "book", "clock", "vase", "scissors", "toothbrush",
]


def _torch_load_compat(path, **kwargs):
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def _normalize_class_name(name):
    return str(name).replace("_", " ").strip().lower()


class VLMDistillLoss:
    """Computes KL distillation loss between region-text similarity and VLM soft targets."""

    def __init__(self, targets_path, num_classes=65, temperature=2.0, weight=1.0, class_names=None):
        logger.info(f"Loading VLM soft targets from {targets_path}")
        self.targets = _torch_load_compat(targets_path, map_location="cpu")
        self.num_classes = num_classes
        self.temperature = temperature
        self.weight = weight
        self.class_names = class_names or self._infer_class_names()
        self.class_to_idx = (
            {_normalize_class_name(name): idx for idx, name in enumerate(self.class_names)}
            if self.class_names is not None
            else None
        )
        logger.info(f"Loaded VLM targets for {len(self.targets)} images")

    def _infer_class_names(self):
        for vlm_list in self.targets.values():
            for target in vlm_list:
                if target is None:
                    continue
                sim_dist = target.get("similarity_distribution", None)
                if sim_dist is None:
                    continue
                if len(sim_dist) == len(OVCOCO_65_CLASSES):
                    return OVCOCO_65_CLASSES
                logger.warning(
                    "VLM target dimension %s has no known class-name mapping; "
                    "falling back to positional alignment.",
                    len(sim_dist),
                )
                return None
        return None

    def get_vlm_targets(self, image_ids):
        """Get VLM targets for a batch of images.

        Returns list of dicts, one per image. Each dict maps ann_id -> vlm_target.
        """
        batch_targets = []
        for img_id in image_ids:
            img_id = int(img_id)
            if img_id in self.targets:
                vlm_list = self.targets[img_id]
                vlm_by_ann = {
                    int(t["ann_id"]): t for t in vlm_list
                    if t is not None and "ann_id" in t
                }
                batch_targets.append(vlm_by_ann)
            else:
                batch_targets.append({})
        return batch_targets

    def compute_loss(self, roi_features, text_features, pred_boxes, targets,
                     indices, batch_vlm_targets, device, current_category_names=None):
        """Compute KL distillation loss.

        Args:
            roi_features: [batch, num_queries, clip_dim] normalized region features
            text_features: [num_classes, clip_dim] text embeddings
            pred_boxes: [batch, num_queries, 4] predicted boxes
            targets: list of target dicts (with ann_ids)
            indices: list of (query_idx, gt_idx) matched pairs
            batch_vlm_targets: output of get_vlm_targets()
            device: torch device

        Returns:
            loss_vlm_distill: scalar tensor
        """
        tau = self.temperature
        total_kl = torch.tensor(0.0, device=device)
        count = 0

        # region-text similarity: [batch, num_queries, num_classes]
        sim_matrix = roi_features @ text_features.t()

        for batch_idx, (query_indices, gt_indices) in enumerate(indices):
            vlm_by_ann = batch_vlm_targets[batch_idx]
            if not vlm_by_ann:
                continue

            target = targets[batch_idx]
            ann_ids = target.get("ann_ids", None)
            if ann_ids is None:
                continue

            for q_idx, g_idx in zip(query_indices, gt_indices):
                ann_id = int(ann_ids[g_idx].item()) if torch.is_tensor(ann_ids[g_idx]) else int(ann_ids[g_idx])
                vlm_t = vlm_by_ann.get(ann_id, None)
                if vlm_t is None:
                    continue

                sim_dist = vlm_t.get("similarity_distribution", None)
                if sim_dist is None:
                    continue

                sim_dist = torch.as_tensor(sim_dist, device=device, dtype=torch.float32)
                if self.class_to_idx is not None and current_category_names is not None:
                    mapped_dist = torch.zeros(sim_matrix.shape[-1], device=device, dtype=torch.float32)
                    for local_idx, class_name in enumerate(current_category_names[:sim_matrix.shape[-1]]):
                        teacher_idx = self.class_to_idx.get(_normalize_class_name(class_name), None)
                        if teacher_idx is not None and teacher_idx < sim_dist.shape[0]:
                            mapped_dist[local_idx] = sim_dist[teacher_idx]
                    sim_dist = mapped_dist
                elif sim_dist.shape[0] != sim_matrix.shape[-1]:
                    # Truncate or pad
                    if sim_dist.shape[0] > sim_matrix.shape[-1]:
                        sim_dist = sim_dist[:sim_matrix.shape[-1]]
                    else:
                        pad = torch.zeros(sim_matrix.shape[-1] - sim_dist.shape[0], device=device)
                        sim_dist = torch.cat([sim_dist, pad])

                if sim_dist.sum() < 1e-6:
                    continue

                # VLM soft target (already a probability distribution)
                vlm_soft = sim_dist / sim_dist.sum()

                # Model prediction
                pred_log_prob = (sim_matrix[batch_idx, q_idx] / tau).log_softmax(dim=-1)

                kl = F.kl_div(pred_log_prob, vlm_soft, reduction='sum')
                total_kl += kl
                count += 1

        if count > 0:
            total_kl = total_kl / count

        return total_kl * self.weight
