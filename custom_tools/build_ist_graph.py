"""
Convert the JSON category relationship graph into a PyTorch adjacency matrix
that can be directly loaded by the IST GAT module.

Output: pretrained/ovcoco_ist_adj.pt
  - adj: [65, 65] float tensor, adj[i][j] = strength score (j -> i edge)
  - cat_names: list of 65 category names in dataset order
  - base_mask: [65] bool tensor, True for base categories
  - novel_mask: [65] bool tensor, True for novel categories

Usage:
    python custom_tools/build_ist_graph.py
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# COCO category ID -> name mapping
COCO_CATS = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
    21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
    27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
    34: 'frisbee', 35: 'skis', 36: 'snowboard', 38: 'kite',
    41: 'skateboard', 42: 'surfboard', 44: 'bottle', 47: 'cup',
    48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana',
    53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot',
    59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
    65: 'bed', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
    75: 'remote', 76: 'keyboard', 78: 'microwave', 79: 'oven',
    80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
    86: 'vase', 87: 'scissors', 90: 'toothbrush'
}

OVCOCO_BASE_CATIDS = {
    70, 2, 53, 7, 73, 57, 4, 79, 62, 74, 9, 38, 20, 19, 54, 85, 72, 27,
    80, 51, 78, 15, 84, 55, 16, 59, 48, 34, 23, 86, 90, 50, 25, 31, 56,
    82, 75, 42, 3, 65, 52, 60, 35, 1, 8, 44, 33, 24,
}
OVCOCO_NOVEL_CATIDS = {
    28, 21, 47, 6, 76, 41, 18, 63, 32, 36, 81, 22, 61, 87, 5, 17, 49
}


def main():
    # Build ordered category list (sorted by COCO ID)
    all_ids = sorted(OVCOCO_BASE_CATIDS | OVCOCO_NOVEL_CATIDS)
    cat_names = [COCO_CATS[cid] for cid in all_ids]
    name_to_idx = {name: i for i, name in enumerate(cat_names)}
    num_cats = len(cat_names)

    base_mask = torch.tensor([cid in OVCOCO_BASE_CATIDS for cid in all_ids])
    novel_mask = torch.tensor([cid in OVCOCO_NOVEL_CATIDS for cid in all_ids])

    # Load relationship graph
    graph_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "pretrained", "ovcoco_category_graph.json"
    )
    with open(graph_path) as f:
        data = json.load(f)

    # Build adjacency matrix: adj[i][j] means node j sends info to node i
    adj = torch.zeros(num_cats, num_cats)

    for target_cat, neighbors in data["graph"].items():
        if target_cat not in name_to_idx:
            print(f"Warning: '{target_cat}' not in category list, skipping")
            continue
        target_idx = name_to_idx[target_cat]
        for neighbor_cat, info in neighbors.items():
            if neighbor_cat not in name_to_idx:
                print(f"Warning: neighbor '{neighbor_cat}' not in category list, skipping")
                continue
            neighbor_idx = name_to_idx[neighbor_cat]
            adj[target_idx][neighbor_idx] = info["strength"]

    # Add self-loops (each node connects to itself)
    adj += torch.eye(num_cats) * 3.0  # self-loop with max strength

    # Statistics
    num_edges = (adj > 0).sum().item() - num_cats  # exclude self-loops
    print(f"Categories: {num_cats} (base={base_mask.sum()}, novel={novel_mask.sum()})")
    print(f"Edges (excl. self-loops): {num_edges}")
    print(f"Avg neighbors per novel: {num_edges / novel_mask.sum().item():.1f}")

    # Print graph for verification
    for target_cat, neighbors in data["graph"].items():
        target_idx = name_to_idx[target_cat]
        neighbor_strs = []
        for n_cat, info in sorted(neighbors.items(), key=lambda x: -x[1]["strength"]):
            neighbor_strs.append(f"{n_cat}({info['strength']})")
        print(f"  [novel] {target_cat:12s} <- {', '.join(neighbor_strs)}")

    # Save
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "pretrained", "ovcoco_ist_adj.pt"
    )
    torch.save({
        "adj": adj,
        "cat_names": cat_names,
        "base_mask": base_mask,
        "novel_mask": novel_mask,
        "coco_ids": all_ids,
    }, out_path)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
