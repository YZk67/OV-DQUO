"""
Build IST adjacency graph for OV-LVIS (1203 categories).

Two modes:
  1. --mode clip    : Use CLIP text embedding cosine similarity (fast, free)
  2. --mode claude  : Use Claude API to mine semantic relationships (better quality, costs ~$2-5)

The Claude mode first pre-filters with CLIP similarity (top-30 candidates per category),
then queries Claude to score relationships, selecting top-K neighbors.

Output: pretrained/lvis_ist_adj.pt
  - adj: [1203, 1203] float tensor
  - cat_names: list of 1203 category names
  - base_mask: [1203] bool tensor (frequent + common)
  - novel_mask: [1203] bool tensor (rare)

Usage:
    # CLIP similarity only (fast)
    python custom_tools/build_lvis_ist_graph.py --mode clip --top_k 16

    # Claude API (better quality, requires ANTHROPIC_API_KEY)
    python custom_tools/build_lvis_ist_graph.py --mode claude --top_k 16

    # Resume interrupted Claude run
    python custom_tools/build_lvis_ist_graph.py --mode claude --top_k 16 --resume
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_categories():
    """Load all 1203 LVIS category names from prompts file."""
    path = os.path.join(ROOT, "pretrained", "lvis_prompts_claude.json")
    with open(path) as f:
        data = json.load(f)
    return list(data.keys())


def load_clip_embeddings():
    """Load precomputed CLIP text embeddings for LVIS categories."""
    path = os.path.join(ROOT, "pretrained", "lvis_multi_prompt_evaclip_vitb_16_fixed.pt")
    embed_dict = torch.load(path, map_location='cpu')
    cat_names = list(embed_dict.keys())
    # Average over 8 prompts per category -> [1203, 512]
    embeddings = torch.stack([embed_dict[name].mean(dim=0) for name in cat_names])
    embeddings = F.normalize(embeddings, dim=-1)
    return cat_names, embeddings


def build_clip_similarity_graph(cat_names, embeddings, top_k=16):
    """Build adjacency matrix from CLIP text embedding cosine similarity."""
    num_cats = len(cat_names)
    # Compute pairwise cosine similarity
    sim = embeddings @ embeddings.t()  # [1203, 1203]

    # Zero out self-similarity for neighbor selection
    sim.fill_diagonal_(0)

    # Select top-K neighbors per category
    adj = torch.zeros(num_cats, num_cats)
    for i in range(num_cats):
        topk_vals, topk_idxs = sim[i].topk(top_k)
        for val, idx in zip(topk_vals, topk_idxs):
            # Map similarity to strength: scale to 1-3 range
            strength = 1.0 + 2.0 * max(0, val.item())
            adj[i][idx] = strength

    # Add self-loops with max strength
    adj += torch.eye(num_cats) * 3.0

    return adj


def get_clip_candidates(embeddings, top_n=30):
    """Pre-filter: get top-N most similar categories per category using CLIP."""
    sim = embeddings @ embeddings.t()
    sim.fill_diagonal_(-1)  # exclude self

    candidates = {}
    for i in range(len(embeddings)):
        topk_vals, topk_idxs = sim[i].topk(top_n)
        candidates[i] = [(idx.item(), val.item()) for idx, val in zip(topk_idxs, topk_vals)]
    return candidates


def query_claude_batch(target_cats, candidate_names_per_target, all_cat_names):
    """Query Claude API for relationship mining of a batch of categories."""
    import anthropic

    client = anthropic.Anthropic()

    results = {}
    for target_cat, candidates in zip(target_cats, candidate_names_per_target):
        target_display = target_cat.replace('_', ' ')
        cand_list = ", ".join([c.replace('_', ' ') for c in candidates])

        prompt = f"""Given the target category "{target_display}", score the semantic relatedness of each candidate category on a scale of 0-3:
- 3: Very strongly related (synonyms, subtypes, parts of same object)
- 2: Moderately related (often co-occur, functionally related, same domain)
- 1: Weakly related (loose semantic connection)
- 0: Not related

Candidate categories: {cand_list}

Respond ONLY in this exact JSON format, no explanation:
{{"scores": {{{", ".join([f'"{c.replace("_", " ")}": <score>' for c in candidates])}}}}}"""

        for attempt in range(3):
            try:
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1024,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}]
                )
                text = response.content[0].text.strip()
                # Parse JSON
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                data = json.loads(text)
                scores = data.get("scores", data)

                # Map back to original cat names
                cat_scores = {}
                for cand in candidates:
                    cand_display = cand.replace('_', ' ')
                    score = scores.get(cand_display, scores.get(cand, 0))
                    if isinstance(score, (int, float)) and score > 0:
                        cat_scores[cand] = min(3, max(0, score))

                results[target_cat] = cat_scores
                break
            except Exception as e:
                print(f"  Attempt {attempt+1} failed for {target_cat}: {e}")
                if attempt < 2:
                    time.sleep(2)
                else:
                    print(f"  Skipping {target_cat}")
                    results[target_cat] = {}

    return results


def build_claude_graph(cat_names, embeddings, top_k=16, top_n_candidates=30,
                       cache_path=None, resume=False):
    """Build adjacency matrix using Claude API with CLIP pre-filtering."""
    num_cats = len(cat_names)
    name_to_idx = {name: i for i, name in enumerate(cat_names)}

    # Load cache if resuming
    all_scores = {}
    if resume and cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            all_scores = json.load(f)
        print(f"Resumed from cache: {len(all_scores)}/{num_cats} categories done")

    # Pre-filter with CLIP similarity
    print("Computing CLIP similarity for pre-filtering...")
    candidates = get_clip_candidates(embeddings, top_n=top_n_candidates)

    # Query Claude in batches
    batch_size = 5
    remaining = [i for i in range(num_cats) if cat_names[i] not in all_scores]
    total_batches = (len(remaining) + batch_size - 1) // batch_size

    print(f"Querying Claude for {len(remaining)} categories in {total_batches} batches...")

    for batch_idx in range(0, len(remaining), batch_size):
        batch = remaining[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1

        target_cats = [cat_names[i] for i in batch]
        candidate_names = [
            [cat_names[idx] for idx, _ in candidates[i]]
            for i in batch
        ]

        print(f"  Batch {batch_num}/{total_batches}: {', '.join(target_cats[:3])}...")

        batch_results = query_claude_batch(target_cats, candidate_names, cat_names)
        all_scores.update(batch_results)

        # Save cache periodically
        if cache_path and batch_num % 10 == 0:
            with open(cache_path, 'w') as f:
                json.dump(all_scores, f, indent=2)
            print(f"  Cache saved ({len(all_scores)}/{num_cats})")

        # Rate limiting
        time.sleep(0.5)

    # Final cache save
    if cache_path:
        with open(cache_path, 'w') as f:
            json.dump(all_scores, f, indent=2)

    # Build adjacency matrix
    adj = torch.zeros(num_cats, num_cats)

    for target_cat, neighbors in all_scores.items():
        if target_cat not in name_to_idx:
            continue
        target_idx = name_to_idx[target_cat]

        # Sort by score, take top-K
        sorted_neighbors = sorted(neighbors.items(), key=lambda x: -x[1])[:top_k]
        for neighbor_cat, score in sorted_neighbors:
            if neighbor_cat in name_to_idx and score > 0:
                neighbor_idx = name_to_idx[neighbor_cat]
                adj[target_idx][neighbor_idx] = score

    # Add self-loops
    adj += torch.eye(num_cats) * 3.0

    return adj


def get_lvis_frequency_split(cat_names):
    """Get base/novel split for LVIS v1.

    Attempts to load from LVIS annotation file.
    Falls back to marking all as base if annotations unavailable.
    """
    # Try loading from annotation file
    ann_path = os.path.join(ROOT, "data", "Annotations", "lvis_v1_val.json")
    if os.path.exists(ann_path):
        print(f"Loading category frequency from {ann_path}")
        with open(ann_path) as f:
            ann = json.load(f)
        rare_names = set()
        for cat in ann['categories']:
            if cat.get('frequency') == 'r':
                rare_names.add(cat['name'])
        base_mask = torch.tensor([name not in rare_names for name in cat_names])
        novel_mask = torch.tensor([name in rare_names for name in cat_names])
        print(f"  Base (frequent+common): {base_mask.sum().item()}, Novel (rare): {novel_mask.sum().item()}")
        return base_mask, novel_mask

    # Try loading from train_norare file (categories present = base)
    train_path = os.path.join(ROOT, "data", "Annotations", "lvis_v1_train_norare.json")
    if os.path.exists(train_path):
        print(f"Loading base categories from {train_path}")
        with open(train_path) as f:
            ann = json.load(f)
        base_names = set(cat['name'] for cat in ann['categories'])
        base_mask = torch.tensor([name in base_names for name in cat_names])
        novel_mask = ~base_mask
        print(f"  Base: {base_mask.sum().item()}, Novel: {novel_mask.sum().item()}")
        return base_mask, novel_mask

    # Fallback: all base (can be updated later on cloud server)
    print("WARNING: LVIS annotations not found locally. Setting all categories as base.")
    print("  Run on cloud server with data/Annotations/ to get proper base/novel split.")
    base_mask = torch.ones(len(cat_names), dtype=torch.bool)
    novel_mask = torch.zeros(len(cat_names), dtype=torch.bool)
    return base_mask, novel_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['clip', 'claude'], default='clip',
                        help='Graph construction mode')
    parser.add_argument('--top_k', type=int, default=16,
                        help='Number of neighbors per category')
    parser.add_argument('--top_n_candidates', type=int, default=30,
                        help='Number of CLIP pre-filter candidates (claude mode)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume interrupted Claude run from cache')
    parser.add_argument('--output', default=None,
                        help='Output path (default: pretrained/lvis_ist_adj.pt)')
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(ROOT, "pretrained", "lvis_ist_adj.pt")

    # Load data
    print("Loading LVIS categories and CLIP embeddings...")
    cat_names, embeddings = load_clip_embeddings()
    num_cats = len(cat_names)
    print(f"  {num_cats} categories, embedding dim={embeddings.shape[1]}")

    # Build adjacency matrix
    if args.mode == 'clip':
        print(f"\nBuilding CLIP similarity graph (top-{args.top_k} neighbors)...")
        adj = build_clip_similarity_graph(cat_names, embeddings, top_k=args.top_k)
    else:
        cache_path = os.path.join(ROOT, "pretrained", "lvis_claude_scores_cache.json")
        print(f"\nBuilding Claude-refined graph (top-{args.top_k}, candidates={args.top_n_candidates})...")
        adj = build_claude_graph(
            cat_names, embeddings,
            top_k=args.top_k,
            top_n_candidates=args.top_n_candidates,
            cache_path=cache_path,
            resume=args.resume,
        )

    # Get base/novel split
    base_mask, novel_mask = get_lvis_frequency_split(cat_names)

    # Statistics
    num_edges = (adj > 0).sum().item() - num_cats
    avg_neighbors = num_edges / num_cats
    print(f"\nGraph statistics:")
    print(f"  Categories: {num_cats}")
    print(f"  Base: {base_mask.sum().item()}, Novel: {novel_mask.sum().item()}")
    print(f"  Edges (excl. self-loops): {int(num_edges)}")
    print(f"  Avg neighbors per category: {avg_neighbors:.1f}")
    print(f"  Edge strength range: [{adj[adj > 0].min():.2f}, {adj.max():.2f}]")

    # Print some examples
    name_to_idx = {name: i for i, name in enumerate(cat_names)}
    examples = ['dog', 'cat', 'airplane', 'chair', 'banana']
    for ex in examples:
        if ex in name_to_idx:
            idx = name_to_idx[ex]
            row = adj[idx].clone()
            row[idx] = 0  # exclude self-loop
            topk_vals, topk_idxs = row.topk(5)
            neighbors = [(cat_names[j], f"{v:.1f}") for j, v in zip(topk_idxs, topk_vals) if v > 0]
            print(f"  {ex:15s} <- {', '.join([f'{n}({s})' for n, s in neighbors])}")

    # Save
    torch.save({
        "adj": adj,
        "cat_names": cat_names,
        "base_mask": base_mask,
        "novel_mask": novel_mask,
    }, args.output)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
