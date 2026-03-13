"""
Encode LVIS concept prompts into EVA-CLIP text embeddings.

Takes the multi-prompt concept JSON (lvis_prompts_claude.json) where each
category has a list of 8 prompts, encodes them with EVA-CLIP text encoder,
averages per category, and saves in the same {name: tensor} dict format
used by build_classifier() in ov_backbone.py.

Usage:
    python custom_tools/encode_lvis_concepts.py \
        --concept_json pretrained/lvis_prompts_claude.json \
        --model_name EVA02-CLIP-B-16 \
        --output pretrained/lvis_concept_evaclip_vitb_16.pt

    python custom_tools/encode_lvis_concepts.py \
        --concept_json pretrained/lvis_prompts_claude.json \
        --model_name EVA02-CLIP-L-14 \
        --output pretrained/lvis_concept_evaclip_vitl_14.pt
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import open_clip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept_json", required=True,
                        help="Path to lvis_prompts_claude.json (dict: name -> list of prompts)")
    parser.add_argument("--model_name", default="EVA02-CLIP-B-16",
                        help="EVA-CLIP model name")
    parser.add_argument("--pretrained_path", default="",
                        help="Local cache dir for EVA-CLIP weights (optional)")
    parser.add_argument("--output", required=True,
                        help="Output .pt file path")
    parser.add_argument("--all_classes", default="",
                        help="Path to lvis_v1_all_classes.json for key alignment verification")
    args = parser.parse_args()

    # Load concept prompts
    with open(args.concept_json, "r") as f:
        concept_data = json.load(f)

    # Support both {name: [prompts]} and {category_to_concept: {name: [prompts]}}
    if isinstance(concept_data, dict) and "category_to_concept" in concept_data:
        concept_data = concept_data["category_to_concept"]

    categories = list(concept_data.keys())
    print(f"Loaded {len(categories)} categories from {args.concept_json}")

    # Verify key alignment with all_classes if provided
    if args.all_classes:
        with open(args.all_classes, "r") as f:
            all_classes = json.load(f)
        missing = [c for c in all_classes if c not in concept_data]
        extra = [c for c in categories if c not in all_classes]
        if missing:
            print(f"WARNING: {len(missing)} classes in all_classes missing from concept_json: {missing[:10]}")
        if extra:
            print(f"WARNING: {len(extra)} classes in concept_json not in all_classes: {extra[:10]}")
        # Use all_classes order for consistent alignment
        categories = all_classes
        print(f"Using all_classes order ({len(categories)} categories)")

    # Load EVA-CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_name} on {device}...")
    if args.pretrained_path:
        model = open_clip.create_model(args.model_name, pretrained="eva",
                                       cache_dir=args.pretrained_path)
    else:
        model = open_clip.create_model(args.model_name, pretrained="eva")
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.model_name)

    # Encode all concepts
    embed_dict = {}
    with torch.no_grad():
        for i, cat_name in enumerate(categories):
            prompts = concept_data.get(cat_name)
            if prompts is None:
                # Fallback: use category name itself (replace _ with space)
                prompts = [cat_name.replace("_", " ")]
                print(f"  WARNING: No concept for '{cat_name}', using name as fallback")
            if isinstance(prompts, str):
                prompts = [prompts]

            tokens = tokenizer(prompts).to(device)
            text_features = model.encode_text(tokens)
            text_features = F.normalize(text_features, p=2, dim=-1)
            # Average across prompts
            avg_feature = text_features.mean(dim=0)
            avg_feature = F.normalize(avg_feature, p=2, dim=0)
            embed_dict[cat_name] = avg_feature.cpu()

            if (i + 1) % 200 == 0:
                print(f"  Encoded {i + 1}/{len(categories)} categories")

    print(f"Encoded all {len(categories)} categories, "
          f"embedding dim = {avg_feature.shape[0]}")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(embed_dict, args.output)
    print(f"Saved concept embeddings to {args.output}")

    # Also save the category order for reference
    meta_path = args.output.replace(".pt", "_meta.json")
    meta = {
        "model_name": args.model_name,
        "num_categories": len(categories),
        "embedding_dim": int(avg_feature.shape[0]),
        "categories": categories,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
