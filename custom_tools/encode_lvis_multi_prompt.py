"""
Encode LVIS concept prompts into EVA-CLIP text embeddings WITHOUT averaging.

Saves per-category multi-prompt embeddings as {name: tensor[K, D]} for TPA use.

Usage:
    python custom_tools/encode_lvis_multi_prompt.py \
        --concept_json pretrained/lvis_prompts_claude.json \
        --model_name EVA02-CLIP-B-16 \
        --output pretrained/lvis_multi_prompt_evaclip_vitb_16.pt \
        --all_classes pretrained/lvis_v1_all_classes.json
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
                        help="Path to lvis_prompts_claude.json")
    parser.add_argument("--model_name", default="EVA02-CLIP-B-16")
    parser.add_argument("--pretrained_path", default="",
                        help="Local cache dir for EVA-CLIP weights")
    parser.add_argument("--output", required=True,
                        help="Output .pt file path")
    parser.add_argument("--all_classes", default="",
                        help="Path to lvis_v1_all_classes.json for key ordering")
    args = parser.parse_args()

    with open(args.concept_json, "r") as f:
        concept_data = json.load(f)
    if isinstance(concept_data, dict) and "category_to_concept" in concept_data:
        concept_data = concept_data["category_to_concept"]

    categories = list(concept_data.keys())
    if args.all_classes:
        with open(args.all_classes, "r") as f:
            categories = json.load(f)
        print(f"Using all_classes order ({len(categories)} categories)")

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

    embed_dict = {}
    with torch.no_grad():
        for i, cat_name in enumerate(categories):
            prompts = concept_data.get(cat_name)
            if prompts is None:
                prompts = [cat_name.replace("_", " ")]
                print(f"  WARNING: No concept for '{cat_name}', using name as fallback")
            if isinstance(prompts, str):
                prompts = [prompts]

            tokens = tokenizer(prompts).to(device)
            text_features = model.encode_text(tokens)
            text_features = F.normalize(text_features, p=2, dim=-1)
            # Save all K prompt embeddings WITHOUT averaging
            embed_dict[cat_name] = text_features.cpu()  # [K, D]

            if (i + 1) % 200 == 0:
                print(f"  Encoded {i + 1}/{len(categories)} categories")

    sample = next(iter(embed_dict.values()))
    print(f"Encoded {len(categories)} categories, "
          f"shape per category = [{sample.shape[0]}, {sample.shape[1]}]")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(embed_dict, args.output)
    print(f"Saved multi-prompt embeddings to {args.output}")


if __name__ == "__main__":
    main()
