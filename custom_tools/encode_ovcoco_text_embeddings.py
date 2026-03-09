"""
Pre-compute text embeddings for OV-COCO using EVA-02-CLIP text encoder.

Generates:
  - pretrained/ovcoco_evaclip_vitb16.pt        (dict: {name: embedding})
  - pretrained/ovcoco_all_classes.json          (list of 65 class names)
  - pretrained/vitb16_ovcoco_object_embbed.pt   (dict: {"object": embedding})
  - pretrained/vitb16_ovcoco_utb_tokens.pt      (tensor [K, 512] for UTB)

Usage:
  python custom_tools/encode_ovcoco_text_embeddings.py
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import open_clip

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.clip.prompts import imagenet_templates

# Standard COCO 80 categories (id -> name), same order as pycocotools
COCO_CATEGORIES = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
    21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
    27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
    34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
    39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard",
    43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup",
    48: "fork", 49: "knife", 50: "spoon", 51: "bowl", 52: "banana",
    53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot",
    58: "hot dog", 59: "pizza", 60: "donut", 61: "cake", 62: "chair",
    63: "couch", 64: "potted plant", 65: "bed", 67: "dining table",
    70: "toilet", 72: "tv", 73: "laptop", 74: "mouse", 75: "remote",
    76: "keyboard", 77: "cell phone", 78: "microwave", 79: "oven",
    80: "toaster", 81: "sink", 82: "refrigerator", 84: "book", 85: "clock",
    86: "vase", 87: "scissors", 88: "teddy bear", 89: "hair drier",
    90: "toothbrush",
}

# OV-COCO split: 48 base + 17 novel = 65 total
BASE_CATIDS = [
    70, 2, 53, 7, 73, 57, 4, 79, 62, 74, 9, 38, 20, 19, 54, 85, 72, 27,
    80, 51, 78, 15, 84, 55, 16, 59, 48, 34, 23, 86, 90, 50, 25, 31, 56,
    82, 75, 42, 3, 65, 52, 60, 35, 1, 8, 44, 33, 24,
]
NOVEL_CATIDS = [28, 21, 47, 6, 76, 41, 18, 63, 32, 36, 81, 22, 61, 87, 5, 17, 49]

# UTB tokens
UTB_TOKENS = ["object", "thing", "animal", "vehicle", "tool", "device", "furniture", "food"]


def encode_text_with_templates(model, tokenizer, category_name, templates, device):
    """Encode a category name using multiple templates, return averaged embedding."""
    texts = [t.format(category_name) for t in templates]
    text_tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = F.normalize(text_features, dim=-1)
        mean_feature = text_features.mean(dim=0)
        mean_feature = F.normalize(mean_feature, dim=-1)
    return mean_feature


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load EVA02-CLIP-B-16
    print("Loading EVA02-CLIP-B-16...")
    model, _, _ = open_clip.create_model_and_transforms(
        "EVA02-CLIP-B-16", pretrained="eva"
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("EVA02-CLIP-B-16")
    print("Model loaded.")

    # Get all 65 category names sorted by cat_id
    all_catids = sorted(BASE_CATIDS + NOVEL_CATIDS)
    all_names = [COCO_CATEGORIES[cid] for cid in all_catids]
    print(f"Total categories: {len(all_names)}")
    print(f"  Base: {len(BASE_CATIDS)}, Novel: {len(NOVEL_CATIDS)}")

    # Encode all categories
    print("Encoding category embeddings...")
    class_embed_dict = {}
    for name in all_names:
        emb = encode_text_with_templates(model, tokenizer, name, imagenet_templates, device)
        class_embed_dict[name] = emb.cpu()
        print(f"  {name}: {emb.shape}")

    # Encode "object" wildcard
    print("Encoding 'object' wildcard...")
    object_emb = encode_text_with_templates(model, tokenizer, "object", imagenet_templates, device)
    object_embed_dict = {"object": object_emb.cpu()}

    # Encode UTB tokens
    print("Encoding UTB tokens...")
    utb_embeddings = []
    for token in UTB_TOKENS:
        emb = encode_text_with_templates(model, tokenizer, token, imagenet_templates, device)
        utb_embeddings.append(emb.cpu())
        print(f"  UTB token '{token}': {emb.shape}")
    utb_tensor = torch.stack(utb_embeddings, dim=0)  # [K, 512]

    # Save outputs
    os.makedirs("pretrained", exist_ok=True)

    # 1. Class embeddings dict
    out_path = "pretrained/ovcoco_evaclip_vitb16.pt"
    torch.save(class_embed_dict, out_path)
    print(f"Saved class embeddings to {out_path}")

    # 2. All class names JSON
    out_path = "pretrained/ovcoco_all_classes.json"
    with open(out_path, "w") as f:
        json.dump(all_names, f, indent=2)
    print(f"Saved class names to {out_path}")

    # 3. Object wildcard embedding
    out_path = "pretrained/vitb16_ovcoco_object_embbed.pt"
    torch.save(object_embed_dict, out_path)
    print(f"Saved object embedding to {out_path}")

    # 4. UTB token embeddings
    out_path = "pretrained/vitb16_ovcoco_utb_tokens.pt"
    torch.save(utb_tensor, out_path)
    print(f"Saved UTB tokens ({utb_tensor.shape}) to {out_path}")

    # Verify
    print("\n--- Verification ---")
    print(f"Class embed dict keys: {len(class_embed_dict)}")
    print(f"Sample embedding dim: {list(class_embed_dict.values())[0].shape}")
    print(f"UTB tensor shape: {utb_tensor.shape}")
    print(f"All class names: {all_names}")


if __name__ == "__main__":
    main()
