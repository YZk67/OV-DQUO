"""
Encode semantic wildcard texts using EVA-CLIP text encoder.
Produces .pt files for ViTB16 and ViTL14.

Usage:
    python custom_tools/encode_semantic_wildcards.py --model EVA02-CLIP-B-16
    python custom_tools/encode_semantic_wildcards.py --model EVA02-CLIP-L-14-336
"""
import argparse
import torch
import torch.nn.functional as F
import open_clip

SEMANTIC_WILDCARDS = [
    "animal", "vehicle", "food", "sports equipment",
    "personal accessory", "furniture", "device", "tool",
]

IMAGENET_TEMPLATES = [
    "a photo of a {}.",
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["EVA02-CLIP-B-16", "EVA02-CLIP-L-14-336"])
    parser.add_argument("--output_dir", type=str, default="pretrained")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model}...")
    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained="eva", device=device)
    tokenizer = open_clip.get_tokenizer(args.model)
    model.eval()

    all_embeddings = {}
    with torch.no_grad():
        for name in SEMANTIC_WILDCARDS:
            texts = [t.format(name) for t in IMAGENET_TEMPLATES]
            tokens = tokenizer(texts).to(device)
            embeddings = model.encode_text(tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            mean_embedding = embeddings.mean(dim=0)
            mean_embedding = mean_embedding / mean_embedding.norm()
            all_embeddings[name] = mean_embedding.cpu()
            print(f"  {name}: {mean_embedding.shape}")

    if "B-16" in args.model:
        out_path = f"{args.output_dir}/vitb16_semantic_wildcards.pt"
    else:
        out_path = f"{args.output_dir}/vitl14_semantic_wildcards.pt"

    torch.save(all_embeddings, out_path)
    print(f"Saved to {out_path}")
    print(f"Keys: {list(all_embeddings.keys())}")


if __name__ == "__main__":
    main()
