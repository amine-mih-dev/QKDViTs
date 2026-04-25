from __future__ import annotations

import argparse

from torchinfo import summary
from transformers import AutoModelForImageClassification, ViTForImageClassification


def inspect_model_architecture(
    vit_model_name: str,
    hybrid_model_name: str,
    show_summary: bool,
) -> None:
    vit_model = ViTForImageClassification.from_pretrained(vit_model_name)
    vit_hybrid_model = AutoModelForImageClassification.from_pretrained(hybrid_model_name)

    print("Loaded ViT model")
    print(vit_model)
    print("\nLoaded hybrid ViT model")
    print(vit_hybrid_model)

    if show_summary:
        print("\nHybrid ViT summary")
        summary(
            vit_hybrid_model,
            (1, 3, 384, 384),
            device="cpu",
            verbose=2,
            col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect ViT and hybrid ViT architectures.")
    parser.add_argument(
        "--vit-model",
        default="google/vit-base-patch16-224",
        help="Hugging Face model id for standard ViT.",
    )
    parser.add_argument(
        "--hybrid-model",
        default="google/vit-hybrid-base-bit-384",
        help="Hugging Face model id for ViT hybrid.",
    )
    parser.add_argument(
        "--show-summary",
        action="store_true",
        help="Print a detailed torchinfo summary for the hybrid model.",
    )
    args = parser.parse_args()

    inspect_model_architecture(args.vit_model, args.hybrid_model, args.show_summary)


if __name__ == "__main__":
    main()
