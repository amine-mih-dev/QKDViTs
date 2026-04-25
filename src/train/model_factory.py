from __future__ import annotations

from collections.abc import Mapping

import timm
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models import resnet50
from transformers import AutoModelForImageClassification, ViTForImageClassification
from vit_pytorch.distill import DistillableViT

from .config import DEFAULT_VIT_ARCHITECTURES, StudentSpec, TeacherSpec


def _get_module_by_path(root_module: nn.Module, path: str) -> nn.Module | None:
    current: object = root_module
    for part in path.split("."):
        if part.isdigit():
            if not isinstance(current, (nn.Sequential, nn.ModuleList)):
                return None
            idx = int(part)
            if idx < 0 or idx >= len(current):
                return None
            current = current[idx]
            continue

        if not hasattr(current, part):
            return None
        current = getattr(current, part)

    return current if isinstance(current, nn.Module) else None


def _set_module_by_path(root_module: nn.Module, path: str, new_module: nn.Module) -> bool:
    if not path:
        return False

    parts = path.split(".")
    parent_path = ".".join(parts[:-1])
    leaf = parts[-1]

    parent_module = (
        root_module if not parent_path else _get_module_by_path(root_module, parent_path)
    )
    if parent_module is None:
        return False

    if leaf.isdigit():
        if not isinstance(parent_module, (nn.Sequential, nn.ModuleList)):
            return False
        idx = int(leaf)
        if idx < 0 or idx >= len(parent_module):
            return False
        parent_module[idx] = new_module
        return True

    setattr(parent_module, leaf, new_module)
    return True


def _get_last_linear_in_sequential(
    module: nn.Sequential,
) -> tuple[int, nn.Linear] | None:
    for idx in range(len(module) - 1, -1, -1):
        layer = module[idx]
        if isinstance(layer, nn.Linear):
            return idx, layer
    return None


def adapt_classifier_if_needed(
    model: nn.Module,
    num_classes: int,
    in_features_hint: int | None = None,
    model_name_hint: str = "model",
) -> nn.Module:
    classifier_paths = [
        "fc",
        "classifier",
        "head",
        "mlp_head",
        "heads.head",
        "head.fc",
    ]

    for path in classifier_paths:
        classifier_layer = _get_module_by_path(model, path)
        if classifier_layer is None:
            continue

        if isinstance(classifier_layer, nn.Linear):
            if classifier_layer.out_features != num_classes:
                current_in_features = classifier_layer.in_features
                final_in_features = (
                    in_features_hint if in_features_hint is not None else current_in_features
                )
                replaced = _set_module_by_path(
                    model,
                    path,
                    nn.Linear(final_in_features, num_classes),
                )
                if not replaced:
                    continue
            return model

        if isinstance(classifier_layer, nn.Sequential):
            final_linear_info = _get_last_linear_in_sequential(classifier_layer)
            if final_linear_info is not None:
                final_linear_idx, final_linear = final_linear_info
                current_in_features = final_linear.in_features
                final_in_features = (
                    in_features_hint if in_features_hint is not None else current_in_features
                )
                if final_linear.out_features != num_classes:
                    classifier_layer[final_linear_idx] = nn.Linear(
                        final_in_features,
                        num_classes,
                    )
                return model

    linear_layers = [
        (name, layer) for name, layer in model.named_modules() if isinstance(layer, nn.Linear)
    ]
    if linear_layers:
        last_name, last_linear = linear_layers[-1]
        if last_linear.out_features == num_classes:
            return model

        if last_name:
            final_in_features = (
                in_features_hint if in_features_hint is not None else last_linear.in_features
            )
            replaced = _set_module_by_path(
                model,
                last_name,
                nn.Linear(final_in_features, num_classes),
            )
            if replaced:
                return model

    print(
        f"Warning: Could not adapt classifier for {model_name_hint}. "
        f"Please verify output dimension is {num_classes}."
    )
    return model


def get_model_by_name(
    backbone_name_str: str,
    num_classes: int,
    pretrained_weights_path: str | None = None,
    timm_pretrained_flag: bool = False,
    actor_name: str = "Model",
) -> tuple[nn.Module, int | None]:
    load_default_pretrained = (
        pretrained_weights_path is None and timm_pretrained_flag
    )
    backbone_feature_dim: int | None = None

    if timm.is_model(backbone_name_str):
        model = timm.create_model(
            backbone_name_str,
            pretrained=load_default_pretrained,
            num_classes=num_classes,
        )
        backbone_feature_dim = getattr(model, "num_features", None)
    elif hasattr(tv_models, backbone_name_str):
        model_fn = getattr(tv_models, backbone_name_str)
        try:
            model = model_fn(weights="DEFAULT" if load_default_pretrained else None)
        except TypeError:
            model = model_fn(pretrained=load_default_pretrained)

        model = adapt_classifier_if_needed(
            model,
            num_classes,
            model_name_hint=actor_name,
        )

        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            backbone_feature_dim = model.fc.in_features
        elif hasattr(model, "classifier"):
            classifier = getattr(model, "classifier")
            if isinstance(classifier, nn.Linear):
                backbone_feature_dim = classifier.in_features
            elif isinstance(classifier, nn.Sequential):
                for layer in classifier:
                    if isinstance(layer, nn.Linear):
                        backbone_feature_dim = layer.in_features
                        break
    elif "vit" in backbone_name_str.lower() or any(
        prefix in backbone_name_str
        for prefix in ("google/", "facebook/", "microsoft/")
    ):
        if "hybrid" in backbone_name_str.lower():
            model = AutoModelForImageClassification.from_pretrained(
                pretrained_weights_path
                if pretrained_weights_path
                else backbone_name_str,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            model = ViTForImageClassification.from_pretrained(
                pretrained_weights_path
                if pretrained_weights_path
                else backbone_name_str,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
            backbone_feature_dim = model.classifier.in_features
    elif backbone_name_str == "resnet50_torchvision":
        model = resnet50(weights="DEFAULT" if load_default_pretrained else None)
        backbone_feature_dim = model.fc.in_features
        model.fc = nn.Linear(backbone_feature_dim, num_classes)
    else:
        raise ValueError(
            f"{actor_name} backbone '{backbone_name_str}' is not supported by model_factory."
        )

    if pretrained_weights_path and pretrained_weights_path.endswith((".pt", ".pth", ".ckpt")):
        state_dict = torch.load(pretrained_weights_path, map_location="cpu", weights_only=False)
        if isinstance(state_dict, Mapping) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        if isinstance(state_dict, Mapping):
            cleaned = {}
            for key, value in state_dict.items():
                normalized = key
                for prefix in ("module.", "model.", "student."):
                    if normalized.startswith(prefix):
                        normalized = normalized[len(prefix) :]
                cleaned[normalized] = value
            model.load_state_dict(cleaned, strict=False)

    return model, backbone_feature_dim


def student_short_name(student_id: str) -> str:
    parts = student_id.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return student_id


def build_distillable_vit(student_id: str, num_classes: int) -> DistillableViT:
    short_name = student_short_name(student_id)
    if short_name not in DEFAULT_VIT_ARCHITECTURES:
        raise ValueError(
            f"Unknown student architecture for '{student_id}'. "
            f"Expected one of {list(DEFAULT_VIT_ARCHITECTURES.keys())}."
        )

    params = {
        "image_size": 224,
        "patch_size": 16,
        "num_classes": num_classes,
        "dropout": 0.0,
        **DEFAULT_VIT_ARCHITECTURES[short_name],
    }
    return DistillableViT(**params)


def build_teacher_from_spec(spec: TeacherSpec, num_classes: int) -> nn.Module:
    model, _ = get_model_by_name(
        backbone_name_str=spec.backbone_name,
        num_classes=num_classes,
        pretrained_weights_path=spec.pretrained_weights_path,
        timm_pretrained_flag=spec.pretrained,
        actor_name=f"Teacher<{spec.id}>",
    )
    return model


def build_student_from_spec(spec: StudentSpec, num_classes: int) -> nn.Module:
    if spec.init_type == "DistillableViT":
        params = {
            "image_size": 224,
            "patch_size": 16,
            "num_classes": num_classes,
            **(spec.params or {}),
        }
        return DistillableViT(**params)

    if spec.init_type == "get_model_by_name":
        if not spec.backbone_name:
            raise ValueError(f"Student spec {spec.id} requires a backbone_name")
        model, _ = get_model_by_name(
            backbone_name_str=spec.backbone_name,
            num_classes=num_classes,
            pretrained_weights_path=spec.pretrained_weights_path,
            timm_pretrained_flag=spec.pretrained,
            actor_name=f"Student<{spec.id}>",
        )
        return model

    raise ValueError(f"Unsupported student init_type: {spec.init_type}")
