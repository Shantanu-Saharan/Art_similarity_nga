import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ViT_B_16_Weights


def get_device():
    # cuda > mps > cpu
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class EmbeddingModel(nn.Module):
    def __init__(self, backbone, embed_dim=512, freeze_backbone=False):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.freeze_backbone = freeze_backbone

        if hasattr(backbone, "hidden_dim"):
            backbone_dim = backbone.hidden_dim
        elif hasattr(backbone, "fc"):
            backbone_dim = backbone.fc.in_features
        else:
            raise ValueError(f"can't figure out backbone dim for {type(backbone).__name__}")

        if hasattr(backbone, "heads"):
            backbone.heads = nn.Identity()
        elif hasattr(backbone, "fc"):
            backbone.fc = nn.Identity()

        self.proj = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim),
            nn.ReLU(),
            nn.Linear(backbone_dim, embed_dim),
        )

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        # returns l2-normalized embeddings
        features = self.backbone(x)
        embeddings = self.proj(features)
        return F.normalize(embeddings, p=2, dim=1)


def build_model(model_name, embed_dim=512, device="auto", freeze_backbone=False):
    # builds the embedding model, auto-selects device if not given
    if device == "auto":
        device = get_device()

    if model_name == "vit":
        backbone = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    elif model_name == "resnet50":
        backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    elif model_name == "resnet18":
        backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"unknown model '{model_name}', use vit/resnet50/resnet18")

    model = EmbeddingModel(backbone, embed_dim, freeze_backbone)
    return model.to(device)


def build_transform():
    # standard imagenet normalization
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_model_info(model_name):
    model = build_model(model_name, embed_dim=512)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "name": model_name,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "backbone": type(model.backbone).__name__,
    }
