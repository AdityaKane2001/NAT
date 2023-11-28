import torch

from timm import create_model
from timm.models.registry import register_model
from tome.patch.timm_prune import apply_patch
from tome import prune


@register_model
def attnprune_vit_small_patch16_224_augreg_in21k_ft_in1k(pretrained=True, **kwargs):
    model = create_model(
        "vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=pretrained
    )

    # apply_patch(model)

    # model.r = 0

    return model


@register_model
def vit_small_patch16_224_augreg_in21k(pretrained=True, **kwargs):
    model = create_model(
        "vit_small_patch16_224.augreg_in21k", pretrained=pretrained, num_classes=1000
    )

    return model

@register_model
def attnprune_vit_small_patch16_224_augreg_in21k(pretrained=True, **kwargs):
    model = create_model(
        "vit_small_patch16_224.augreg_in21k", pretrained=pretrained, num_classes=1000
    )

    apply_patch(model)

    model.r = 4

    return model
