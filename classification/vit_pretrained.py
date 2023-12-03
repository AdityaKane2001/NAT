import torch

from timm import create_model
from timm.models.registry import register_model
from tome.patch import timm, timm_prune, timm_prunemap, timm_attnsum_tail


from replace_na import patch_timm
from tome import prune



@register_model
def vit_small_patch16_224_augreg_in21k_ft_in1k(pretrained=True, **kwargs):
    model = create_model(
        "vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=pretrained
    )
    return model


@register_model
def vit_small_patch16_224_augreg_in21k(pretrained=True, **kwargs):
    model = create_model(
        "vit_small_patch16_224.augreg_in21k", pretrained=pretrained, num_classes=1000
    )

    return model


@register_model
def tome_vit_small_patch16_224_augreg_in21k_f1_in1k(pretrained=True, **kwargs):
    model = create_model(
        "vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=pretrained
    )

    timm.apply_patch(model)

    model.r = 4

    return model

@register_model
def attnmap_merge_tail_vit_small_patch16_224_augreg_in21k_f1_in1k(pretrained=True, **kwargs):
    model = create_model(
        "vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=pretrained
    )

    timm_attnsum_tail.apply_patch(model)

    model.r = 4

    return model

@register_model
def attnprune_vit_small_patch16_224_augreg_in21k(pretrained=True, **kwargs):
    model = create_model(
        "vit_small_patch16_224.augreg_in21k", pretrained=pretrained, num_classes=1000
    )

    timm_prune.apply_patch(model)

    model.r = 4

    return model

@register_model
def prunemap_vit_small_patch16_224_augreg_in21k(pretrained=True, **kwargs):
    model = create_model(
        "vit_small_patch16_224.augreg_in21k", pretrained=pretrained, num_classes=1000
    )

    timm_prunemap.apply_patch(model)

    model.r = 4

    return model

@register_model
def replaceNA_vit_small_patch16_224_augreg_in21k_ft_in1k(pretrained=True, **kwargs):
    model = create_model(
        "vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=pretrained
    )

    patch_timm.apply_patch(model)

    # model.r = 0

    return model
