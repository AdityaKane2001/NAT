# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------


from typing import Tuple

import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer

from ..prune import attentive_pruning, prune_source
from ..utils import parse_r
import math


class AttnMapPruneAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4) # (3, B, Heads, N, C // Heads)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply proportional attention
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        r = self._tome_info["r"].pop(0)
        retain_idxs = torch.arange(start=0, end=N, step=1)
        if r > 0:
            with torch.no_grad():
                _attn = attn.detach()
                
                _attn = attn.mean(dim=1)
                # print(f"{attn.shape=}")
                t = _attn.shape[1]
                attnsum = _attn.sum(dim=1)
                if self._tome_info["class_token"]:
                    attnsum[:, 0] = math.inf
                if self._tome_info["distill_token"]:
                    attnsum[:, -1] = math.inf

                # print(f"{attnsum.shape=}")
                # print(attnsum.argsort(descending=True))
                retain_idxs = attnsum.argsort(descending=True)[:, : t - r]
                # print(retain_idxs.shape)
                # print(f"{retain_idxs.shape=}")

                retain_idxs = retain_idxs.sort()[0]
        print(attn.shape)
        # attn = attn[, , :]
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N - r, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Return k as well here
        return x


def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tome_info["r"] = parse_r(len(self.blocks), self.r)
            self._tome_info["size"] = None
            self._tome_info["source"] = None

            return super().forward(*args, **kwdargs)

    return ToMeVisionTransformer


def apply_patch(
    model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    ToMeVisionTransformer = make_tome_class(model.__class__)

    model.__class__ = ToMeVisionTransformer
    model.r = 0
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, Attention):
            module._tome_info = model._tome_info
            module.__class__ = AttnMapPruneAttention
