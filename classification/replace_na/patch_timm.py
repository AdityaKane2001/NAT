from typing import Tuple

import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer
from natten.functional import natten1dqkrpb, natten1dav

class PartNA(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        # attn = (q @ k.transpose(-2, -1)) * self.scale

        # # Apply proportional attention

        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        attn = natten1dqkrpb(q, k, None, 3, 1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = natten1dav(attn, v, 3, 1)
        
        x = x.permute(0, 2, 1, 3).reshape(B, N, C)

        
        x = self.proj(x)
        x = self.proj_drop(x)

        # Return k as well here
        return x


def make_tome_class(transformer_class):
    class PartNAVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            return super().forward(*args, **kwdargs)

    return PartNAVisionTransformer


def apply_patch(
    model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True
):
    """
    """
    PartNAVisionTransformer = make_tome_class(model.__class__)

    model.__class__ = PartNAVisionTransformer

    for key, module in model.named_modules():
        # print(key)
        if isinstance(module, Attention):
            module.__class__ = PartNA