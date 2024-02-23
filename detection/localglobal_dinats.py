"""
Dilated Neighborhood Attention Transformer.
https://arxiv.org/abs/2209.15001

DiNAT_s -- our alternative model.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
import torch.nn as nn
from torch.nn.functional import pad
from torch import Tensor
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from timm.models.registry import register_model
from natten.functional import natten2dqkrpb, natten2dav
from natten.natten2d import na2d_qk_with_bias, na2d_av
from typing import Optional

from mmcv.runner import load_checkpoint
from mmdet.utils import get_root_logger
from mmdet.models.builder import BACKBONES

import math

from merge import bipartite_soft_matching_random2d

model_urls = {
    # ImageNet-1K
    "localglobal_dinat_s_tiny_1k": "https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_s_tiny_in1k_224.pth",
    "localglobal_dinat_s_small_1k": "https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_s_small_in1k_224.pth",
    "localglobal_dinat_s_base_1k": "https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_s_base_in1k_224.pth",
    "localglobal_dinat_s_large_1k": "https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_s_large_in1k_224.pth",
    "localglobal_dinat_s_large_1k_384": "https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_s_large_in1k_384.pth",
    # ImageNet-22K
    "localglobal_dinat_s_large_21k": "https://shi-labs.com/projects/dinat/checkpoints/imagenet22k/dinat_s_large_in22k_224.pth",
}


class ToMeAidedNeighborhoodAttention2D(nn.Module):
    """
    Neighborhood Attention 2D Module
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        global_sampling_stride=None
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.gs_stride = global_sampling_stride
        assert (
            kernel_size > 1 and kernel_size % 2 == 1
        ), f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        assert (
            dilation is None or dilation >= 1
        ), f"Dilation must be greater than or equal to 1, got {dilation}."
        self.dilation = dilation or 1
        self.window_size = self.kernel_size * self.dilation

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if bias:
            self.rpb = nn.Parameter(
                torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1))
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"NeighborhoodAttention2D expected a rank-4 input tensor; got {x.dim()=}."
            )

        B, H, W, C = x.shape
        
        # Pad if the input is small than the minimum supported size
        H_padded, W_padded = H, W
        padding_h = padding_w = 0       
        if H < self.window_size or W < self.window_size:
            padding_h = max(0, self.window_size - H_padded)
            padding_w = max(0, self.window_size - W_padded)
            x = pad(x, (0, 0, 0, padding_w, 0, padding_h))
            _, H_padded, W_padded, _ = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, H_padded, W_padded, 3, self.num_heads, self.head_dim)
            .permute(3, 0, 4, 1, 2, 5)
        )
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if self.gs_stride is not None:
            # Perform aggressive ToMe over the entire image. Merge keys and values
            # using the resulting merge fn. Pass the merged KV as additional KV.
            with torch.no_grad():
                x_metric_view = x.view(B, H_padded * W_padded, self.num_heads, self.head_dim).mean(-2)
                num_global_tokens = (H_padded // self.gs_stride) * (W_padded // self.gs_stride)
                r = (H_padded * W_padded) - num_global_tokens
                perfect_merge, imperfect_merge, _ = bipartite_soft_matching_random2d(
                    metric=x_metric_view,
                    h=H_padded,  # 56
                    w=W_padded,  # 56
                    sy=self.gs_stride, # maximum possible dilation
                    sx=self.gs_stride, # maximum possible dilation
                    r=r # 
                    # merge all tokens from src to dst
                )
                # FIXME, need to find conditions for perfect or imperfect merge
                if True: 
                    merge_fn = perfect_merge
                else:
                    merge_fn = imperfect_merge
        
        # k: [B, self.num_heads, H, W, self.head_dim]
        # k_view: [B, N, self.num_heads * self.head_dim]
        # global_k (before view): [B, N', self.num_heads * self.head_dim]
        
        if self.gs_stride is not None:
            k_view = k.permute(0, 2, 3, 1, 4).reshape(B, -1, self.num_heads * self.head_dim)
            v_view = v.permute(0, 2, 3, 1, 4).reshape(B, -1, self.num_heads * self.head_dim)
            
            global_k = (merge_fn(k_view)
                        .reshape(B, -1, self.num_heads, self.head_dim)
                        .permute(0, 2, 1, 3)
            )
            global_v = (merge_fn(v_view)
                        .reshape(B, -1, self.num_heads, self.head_dim)
                        .permute(0, 2, 1, 3)
            )
        else:
            global_k = None
            global_v = None
            num_global_tokens = 0
        
        
        kernel_area = self.kernel_size ** 2
        entropy_scale = math.log(kernel_area + num_global_tokens, kernel_area)**.5
        
        q = q * self.scale * entropy_scale
        attn = na2d_qk_with_bias(q, k, self.rpb, self.kernel_size, self.dilation,
                                 additional_keys=global_k)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = na2d_av(attn, v, self.kernel_size, self.dilation, 
                    additional_values=global_v)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H_padded, W_padded, C)

        # Remove padding, if added any
        if padding_h or padding_w:
            x = x[:, :H, :W, :]

        return self.proj_drop(self.proj(x))

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, dilation={self.dilation}, "
            + f"has_bias={self.rpb is not None}"
        )

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class NATransformerLayer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        kernel_size=7,
        dilation=1,
        mlp_ratio=4.0,
        rpb: bool = True,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        global_sampling_stride=None,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = ToMeAidedNeighborhoodAttention2D(
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            num_heads=num_heads,
            bias=rpb,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            global_sampling_stride=global_sampling_stride
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    """
    Based on Swin Transformer
    https://arxiv.org/abs/2103.14030
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = pad(x, (0, 0, 0, W % 2, 0, H % 2))
            _, H, W, _ = x.shape

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, (H + 1) // 2, (W + 1) // 2, 4 * C)  # B H/2 W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    """
    Based on Swin Transformer
    https://arxiv.org/abs/2103.14030
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        kernel_size,
        dilations=None,
        mlp_ratio=4.0,
        rpb=False,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        global_sampling_stride=None,
        norm_layer=nn.LayerNorm,
        downsample=None,
    ):

        super().__init__()
        self.dim = dim
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList(
            [
                NATransformerLayer(
                    dim=dim,
                    num_heads=num_heads,
                    kernel_size=kernel_size,
                    dilation=1 if dilations is None else dilations[i],
                    mlp_ratio=mlp_ratio,
                    rpb=rpb,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    global_sampling_stride=global_sampling_stride
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x_down = self.downsample(x)
            return x, x_down
        return x, x


class PatchEmbed(nn.Module):
    """
    From Swin Transformer
    https://arxiv.org/abs/2103.14030
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.norm = None if norm_layer is None else norm_layer(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        if W % self.patch_size[1] != 0:
            x = pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x

@BACKBONES.register_module()
class LocalGlobalDiNAT_s(nn.Module):
    def __init__(
        self,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        kernel_size=7,
        dilations=None,
        mlp_ratio=4.0,
        rpb=True,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        pretrained=None,
        frozen_stages=-1,
        pretrain_img_size=224,
        **kwargs
    ):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        global_sampling_strides = [8, 4, None, None]
        
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                kernel_size=kernel_size,
                dilations=None if dilations is None else dilations[i_layer],
                mlp_ratio=self.mlp_ratio,
                rpb=rpb,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                global_sampling_stride=global_sampling_strides[i_layer]
            )
            self.layers.append(layer)

        
        num_features = [int(embed_dim * 2**i) for i in range(self.num_layers)]
        self.num_features = num_features
        
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

        self._freeze_stages()
        if pretrained is not None:
            self.init_weights(pretrained)

    
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False


    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            pass
        else:
            raise TypeError("pretrained must be a str or None")


    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, x = layer(x)

            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x_out)

                out = x_out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)


    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super().train(mode)
        self._freeze_stages()
