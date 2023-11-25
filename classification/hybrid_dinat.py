"""
Dilated Neighborhood Attention Transformer.
https://arxiv.org/abs/2209.15001

HybridDiNAT_s -- our alternative model.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
import torch.nn as nn
from torch.nn.functional import pad
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from timm.models.registry import register_model
from natten import NeighborhoodAttention2D as NeighborhoodAttention
from nat import Mlp

from typing import Tuple, Optional

from timm.models.vision_transformer import Attention, Block
from merge import (
    bipartite_soft_matching,
    merge_source,
    merge_wavg,
    parse_r,
    compute_r_sched,
)


model_urls = {
    # ImageNet-1K
    "hybrid_dinat_s_tiny_1k": "",
    "hybrid_dinat_s_small_1k": "",
    "hybrid_dinat_s_base_1k": "",
    "hybrid_dinat_s_large_1k": "",
    "hybrid_dinat_s_large_1k_384": "",
    # ImageNet-22K
    "hybrid_dinat_s_large_21k": "",
}


class ToMeAttention(Attention):
    # Copied over from ToMe/tome/patch/timm.py
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
            .permute(2, 0, 3, 1, 4)
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
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Return k as well here
        return x, k.mean(1)


class LayerScale(nn.Module):
    # Copied over from timm/models/vision_transformer.py
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class GlobalBlock(nn.Module):
    # Borrowed with modifications from timm/models/vision_transformer.py
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ToMeAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        # self.ls1 = (
        #     LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        # self.ls2 = (
        #     LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, r: int = None) -> torch.Tensor:
        # x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        # x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        x_attn, metric = self.attn(self.norm1(x), size=self.reduction_info["size"])
        x = x + self.drop_path1(x_attn)

        new_source = None
        new_size = None

        if r > 0:
            # Apply ToMe here
            merge, _ = bipartite_soft_matching(
                metric,
                r,
                self.reduction_info["has_cls"],
                self.reduction_info["has_distill"],
            )
            if self.reduction_info["source"] is not None:
                new_source = merge_source(merge, x, self.reduction_info["source"])
            x, new_size = merge_wavg(merge, x, self.reduction_info["size"])
            self.reduction_info["size"] = new_size
            self.reduction_info["source"] = new_source

        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class OldToMeLayer(Block):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        x = x + self._drop_path1(x_attn)

        r = self._tome_info["r"].pop(0)
        if r > 0:
            # Apply ToMe here
            merge, _ = bipartite_soft_matching(
                metric,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, x, self._tome_info["source"]
                )
            x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])

        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        return x


class NATransformerLayer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        kernel_size=7,
        dilation=1,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = NeighborhoodAttention(
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
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


class LocalLayer(nn.Module):
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
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
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
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
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
            x = self.downsample(x)
        return x


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


class GlobalLayer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        depth,
        mlp_ratio=4.0,
        qkv_bias=True,
        proj_drop=None,
        attn_drop=None,
        drop_path=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_layer=Mlp,
        downsample=None,
        upsample=None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList(
            [
                GlobalBlock(
                    dim=self.dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop=proj_drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[d_idx],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    mlp_layer=mlp_layer,
                )
                for d_idx in range(depth)
            ]
        )

        self.downsample = downsample

        if upsample:
            self.upsample = Mlp(self.dim, out_features=self.dim * 2)
        else:
            self.upsample = None

    def forward(self, x):
        B, T, D = x.shape

        if self.downsample:
            r_sched = compute_r_sched(self.depth, T, T // 4)
            # print(r_sched)
        else:
            r_sched = [0 for _ in range(self.depth)]
        # print(f"Before layer: {x.shape}")

        for idx, blk in enumerate(self.blocks):
            blk.reduction_info = self.reduction_info
            x = blk(x, r=r_sched[idx])
            self.reduction_info = blk.reduction_info
            # print(f"Layer's reduction info: {self.reduction_info}")
            # print(f"Block's reduction info: {blk.reduction_info}")
            # print(f"\tAfter {idx} block: {x.shape}")

        if self.upsample:
            x = self.upsample(x)

        return x


class HybridDiNAT_s(nn.Module):
    def __init__(
        self,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_local_stages=2,
        num_heads=[3, 6, 12, 24],
        kernel_size=7,
        dilations=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        **kwargs,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.num_local_stages = num_local_stages
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

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

        self.reduction_info = dict()

        # build layers
        self.local_layers = nn.ModuleList()
        for i_layer in range(self.num_local_stages):
            layer = LocalLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                kernel_size=kernel_size,
                dilations=None if dilations is None else dilations[i_layer],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
            )
            self.local_layers.append(layer)

        self.global_layers = nn.ModuleList()
        for i_layer in range(self.num_layers - self.num_local_stages):
            layer = GlobalLayer(
                dim=int(embed_dim * 2 ** (self.num_local_stages + i_layer)),
                num_heads=num_heads[i_layer],
                depth=depths[self.num_local_stages + i_layer],
                mlp_ratio=4.0,
                qkv_bias=qkv_bias,
                proj_drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[: i_layer + self.num_local_stages]) : sum(
                        depths[: i_layer + +self.num_local_stages + 1]
                    )
                ],
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm,
                mlp_layer=Mlp,
                downsample=True
                if i_layer != (self.num_layers - self.num_local_stages) - 1
                else False,
                # downsample=False,
                upsample=True
                if i_layer != (self.num_layers - self.num_local_stages) - 1
                else False,
            )
            layer.reduction_info = self.reduction_info
            self.global_layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"rpb"}

    def forward_features(self, x):
        self.reduction_info["size"] = None
        self.reduction_info["source"] = None
        self.reduction_info["has_cls"] = False
        self.reduction_info["has_distill"] = False
        
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for idx, layer in enumerate(self.local_layers):
            x = layer(x)

        x = x.view(x.shape[0], -1, x.shape[-1])

        for idx, layer in enumerate(self.global_layers):
            x = layer(x)
        
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@register_model
def hybrid_dinat_s_tiny(pretrained=False, **kwargs):
    model = HybridDiNAT_s(
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        embed_dim=96,
        mlp_ratio=4,
        drop_path_rate=0.2,
        kernel_size=7,
        dilations=[
            [1, 8],
            [1, 4],
            [1, 2, 1, 2, 1, 2],
            [1, 1],
        ],
        **kwargs,
    )
    if pretrained:
        url = model_urls["hybrid_dinat_s_tiny_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def hybrid_dinat_s_small(pretrained=False, **kwargs):
    model = HybridDiNAT_s(
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        embed_dim=96,
        mlp_ratio=4,
        drop_path_rate=0.3,
        kernel_size=7,
        dilations=[
            [1, 8],
            [1, 4],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [1, 1],
        ],
        **kwargs,
    )
    if pretrained:
        url = model_urls["hybrid_dinat_s_small_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def hybrid_dinat_s_base(pretrained=False, **kwargs):
    model = HybridDiNAT_s(
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        embed_dim=128,
        mlp_ratio=4,
        drop_path_rate=0.5,
        kernel_size=7,
        dilations=[
            [1, 8],
            [1, 4],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [1, 1],
        ],
        **kwargs,
    )
    if pretrained:
        url = model_urls["hybrid_dinat_s_base_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def hybrid_dinat_s_large(pretrained=False, **kwargs):
    model = HybridDiNAT_s(
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        embed_dim=192,
        mlp_ratio=4,
        drop_path_rate=0.35,
        kernel_size=7,
        dilations=[
            [1, 8],
            [1, 4],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [1, 1],
        ],
        **kwargs,
    )
    if pretrained:
        url = model_urls["hybrid_dinat_s_large_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def hybrid_dinat_s_large_384(pretrained=False, **kwargs):
    model = HybridDiNAT_s(
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        embed_dim=192,
        mlp_ratio=4,
        drop_path_rate=0.35,
        kernel_size=7,
        dilations=[
            [1, 13],
            [1, 6],
            [1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3],
            [1, 1],
        ],
        **kwargs,
    )
    if pretrained:
        url = model_urls["hybrid_dinat_s_large_1k_384"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def hybrid_dinat_s_large_21k(pretrained=False, **kwargs):
    model = HybridDiNAT_s(
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        embed_dim=192,
        mlp_ratio=4,
        drop_path_rate=0.2,
        kernel_size=7,
        dilations=[
            [1, 8],
            [1, 4],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [1, 1],
        ],
        **kwargs,
    )
    if pretrained:
        url = model_urls["hybrid_dinat_s_large_21k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model
