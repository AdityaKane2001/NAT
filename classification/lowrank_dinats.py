"""
Dilated Neighborhood Attention Transformer.
https://arxiv.org/abs/2209.15001

lowrank_DiNAT_s -- our alternative model.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
import torch.nn as nn
from torch.nn.functional import pad
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from timm.models.registry import register_model
from natten.functional import natten2dqkrpb, natten2dav
from nat import Mlp

model_urls = {
    # ImageNet-1K
    "lowrank_dinat_s_tiny_1k": "https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/lowrank_dinat_s_tiny_in1k_224.pth",
    "lowrank_dinat_s_small_1k": "https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/lowrank_dinat_s_small_in1k_224.pth",
    "lowrank_dinat_s_base_1k": "https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/lowrank_dinat_s_base_in1k_224.pth",
    "lowrank_dinat_s_large_1k": "https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/lowrank_dinat_s_large_in1k_224.pth",
    "lowrank_dinat_s_large_1k_384": "https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/lowrank_dinat_s_large_in1k_384.pth",
    # ImageNet-22K
    "lowrank_dinat_s_large_21k": "https://shi-labs.com/projects/dinat/checkpoints/imagenet22k/lowrank_dinat_s_large_in22k_224.pth",
}



class NeighborhoodAttention2D(nn.Module):
    """
    Neighborhood Attention 2D Module
    """

    def __init__(
        self,
        dim,
        num_heads,
        kernel_size,
        dilation=1,
        bias=True,
        low_rank=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.low_rank = low_rank
        self.scale = qk_scale or self.head_dim**-0.5
        assert (
            kernel_size > 1 and kernel_size % 2 == 1
        ), f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        assert (
            dilation is None or dilation >= 1
        ), f"Dilation must be greater than or equal to 1, got {dilation}."
        self.dilation = dilation or 1
        self.window_size = self.kernel_size * self.dilation

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q = nn.Linear(dim, num_heads * low_rank, bias=qkv_bias)
        self.k = nn.Linear(dim, num_heads * low_rank, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
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

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = int(Hp), int(Wp)
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.window_size or W < self.window_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.window_size - W)
            pad_b = max(0, self.window_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        # qkv = (
        #     self.qkv(x)
        #     .reshape(B, H, W, 3, self.num_heads, self.head_dim)
        #     .permute(3, 0, 4, 1, 2, 5) # (3, B, self.num_heads, H, W, self.head_dim)
        # )
        # q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = self.q(x).reshape(B, self.num_heads, H, W, self.low_rank)
        k = self.k(x).reshape(B, self.num_heads, H, W, self.low_rank)
        v = self.v(x).reshape(B, self.num_heads, H, W, self.head_dim)
        
        q = q * self.scale
        attn = natten2dqkrpb(q, k, self.rpb, self.kernel_size, self.dilation)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = natten2dav(attn, v, self.kernel_size, self.dilation)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, dilation={self.dilation}, "
            + f"rel_pos_bias={self.rpb is not None}"
        )

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
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = NeighborhoodAttention2D(
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


class lowrank_DiNAT_s(nn.Module):
    def __init__(
        self,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[2, 2, 6, 2],
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
        **kwargs
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
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

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
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
            self.layers.append(layer)

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
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x).flatten(1, 2)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@register_model
def lowrank_dinat_s_tiny(pretrained=False, **kwargs):
    model = lowrank_DiNAT_s(
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
        **kwargs
    )
    if pretrained:
        url = model_urls["lowrank_dinat_s_tiny_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def lowrank_dinat_s_small(pretrained=False, **kwargs):
    model = lowrank_DiNAT_s(
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
        **kwargs
    )
    if pretrained:
        url = model_urls["lowrank_dinat_s_small_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def lowrank_dinat_s_base(pretrained=False, **kwargs):
    model = lowrank_DiNAT_s(
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
        **kwargs
    )
    if pretrained:
        url = model_urls["lowrank_dinat_s_base_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def lowrank_dinat_s_large(pretrained=False, **kwargs):
    model = lowrank_DiNAT_s(
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
        **kwargs
    )
    if pretrained:
        url = model_urls["lowrank_dinat_s_large_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def lowrank_dinat_s_large_384(pretrained=False, **kwargs):
    model = lowrank_DiNAT_s(
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
        **kwargs
    )
    if pretrained:
        url = model_urls["lowrank_dinat_s_large_1k_384"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def lowrank_dinat_s_large_21k(pretrained=False, **kwargs):
    model = lowrank_DiNAT_s(
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
        **kwargs
    )
    if pretrained:
        url = model_urls["lowrank_dinat_s_large_21k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model
