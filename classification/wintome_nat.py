"""
Neighborhood Attention Transformer.
To appear in CVPR 2023.
https://arxiv.org/abs/2204.07143

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
import torch.nn as nn
from torch.nn.functional import pad

from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from natten.functional import natten2dav, natten2dqkrpb

from merge import windowed_unequal_bipartite_soft_matching

model_urls = {
    "nat_mini_1k": "https://shi-labs.com/projects/nat/checkpoints/CLS/nat_mini.pth",
    "nat_tiny_1k": "https://shi-labs.com/projects/nat/checkpoints/CLS/nat_tiny.pth",
    "nat_small_1k": "https://shi-labs.com/projects/nat/checkpoints/CLS/nat_small.pth",
    "nat_base_1k": "https://shi-labs.com/projects/nat/checkpoints/CLS/nat_base.pth",
}


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(B, H // window_size, W // window_size, -1, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (B, num_windows, num_windows, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """

    B = windows.shape[0]  # / (H * W / window_size / window_size))
    H = windows.shape[1] * window_size
    W = windows.shape[2] * window_size

    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WinToMeNeighborhoodAttention2D(nn.Module):
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
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
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

    def forward(self, x, return_keys=False):
        B, Hp, Wp, C = x.shape
        H, W = int(Hp), int(Wp)
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.window_size or W < self.window_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.window_size - W)
            pad_b = max(0, self.window_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, H, W, 3, self.num_heads, self.head_dim)
            .permute(3, 0, 4, 1, 2, 5)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        attn = natten2dqkrpb(q, k, self.rpb, self.kernel_size, self.dilation)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = natten2dav(attn, v, self.kernel_size, self.dilation)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        if return_keys:
            return self.proj_drop(self.proj(x)), k
        return self.proj_drop(self.proj(x))

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, dilation={self.dilation}, "
            + f"rel_pos_bias={self.rpb is not None}"
        )


class ConvTokenizer(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_chans,
                embed_dim // 2,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
            ),
            nn.Conv2d(
                embed_dim // 2,
                embed_dim,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
            ),
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class ConvDownsampler(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Conv2d(
            dim, 2 * dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        x = self.reduction(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.norm(x)
        return x


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


class NATBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        kernel_size=7,
        dilation=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        layer_scale=None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WinToMeNeighborhoodAttention2D(
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
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )
            self.gamma2 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )

    def forward(self, x):
        if not self.layer_scale:
            shortcut = x
            x = self.norm1(x)
            x = self.attn(x)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class NATReductionBlock(NATBlock):
    def __init__(
        self,
        dim,
        num_heads,
        kernel_size=7,
        reduction_window_size=4,
        dilation=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        layer_scale=None,
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            kernel_size=kernel_size,
            dilation=dilation,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            layer_scale=layer_scale,
        )
        self.reduction_window_size = reduction_window_size
        self.mlp = Mlp(
            in_features=dim,
            out_features=dim * 2,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        B, H, W, C = x.shape
        if not self.layer_scale:
            shortcut = x
            x = self.norm1(x)
            x, k = self.attn(x, return_keys=True)
            x = shortcut + self.drop_path(x)

            # Apply tome here, after first residual connection, before the
            # second
            # TODO: Figure out a way to make a neat API, needs to windowfy
            # the input and perform merging

            windowed_x = window_partition(x, window_size=self.reduction_window_size)
            windowed_k = window_partition(
                k.mean(dim=1), window_size=self.reduction_window_size
            )
            reduction_factor = 3 * self.reduction_window_size**2 // 4
            m, u = windowed_unequal_bipartite_soft_matching(
                windowed_k, r=reduction_factor
            )
            merged_x = m(windowed_x)

            x = window_reverse(
                merged_x,
                window_size=self.reduction_window_size // 2,
                H=H // 2,
                W=W // 2,
            )
            x = self.mlp(self.norm2(x))
            return x
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(self.gamma1 * x)

        # Apply tome here, after first residual connection, before the
        # second

        x = self.gamma2 * self.mlp(self.norm2(x))
        return x


class WinToMeNATLevel(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        kernel_size,
        dilations=None,
        downsample=True,
        downsample_method="tome",
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        layer_scale=None,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.downsample = downsample
        self.downsample_method = downsample_method

        module_list = [
            NATBlock(
                dim=dim,
                num_heads=num_heads,
                kernel_size=kernel_size,
                dilation=None if dilations is None else dilations[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                layer_scale=layer_scale,
            )
            for i in range(depth - 1)
        ]

        if self.downsample and self.downsample_method == "conv":
            module_list.append(
                NATBlock(
                    dim=dim,
                    num_heads=num_heads,
                    kernel_size=kernel_size,
                    dilation=None if dilations is None else dilations[depth - 1],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[depth - 1]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    layer_scale=layer_scale,
                )
            )

            self.downsampler = ConvDownsampler(dim=dim, norm_layer=norm_layer)

        elif self.downsample and self.downsample_method == "tome":
            module_list.append(
                NATReductionBlock(
                    dim=dim,
                    num_heads=num_heads,
                    kernel_size=kernel_size,
                    dilation=None if dilations is None else dilations[depth - 1],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[depth - 1]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    layer_scale=layer_scale,
                )
            )
            self.downsampler = None
        
        elif not downsample:
            module_list.append(
                NATBlock(
                    dim=dim,
                    num_heads=num_heads,
                    kernel_size=kernel_size,
                    dilation=None if dilations is None else dilations[depth - 1],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[depth - 1]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    layer_scale=layer_scale,
                )
            )
            self.downsampler = None
        

        self.blocks = nn.ModuleList(module_list)

    def forward(self, x):
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
        if self.downsample_method == "conv" and self.downsample:
            return self.downsampler(x)
        return x


class WinToMeNAT(nn.Module):
    def __init__(
        self,
        embed_dim,
        mlp_ratio,
        depths,
        num_heads,
        drop_path_rate=0.2,
        in_chans=3,
        kernel_size=7,
        dilations=None,
        num_classes=1000,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        layer_scale=None,
        **kwargs,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_levels = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_levels - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = ConvTokenizer(
            in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = WinToMeNATLevel(
                dim=int(embed_dim * 2**i),
                depth=depths[i],
                num_heads=num_heads[i],
                kernel_size=kernel_size,
                dilations=None if dilations is None else dilations[i],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                norm_layer=norm_layer,
                downsample=(i < self.num_levels - 1), # 0, 1, 2
                downsample_method="tome" if (i < self.num_levels - 2) else "conv",
                # "tome" if i = 0, 1 ; "conv" if i = 2
                layer_scale=layer_scale,
            )
            self.levels.append(level)

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
        for idx, level in enumerate(self.levels):
            x = level(x)

        x = self.norm(x).flatten(1, 2)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@register_model
def wintome_nat_mini(pretrained=False, **kwargs):
    model = WinToMeNAT(
        depths=[3, 4, 6, 5],
        num_heads=[2, 4, 8, 16],
        embed_dim=64,
        mlp_ratio=3,
        drop_path_rate=0.2,
        kernel_size=7,
        **kwargs,
    )
    if pretrained:
        url = model_urls["nat_mini_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def wintome_nat_tiny(pretrained=False, **kwargs):
    model = WinToMeNAT(
        depths=[3, 4, 18, 5],
        num_heads=[2, 4, 8, 16],
        embed_dim=64,
        mlp_ratio=3,
        drop_path_rate=0.2,
        kernel_size=7,
        **kwargs,
    )
    if pretrained:
        url = model_urls["nat_tiny_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def wintome_nat_small(pretrained=False, **kwargs):
    model = WinToMeNAT(
        depths=[3, 4, 18, 5],
        num_heads=[3, 6, 12, 24],
        embed_dim=96,
        mlp_ratio=2,
        drop_path_rate=0.3,
        layer_scale=1e-5,
        kernel_size=7,
        **kwargs,
    )
    if pretrained:
        url = model_urls["nat_small_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def wintome_nat_base(pretrained=False, **kwargs):
    model = WinToMeNAT(
        depths=[3, 4, 18, 5],
        num_heads=[4, 8, 16, 32],
        embed_dim=128,
        mlp_ratio=2,
        drop_path_rate=0.5,
        layer_scale=1e-5,
        kernel_size=7,
        **kwargs,
    )
    if pretrained:
        url = model_urls["nat_base_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model
