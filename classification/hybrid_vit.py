"""
Dilated Neighborhood Attention Transformer.
https://arxiv.org/abs/2209.15001

Isotropic models.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from natten import NeighborhoodAttention2D as NeighborhoodAttention
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from torch.nn.functional import pad


model_urls = {
    # ImageNet-1K
    "vitrpb_small_1k": "https://shi-labs.com/projects/dinat/checkpoints/isotropic/vitrpb_small_in1k_224.pth",
    "nat_isotropic_small_1k": "https://shi-labs.com/projects/dinat/checkpoints/isotropic/nat_isotropic_small_in1k_224.pth",
    "dinat_isotropic_small_1k": "https://shi-labs.com/projects/dinat/checkpoints/isotropic/dinat_isotropic_small_in1k_224.pth",
    "vitrpb_base_1k": "https://shi-labs.com/projects/dinat/checkpoints/isotropic/vitrpb_base_in1k_224.pth",
    "nat_isotropic_base_1k": "https://shi-labs.com/projects/dinat/checkpoints/isotropic/nat_isotropic_base_in1k_224.pth",
    "dinat_isotropic_base_1k": "https://shi-labs.com/projects/dinat/checkpoints/isotropic/dinat_isotropic_base_in1k_224.pth",
}


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


class MHSARPB(nn.Module):
    """
    Self Attention + RPB
    """

    def __init__(
        self,
        dim,
        input_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.input_size = input_size[0] if type(input_size) is tuple else input_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.rpb = nn.Parameter(
            torch.zeros(num_heads, (2 * self.input_size - 1), (2 * self.input_size - 1))
        )
        trunc_normal_(self.rpb, std=0.02)
        coords_h = torch.arange(self.input_size)
        coords_w = torch.arange(self.input_size)
        coords = torch.stack(
            torch.meshgrid([coords_h, coords_w], indexing="ij")
        )  # 2, L, L
        coords_flatten = torch.flatten(coords, 1)  # 2, L^2
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, L^2, L^2
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # L^2, L^2, 2
        relative_coords[:, :, 0] += self.input_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.input_size - 1
        relative_coords[:, :, 0] *= 2 * self.input_size - 1
        relative_position_index = torch.flipud(
            torch.fliplr(relative_coords.sum(-1))
        )  # L^2, L^2
        self.register_buffer("relative_position_index", relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def apply_pb(self, attn):
        relative_position_bias = (
            self.rpb.permute(1, 2, 0)
            .flatten(0, 1)[self.relative_position_index.view(-1)]
            .view(self.input_size**2, self.input_size**2, -1)
        )
        relative_position_bias = (
            relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        )
        return attn + relative_position_bias

    def forward(self, x):
        B, N, C = x.shape
        # N = H * W
        # num_tokens = int(self.input_size**2)
        # if N != num_tokens:
        #     raise RuntimeError(
        #         f"Feature map size ({H} x {W}) is not equal to "
        #         + f"expected size ({self.input_size} x {self.input_size}). "
        #     )
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = self.apply_pb(attn)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # print(x.shape)

        return self.proj_drop(self.proj(x))


class MHSA(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class VisionTransformerLayer(nn.Module):
    def __init__(
        self,
        dim,
        input_size,
        num_heads,
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
        self.attn = MHSA(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
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


class VisionTransformerRPBLayer(nn.Module):
    def __init__(
        self,
        dim,
        input_size,
        num_heads,
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
        self.attn = MHSARPB(
            dim,
            input_size=input_size,
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


class HybridNATIsotropic(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=384,
        local_depth=12,
        global_depth=0,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        kernel_size=7,
        dilation=2,
        local_layer=NATransformerLayer,
        global_layer=VisionTransformerRPBLayer,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.img_size = img_size
        self.feature_map_size = img_size // patch_size

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
            x.item()
            for x in torch.linspace(0, drop_path_rate, local_depth + global_depth)
        ]  # stochastic depth decay rule

        # build layers
        self.local_layers = nn.ModuleList()
        self.global_layers = nn.ModuleList()
        for i in range(local_depth):
            _layer = local_layer(
                dim=embed_dim,
                input_size=self.feature_map_size,
                num_heads=num_heads,
                kernel_size=kernel_size,
                dilation=1 if i % 2 == 0 else dilation,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            self.local_layers.append(_layer)

        for i in range(global_depth):
            _layer = global_layer(
                dim=embed_dim,
                input_size=self.feature_map_size,
                num_heads=num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[local_depth + i],
                norm_layer=norm_layer,
            )
            self.global_layers.append(_layer)

        self.norm = norm_layer(embed_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
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
        B = x.shape[0]
        x = self.patch_embed(x).reshape(
            B, self.feature_map_size, self.feature_map_size, self.embed_dim
        )
        x = self.pos_drop(x)

        for layer in self.local_layers:
            x = layer(x)

        x = x.view(x.shape[0], -1, x.shape[-1])

        for layer in self.global_layers:
            x = layer(x)

        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@register_model
def hybrid_nat_norpb_isotropic_small(pretrained=False, **kwargs):
    model = HybridNATIsotropic(
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=384,
        local_depth=6,
        global_depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        kernel_size=7,
        dilation=1,
        local_layer=NATransformerLayer,
        global_layer=VisionTransformerLayer,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        **kwargs,
    )
    if pretrained:
        url = model_urls["nat_norpb_isotropic_small_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def hybrid_dinat_norpb_isotropic_small(pretrained=False, **kwargs):
    model = HybridNATIsotropic(
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=384,
        local_depth=6,
        global_depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        kernel_size=7,
        dilation=2,
        local_layer=NATransformerLayer,
        global_layer=VisionTransformerLayer,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        **kwargs,
    )
    if pretrained:
        url = model_urls["dinat_norpb_isotropic_small_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def hybrid_nat_norpb_isotropic_base(pretrained=False, **kwargs):
    model = HybridNATIsotropic(
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        local_depth=6,
        global_depth=6,
        num_heads=24,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        kernel_size=7,
        dilation=1,
        local_layer=NATransformerLayer,
        global_layer=VisionTransformerLayer,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.4,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        **kwargs,
    )
    if pretrained:
        url = model_urls["nat_norpb_isotropic_base_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def hybrid_dinat_norpb_isotropic_base(pretrained=False, **kwargs):
    model = HybridNATIsotropic(
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        local_depth=6,
        global_depth=6,
        num_heads=24,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        kernel_size=7,
        dilation=2,
        local_layer=NATransformerLayer,
        global_layer=VisionTransformerLayer,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.4,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        **kwargs,
    )
    if pretrained:
        url = model_urls["dinat_norpb_isotropic_base_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def hybrid_nat_rpb_isotropic_small(pretrained=False, **kwargs):
    model = HybridNATIsotropic(
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=384,
        local_depth=6,
        global_depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        kernel_size=7,
        dilation=1,
        local_layer=NATransformerLayer,
        global_layer=VisionTransformerRPBLayer,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        **kwargs,
    )
    if pretrained:
        url = model_urls["nat_rpb_isotropic_small_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def hybrid_dinat_rpb_isotropic_small(pretrained=False, **kwargs):
    model = HybridNATIsotropic(
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=384,
        local_depth=6,
        global_depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        kernel_size=7,
        dilation=2,
        local_layer=NATransformerLayer,
        global_layer=VisionTransformerRPBLayer,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        **kwargs,
    )
    if pretrained:
        url = model_urls["dinat_rpb_isotropic_small_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def hybrid_nat_rpb_isotropic_base(pretrained=False, **kwargs):
    model = HybridNATIsotropic(
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        local_depth=6,
        global_depth=6,
        num_heads=24,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        kernel_size=7,
        dilation=1,
        local_layer=NATransformerLayer,
        global_layer=VisionTransformerRPBLayer,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.4,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        **kwargs,
    )
    if pretrained:
        url = model_urls["nat_rpb_isotropic_base_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def hybrid_dinat_rpb_isotropic_base(pretrained=False, **kwargs):
    model = HybridNATIsotropic(
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        local_depth=6,
        global_depth=6,
        num_heads=24,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        kernel_size=7,
        dilation=2,
        local_layer=NATransformerLayer,
        global_layer=VisionTransformerRPBLayer,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.4,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        **kwargs,
    )
    if pretrained:
        url = model_urls["dinat_rpb_isotropic_base_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model
