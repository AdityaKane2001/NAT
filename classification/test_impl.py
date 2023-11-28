import torch
from tome.merge import attentive_pruning

x = torch.rand(4, 10, 16)
attn = torch.randn(4, 10, 10)
attn = attn.softmax(dim=-1)

p, _ = attentive_pruning(attn, r=4, class_token=True, distill_token=True)

xp = p(x)
print(xp.shape)
# from wintome_nat import WinToMeNATLevel, WinToMeNeighborhoodAttention2D

# # from natten import NeighborhoodAttention2D as NeighborhoodAttention


# from torchsummary import summary
# import torch
# import sys

# from merge import (
#     windowed_unequal_bipartite_soft_matching,
#     windowed_bipartite_soft_matching,
# )


# def trap(message=""):
#     sys.exit(message)


# def window_partition(x, window_size):
#     """
#     Args:
#         x: (B, H, W, C)
#         window_size (int): window size

#     Returns:
#         windows: (num_windows*B, window_size, window_size, C)
#     """
#     B, H, W, C = x.shape
#     x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
#     windows = (
#         x.permute(0, 1, 3, 2, 4, 5)
#         .contiguous()
#         .view(B, H // window_size, W // window_size, -1, C)
#     )
#     return windows


# def window_reverse(windows, window_size, H, W):
#     """
#     Args:
#         windows: (B, num_windows, num_windows, window_size, window_size, C)
#         window_size (int): Window size
#         H (int): Height of image
#         W (int): Width of image

#     Returns:
#         x: (B, H, W, C)
#     """
#     B = windows.shape[0]  # / (H * W / window_size / window_size))
#     H = windows.shape[1] * windows.shape[3]
#     W = windows.shape[2] * windows.shape[4]
    
#     x = windows.view(
#         B, H // window_size, W // window_size, window_size, window_size, -1
#     )
#     x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
#     return x


# attn = WinToMeNeighborhoodAttention2D(10, 2, kernel_size=3)

# a, k = attn(torch.rand(3, 28, 28, 10), return_keys=True)

# print(f"{a.shape = }")
# print(f"{k.shape = }")

# # trap()

# bs = 1
# p = 6
# c = 3
# window_size = 4

# # mock_a = torch.arange(1, p * p + 1).view(p, p).float()
# # mock_a = mock_a.unsqueeze(-1)
# # mock_a = mock_a.repeat(1, 1, c)
# # mock_a = mock_a.repeat(bs, 1, 1, 1)
# # print(f"{mock_a.shape = }")

# windowed_a = window_partition(a, window_size)
# windowed_keys = window_partition(k.mean(dim=1), window_size)


# print(f"{windowed_a.shape = }")
# print(f"{windowed_keys.shape = }")

# # for k in range(window_size**2):
# # try:
# k = 3 * ((window_size**2) // 4)
# print(f"ToMe output where #tokens is reduced by {k} tokens:")
# m, u = windowed_unequal_bipartite_soft_matching(windowed_keys, r=k)
# merged, once_merged_k = m(windowed_a, return_merged_metric=True)
# print(f"\t{merged.shape = }")
# x = window_reverse(merged, window_size=2, H=a.shape[1] // 2, W=a.shape[2] // 2)
# # print()
# print(f"{a.shape = }")
# print(f"{windowed_a.shape = }")
# print(f"{x.shape = }")
# # print(f"{once_merged_k.shape = }")
# # except:
# #     print(f"err at {k=}")

# # m, u = windowed_bipartite_soft_matching(windowed_keys, r=8)

# # m, u = nat_windowed_bipartite_soft_matching(once_merged_k, r=4)
# # twice_merged_a, twice_merged_k = m(once_merged_a, return_merged_metric=True)
# # print(f"{twice_merged_a.shape = }")
# # print(f"{twice_merged_k.shape = }")

# # print(f"{merged[0,0,0] = }  ")

# # model = nat_tiny()
# # x = torch.rand(2, 3, 224, 224)
# # out = model(x)
# # print(out.shape)
