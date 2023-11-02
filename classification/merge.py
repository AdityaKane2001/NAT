# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
from typing import Callable, Tuple

import torch

# print(f"{tensor.shape = }")


def do_nothing(x, *args, return_merged_metric=False, **kwargs):
    if return_merged_metric:
        return x, x
    return x


def windowed_unequal_bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
):
    """
    metric: tensor corresponding to vectors for each token to calculate merging
        with
    r: number of tokens to reduce from existing tokens 
    (eg. if no. of input tokens is 100 and r is 75, the resulting tensor will 
    have 25 tokens)

    Assumes x to be of the shape (bs, p, p, t, c)
    &  metric to be of the shape (bs, p, p, t, c // num_heads)

    """
    protected = 0
    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[-2]
    # print(f"{t=}")
    # r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        randperm = torch.randperm(t)
        # r = 75% of total tokens
        src_indices, dst_indices = randperm[t - r :], randperm[: t - r]
        a, b = metric[..., src_indices, :], metric[..., dst_indices, :]
        # a: ..., 75, c ; b: ..., 25, c
        scores = a @ b.transpose(-1, -2)
        # print(f"{scores.shape = }")
        node_max, node_idx = scores.max(dim=-1)
        # print(f"{node_max.shape = }")
        # print(f"{node_idx.shape = }")

        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        # print(f"{edge_idx.shape = }")
        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        # print(f"{unm_idx.shape = }")
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        # print(f"{src_idx.shape = }")

        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
        # print(f"{dst_idx.shape = }")

    def merge(x: torch.Tensor, mode="mean", return_merged_metric=False) -> torch.Tensor:
        src, dst = x[..., src_indices, :], x[..., dst_indices, :]

        n, hwin, wwin, _, c = src.shape

        # print(f"{src.shape = }")
        # print(f"{dst.shape = }")

        unm = src.gather(
            dim=-2, index=unm_idx.expand(n, hwin, wwin, unm_idx.shape[-2], c)
        )
        src = src.gather(dim=-2, index=src_idx.expand(n, hwin, wwin, r, c))
        # print(f"{src[0,0,0] = }")
        dst = dst.scatter_reduce(
            -2, dst_idx.expand(n, hwin, wwin, r, c), src, reduce=mode
        )

        if return_merged_metric:
            src_metric, dst_metric = (
                metric[..., src_indices, :],
                metric[..., dst_indices, :],
            )
            metric_c = metric.shape[-1]
            unm_metric = src_metric.gather(
                dim=-2, index=unm_idx.expand(n, hwin, wwin, unm_idx.shape[-2], metric_c)
            )
            src_metric = src_metric.gather(
                dim=-2, index=src_idx.expand(n, hwin, wwin, r, metric_c)
            )

            dst_metric = dst_metric.scatter_reduce(
                -2, dst_idx.expand(n, hwin, wwin, r, metric_c), src_metric, reduce=mode
            )

            return torch.cat([unm, dst], dim=-2), torch.cat(
                [unm_metric, dst_metric], dim=-2
            )
        return torch.cat([unm, dst], dim=-2)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


def windowed_bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
):
    """
    Assumes x to be of the shape (bs, p, p, t, c)
    &  metric to be of the shape (bs, p, p, t, c // num_heads)

    Caveats:
        - Currently supports only square images and square merge windows
        - Manually merges tokens when merge_window_size is less than 3
    """
    protected = 0
    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[-2]
    # print(f"{t=}")
    # r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)
        # print(f"{scores.shape = }")

        node_max, node_idx = scores.max(dim=-1)

        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        # print(f"{edge_idx.shape = }")

        # print(f"{edge_idx[0,0,0] = }")
        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        # print(f"{unm_idx.shape = }")
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        # print(f"{src_idx = }")

        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
        # print(f"{dst_idx = }")

    def merge(x: torch.Tensor, mode="mean", return_merged_metric=False) -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        # print(f"{src.shape = }")
        # print(f"{dst.shape = }")
        n, hwin, wwin, t1, c = src.shape

        # print(f"{unm_idx.shape = }")
        unm = src.gather(dim=-2, index=unm_idx.expand(n, hwin, wwin, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, hwin, wwin, r, c))
        # print(f"{src[0,0,0] = }")
        dst = dst.scatter_reduce(
            -2, dst_idx.expand(n, hwin, wwin, r, c), src, reduce=mode
        )

        if return_merged_metric:
            src_metric, dst_metric = metric[..., ::2, :], metric[..., 1::2, :]
            metric_c = metric.shape[-1]
            unm_metric = src_metric.gather(
                dim=-2, index=unm_idx.expand(n, hwin, wwin, t1 - r, metric_c)
            )
            src_metric = src_metric.gather(
                dim=-2, index=src_idx.expand(n, hwin, wwin, r, metric_c)
            )

            dst_metric = dst_metric.scatter_reduce(
                -2, dst_idx.expand(n, hwin, wwin, r, metric_c), src, reduce=mode
            )

            return torch.cat([unm, dst], dim=-2), torch.cat(
                [unm_metric, dst_metric], dim=-2
            )
        return torch.cat([unm, dst], dim=-2)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    metric: the keys from the previous attention op in the default case, can
        be anything

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    # print(f"{t=}")
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        # print(f"{a.shape = }")
        # print(f"{b.shape = }")
        scores = a @ b.transpose(-1, -2)
        # print(f"{scores = }")

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        # print(f"{node_max = }")
        # print(f"{node_idx = }")

        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        # print(f"{edge_idx = }")

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        # print(f"{src_idx = }")

        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
        # print(f"{dst_idx = }")

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


def kth_bipartite_soft_matching(
    metric: torch.Tensor, k: int
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with the two sets as (every kth element, the rest).
    If n is the number of tokens, resulting number of tokens will be n // z.

    Input size is [batch, tokens, channels].
    z indicates the stride for the first set.
    z = 2 is equivalent to regular bipartite_soft_matching with r = 0.5 * N
    """
    if k <= 1:
        return do_nothing, do_nothing

    def split(x):
        t_rnd = (x.shape[1] // k) * k
        x = x[:, :t_rnd, :].view(x.shape[0], -1, k, x.shape[2])
        a, b = (
            x[:, :, : (k - 1), :].contiguous().view(x.shape[0], -1, x.shape[-1]),
            x[:, :, (k - 1), :],
        )
        return a, b

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        r = a.shape[1]
        scores = a @ b.transpose(-1, -2)

        _, dst_idx = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, _, c = src.shape
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        n, _, c = x.shape
        dst = x

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c)).to(x.dtype)

        src = src.view(n, -1, (k - 1), c)
        dst = dst.view(n, -1, 1, c)

        out = torch.cat([src, dst], dim=-2)
        out = out.contiguous().view(n, -1, c)

        return out

    return merge, unmerge


def random_bipartite_soft_matching(
    metric: torch.Tensor, r: int
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with the two sets as (r chosen randomly, the rest).
    Input size is [batch, tokens, channels].

    This will reduce the number of tokens by r.
    """
    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        B, N, _ = metric.shape
        rand_idx = torch.rand(B, N, 1, device=metric.device).argsort(dim=1)

        a_idx = rand_idx[:, :r, :]
        b_idx = rand_idx[:, r:, :]

        def split(x):
            C = x.shape[-1]
            a = x.gather(dim=1, index=a_idx.expand(B, r, C))
            b = x.gather(dim=1, index=b_idx.expand(B, N - r, C))
            return a, b

        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        _, dst_idx = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        C = src.shape[-1]
        dst = dst.scatter_reduce(-2, dst_idx.expand(B, r, C), src, reduce=mode)

        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        C = x.shape[-1]
        dst = x
        src = dst.gather(dim=-2, index=dst_idx.expand(B, r, C))

        out = torch.zeros(B, N, C, device=x.device, dtype=x.dtype)

        out.scatter_(dim=-2, index=a_idx.expand(B, r, C), src=src)
        out.scatter_(dim=-2, index=b_idx.expand(B, N - r, C), src=dst)

        return out

    return merge, unmerge


def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x, size


def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source
