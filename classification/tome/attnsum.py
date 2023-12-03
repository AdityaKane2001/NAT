import math
import torch

from typing import Tuple, Callable


def attnsum_merge_tail(
    r: int,
    attn: torch.Tensor,
    metric: torch.Tensor,
    has_class_token: bool = False,
    has_distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    
    protected = 0
    if has_class_token:
        protected += 1
    if has_distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    n, t, c = metric.shape
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return lambda x, mode="mean": x, lambda x, mode="mean": x

    with torch.no_grad():
        attn = attn.mean(dim=1)
        t = attn.shape[1]
        attnsum = attn.sum(dim=1)
        if has_class_token:
            attnsum[:, 0] = math.inf
        if has_distill_token:
            attnsum[:, -1] = math.inf

        imp_idx = attnsum.argsort(descending=True)
        
        retain_idx = imp_idx[:, :-2*r]
        retain_idx = torch.sort(retain_idx)[0]
        
        # print(retain_idx)
        
        
        potential_src_dst_idx = imp_idx[:, -2*r:]
        potential_src_idx = potential_src_dst_idx[:, 0::2, None].expand(n, r, c)
        potential_dst_idx = potential_src_dst_idx[:, 1::2, None].expand(n, r, c)
        
        metric = metric / metric.norm(dim=-1, keepdim=True)
        
        metric_a = torch.gather(metric, dim=1, index=potential_src_idx)
        metric_b = torch.gather(metric, dim=1, index=potential_dst_idx)
        
        # scores = metric[..., potential_src_idx, :] @ metric[..., potential_dst_idx, :].transpose(-1, -2)

        scores = metric_a @ metric_b.transpose(-1, -2)
        
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        
        src_idx = edge_idx[..., :r, :] 
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        
    def merge(x, mode="mean"):
        
        n, t, c = x.shape
        
        src, dst = x[..., src_idx, :], x[..., dst_idx, :]
        src = torch.gather(x, dim=1, index=src_idx.expand(n, r, c))
        dst = torch.gather(x, dim=1, index=dst_idx.expand(n, r, c))
        
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        
        if has_distill_token:
            distill_token = x[..., -1, :].unsqueeze(-2)
            x = torch.gather(x, dim=-2, index=retain_idx[:, :-1, None].expand(n, t - (2 * r) - 1, c))
            x = torch.cat([x, dst, distill_token], dim=-2)
        else:
            x = torch.gather(x, dim=-2, index=retain_idx[..., None].expand(n, t - (2 * r), c))
            x = torch.cat([x, dst], dim=-2)
            
        return x
        
    
    def unmerge(x):
        n, t, c = x.shape
        return x
    
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
