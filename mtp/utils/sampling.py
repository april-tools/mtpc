import torch


# Truncation used in Nucleus sampling
def truncate_probs_top_p(probs, p):
    # performs the operation on the last dimension
    argsorts = probs.argsort(dim=-1, descending=True)
    sorted_probs = torch.gather(probs, -1, argsorts)
    keep_mask = sorted_probs.cumsum(dim=-1) <= p
    # Also keep prob of first token that surpasses the threshold
    keep_mask = torch.scatter(keep_mask, -1, keep_mask.sum(axis=-1, keepdims=True), True)
    # Map mask back to original ids with inverse perm
    inv_argsorts = argsorts.argsort(dim=-1)
    keep_mask = torch.gather(keep_mask, -1, inv_argsorts)
    truncated_probs = torch.where(keep_mask, probs, torch.zeros_like(probs))
    truncated_probs = truncated_probs / truncated_probs.sum(dim=-1, keepdims=True)
    return truncated_probs


# Same as above but applied to logprobs
def truncate_logprobs_top_p(logprobs, p):
    # performs the operation on the last dimension
    probs = torch.exp(logprobs)
    argsorts = probs.argsort(dim=-1, descending=True)
    sorted_probs = torch.gather(probs, -1, argsorts)
    keep_mask = sorted_probs.cumsum(dim=-1) <= p
    # Also keep prob of first token that surpasses the threshold
    keep_mask = torch.scatter(keep_mask, -1, keep_mask.sum(axis=-1, keepdims=True), True)
    # Map mask back to original ids with inverse perm
    inv_argsorts = argsorts.argsort(dim=-1)
    keep_mask = torch.gather(keep_mask, -1, inv_argsorts)
    truncated_logprobs = torch.where(keep_mask, logprobs, torch.full_like(probs, -torch.inf))
    truncated_logprobs = truncated_logprobs - truncated_logprobs.logsumexp(dim=-1, keepdims=True)
    return truncated_logprobs
