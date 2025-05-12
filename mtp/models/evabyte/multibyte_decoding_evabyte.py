import torch


def multi_byte_pred_prepare_attn_mask(
    config,
    past_seen_tokens: int,
    last_new_tokens: int,
    batch_size: int = 1,
    *,
    device: torch.device
):
    # NOTE: past_key_values has been updated so now
    # seen_tokens incldues new tokens from the last tree iteration
    assert past_seen_tokens > 0
    # so one iteration would not cross two windows
    assert last_new_tokens < config.window_size
    if past_seen_tokens < config.window_size:
        attn_mask = torch.ones(
            (batch_size, 1, last_new_tokens, past_seen_tokens + last_new_tokens),
            dtype=torch.bool,
            device=device
        )
        attn_mask.tril_(past_seen_tokens)
    else:
        # we initialize attn mask each time when
        # 1. the model crosses the window bounary, or
        # 2. after prefilling
        chunks_per_window = int(config.window_size // config.chunk_size)

        window_tokens = past_seen_tokens % config.window_size
        num_windows_seen_so_far = past_seen_tokens // config.window_size
        attn_mask_len = num_windows_seen_so_far * chunks_per_window + window_tokens
        attn_mask = torch.ones(
            (batch_size, 1, last_new_tokens, attn_mask_len + last_new_tokens),
            dtype=torch.bool,
            device=device
        )
        attn_mask.tril_(attn_mask_len)

    return attn_mask
