import torch
import torch.nn.functional as F

from mtp.utils.extern import log1mexp


IGNORE_TOKEN_ID = -100


def compute_valid_mask(yy: torch.Tensor):
    return yy != IGNORE_TOKEN_ID


def compute_num_valid_tokens(mask: torch.Tensor, dim=None):
    num_valid_tokens = mask.sum(dim=dim)
    # Make sure we do not divide by zero
    num_valid_tokens = torch.clamp(num_valid_tokens, min=1)
    return num_valid_tokens


def compute_full_kl(draft_log_probs: torch.Tensor,
                    teacher_log_probs: torch.Tensor,
                    kl_type: str,
                    valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Computes the Kullback–Leibler (KL) divergence between two distributions.

    Args:
        draft_log_probs (torch.Tensor): Log probabilities from the draft model,
        shape (H, B, S, V), where:
            H: is the # of tokens in the MTP window
            B: is the batch size
            S: is the seq len
            V: is the vocabulary size
        teacher_log_probs (torch.Tensor): Log probabilities from the
        teacher model, shape (H, B, S, V).
        kl_type (str): Specifies the type of KL divergence to compute
        ('forward' or 'reverse').
        valid_mask (torch.Tensor): A boolean mask specifying whether each output should
        be predicted or not (affects mean loss computation), shape (H, B, S)

    Returns:
        kl_losses (torch.Tensor of shape H): KL divergence loss computed
        for each of the H positions in the MTP window. The value for each
        position is the average across the valid sequence and batch dimension
        (not vocab).

    Raises:
        AssertionError: If shapes of `draft_log_probs` and `teacher_log_probs`
            do not match.
        ValueError: If `kl_type` is not 'forward' or 'reverse'.

    Note:
        The forward KL divergence is computed as D_{KL}(P || Q) and the reverse
        as D_{KL}(Q || P).
    """
    assert draft_log_probs.shape == teacher_log_probs.shape
    assert kl_type in ('forward', 'reverse')
    H, B, S, V = draft_log_probs.shape
    assert valid_mask is None or valid_mask.shape == (H, B, S)

    if valid_mask is None:
        valid_mask = torch.ones((H, B, S), dtype=torch.bool)

    # Broadcast valid mask to V dim (cannot have subset of vocab be invalid)
    bc_valid_mask = valid_mask.unsqueeze(-1).broadcast_to(H, B, S, V)
            
    if kl_type == 'forward':
        # We want to compute KL(p || q) = sum_k p_k log p_k/q_k
        # where p = teacher and q = draft
        # The above is written kl_div(q, p) for the pt function
        # We use reduction=none as we need to deal with valid tokens
        # so we aggregate later
        kl_losses = F.kl_div(draft_log_probs,
                             teacher_log_probs,
                             log_target=True,
                             reduction='none')
    else:
        kl_losses = F.kl_div(teacher_log_probs,
                             draft_log_probs,
                             log_target=True,
                             reduction='none')

    # Replace invalid kl losses with 0. so they do not affect sum
    # H, B, S, V
    valid_kl_losses = torch.where(bc_valid_mask, kl_losses, 0.)

    # H, B
    valid_kl_losses = valid_kl_losses.sum(axis=(2,3))
    # H, B
    num_valid_tokens = compute_num_valid_tokens(valid_mask, dim=2)

    valid_kl_losses = (valid_kl_losses / num_valid_tokens).sum(axis=1)

    return valid_kl_losses


def compute_binary_approx_kl(draft_log_probs: torch.Tensor,
                             teacher_log_probs: torch.Tensor,
                             kl_type: str = 'forward',
                             mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Computes an approximate KL divergence between two distributions by grouping
    the probabilities into two categories: target and rest, and computing a KL
    between Bernoulli RVs.

    Args:
        draft_log_probs (torch.Tensor): Log probabilities for the target
        category under the draft model, shape (H, BS), where:
            H: is the # of tokens in the MTP window
            BS: is the seq len * batch size (collapsed)
        teacher_log_probs (torch.Tensor): Log probabilities from the teacher
            model, shape (H, BS).
        kl_type (str): Specifies the KL divergence to compute
            ('forward' or 'reverse').

    Returns:
        kl_losses (torch.Tensor of shape H): Approx KL divergence loss computed
        for each of the H positions in the MTP window. The value for each
        position is the average across the sequence and batch dimension
        (not vocab).

    Raises:
        AssertionError: If shapes of `draft_log_probs` and `teacher_log_probs` do not match.
        ValueError: If `kl_type` is not 'forward' or 'reverse'.
    """
    raise NotImplementedError('For Approx KL, need to deal with new dimensions')
    assert draft_log_probs.shape == teacher_log_probs.shape
    assert kl_type in ('forward', 'reverse')
    assert draft_log_probs.shape == teacher_log_probs.shape
    H, BS = teacher_log_probs.shape
    assert mask is None or mask.shape == (BS, H)

    # Clamp log probs to avoid NaNs
    assert draft_log_probs.dtype == teacher_log_probs.dtype
    # Smallest representable positive number
    epsilon = torch.finfo(teacher_log_probs.dtype).tiny
    max_float = torch.finfo(teacher_log_probs.dtype).max
    # Clamp 0 -> -1.1754943508222875e-38 (for float32)
    # Clamp -inf -> -3.4028234663852886e+38 (for float32)
    draft_log_probs = torch.clamp(draft_log_probs, min=-max_float, max=-epsilon)
    teacher_log_probs = torch.clamp(teacher_log_probs, min=-max_float, max=-epsilon)

    # Compute log (1 - p) where p is given as logprob
    # Use log1mexp for numerical stability
    rest_draft_log_probs = log1mexp(draft_log_probs)
    rest_teacher_log_probs = log1mexp(teacher_log_probs)

    if kl_type == 'forward':
        kl = torch.exp(teacher_log_probs) * (teacher_log_probs - draft_log_probs) + \
            torch.exp(rest_teacher_log_probs) * (rest_teacher_log_probs - rest_draft_log_probs)
    else:
        kl = torch.exp(draft_log_probs) * (draft_log_probs - teacher_log_probs) + \
            torch.exp(rest_draft_log_probs) * (rest_draft_log_probs - rest_teacher_log_probs)
    if mask is None:
        kl_losses = kl.mean(axis=-1)
    else:
        kl_losses = kl.sum(axis=-1) / compute_num_valid_tokens(mask)
    return kl_losses


def compute_cross_entropy(draft_log_probs: torch.Tensor,
                          yy: torch.Tensor):
    """
    Computes the cross-entropy loss for a batch of sequences with given log
    probs and targets. We also support computing cross entropy when
    draft_log_probs contain the log probs for the true categories.

    Args:
        draft_log_probs (torch.Tensor): The logits for the predicted tokens,
        with shape (H, B, S, V) or (H, B, S), where:
            H: is the # of tokens in the MTP window
            BS: is the seq len * batch size (collapsed)
            V: is the vocabulary size
        yy (torch.Tensor): The targets with shape (H, B, S), containing
        the indices of the correct token or IGNORE_TOKEN_ID for tokens that
        should not be predicted.
        If draft_log_probs is (H, B, S, V), yy is expected to index the prob of
        true category. Else, probs should be true log probs.

    Returns:
        Tensor: cross_entropy_loss of each head, shape (H,)
    """
    H, B, S = yy.shape
    assert draft_log_probs.shape[:3] == (H, B, S)

    if len(draft_log_probs.shape) == 3:
        mask = compute_valid_mask(yy)
        # Compute number of valid tokens in the sequence
        num_valid_tokens = compute_num_valid_tokens(mask, dim=2)
        # Compute loss along sequence
        batch_loss_per_head = - draft_log_probs.sum(dim=2)
        # Cross-entropy with one-hot targets == negative log-likelihood
        # The circuit has only computed the log probs for the targets
        ce_losses = (batch_loss_per_head / num_valid_tokens).sum(axis=1)
    elif len(draft_log_probs.shape) == 4:
        # NOTE: log_probs are logits, but not vice-versa
        # we compute cross entropy across the S dimension
        # ce_losses is (H, B)  (because we apply cross ent on the two outer dims via vmap)
        ce_losses = torch.vmap(torch.vmap(F.cross_entropy))(draft_log_probs, yy, ignore_index=IGNORE_TOKEN_ID, reduction='mean')
        # Sum across the batch dimension
        ce_losses = ce_losses.sum(axis=1)
    else:
        raise ValueError('Expected draft_log_probs to be shape (H, B, S, [V]), got %r' % draft_log_probs.shape)
    assert ce_losses.shape == (H,)
    return ce_losses
