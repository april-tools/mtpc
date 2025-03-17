import torch
import torch.nn.functional as F


def compute_full_kl(draft_log_probs: torch.Tensor,
                    teacher_log_probs: torch.Tensor,
                    kl_type: str) -> torch.Tensor:
    """
    Computes the Kullback–Leibler (KL) divergence between two distributions.

    Args:
        draft_log_probs (torch.Tensor): Log probabilities from the draft model,
        shape (H, BS, V), where:
            H: is the # of tokens in the MTP window
            BS: is the seq len * batch size (collapsed)
            V: is the vocabulary size
        teacher_log_probs (torch.Tensor): Log probabilities from the
        teacher model, shape (H, BS, V).
        kl_type (str): Specifies the type of KL divergence to compute
        ('forward' or 'reverse').

    Returns:
        kl_losses (torch.Tensor of shape H): KL divergence loss computed
        for each of the H positions in the MTP window. The value for each
        position is the average across the sequence and batch dimension
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
    H, BS, V = draft_log_probs.shape

    kl_losses = torch.zeros(H, device=teacher_log_probs.device)
    for h in range(H):
        if kl_type == 'forward':
            # We want to compute KL(target_model || draft_model)
            # For usual order: input, target, pt computes forward KL.
            # target = torch.softmax(torch.randn(3, 5), dim=-1)
            # draft = torch.softmax(torch.randn(3, 5), dim=-1)
            # kl_f = (target * torch.log(target / draft)).sum(axis=1).mean()
            # # NOTE: the flip in order arguments for pt below
            # kl_f_pt = F.kl_div(torch.log(draft), torch.log(target), log_target=True, reduction='batchmean')
            # assert torch.allclose(kl_f, kl_f_pt)
            # So we do draft, target order for forward KL:
            kl_losses[h] = F.kl_div(draft_log_probs[h],
                                    teacher_log_probs[h],
                                    log_target=True,
                                    reduction='batchmean')
        else:
            kl_losses[h] = F.kl_div(teacher_log_probs[h],
                                    draft_log_probs[h],
                                    log_target=True,
                                    reduction='batchmean')
    return kl_losses


def compute_binary_approx_kl(draft_log_probs: torch.Tensor,
                             teacher_log_probs: torch.Tensor,
                             kl_type: str) -> torch.Tensor:
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
    assert draft_log_probs.shape == teacher_log_probs.shape
    assert kl_type in ('forward', 'reverse')
    assert draft_log_probs.shape == teacher_log_probs.shape
    H, BS = teacher_log_probs.shape

    epsilon = torch.finfo(teacher_log_probs.dtype).eps
    # Compute log probs of 1 - p(x), use log1p for num stability
    rest_draft_log_probs = torch.log1p(-torch.exp(torch.clamp(draft_log_probs, max=-epsilon)))
    rest_teacher_log_probs = torch.log1p(-torch.exp(torch.clamp(teacher_log_probs, max=-epsilon)))

    # kl_losses = torch.zeros(H, device=teacher_log_probs.device)
    # for h in range(H):
    if kl_type == 'forward':
        kl = torch.exp(teacher_log_probs) * (teacher_log_probs - draft_log_probs)
        kl += torch.exp(rest_teacher_log_probs) * (rest_teacher_log_probs - rest_draft_log_probs)
        kl_losses = kl.mean(axis=-1)
    else:
        kl = torch.exp(draft_log_probs) * (draft_log_probs - teacher_log_probs)
        kl += torch.exp(rest_draft_log_probs) * (rest_draft_log_probs - rest_teacher_log_probs)
        kl_losses = kl.mean(axis=-1)
    return kl_losses


def compute_cross_entropy(draft_log_probs: torch.Tensor,
                          yy: torch.Tensor | None = None):
    """
    Computes the cross-entropy loss for a batch of sequences with given log
    probs and targets. We also support computing cross entropy when
    draft_log_probs contain the log probs for the true categories.

    Args:
        draft_log_probs (torch.Tensor): The logits for the predicted tokens,
        with shape (H, BS, V) or (H, BS), where:
            H: is the # of tokens in the MTP window
            BS: is the seq len * batch size (collapsed)
            V: is the vocabulary size
        yy (torch.Tensor | None): The targets with shape (BS, 1, H), containing
        the indices of the correct token, or None.
        If draft_log_probs is (H, BS, V), yy is expected to index the prob of
        true category. Else, probs should be true log probs and yy=None is
        expected.
    """
    H, BS = draft_log_probs.shape[:2]
    if yy is None:
        assert len(draft_log_probs.shape) == 2
    else:
        assert len(draft_log_probs.shape) == 3
        assert yy.shape == (BS, 1, H)
    ce_losses = torch.zeros(H, device=draft_log_probs.device)
    for h in range(H):
        if yy is None:
            # Cross-entropy with one-hot targets == negative log-likelihood
            # The circuit has only computed the log probs for the targets
            ce_losses[h] = -draft_log_probs[h].mean()
        else:
            # NOTE: log_probs are logits, but not vice-versa
            ce_losses[h] = F.cross_entropy(draft_log_probs[h], yy[:, :, h].ravel())
    return ce_losses
