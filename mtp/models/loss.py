import torch
import torch.nn.functional as F


def compute_full_kl(draft_log_probs: torch.Tensor,
                    teacher_log_probs: torch.Tensor,
                    kl_type: str):
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
                             kl_type: str):
    assert draft_log_probs.shape == teacher_log_probs.shape
    assert kl_type in ('forward', 'reverse')
    assert draft_log_probs.shape == teacher_log_probs.shape
    H, BS = teacher_log_probs.shape

    # TODO: Implement forward and reverse KL
    kl_losses = torch.zeros(H, device=teacher_log_probs.device)
    for h in range(H):
        # Compute log probs of 1 - p(x), use log1p for num stability
        rest_draft_log_probs = torch.log1p(-torch.exp(draft_log_probs))
        rest_teacher_log_probs = torch.log1p(-torch.exp(teacher_log_probs))
        if kl_type == 'forward':
            kl = torch.exp(teacher_log_probs) * (teacher_log_probs - draft_log_probs)
            kl += torch.exp(rest_teacher_log_probs) * (rest_teacher_log_probs - rest_draft_log_probs)
            kl_losses[h] = kl.mean()
        else:
            kl = torch.exp(draft_log_probs) * (draft_log_probs - teacher_log_probs)
            kl += torch.exp(rest_draft_log_probs) * (rest_draft_log_probs - rest_teacher_log_probs)
            kl_losses[h] = kl.mean()
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
        If we shape is (H, BS, V), yy is expected to index the prob of true
        category. Else, probs should be true log probs and yy=None is expected.
        yy (torch.Tensor | None): The targets with shape (BS, 1, H), containing
        the indices of the correct token, or None.
    """
    BS, _, H = yy.shape
    if yy is None:
        assert draft_log_probs.shape == (BS, H)
    else:
        assert draft_log_probs.shape[:2] == (BS, H)
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
