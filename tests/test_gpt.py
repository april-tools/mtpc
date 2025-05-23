import itertools

import pytest
import torch

from mtp.models.gpt import GPT
from mtp.models.lm import LM


@pytest.fixture
def gpt(
    vocab_size: int = 2,
    n_embd = 8,
    n_layer: int = 1,
    n_head: int = 2
) -> LM:
    gpt = GPT(vocab_size, n_embd, n_layer, n_head)
    lm = LM(gpt, ref_enc='encoder', ref_head='head', encoder_only=False, freeze=False)
    return lm


def test_gpt_forward(gpt: LM):
    batch_size, seq_length = 8, 12
    seq = torch.randint(high=2, size=(batch_size, seq_length + 1))
    xx = seq[:, :seq_length]
    yy = seq[:, 1:].contiguous()
    results = gpt(xx, yy, return_logits=False)
    assert torch.isfinite(results['loss'])
    assert results['loss'] >= 0.0


def test_gpt_generate(gpt: LM):
    BOS = 1
    # Sample a bunch of short sentences
    # We will use these samples to get empirical estimates of the sentences distribution
    num_seqs, max_seq_length = 2 ** 17, 4
    seqs = torch.full(size=(num_seqs, 1), fill_value=BOS, dtype=torch.int64)
    while seqs.shape[1] < max_seq_length:
        gresult = gpt.generate(seqs, use_argmax=False, use_cache=False)
        toks = gresult['tokens']
        assert toks.shape == (num_seqs, 1)
        seqs = torch.concat([seqs, toks], dim=1)
    assert torch.all(torch.isin(seqs, torch.tensor(list(range(gpt.encoder.vocab_size)))))
    # Map samples to indices of the probabilities computed above
    # seqs_idx: (num_seqs,)
    seqs_idx = torch.sum(seqs * torch.tensor([0] + list(reversed([2 ** i for i in range(max_seq_length - 1)]))), dim=-1)
    # Compute ratios and compare with the probabilities
    _, counts = torch.unique(seqs_idx, return_counts=True)
    ratios = counts / num_seqs
    assert len(ratios) == 2 ** (max_seq_length - 1)

    # Compute the likelihood of the sentence and check it matches with empirical estimates
    # obtained by sampling sentences (see above)
    worlds = torch.tensor(list(itertools.product([0, 1], repeat=max_seq_length - 1)))
    worlds_seqs = torch.cat([torch.full(size=(worlds.shape[0], 1), fill_value=BOS, dtype=torch.int64), worlds], dim=1)
    xx = worlds_seqs[:, :-1].contiguous()
    yy = worlds_seqs[:, 1:].contiguous()
    results = gpt(xx, labels=yy, return_logits=True)
    assert results['logits'].shape[1] == max_seq_length - 1
    log_probs = torch.log_softmax(results['logits'], dim=-1).double()
    worlds_log_probs = torch.gather(log_probs, dim=2, index=yy.unsqueeze(dim=2)).squeeze(dim=2)
    worlds_log_probs = torch.sum(worlds_log_probs, dim=1)
    worlds_probs = torch.exp(worlds_log_probs)
    assert torch.isclose(torch.sum(ratios), torch.tensor(1.0))
    assert torch.isclose(torch.sum(worlds_probs), torch.tensor(1.0))
    assert torch.allclose(ratios, worlds_probs, rtol=3e-2)
