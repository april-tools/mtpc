import itertools

import pytest
import torch

from mtp.models.circuits import CircuitCP
from mtp.models.gpt import GPT
from mtp.models.mtp_head import MultiTokenHead, OutputHead
from mtp.models.mtp_head import TransformerEncoderHead, ExpanderHead
from mtp.models.mtp import MultiTokenLM
from mtp.models.lm import LM


@pytest.fixture
def mtp_cp(
    vocab_size: int = 2,
    n_embd: int = 8,
    n_layer: int = 1,
    n_head: int = 2,
    n_component: int = 2,
    n_token: int = 2
) -> MultiTokenLM:
    lm = LM(
        lm=GPT(vocab_size, n_embd, n_layer, n_head),
        ref_enc='encoder',
        ref_head='head',
        encoder_only=False
    )
    token_head = OutputHead(
        encoder=TransformerEncoderHead(
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer
        ),
        expander=ExpanderHead(
            expander_type='linear',
            n_embd=n_embd,
            n_component=n_component,
            n_layer=1
        ))
    sum_head = OutputHead(
        encoder=TransformerEncoderHead(
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer
        ),
        expander=torch.nn.Linear(n_embd, n_component)
    )
    mt_head = MultiTokenHead(
        token_head=token_head,
        sum_weight_head=sum_head,
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_component=n_component,
        n_token=n_token
    )
    circuit = CircuitCP(vocab_size, n_token, n_component)
    mtp = MultiTokenLM(lm, mt_head, circuit)
    return mtp


def test_mtp_cp_forward(mtp_cp: MultiTokenLM):
    batch_size, seq_length = 8, 12
    seq = torch.randint(high=2, size=(batch_size, seq_length + 1))
    xx = seq[:, :seq_length]
    yy = seq[:, 1:]
    results = mtp_cp(xx, yy)
    assert torch.isfinite(results['loss'])
    assert results['loss'] >= 0.0


def test_mtp_cp_generate(mtp_cp: MultiTokenLM):
    # TODO: which value is the "beginning of sentence"?
    BOS = 1
    # Sample a bunch of short sentences
    # We will use these samples to get empirical estimates of the sentences distribution
    num_steps = 2
    num_seqs, max_seq_length = 2 ** 17, mtp_cp.mt_head.n_token * num_steps + 1
    seqs = torch.full(size=(num_seqs, 1), fill_value=BOS, dtype=torch.int64)
    for _ in range(num_steps):
        toks = mtp_cp.generate(seqs)['tokens']
        assert toks.shape == (num_seqs, mtp_cp.mt_head.n_token)
        seqs = torch.cat([seqs, toks], dim=1)
    assert torch.all(torch.isin(seqs, torch.tensor(list(range(mtp_cp.lm.lm.vocab_size)))))
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
    results = mtp_cp(xx, yy, return_log_probs=True)
    log_probs = results['log_probs'].view(worlds.shape[0], -1)
    assert log_probs.shape[1] == max_seq_length - mtp_cp.mt_head.n_token
    worlds_log_probs = torch.sum(log_probs[:, [i * mtp_cp.mt_head.n_token for i in range(num_steps)]], dim=1)
    worlds_probs = torch.exp(worlds_log_probs)
    assert torch.isclose(torch.sum(ratios), torch.tensor(1.0))
    assert torch.isclose(torch.sum(worlds_probs), torch.tensor(1.0))
    assert torch.allclose(ratios, worlds_probs, rtol=4e-2), \
        torch.max(torch.abs(worlds_probs / ratios - 1.0))


def test_mtp_cp_self_speculative_generate(mtp_cp: MultiTokenLM):
    # TODO: which value is the "beginning of sentence"?
    BOS = 1
    # Sample a bunch of short sentences
    # We will use these samples to get empirical estimates of the sentences distribution
    num_seqs, max_seq_length = 2 ** 18, mtp_cp.mt_head.n_token * 2 + 1
    seqs = torch.zeros(size=(num_seqs, max_seq_length), dtype=torch.int64)
    num_accepted_tokens = []
    for i in range(num_seqs):
        seq = torch.full(size=(1, 1), fill_value=BOS, dtype=torch.int64)
        while seq.shape[1] < max_seq_length:
            toks = mtp_cp.self_speculative_generate(seq)['tokens']
            assert len(toks.shape) == 2 and toks.shape[0] == 1
            assert 1 <= toks.shape[1] <= mtp_cp.mt_head.n_token + 1
            seq = torch.concat([seq, toks], dim=1)
            num_accepted_tokens.append(toks.shape[1] - 1)
        seq = seq[:, :max_seq_length]
        assert seq.shape == (1, max_seq_length)
        assert torch.all(torch.isin(seq, torch.tensor(list(range(mtp_cp.lm.lm.vocab_size)))))
        seqs[i] = seq.squeeze(dim=0)
    assert any(j != 0 and j != mtp_cp.mt_head.n_token for j in num_accepted_tokens)
    # Map samples to indices of the probabilities computed above
    # seqs_idx: (num_seqs,)
    seqs_idx = torch.sum(seqs * torch.tensor([0] + list(reversed([2 ** i for i in range(max_seq_length - 1)]))), dim=-1)
    # Compute ratios and compare with the probabilities
    _, counts = torch.unique(seqs_idx, return_counts=True)
    ratios = counts / num_seqs
    assert len(ratios) == 2 ** (max_seq_length - 1)

    # Compute the likelihood of the sentence and check it matches with empirical estimates
    # obtained by sampling sentences (see above)
    #
    # Since self-speculative decoding as implemented by Leviathan et al. should sample
    # the same distributions that the target model would generate (in expectation),
    # the sentences likelihood and the empirical distribution estimates should match
    worlds = torch.tensor(list(itertools.product([0, 1], repeat=max_seq_length - 1)))
    worlds_seqs = torch.cat([torch.full(size=(worlds.shape[0], 1), fill_value=BOS, dtype=torch.int64), worlds], dim=1)
    xx = worlds_seqs[:, :-1].contiguous()
    yy = worlds_seqs[:, 1:].contiguous()
    results = mtp_cp.lm(xx, targets=yy, return_logits=True)
    assert results['logits'].shape[1] == max_seq_length - 1
    log_probs = torch.log_softmax(results['logits'], dim=-1)
    worlds_log_probs = torch.gather(log_probs, dim=2, index=yy.unsqueeze(dim=2)).squeeze(dim=2)
    worlds_log_probs = torch.sum(worlds_log_probs, dim=1)
    worlds_probs = torch.exp(worlds_log_probs)
    assert torch.isclose(torch.sum(ratios), torch.tensor(1.0))
    assert torch.isclose(torch.sum(worlds_probs), torch.tensor(1.0))
    assert torch.allclose(ratios, worlds_probs, rtol=3e-2), \
        torch.max(torch.abs(worlds_probs / ratios - 1.0))
