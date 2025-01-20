import itertools

import pytest
import torch

from nanogpt.models.circuit import MultiTokenHead, CircuitCP
from nanogpt.models.gpt import GPT
from nanogpt.models.mt import MultiTokenLM


@pytest.fixture
def mtp_cp(
    vocab_size: int = 2,
    n_embd = 12,
    n_layer: int = 2,
    n_head: int = 2,
    n_component: int = 2,
    n_token: int = 3
) -> MultiTokenLM:
    gpt = GPT(vocab_size, n_embd, n_layer, n_head)
    mt_head = MultiTokenHead(vocab_size, n_embd, n_component, n_token)
    circuit = CircuitCP(vocab_size, n_token, n_component)
    mtp = MultiTokenLM(gpt, mt_head, circuit)
    return mtp


def test_mtp_cp_forward(mtp_cp: MultiTokenLM):
    batch_size, seq_length = 8, 12
    seq = torch.randint(high=2, size=(batch_size, seq_length + 1))
    xx = seq[:, :seq_length]
    yy = seq[:, 1:]
    loss = mtp_cp(xx, yy)
    assert torch.isfinite(loss)
    assert loss >= 0.0


def test_mtp_cp_generate(mtp_cp: MultiTokenLM):
    # TODO: which value is the "beginning of sentence"?
    BOS = 1
    seq = torch.full(size=(1, 1), fill_value=BOS, dtype=torch.int64)
    n_steps = 10
    for _ in range(n_steps):
        toks = mtp_cp.generate(seq)
        seq = torch.concat([seq, toks], dim=1)
    assert seq.shape[0] == 1 and seq.shape[1] == 1 + n_steps * mtp_cp.mt_head.n_token
    assert torch.all(torch.isin(seq, torch.tensor(list(range(mtp_cp.gpt.vocab_size)))))


def test_mtp_cp_self_speculative_generate(mtp_cp: MultiTokenLM):
    # TODO: which value is the "beginning of sentence"?
    BOS = 1
    seq = torch.full(size=(1, 1), fill_value=BOS, dtype=torch.int64)
    n_steps = 10
    for _ in range(n_steps):
        toks = mtp_cp.self_speculative_generate(seq)
        seq = torch.concat([seq, toks], dim=1)
        assert 0 <= len(toks) <= mtp_cp.mt_head.n_token + 1
    assert seq.shape[0] == 1 and seq.shape[1] <= 1 + n_steps * (mtp_cp.mt_head.n_token + 1)
    assert torch.all(torch.isin(seq, torch.tensor(list(range(mtp_cp.gpt.vocab_size)))))


def test_mtp_cp_self_speculative_generate_correctness(mtp_cp: MultiTokenLM):
    # TODO: which value is the "beginning of sentence"?
    BOS = 1
    num_seqs, max_seq_length = 10_000, 3
    seqs = torch.zeros(size=(num_seqs, max_seq_length), dtype=torch.int64)
    for i in range(num_seqs):
        seq = torch.full(size=(1, 1), fill_value=BOS, dtype=torch.int64)
        while seq.shape[1] < max_seq_length:
            toks = mtp_cp.self_speculative_generate(seq)
            seq = torch.concat([seq, toks], dim=1)
            assert 0 <= len(toks) <= mtp_cp.mt_head.n_token + 1
        seq = seq[:, :max_seq_length]
        assert seq.shape == (1, max_seq_length)
        assert torch.all(torch.isin(seq, torch.tensor(list(range(mtp_cp.gpt.vocab_size)))))
        seqs[i] = seq.squeeze(dim=0)
    # Map samples to indices of the probabilities computed above
    # seqs_idx: (num_seqs,)
    seqs_idx = torch.sum(seqs * torch.tensor([0] + [2 ** i for i in range(max_seq_length - 1)]), dim=-1)
    # Compute ratios and compare with the probabilities
    _, counts = torch.unique(seqs_idx, return_counts=True)
    ratios = counts / num_seqs
    assert len(ratios) == 2 ** (max_seq_length - 1)

    # logits: (num_seqs, 1, max_seq_length)
    worlds = torch.tensor(list(itertools.product([0, 1], repeat=max_seq_length - 1)))
    worlds = torch.cat([torch.full(size=(2**(max_seq_length - 1), 1), fill_value=BOS, dtype=torch.int64), worlds], dim=1)
    yy = torch.cat([worlds[:, 1:], torch.zeros((worlds.shape[0], 1), dtype=torch.int64)], dim=1)
    logits, _ = mtp_cp.gpt(worlds, targets=yy, return_logits=True)
    logits = logits[:, :max_seq_length - 1]
    log_probs = torch.log_softmax(logits, dim=-1)
    worlds_log_probs = torch.gather(log_probs, dim=2, index=worlds[:, 1:].unsqueeze(dim=2)).squeeze(dim=2)
    worlds_log_probs = torch.sum(worlds_log_probs, dim=1)
    worlds_probs = torch.exp(worlds_log_probs)
    assert torch.isclose(torch.sum(ratios), torch.tensor(1.0))
    assert torch.isclose(torch.sum(worlds_probs), torch.tensor(1.0))
    assert torch.allclose(ratios, worlds_probs, rtol=1e-3, atol=1e-3)
