import pytest
import torch

from nanogpt.models.circuit import MultiTokenHead, CircuitCP
from nanogpt.models.gpt import GPT
from nanogpt.models.mt import MultiTokenLM


@pytest.fixture
def mtp_cp(
    vocab_size: int = 2,
    n_embd = 12,
    n_layer: int = 3,
    n_head: int = 2,
    n_component: int = 2,
    n_token: int = 4
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
