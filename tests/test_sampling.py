import torch

from mtp.utils.sampling import truncate_probs_top_p, truncate_logprobs_top_p


def test_categorical_distr_truncation():
    probs = torch.tensor([
        [.05, .5, .2, .25],
        [.1, .4, .4, .1]
    ])
    out = truncate_probs_top_p(probs, p=.5)

    expected = torch.tensor([
        [0., 2/3, 0., 1/3],
        [0., .5, .5, .0]
    ])

    assert torch.allclose(out, expected)


def test_truncating_probs_and_logprobs_equiv():
    probs = torch.tensor([
        [.05, .5, .2, .25],
        [.1, .4, .4, .1]
    ])

    logprobs = torch.log(probs)
    out = truncate_logprobs_top_p(logprobs, p=.5)

    expected = torch.tensor([
        [0., 2/3, 0., 1/3],
        [0., .5, .5, .0]
    ])

    assert torch.allclose(torch.exp(out), expected)


def test_categorical_distr_truncation_one_hot():

    probs = torch.eye(10)
    assert torch.allclose(probs, truncate_probs_top_p(probs, p=.1))
    assert torch.allclose(probs, truncate_probs_top_p(probs, p=.9))
