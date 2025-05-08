import os
import math
import copy
import torch
import pytest

from mtp.models.loss import compute_full_kl, compute_binary_approx_kl, compute_cross_entropy
from mtp.models.loss import IGNORE_TOKEN_ID
from mtp.models.mtp import MultiTokenLM
from mtp.models.lm import LM
from mtp.models.gpt import GPT
from mtp.train import training_step
from mtp.utils.distributed import setup_distributed, wrap_model_distributed

from tests.test_circuit_model import build_circuit


class MockDataLoader:
    def __init__(self, input_ids, labels, batch_size=1):
        self.input_ids = input_ids
        self.labels = labels
        self.batch_size = batch_size
        self.i = 0

    def next_batch(self):

        if self.i < len(self.input_ids):

            out =  {
                "input_ids": self.input_ids[self.i:self.i+self.batch_size],
                "labels": self.labels[self.i:self.i+self.batch_size]
            }
            self.i += self.batch_size
            return out
        else:
            raise StopIteration()



def create_model():
    vocab_size = 5
    n_embd = 8
    n_layer = 1
    n_head = 2
    kind, n_component = 'cp', 2
    n_token = 3
    lm = LM(
        lm=GPT(vocab_size, n_embd, n_layer, n_head),
        ref_enc='encoder',
        ref_head='head',
        encoder_only=True
    )
    circuit = build_circuit(vocab_size, n_token, n_component, kind=kind)
    mtp = MultiTokenLM(
        lm,
        circuit,
        mt_head_kwargs={'n_embd': n_embd, 'n_head': n_head, 'transformer_n_layer': 1},
        beta=0
    )
    return mtp


def test_cross_entropy_with_mask():

    pp = torch.tensor([[.1, .7, .2],
                       [.6, .3, .1],
                       [.2, .7, .1],
                       [.4, .5, .1],
                       [.2, .7, .1],
                       [.1, .85, .05]])
    logprobs = torch.log(pp).reshape(2, 1, 3, 3)
    yy = torch.tensor([0, 1, 0, 2, 1, 0], dtype=torch.long)
    yys = yy.reshape(2, 1, 3)
    yys[0, :, 2:] = IGNORE_TOKEN_ID
    yys[1, :, :-1] = IGNORE_TOKEN_ID

    masked_loss = compute_cross_entropy(logprobs, yys)

    correct_loss = - torch.tensor([math.log(.03)/2, math.log(.1)])
    assert torch.allclose(masked_loss, correct_loss)


def test_cross_entropy_single_log_prob_with_mask():

    # We would rely on circuit marginalisation to assign prob 1.
    # to every output that should not be predicted
    pp = torch.tensor([[.1, .3, 1.], [1., 1., .1]])
    logprobs = torch.log(pp.reshape(2, 1, 3))
    yy = torch.zeros((2 * 3), dtype=torch.long)
    yys = yy.reshape(2, 1, 3)
    yys[0, :, 2:] = IGNORE_TOKEN_ID
    yys[1, :, :-1] = IGNORE_TOKEN_ID

    masked_loss = compute_cross_entropy(logprobs, yys)

    correct_loss = - torch.tensor([math.log(.03)/2, math.log(.1)])
    assert torch.allclose(masked_loss, correct_loss)


# def test_device_batch_length_does_not_affect_loss():
#     input_ids = torch.tensor([[2, 1, 3, 2, 0, 0, 0], [1, 3, 0, 0, 0, 0, 0]])
#     labels = torch.tensor([[2, 1, 3, 2, 1, 1, -100], [1, 3, 2, 1,-100, -100, -100]])
#
#     dl = MockDataLoader(input_ids, labels, batch_size=1)
#
#     model = create_model()
#     model.device = 'cpu'
#
#     # Create Adam optimizer with default settings
#     optimizer = torch.optim.Adam(model.parameters())
#
#
#     ctx = torch.autocast(device_type='cpu')
#
#     loss_a, _ = training_step(model, dl, train_accumulation_steps=2, batch_size=2, optimizer=optimizer, scheduler=None, ctx=ctx)
#
#     dl2 = MockDataLoader(input_ids, labels, batch_size=2)
#     loss_b, _ = training_step(model_cp, dl2, train_accumulation_steps=1, batch_size=2, optimizer=optimizer, scheduler=None, ctx=ctx)
#
#     assert torch.allclose(loss_a, loss_b)


def test_padding_does_not_affect_loss():
    input_ids = torch.tensor([[2, 1, 3, 2, 0, 4, 0]])
    labels = torch.tensor([[1, 3, 2, 0, 4, 0, 1]])
    attention = torch.tensor([[1, 1, 1, 1, 1, 1, 1]])

    model = create_model()
    model.device = 'cpu'

    loss_a = model(input_ids=input_ids, labels=labels, attention_mask=attention)['loss']

    input_ids = torch.tensor([[2, 1, 3, 2, 0, 4, 0, 0, 0]])
    labels = torch.tensor([[1, 3, 2, 0, 4, 0, 1, -100, -100]])
    attention = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]])

    loss_b = model(input_ids=input_ids, labels=labels, attention_mask=attention)['loss']

    assert torch.allclose(loss_a, loss_b)


# TODO: Add below back when we correct the ordering
# def test_zero_kl():
#
#     pp = torch.tensor([[.1, .7, .2],
#                        [.4, .5, .1]])
#     tt = torch.log(pp).reshape(1, -1, 3)
#     dd = torch.log(pp).reshape(1, -1, 3)
#
#     fkl = compute_full_kl(tt, dd, 'forward')
#     assert torch.allclose(fkl, torch.zeros(1))
#
#     rkl = compute_full_kl(tt, dd, 'reverse')
#     assert torch.allclose(rkl, torch.zeros(1))
#
#
# def test_binary_zero_not_nan():
#
#     tt = torch.tensor([0., -torch.inf]).reshape(2, 1)
#     dd = torch.tensor([0., -torch.inf]).reshape(2, 1)
#
#     fkl = compute_binary_approx_kl(tt, dd, 'forward')
#     assert torch.allclose(fkl, torch.zeros_like(fkl))
#
#     rkl = compute_binary_approx_kl(tt, dd, 'reverse')
#     assert torch.allclose(rkl, torch.zeros_like(rkl))
#
#
# def test_approx_equals_full_forward_seq():
#
#     pp = torch.tensor([[.3, .7],
#                        [.2, .8]])
#     tt = torch.log(pp).reshape(1, -1, 2)
#
#     ppd = torch.tensor([[.4, .6],
#                         [.1, .9]])
#     dd = torch.log(ppd).reshape(1, -1, 2)
#
#     fkl = compute_full_kl(tt, dd, 'forward')
#
#     akl = compute_binary_approx_kl(tt[:, :, 0], dd[:, :, 0], 'forward')
#     assert torch.allclose(fkl, akl)
#
#     akl = compute_binary_approx_kl(tt[:, :, 1], dd[:, :, 1], 'forward')
#     assert torch.allclose(fkl, akl)
#
#
# def test_approx_equals_full_forward_heads():
#
#     pp = torch.tensor([[.3, .7],
#                        [.2, .8]])
#     tt = torch.log(pp).reshape(-1, 1, 2)
#
#     ppd = torch.tensor([[.4, .6],
#                         [.1, .9]])
#     dd = torch.log(ppd).reshape(-1, 1, 2)
#
#     fkl = compute_full_kl(tt, dd, 'forward')
#
#     akl = compute_binary_approx_kl(tt[:, :, 0], dd[:, :, 0], 'forward')
#     assert torch.allclose(fkl, akl)
#
#     akl = compute_binary_approx_kl(tt[:, :, 1], dd[:, :, 1], 'forward')
#     assert torch.allclose(fkl, akl)
#
#
# def test_approx_equals_full_forward_both():
#
#     pp = torch.tensor([[[.3, .7],
#                         [.2, .8]],
#                        [[.1, .9],
#                         [.05, .95]]])
#     tt = torch.log(pp)
#
#     ppd = torch.tensor([[[.4, .6],
#                          [.1, .9]],
#                         [[.2, .8],
#                          [.25, .75]]])
#     dd = torch.log(ppd)
#
#     fkl = compute_full_kl(tt, dd, 'forward')
#
#     akl = compute_binary_approx_kl(tt[:, :, 0], dd[:, :, 0], 'forward')
#     assert torch.allclose(fkl, akl)
#
#     akl = compute_binary_approx_kl(tt[:, :, 1], dd[:, :, 1], 'forward')
#     assert torch.allclose(fkl, akl)
#
#
# def test_approx_equals_full_reverse_seq():
#
#     pp = torch.tensor([[.3, .7],
#                        [.2, .8]])
#     tt = torch.log(pp).reshape(1, -1, 2)
#
#     ppd = torch.tensor([[.4, .6],
#                         [.1, .9]])
#     dd = torch.log(ppd).reshape(1, -1, 2)
#
#     fkl = compute_full_kl(tt, dd, 'reverse')
#
#     akl = compute_binary_approx_kl(tt[:, :, 0], dd[:, :, 0], 'reverse')
#     assert torch.allclose(fkl, akl)
#
#     akl = compute_binary_approx_kl(tt[:, :, 1], dd[:, :, 1], 'reverse')
#     assert torch.allclose(fkl, akl)
#
#
# def test_approx_equals_full_reverse_heads():
#
#     pp = torch.tensor([[.3, .7],
#                        [.2, .8]])
#     tt = torch.log(pp).reshape(-1, 1, 2)
#
#     ppd = torch.tensor([[.4, .6],
#                         [.1, .9]])
#     dd = torch.log(ppd).reshape(-1, 1, 2)
#
#     fkl = compute_full_kl(tt, dd, 'reverse')
#
#     akl = compute_binary_approx_kl(tt[:, :, 0], dd[:, :, 0], 'reverse')
#     assert torch.allclose(fkl, akl)
#
#     akl = compute_binary_approx_kl(tt[:, :, 1], dd[:, :, 1], 'reverse')
#     assert torch.allclose(fkl, akl)
#
#
# def test_approx_equals_full_forward_rounding():
#
#     pp = torch.tensor([[5e-8, 1 - 5e-8],
#                        [3e-8, 1 - 3e-8]])
#     tt = torch.log(pp).reshape(1, -1, 2)
#
#     ppd = torch.tensor([[4e-8, 1 - 4e-8],
#                         [2e-8, 1 - 2e-8]])
#     dd = torch.log(ppd).reshape(1, -1, 2)
#
#     fkl = compute_full_kl(tt, dd, 'forward')
#
#     akl = compute_binary_approx_kl(tt[:, :, 0], dd[:, :, 0], 'forward')
#     assert torch.allclose(fkl, akl), '%.5f %.5f' % (fkl, akl)
#
#     akl = compute_binary_approx_kl(tt[:, :, 1], dd[:, :, 1], 'forward')
#     assert torch.allclose(fkl, akl), '%.5f %.5f' % (fkl, akl)
