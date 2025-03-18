import torch

from mtp.models.loss import compute_full_kl, compute_binary_approx_kl


def test_zero_kl():

    pp = torch.tensor([[.1, .7, .2],
                       [.4, .5, .1]])
    tt = torch.log(pp).reshape(1, -1, 3)
    dd = torch.log(pp).reshape(1, -1, 3)

    fkl = compute_full_kl(tt, dd, 'forward')
    assert torch.allclose(fkl, torch.zeros(1))

    rkl = compute_full_kl(tt, dd, 'reverse')
    assert torch.allclose(rkl, torch.zeros(1))


def test_approx_equals_full_forward_seq():

    pp = torch.tensor([[.3, .7],
                       [.2, .8]])
    tt = torch.log(pp).reshape(1, -1, 2)

    ppd = torch.tensor([[.4, .6],
                        [.1, .9]])
    dd = torch.log(ppd).reshape(1, -1, 2)

    fkl = compute_full_kl(tt, dd, 'forward')

    akl = compute_binary_approx_kl(tt[:, :, 0], dd[:, :, 0], 'forward')
    assert torch.allclose(fkl, akl)

    akl = compute_binary_approx_kl(tt[:, :, 1], dd[:, :, 1], 'forward')
    assert torch.allclose(fkl, akl)


def test_approx_equals_full_forward_heads():

    pp = torch.tensor([[.3, .7],
                       [.2, .8]])
    tt = torch.log(pp).reshape(-1, 1, 2)

    ppd = torch.tensor([[.4, .6],
                        [.1, .9]])
    dd = torch.log(ppd).reshape(-1, 1, 2)

    fkl = compute_full_kl(tt, dd, 'forward')

    akl = compute_binary_approx_kl(tt[:, :, 0], dd[:, :, 0], 'forward')
    assert torch.allclose(fkl, akl)

    akl = compute_binary_approx_kl(tt[:, :, 1], dd[:, :, 1], 'forward')
    assert torch.allclose(fkl, akl)


def test_approx_equals_full_forward_both():

    pp = torch.tensor([[[.3, .7],
                        [.2, .8]],
                       [[.1, .9],
                        [.05, .95]]])
    tt = torch.log(pp)

    ppd = torch.tensor([[[.4, .6],
                         [.1, .9]],
                        [[.2, .8],
                         [.25, .75]]])
    dd = torch.log(ppd)

    fkl = compute_full_kl(tt, dd, 'forward')

    akl = compute_binary_approx_kl(tt[:, :, 0], dd[:, :, 0], 'forward')
    assert torch.allclose(fkl, akl)

    akl = compute_binary_approx_kl(tt[:, :, 1], dd[:, :, 1], 'forward')
    assert torch.allclose(fkl, akl)


def test_approx_equals_full_reverse_seq():

    pp = torch.tensor([[.3, .7],
                       [.2, .8]])
    tt = torch.log(pp).reshape(1, -1, 2)

    ppd = torch.tensor([[.4, .6],
                        [.1, .9]])
    dd = torch.log(ppd).reshape(1, -1, 2)

    fkl = compute_full_kl(tt, dd, 'reverse')

    akl = compute_binary_approx_kl(tt[:, :, 0], dd[:, :, 0], 'reverse')
    assert torch.allclose(fkl, akl)

    akl = compute_binary_approx_kl(tt[:, :, 1], dd[:, :, 1], 'reverse')
    assert torch.allclose(fkl, akl)


def test_approx_equals_full_reverse_heads():

    pp = torch.tensor([[.3, .7],
                       [.2, .8]])
    tt = torch.log(pp).reshape(-1, 1, 2)

    ppd = torch.tensor([[.4, .6],
                        [.1, .9]])
    dd = torch.log(ppd).reshape(-1, 1, 2)

    fkl = compute_full_kl(tt, dd, 'reverse')

    akl = compute_binary_approx_kl(tt[:, :, 0], dd[:, :, 0], 'reverse')
    assert torch.allclose(fkl, akl)

    akl = compute_binary_approx_kl(tt[:, :, 1], dd[:, :, 1], 'reverse')
    assert torch.allclose(fkl, akl)


def test_approx_equals_full_forward_rounding():

    pp = torch.tensor([[5e-8, 1 - 5e-8],
                       [3e-8, 1 - 3e-8]])
    tt = torch.log(pp).reshape(1, -1, 2)

    ppd = torch.tensor([[4e-8, 1 - 4e-8],
                        [2e-8, 1 - 2e-8]])
    dd = torch.log(ppd).reshape(1, -1, 2)

    fkl = compute_full_kl(tt, dd, 'forward')

    akl = compute_binary_approx_kl(tt[:, :, 0], dd[:, :, 0], 'forward')
    assert torch.allclose(fkl, akl), '%.5f %.5f' % (fkl, akl)

    akl = compute_binary_approx_kl(tt[:, :, 1], dd[:, :, 1], 'forward')
    assert torch.allclose(fkl, akl), '%.5f %.5f' % (fkl, akl)
