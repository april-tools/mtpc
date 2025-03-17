import pytest
import torch

from .test_circuit_model import build_circuit


@pytest.mark.parametrize(
    "kind", [('fully_factorized',), ('cp',), ('hmm',)]
)
def test_circuit_marginalisation_with_logits(kind: str):
    BS, H, R, V = 8, 4, 2, 5
    marg_idx = 2

    cc = build_circuit(V, H, R, kind=kind)

    sum_layer = cc.circuit.layers[cc.sum_layer_idx]
    cat_layer = cc.circuit.layers[cc.cat_layer_idx]

    cat_layer.log_probs = torch.log_softmax(torch.randn(H, BS, R, V), dim=-1)
    sum_layer.weight = torch.softmax(torch.randn(1, BS, 1, R), dim=-1)

    yy = torch.randint(V, (BS, 1, H))

    # Just evaluate the idx
    log_probs = cc.univariate_marginal_at_k(marg_idx, yy, with_logits=False)

    # SET MASK to get all logits for marg_idx
    v_idxs = yy[:, :, marg_idx].ravel().clone()

    log_probs_all = cc.univariate_marginal_at_k(marg_idx, with_logits=True)
    # Assert entries we get without all logits agree with all logits case
    assert torch.allclose(log_probs_all[torch.arange(BS), v_idxs], log_probs)
    # Assert we are getting prob distributions
    assert torch.allclose(torch.exp(log_probs_all).sum(axis=1), torch.ones(BS))


def test_circuit_conditionals_with_logits():
    BS, H, R, V = 8, 4, 1, 5

    cc = CircuitCP(V, H, R)

    sum_layer = cc.circuit.layers[cc.sum_layer_idx]
    cat_layer = cc.circuit.layers[cc.cat_layer_idx]

    cat_layer.log_probs = torch.log_softmax(torch.randn(H, BS, R, V), dim=-1)
    sum_layer.weight = torch.softmax(torch.randn(1, BS, 1, R), dim=-1)

    yy = torch.randint(V, (BS, 1, H))

    log_probs = cc.autoregressive_conditionals(yy, with_logits=False)
    log_probs = torch.stack(log_probs)

    log_probs_all = cc.autoregressive_conditionals(yy, with_logits=True)
    log_probs_all = torch.stack(log_probs_all)
    match = log_probs_all[torch.arange(H)[:, None], torch.arange(BS)[None, :], yy.squeeze().permute(1, 0)]

    # Assert entries we get without all logits agree with all logits case
    assert torch.allclose(log_probs, match)


def test_circuit_ntp_probs():
    BS, H, R, V = 8, 4, 1, 5

    cc = CircuitCP(V, H, R)

    sum_layer = cc.circuit.layers[cc.sum_layer_idx]
    cat_layer = cc.circuit.layers[cc.cat_layer_idx]

    cat_layer.log_probs = torch.log_softmax(torch.randn(H, BS, R, V), dim=-1)
    sum_layer.weight = torch.softmax(torch.randn(1, BS, 1, R), dim=-1)

    # Compute probs using matmuls
    # See https://arxiv.org/pdf/2410.17765, eq. 11
    # (H, B * S', R, V)
    next_token_cats = torch.exp(cat_layer.log_probs[0, :, :, :])
    # sum_layer_weight is (1, B * S', 1, R)
    # (B * S', V)
    next_token_probs = (sum_layer.weight @ next_token_cats).squeeze(0, 2)

    # Compute what Circuit returns
    next_token_log_probs = cc.univariate_marginal_at_k(0, with_logits=True)
    assert torch.allclose(next_token_probs, torch.exp(next_token_log_probs))


def test_circuit_ntp_equals_univariate():
    BS, H, R, V = 8, 4, 2, 5
    marg_idx = 0

    cc = CircuitCP(V, H, R)

    sum_layer = cc.circuit.layers[cc.sum_layer_idx]
    cat_layer = cc.circuit.layers[cc.cat_layer_idx]

    cat_layer.log_probs = torch.log_softmax(torch.randn(H, BS, R, V), dim=-1)
    sum_layer.weight = torch.softmax(torch.randn(1, BS, 1, R), dim=-1)

    yy = torch.randint(V, (BS, 1, H))

    # Just evaluate the idx
    marg_log_probs = cc.univariate_marginal_at_k(marg_idx, with_logits=True)

    cond_log_probs = cc.autoregressive_conditionals(yy, with_logits=True)[0]
    assert torch.allclose(marg_log_probs, cond_log_probs)


def test_circuit_joint():
    BS, H, R, V = 8, 4, 2, 5

    cc = CircuitCP(V, H, R)

    sum_layer = cc.circuit.layers[cc.sum_layer_idx]
    cat_layer = cc.circuit.layers[cc.cat_layer_idx]

    cat_layer.log_probs = torch.log_softmax(torch.randn(H, BS, R, V), dim=-1)
    sum_layer.weight = torch.softmax(torch.randn(1, BS, 1, R), dim=-1)

    yy = torch.randint(V, (BS, 1, H))

    # The product of the conditionals should be equal to the joint
    cond_log_probs = cc.autoregressive_conditionals(yy, with_logits=False)
    joint_log_probs_from_cond = sum(cond_log_probs)

    joint = cc(yy)

    assert torch.allclose(joint_log_probs_from_cond, joint)


def test_circuit_joint_with_logits():
    BS, H, R, V = 8, 4, 2, 5

    cc = CircuitCP(V, H, R)

    sum_layer = cc.circuit.layers[cc.sum_layer_idx]
    cat_layer = cc.circuit.layers[cc.cat_layer_idx]

    cat_layer.log_probs = torch.log_softmax(torch.randn(H, BS, R, V), dim=-1)
    sum_layer.weight = torch.softmax(torch.randn(1, BS, 1, R), dim=-1)

    yy = torch.randint(V, (BS, 1, H))

    # The product of the conditionals should be equal to the joint
    cond_log_probs = cc.autoregressive_conditionals(yy, with_logits=True)
    match = torch.zeros(BS)
    for i, clp in enumerate(cond_log_probs):
        match += clp[torch.arange(BS), yy[:, :, i].ravel()]

    joint = cc(yy)

    assert torch.allclose(match, joint)


def test_circuit_conditional_independence():
    # NOTE : we set R = 1, so this is just a product of categoricals
    BS, H, R, V = 8, 4, 1, 5
    marg_idx = 2

    cc = CircuitCP(V, H, R)

    sum_layer = cc.circuit.layers[cc.sum_layer_idx]
    cat_layer = cc.circuit.layers[cc.cat_layer_idx]

    cat_layer.log_probs = torch.log_softmax(torch.randn(H, BS, R, V), dim=-1)
    sum_layer.weight = torch.softmax(torch.randn(1, BS, 1, R), dim=-1)

    yy = torch.randint(V, (BS, 1, H))

    # Just evaluate the idx
    marg_log_probs = cc.univariate_marginal_at_k(marg_idx, yy, with_logits=False)
    cond_log_probs = cc.autoregressive_conditionals(yy, with_logits=False)[marg_idx]
    assert torch.allclose(marg_log_probs, cond_log_probs)


def test_circuit_dependence():
    # NOTE : we set R = 2, so this is not just a product of categoricals
    BS, H, R, V = 8, 4, 2, 5
    marg_idx = 2

    cc = CircuitCP(V, H, R)

    sum_layer = cc.circuit.layers[cc.sum_layer_idx]
    cat_layer = cc.circuit.layers[cc.cat_layer_idx]

    cat_layer.log_probs = torch.log_softmax(torch.randn(H, BS, R, V), dim=-1)
    sum_layer.weight = torch.softmax(torch.randn(1, BS, 1, R), dim=-1)

    yy = torch.randint(V, (BS, 1, H))

    # Just evaluate the idx
    marg_log_probs = cc.univariate_marginal_at_k(marg_idx, yy, with_logits=False)
    cond_log_probs = cc.autoregressive_conditionals(yy, with_logits=False)[marg_idx]
    assert not torch.allclose(marg_log_probs, cond_log_probs)


def test_circuit_conditional_independence_with_logits():
    # NOTE : we set R = 1, so this is just a product of categoricals
    BS, H, R, V = 8, 4, 1, 5

    cc = CircuitCP(V, H, R)

    sum_layer = cc.circuit.layers[cc.sum_layer_idx]
    cat_layer = cc.circuit.layers[cc.cat_layer_idx]

    cat_layer.log_probs = torch.log_softmax(torch.randn(H, BS, R, V), dim=-1)
    sum_layer.weight = torch.softmax(torch.randn(1, BS, 1, R), dim=-1)

    yy = torch.randint(V, (BS, 1, H))

    cond_log_probs = cc.autoregressive_conditionals(yy, with_logits=True)
    # Just evaluate the idx
    for marg_idx in range(H):
        marg_log_probs = cc.univariate_marginal_at_k(marg_idx, with_logits=True)
        assert torch.allclose(marg_log_probs, cond_log_probs[marg_idx])


def test_circuit_dependence_with_logits():
    # NOTE : we set R = 1, so this is just a product of categoricals
    BS, H, R, V = 8, 4, 2, 5
    marg_idx = 1

    cc = CircuitCP(V, H, R)

    sum_layer = cc.circuit.layers[cc.sum_layer_idx]
    cat_layer = cc.circuit.layers[cc.cat_layer_idx]

    cat_layer.log_probs = torch.log_softmax(torch.randn(H, BS, R, V), dim=-1)
    sum_layer.weight = torch.softmax(torch.randn(1, BS, 1, R), dim=-1)

    yy = torch.randint(V, (BS, 1, H))

    cond_log_probs = cc.autoregressive_conditionals(yy, with_logits=True)
    # Start from second index, as first is actually the same
    for marg_idx in range(1, H):
        marg_log_probs = cc.univariate_marginal_at_k(marg_idx, with_logits=True)
        assert not torch.allclose(marg_log_probs, cond_log_probs[marg_idx])


if __name__ == "__main__":

    test_circuit_marginalisation_with_logits()
    test_circuit_conditionals_with_logits()
    test_circuit_ntp_equals_univariate()
    test_circuit_joint()
    test_circuit_joint_with_logits()
    test_circuit_conditional_independence()
    test_circuit_dependence()
    test_circuit_conditional_independence_with_logits()
    test_circuit_dependence_with_logits()
