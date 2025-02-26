import torch

from mtp.models.circuits import CircuitCP


def test_circuit_marginalisation_with_logits():
    BS, H, R, V = 8, 4, 2, 5
    marg_idx = 2

    cc = CircuitCP(V, H, R)

    sum_layer = cc.circuit.layers[cc.sum_layer_idx]
    cat_layer = cc.circuit.layers[cc.cat_layer_idx]

    cat_layer.log_probs = torch.log_softmax(torch.randn(H, BS, R, V), dim=-1)
    sum_layer.weight = torch.softmax(torch.randn(1, BS, 1, R), dim=-1)

    yy = torch.randint(V, (BS, 1, H))

    # Just evaluate the idx
    log_probs = cc.univariate_marginal_at_k(marg_idx, yy, with_logits=False)

    # SET MASK to get all logits for marg_idx
    v_idxs = yy[:, :, marg_idx].ravel().clone()
    yy[:, :, marg_idx] = -1

    log_probs_all = cc.univariate_marginal_at_k(marg_idx, yy, with_logits=True)
    # Assert entries weget without all logits agree with all logits case
    assert torch.allclose(log_probs_all[torch.arange(BS), v_idxs], log_probs)
    # Assert we are getting prob distributions
    assert torch.allclose(torch.exp(log_probs_all).sum(axis=1), torch.ones(BS))


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
    marg_log_probs = cc.univariate_marginal_at_k(marg_idx, yy, with_logits=True)

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
    joint_log_probs_from_cond = cond_log_probs.sum(dim=0)

    joint = cc(yy)

    assert torch.allclose(joint_log_probs_from_cond, joint)


if __name__ == "__main__":

    test_circuit_marginalisation_with_logits()
    test_circuit_ntp_equals_univariate()
    test_circuit_joint()
