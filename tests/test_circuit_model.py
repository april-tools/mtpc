import pytest
import torch

from mtp.models.circuits import CircuitModel
from mtp.models.loss import IGNORE_TOKEN_ID


BATCH_SIZE = 8


def build_circuit(vocab_size: int, n_token: int, n_component: int, kind: str) -> CircuitModel:
    return CircuitModel(
        vocab_size,
        n_token,
        n_component,
        kind=kind
    )


@pytest.fixture(params=[('cp', 1), ('cp', 2), ('hmm', 2)])
def circuit(request):
    vocab_size = 5
    kind, n_component = request.param
    n_token = 4
    circuit = build_circuit(vocab_size, n_token, n_component, kind=kind)
    parameters_config = circuit.parameters_config
    # Set the parameters of the circuit
    for layer, log_probs_shape in \
        zip(parameters_config.categorical_layers, parameters_config.categorical_log_probs_shapes):
        log_probs_shape = (log_probs_shape[0], BATCH_SIZE, *log_probs_shape[1:])
        layer.log_probs = torch.log_softmax(torch.randn(*log_probs_shape), dim=-1)
    for layer, sum_weights_shape in \
        zip(parameters_config.sum_layers, parameters_config.sum_weights_shapes):
        sum_weights_shape = (sum_weights_shape[0], BATCH_SIZE, *sum_weights_shape[1:])
        layer.weight = torch.softmax(torch.randn(*sum_weights_shape), dim=-1)
    yield circuit


def test_circuit_marginalisation_with_logits(circuit: CircuitModel):
    marg_idx = 2
    yy = torch.randint(circuit.vocab_size, (BATCH_SIZE, circuit.n_token))

    # Just evaluate the idx
    log_probs = circuit.univariate_marginal_at_k(marg_idx, yy, with_logits=False)

    # SET MASK to get all logits for marg_idx
    v_idxs = yy[:, marg_idx].ravel().clone()

    log_probs_all = circuit.univariate_marginal_at_k(marg_idx, with_logits=True)
    # Assert entries we get without all logits agree with all logits case
    assert torch.allclose(log_probs_all[torch.arange(BATCH_SIZE), v_idxs], log_probs)
    # Assert we are getting prob distributions
    assert torch.allclose(torch.exp(log_probs_all).sum(axis=1), torch.ones(BATCH_SIZE))


def test_circuit_conditionals_mask_batch(circuit: CircuitModel):
    yy = torch.randint(circuit.vocab_size, (BATCH_SIZE, circuit.n_token))

    log_probs_all = circuit.autoregressive_conditionals(yy, with_logits=False)

    # All heads for Batch 2
    yy[2] = IGNORE_TOKEN_ID
    # Single head for Batch 5 (we marginalise out last head - since if we marginalise out earlier
    # it affects the follow-up probs
    yy[5, -1] = IGNORE_TOKEN_ID
    marg_mask = (yy == IGNORE_TOKEN_ID)
    log_probs_all_but_one = circuit.autoregressive_conditionals(yy, marg_mask=marg_mask, with_logits=False)

    assert torch.allclose(log_probs_all_but_one[:, 2], torch.zeros_like(log_probs_all[:, 2]))
    assert torch.allclose(log_probs_all_but_one[-1, 5], torch.zeros_like(log_probs_all[-1, 5]))
    # Assert entries we get without all logits agree with all logits case
    not_marginalised = ~marg_mask.permute(1, 0)
    assert torch.allclose(log_probs_all[not_marginalised], log_probs_all_but_one[not_marginalised])


def test_circuit_conditionals_with_logits_mask_batch(circuit: CircuitModel):
    yy = torch.randint(circuit.vocab_size, (BATCH_SIZE, circuit.n_token))

    log_probs_all = circuit.autoregressive_conditionals(yy, with_logits=True)

    # All heads for Batch 2
    yy[2] = IGNORE_TOKEN_ID
    # Single head for Batch 5 (we marginalise out last head - since if we marginalise out earlier
    # it affects the follow-up probs
    yy[5, -1] = IGNORE_TOKEN_ID
    marg_mask = (yy == IGNORE_TOKEN_ID)

    log_probs_all_but_one = circuit.autoregressive_conditionals(yy, marg_mask=marg_mask, with_logits=True)

    assert torch.allclose(log_probs_all_but_one[:, 2], torch.zeros_like(log_probs_all[:, 2]))
    assert torch.allclose(log_probs_all_but_one[-1, 5], torch.zeros_like(log_probs_all[-1, 5]))
    # Assert entries we get without all logits agree with all logits case
    not_marginalised = ~marg_mask.permute(1, 0)
    assert torch.allclose(log_probs_all[not_marginalised, :], log_probs_all_but_one[not_marginalised, :])


def test_circuit_conditionals_with_masked_logits(circuit: CircuitModel):
    yy = torch.randint(circuit.vocab_size, (BATCH_SIZE, circuit.n_token))

    log_probs = circuit.autoregressive_conditionals(yy, with_logits=False)

    log_probs_all = circuit.autoregressive_conditionals(yy, with_logits=True)
    should_match = log_probs_all[
        torch.arange(circuit.n_token)[:, None],
        torch.arange(BATCH_SIZE)[None, :],
        yy.permute(1, 0)
    ]

    # Assert entries we get without all logits agree with all logits case
    assert torch.allclose(log_probs, should_match)


def test_circuit_ntp_equals_univariate(circuit: CircuitModel):
    marg_idx = 0
    yy = torch.randint(circuit.vocab_size, (BATCH_SIZE, circuit.n_token))

    # Just evaluate the idx
    marg_log_probs = circuit.univariate_marginal_at_k(marg_idx, with_logits=True)

    cond_log_probs = circuit.autoregressive_conditionals(yy, with_logits=True)[0]
    assert torch.allclose(marg_log_probs, cond_log_probs)


def test_circuit_joint(circuit: CircuitModel):
    yy = torch.randint(circuit.vocab_size, (BATCH_SIZE, circuit.n_token))

    # The product of the conditionals should be equal to the joint
    cond_log_probs = circuit.autoregressive_conditionals(yy, with_logits=False)
    joint_log_probs_from_cond = sum(cond_log_probs)

    joint = circuit(yy)

    assert torch.allclose(joint_log_probs_from_cond, joint)


def test_circuit_joint_with_batch_mask_all_out(circuit: CircuitModel):
    yy = torch.randint(circuit.vocab_size, (BATCH_SIZE, circuit.n_token))

    marg_mask = torch.ones_like(yy, dtype=torch.bool)

    # The product of the conditionals should be equal to the joint
    cond_log_probs = circuit.autoregressive_conditionals(yy, marg_mask=marg_mask, with_logits=False)
    joint_log_probs_from_cond = sum(cond_log_probs)

    assert torch.allclose(joint_log_probs_from_cond, torch.zeros_like(joint_log_probs_from_cond))


def test_circuit_joint_with_logits(circuit: CircuitModel):
    yy = torch.randint(circuit.vocab_size, (BATCH_SIZE, circuit.n_token))

    # The product of the conditionals should be equal to the joint
    cond_log_probs = circuit.autoregressive_conditionals(yy, with_logits=True)
    should_match = torch.zeros(BATCH_SIZE)
    for i, clp in enumerate(cond_log_probs):
        should_match += clp[torch.arange(BATCH_SIZE), yy[:, i].ravel()]

    joint = circuit(yy)

    assert torch.allclose(should_match, joint)


def test_circuit_conditional_dependency(circuit: CircuitModel):
    marg_idx = 2
    yy = torch.randint(circuit.vocab_size, (BATCH_SIZE, circuit.n_token))

    # Just evaluate the idx
    marg_log_probs = circuit.univariate_marginal_at_k(marg_idx, yy, with_logits=False)
    cond_log_probs = circuit.autoregressive_conditionals(yy, with_logits=False)[marg_idx]

    if circuit.kind == 'cp' and circuit.n_component == 1:
        # The circuit is a product of independent categoricals
        assert torch.allclose(marg_log_probs, cond_log_probs)
    else:
        # The circuit is not a product of independent categoricals
        assert not torch.allclose(marg_log_probs, cond_log_probs)


def test_circuit_conditional_dependency_with_logits(circuit: CircuitModel):
    yy = torch.randint(circuit.vocab_size, (BATCH_SIZE, circuit.n_token))

    cond_log_probs = circuit.autoregressive_conditionals(yy, with_logits=True)

    if circuit.kind == 'cp' and circuit.n_component == 1:
        for marg_idx in range(circuit.n_token):
            marg_log_probs = circuit.univariate_marginal_at_k(marg_idx, with_logits=True)
            # The circuit is a product of independent categoricals
            assert torch.allclose(marg_log_probs, cond_log_probs[marg_idx])
    else:
        # Start from second index, as first is actually the same
        for marg_idx in range(1, circuit.n_token):
            marg_log_probs = circuit.univariate_marginal_at_k(marg_idx, with_logits=True)
            # The circuit is not a product of independent categoricals
            assert not torch.allclose(marg_log_probs, cond_log_probs[marg_idx])


# def test_circuit_ntp_probs():
#     BS, H, R, V = 8, 4, 1, 5

#     cc = CircuitCP(V, H, R)

#     sum_layer = cc.circuit.layers[cc.sum_layer_idx]
#     cat_layer = cc.circuit.layers[cc.cat_layer_idx]

#     cat_layer.log_probs = torch.log_softmax(torch.randn(H, BS, R, V), dim=-1)
#     sum_layer.weight = torch.softmax(torch.randn(1, BS, 1, R), dim=-1)

#     # Compute probs using matmuls
#     # See https://arxiv.org/pdf/2410.17765, eq. 11
#     # (H, B * S', R, V)
#     next_token_cats = torch.exp(cat_layer.log_probs[0, :, :, :])
#     # sum_layer_weight is (1, B * S', 1, R)
#     # (B * S', V)
#     next_token_probs = (sum_layer.weight @ next_token_cats).squeeze(0, 2)

#     # Compute what Circuit returns
#     next_token_log_probs = cc.univariate_marginal_at_k(0, with_logits=True)
#     assert torch.allclose(next_token_probs, torch.exp(next_token_log_probs))


if __name__ == "__main__":
    test_circuit_marginalisation_with_logits()
    test_circuit_conditionals_with_logits()
    test_circuit_ntp_equals_univariate()
    test_circuit_joint()
    test_circuit_joint_with_logits()
    test_circuit_conditional_dependency()
    test_circuit_conditional_dependency_with_logits()
