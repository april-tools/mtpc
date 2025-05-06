import torch

from mtp.utils.sampling import truncate_probs_top_p, truncate_logprobs_top_p

from tests.test_circuit_model import build_circuit


def test_categorical_distr_truncation():
    probs = torch.tensor([[0.05, 0.5, 0.2, 0.25], [0.1, 0.4, 0.4, 0.1]])
    out = truncate_probs_top_p(probs, p=0.5)

    expected = torch.tensor([[0.0, 2 / 3, 0.0, 1 / 3], [0.0, 0.5, 0.5, 0.0]])

    assert torch.allclose(out, expected)


def test_truncating_probs_and_logprobs_equiv():
    probs = torch.tensor([[0.05, 0.5, 0.2, 0.25], [0.1, 0.4, 0.4, 0.1]])

    logprobs = torch.log(probs)
    out = truncate_logprobs_top_p(logprobs, p=0.5)

    expected = torch.tensor([[0.0, 2 / 3, 0.0, 1 / 3], [0.0, 0.5, 0.5, 0.0]])

    assert torch.allclose(torch.exp(out), expected)


def test_categorical_distr_truncation_one_hot():

    probs = torch.eye(10)
    assert torch.allclose(probs, truncate_probs_top_p(probs, p=0.1))
    assert torch.allclose(probs, truncate_probs_top_p(probs, p=0.9))


def test_truncation_at_zero_is_approx_argmax():
    circuit = build_circuit(vocab_size=10, n_token=6, n_component=5, kind="cp")

    BATCH_SIZE = 1

    torch.manual_seed(42)

    parameters_config = circuit.parameters_config
    # Set the parameters of the circuit
    for layer, log_probs_shape in zip(
        parameters_config.categorical_layers,
        parameters_config.categorical_log_probs_shapes,
    ):
        log_probs_shape = (log_probs_shape[0], BATCH_SIZE, *log_probs_shape[1:])
        layer.log_probs = torch.log_softmax(torch.randn(*log_probs_shape), dim=-1)
        categorical_argmax = layer.log_probs.argmax(dim=-1)
    for layer, sum_weights_shape in zip(
        parameters_config.sum_layers, parameters_config.sum_weights_shapes
    ):
        sum_weights_shape = (sum_weights_shape[0], BATCH_SIZE, *sum_weights_shape[1:])
        layer.weight = torch.softmax(torch.randn(*sum_weights_shape), dim=-1)
        sum_argmax = layer.weight.argmax(dim=-1)
        print(sum_argmax.shape)

    approx_argmax_tokens = categorical_argmax[:, :, sum_argmax.squeeze()]
    approx_argmax_tokens = approx_argmax_tokens.squeeze()
    print(approx_argmax_tokens)

    torch.manual_seed(42)

    # Set the parameters of the circuit
    for layer, log_probs_shape in zip(
        parameters_config.categorical_layers,
        parameters_config.categorical_log_probs_shapes,
    ):
        log_probs_shape = (log_probs_shape[0], BATCH_SIZE, *log_probs_shape[1:])
        layer.log_probs = torch.log_softmax(torch.randn(*log_probs_shape), dim=-1)
        layer.log_probs = truncate_logprobs_top_p(layer.log_probs, p=0.0)
    for layer, sum_weights_shape in zip(
        parameters_config.sum_layers, parameters_config.sum_weights_shapes
    ):
        sum_weights_shape = (sum_weights_shape[0], BATCH_SIZE, *sum_weights_shape[1:])
        layer.weight = torch.softmax(torch.randn(*sum_weights_shape), dim=-1)
        layer.weight = truncate_probs_top_p(layer.weight, p=0.0)

    approx_argmax_sample, _ = circuit.sample(1)
    approx_argmax_sample = approx_argmax_sample.squeeze()
    assert torch.allclose(approx_argmax_tokens, approx_argmax_sample)

    approx_argmax_sample, _ = circuit.sample(100)
    assert torch.allclose(torch.tile(approx_argmax_tokens, (100, 1)), approx_argmax_sample)
