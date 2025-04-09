import hydra
import itertools

import pytest


def load_mtp(overrides):
    with hydra.initialize(version_base=None, config_path="../configs", job_name=None):
        cfg = hydra.compose(config_name="config", overrides=overrides)
        model = hydra.utils.instantiate(cfg.model).model
        return cfg, model


# Check that when we use n_layer=0 we get expected behaviour
@pytest.mark.parametrize("expander, transf_nlayer", itertools.product(["linear", "mlp"], [0, 1]))
def test_zero_layer_encoder(expander, transf_nlayer: int):
    cfg, model = load_mtp([
        'model=mtp',
        'model.beta=0',
        'lm=nanogpt',
        'lm.n_layer=2',
        'lm.n_head=2',
        'lm.n_embd=32',
        'circuit=cp',
        'circuit.n_component=2',
        'circuit.n_token=3',
        'mt_head=transformer',
        f'mt_head.hyperparameters.transformer_n_layer={transf_nlayer}',
        f'mt_head.hyperparameters.expander_type={expander}'
    ])

    token_heads = model.mt_head.token_heads
    for head in token_heads:
        transformer_found = False
        for name, p in head.named_parameters():
            if 'transformer' in name:
                transformer_found = True
        if transf_nlayer == 0:
            assert not transformer_found
        else:
            assert transformer_found

    transformer_found = False
    sum_weight_heads = model.mt_head.sum_weight_heads
    for head in sum_weight_heads:
        for name, p in head.named_parameters():
            if 'transformer' in name:
                transformer_found = True
    if transf_nlayer == 0:
        assert not transformer_found
    else:
        assert transformer_found


if __name__ == "__main__":
    test_zero_layer_encoder('mlp', 1, 1)
