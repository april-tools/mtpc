import copy
import hydra
import itertools

import pytest
import torch


def load_mtp(overrides):
    with hydra.initialize(version_base=None, config_path="../configs", job_name=None):
        cfg = hydra.compose(config_name="config", overrides=overrides)
        model = hydra.utils.instantiate(cfg.model).model
        return cfg, model


# Check that when we use n_layer=0 we get expected behaviour
@pytest.mark.parametrize("expander, th_nlayer, swh_nlayer", itertools.product(["linear", "mlp"], [0, 1], [0, 1]))
def test_zero_layer_encoder(expander, th_nlayer: int, swh_nlayer: int):
    cfg, model = load_mtp([
        'model=mtp',
        'lm.n_layer=2',
        'lm.n_head=2',
        'lm.n_embd=32',
        'model.n_component=2',
        'model.n_token=3',
        'model.beta=0',
        f'model.mt_head_hparams.tok_transformer_n_layer={th_nlayer}',
        f'model.mt_head_hparams.sum_transformer_n_layer={swh_nlayer}',
        f'model.mt_head_hparams.expander_type={expander}']
    )

    token_heads = model.mt_head.token_heads
    for head in token_heads:
        transformer_found = False
        for name, p in head.named_parameters():
            if 'transformer' in name:
                transformer_found = True
        if th_nlayer == 0:
            assert not transformer_found
        else:
            assert transformer_found

    transformer_found = False
    sum_weight_heads = model.mt_head.sum_weight_heads
    for head in sum_weight_heads:
        for name, p in head.named_parameters():
            if 'transformer' in name:
                transformer_found = True
    if swh_nlayer == 0:
        assert not transformer_found
    else:
        assert transformer_found


if __name__ == "__main__":
    test_mtp_head_params_differ('linear')
    test_mtp_head_params_differ('mlp')
    test_mtp_train_params_differ('mlp')
    test_zero_layer_encoder('mlp', 1, 1)
