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


# Check that the initialised parameters all differ
@pytest.mark.parametrize("expander", ["linear", "mlp"])
def test_mtp_head_params_differ(expander):
    cfg, mt = load_mtp(['model=mtp',
                        'lm.n_layer=2',
                        'lm.n_head=2',
                        'lm.n_embd=32',
                        'model.n_component=2',
                        'model.n_token=3',
                        'model.beta=0',
                        'model.token_head.expander.expander_type=%s' % expander])

    token_heads = mt.mt_head.token_heads
    first_head = token_heads[0]
    first_head_params = first_head.named_parameters()
    for head in token_heads[1:]:
        params = head.named_parameters()
        for (ln, lp), (rn, rp) in zip(first_head_params, params):
            if expander == 'mlp':
                # Medusa MLP linear layer is initialised to zero
                if 'mlps' in ln and 'bias' not in ln:
                    assert torch.all(lp == torch.zeros_like(lp)) and torch.all(rp == torch.zeros_like(rp))
                else:
                    assert not torch.allclose(lp, rp)
            # Below is no longer true now that we switched to identity init
            # else:
            #     assert not torch.allclose(lp, rp)

# Check that after training a single step, all params are updated
@pytest.mark.parametrize("expander, freeze_lm", itertools.product(["linear", "mlp"], [True, False]))
def test_mtp_train_params_differ(expander, freeze_lm):
    torch.set_grad_enabled(True)
    torch.manual_seed(13)
    cfg, model = load_mtp(['model=mtp',
                           'lm.n_layer=2',
                           'lm.n_head=2',
                           'lm.n_embd=32',
                           'model.n_component=2',
                           'model.n_token=3',
                           'model.beta=0',
                           'lm.model.freeze=%s' % freeze_lm,
                           'model.token_head.expander.expander_type=%s' % expander])
    mcopy = copy.deepcopy(model)

    model.to(cfg.device)
    mcopy.to(cfg.device)

    optimiser = torch.optim.Adam(model.parameters(), lr=.1)
    batch_size, seq_length = 8, 12
    xx = torch.randint(high=4, size=(batch_size, seq_length + 1), device=cfg.device)
    yy = xx[:, 1:].contiguous()
    xx = xx[:, :-1]
    model.train()
    for i in range(2):
        results = model(xx, yy)
        loss = results['loss']
        loss.backward()
        optimiser.step()

    for (ln, lp), (rn, rp) in zip(model.named_parameters(), mcopy.named_parameters()):
        if not freeze_lm:
            assert not torch.allclose(lp, rp)
        else:
            if 'lm' not in ln:
                assert not torch.allclose(lp, rp)
            else:
                assert torch.allclose(lp, rp)
    torch.set_grad_enabled(False)


# Check that when we use n_layer=0 we get expected behaviour
@pytest.mark.parametrize("expander, th_nlayer, swh_nlayer", itertools.product(["linear", "mlp"], [0, 1], [0, 1]))
def test_zero_layer_encoder(expander, th_nlayer: int, swh_nlayer: int):
    cfg, model = load_mtp(['model=mtp',
                           'lm.n_layer=2',
                           'lm.n_head=2',
                           'lm.n_embd=32',
                           'model.n_component=2',
                           'model.n_token=1',
                           'model.beta=0',
                           'model.token_head.encoder.n_layer=%s' % th_nlayer,
                           'model.sum_weight_head.encoder.n_layer=%s' % swh_nlayer,
                           'model.token_head.expander.expander_type=%s' % expander])
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
    for name, p in model.mt_head.sum_weight_head.named_parameters():
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
