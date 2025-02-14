import hydra

import pytest
import torch


def load_mtp(overrides):
    with hydra.initialize(version_base=None, config_path="../configs", job_name=None):
        cfg = hydra.compose(config_name="config", overrides=overrides)

        model = hydra.utils.instantiate(cfg.model).model
        return model


# Check that the initialised parameters all differ
@pytest.mark.parametrize("expander", ["linear", "mlp"])
def test_mtp_head_params_differ(expander):
    mt = load_mtp(['model=mtp',
                   'model.n_layer=2',
                   'model.n_head=2',
                   'model.n_embd=32',
                   'model.n_component=2',
                   'model.n_token=3',
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
            else:
                assert not torch.allclose(lp, rp)


if __name__ == "__main__":
    test_mtp_head_params_differ('linear')
    test_mtp_head_params_differ('mlp')
