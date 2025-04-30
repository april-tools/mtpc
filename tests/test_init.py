
import copy
import hydra
import itertools
import pytest
import torch
import numpy

from transformers import AutoTokenizer


def load_mtp(overrides):
    with hydra.initialize(version_base=None, config_path="../configs", job_name=None):
        cfg = hydra.compose(config_name="config", overrides=overrides)
        model = hydra.utils.instantiate(cfg.model).model
        return cfg, model


# Check that the initialised parameters all differ
@pytest.mark.parametrize("expander", ["linear", "mlp"])
def test_mtp_head_params_differ(expander):
    cfg, mt = load_mtp([
        'model=mtp',
        'model.beta=0',
        'lm.n_layer=2',
        'lm.n_head=2',
        'lm.n_embd=32',
        'circuit=cp',
        'circuit.n_component=2',
        'circuit.n_token=3',
        'mt_head=linear',
        f'mt_head.hyperparameters.expander_type={expander}'
    ])

    token_heads = mt.mt_head.token_heads

    for ll, rr in itertools.combinations(range(len(token_heads)), 2):
        left_head = token_heads[ll]
        right_head = token_heads[rr]
        for (ln, lp), (rn, rp) in zip(left_head.named_parameters(), right_head.named_parameters()):
            if expander == 'mlp':
                # Medusa MLP linear layer is initialised to zero
                if 'mlps' in ln and 'bias' not in ln:
                    if ll == 0:
                        assert torch.all(lp == torch.zeros_like(lp))
                    assert torch.all(rp != lp)
                else:
                    assert not torch.allclose(lp, rp)
            elif expander == 'linear':
                if 'Wr' in ln:
                    if ll == 0:
                        for i in range(lp.shape[0]):
                            assert torch.allclose(lp[i], torch.eye(lp.shape[1]))
                    assert torch.all(rp != lp)


@pytest.mark.parametrize("expander", ["linear", "mlp"])
def test_uniform_init_for_future_tokens(expander):
    torch.set_default_dtype(torch.bfloat16)  # type: ignore[no-untyped-call]

    cfg, model = load_mtp([
        'model=mtp',
        'data=fineweb10B',
        'lm=finewebedu',
        'lm.model.encoder_only=false',
        'circuit=fully_factorized',
        'circuit.n_token=4',
        'mt_head=linear',
        f'mt_head.hyperparameters.expander_type={expander}'
    ])

    model.to('cuda')
    tokeniser = AutoTokenizer.from_pretrained(cfg.lm.model.from_huggingface)
    xx = tokeniser.encode('The only problem with having too many', return_tensors='pt')
    xx = xx.to('cuda')
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        out = model(xx[:, :-1], xx[:, 1:], return_log_probs=True)
        probs_after_first = out['full_log_probs'][1:]
        assert torch.allclose(probs_after_first,
                              torch.ones_like(probs_after_first) * -numpy.log(cfg.data.vocab_size),
                              rtol=1e-1)
    torch.set_default_dtype(torch.float64)  # type: ignore[no-untyped-call]


# Check that after training a single step, all params are updated
@pytest.mark.parametrize("expander, freeze_lm, freeze_unembed",
                         itertools.product(["linear", "mlp"], [True, False], [True, False]))
def test_mtp_train_params_differ(expander, freeze_lm, freeze_unembed):
    torch.set_grad_enabled(True)
    torch.manual_seed(13)
    cfg, model = load_mtp([
        'model=mtp',
        'lm.n_layer=2',
        'lm.n_head=2',
        'lm.n_embd=32',
        f'lm.model.freeze={freeze_lm}',
        'circuit=cp',
        'circuit.n_component=2',
        'circuit.n_token=3',
        'mt_head=linear',
        f'mt_head.hyperparameters.freeze_vocab_unembedding={freeze_unembed}',
        f'mt_head.hyperparameters.expander_type={expander}'
    ])
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
        if ln.endswith('.vocab_proj.weight'):
            if freeze_unembed:
                assert torch.allclose(lp, rp)
            else:
                assert not torch.allclose(lp, rp)
        else:
            if not freeze_lm:
                assert not torch.allclose(lp, rp)
            else:
                if 'lm' not in ln:
                    assert not torch.allclose(lp, rp)
                else:
                    assert torch.allclose(lp, rp)
    torch.set_grad_enabled(False)


if __name__ == "__main__":
    # test_mtp_head_params_differ('linear')
    # test_mtp_head_params_differ('mlp')
    test_uniform_init_for_future_tokens('linear')
    test_uniform_init_for_future_tokens('mlp')
