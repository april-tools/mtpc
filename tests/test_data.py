import torch
from mtp.data import DistributedDataLoader
from mtp.models.loss import IGNORE_TOKEN_ID


def test_hf_dataloader_seek():
    dl = DistributedDataLoader.resolve(
        "allenai/tulu-3-sft-mixture", "EvaByte/EvaByte", 20, 2048, 0, 1
    )
    batch = dl.next_batch()
    batch = dl.next_batch()
    batch = dl.next_batch()

    dl.seek(2)
    found = dl.next_batch()
    for k in found:
        assert torch.allclose(found[k], batch[k])


def test_hf_dataloader_tulu_shuffle():
    # Make sure we are seeing instances from all tulu sources
    dl = DistributedDataLoader.resolve(
        "allenai/tulu-3-sft-mixture", "EvaByte/EvaByte", 1, 2048, 0, 1
    )

    di = iter(dl.dataset)

    sources = set()
    for i, batch in enumerate(di):
        sources.add(batch['source'][0])
        if i == 1000:
            break
    # We ignore Tulu 3 hardcoded, it has 240 prompts
    assert len(sources) == 17


def test_hf_dataloader_tulu_label_masking():
    # Make sure we are seeing instances from all tulu sources
    dl = DistributedDataLoader.resolve(
        "allenai/tulu-3-sft-mixture", "EvaByte/EvaByte", 1, 2048, 0, 1
    )

    batch = dl.next_batch()

    assert torch.all(batch['input_ids'] != IGNORE_TOKEN_ID)

    seq_labels = batch['labels'][:, :-1][batch['attention_mask'][:, 1:].bool()]
    assert torch.any(seq_labels == IGNORE_TOKEN_ID)

    pred_labels = batch['labels'][batch['labels'] != -100]

    assert torch.allclose(pred_labels, seq_labels[-len(pred_labels):])


def test_hf_dataloader_batching():

    dl = DistributedDataLoader.resolve(
        "allenai/tulu-3-sft-mixture", "EvaByte/EvaByte", 2, 2048, 0, 1
    )
    out = dl.next_batch()
    out = dl.next_batch()

    dl = DistributedDataLoader.resolve(
        "allenai/tulu-3-sft-mixture", "EvaByte/EvaByte", 1, 2048, 0, 1
    )
    out2 = dl.next_batch()
    out2 = dl.next_batch()
    out2 = dl.next_batch()
    for k in out:
        assert torch.allclose(out2[k][0], out[k][0])
    out2 = dl.next_batch()
    for k in out:
        assert torch.allclose(out2[k][0], out[k][1])


def test_hf_dataloader_ddp():

    dl = DistributedDataLoader.resolve(
        "allenai/tulu-3-sft-mixture", "EvaByte/EvaByte", 2, 2048, 0, 2
    )
    out = dl.next_batch()

    dl = DistributedDataLoader.resolve(
        "allenai/tulu-3-sft-mixture", "EvaByte/EvaByte", 2, 2048, 1, 2
    )
    out2 = dl.next_batch()
    for k in out:
        assert not torch.allclose(out2[k][0], out[k][0])
