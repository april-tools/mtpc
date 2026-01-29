import torch
from itertools import chain
from transformers import AutoTokenizer, AutoModelForCausalLM

from mtp.utils.packing import build_attention_mask, build_position_ids


def test_packing_llama():

    tokenizer = AutoTokenizer.from_pretrained("benjamin/Llama3-2-3B-IT-Byte")
    print("Vocab Size:", len(tokenizer))  # 256 bytes + some special tokens

    model = AutoModelForCausalLM.from_pretrained(
        "benjamin/Llama3-2-3B-IT-Byte", trust_remote_code=True, torch_dtype=torch.float32
    )
    model.cuda()

    examples = ["I like Evabyte", "This should work", "blah"]

    EOS_TOKEN_ID = tokenizer.added_tokens_encoder['<|end_of_text|>']

    # Padding case
    docs = tokenizer.apply_chat_template([[{"role": "user", "content": e}] for e in examples], padding=True, return_tensors='pt')
    docs = docs.cuda()
    outs_pad = model.model(input_ids=docs)

    # Packing case
    docs = tokenizer.apply_chat_template([[{"role": "user", "content": e}] for e in examples])

    seq_lens, packed = [], []
    for doc in docs[:-1]:
        packed.append(doc)
        seq_lens.append(len(doc))
        packed.append([EOS_TOKEN_ID])
    packed.append(docs[-1])
    seq_lens.append(len(docs[-1]))

    packed = torch.tensor(list(chain.from_iterable(packed))).reshape(1, -1)

    position_ids = build_position_ids(packed, EOS_TOKEN_ID)

    packed = packed.cuda()
    position_ids = position_ids.cuda()

    attention_mask = build_attention_mask(position_ids, dtype=outs_pad['last_hidden_state'].dtype)

    outs_pack = model.model(input_ids=packed, position_ids=position_ids, attention_mask=attention_mask)
    outs_pack = outs_pack["last_hidden_state"].squeeze(0)
    # outs_pack = model.model(input_ids=packed, position_ids=position_ids)

    # Check outputs are the same

    # Remove padding from pad_states
    pad_states = [outs_pad["last_hidden_state"][i, :seq_len] for i, seq_len in enumerate(seq_lens)]
    pad_states = torch.concat(pad_states, dim=0)

    # Need to remove the representations obtained for EOD
    pack_states = []
    start = 0
    for sl in seq_lens:
        end = start + sl
        # Skip the next <|end_of_text|>
        pack_states.append(outs_pack[start: end])
        start = end + 1
    pack_states = torch.concat(pack_states, dim=0)

    assert torch.allclose(pad_states, pack_states, rtol=1e-4, atol=1e-4)
