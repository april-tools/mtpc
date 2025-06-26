import os
import torch
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from torch import autocast

from mtp.utils.checkpoint import Checkpoint
from mtp.train import set_deterministic


def score_autoregressive(model, input_ids, samples, device='cuda', batch_size=5):
    assert len(input_ids.shape) == len(samples.shape) == 2
    num_samples, sample_len = samples.shape
    # Prepare context
    context = input_ids.repeat(samples.shape[0], 1)
    inputs = torch.cat((context, samples), dim=1)
    outputs = []
    with torch.no_grad(), autocast(device_type=device, dtype=torch.bfloat16) as ctx:
        for i in range(0, num_samples, batch_size):
            outs = model(input_ids=inputs[i:min(i+batch_size, num_samples)], use_cache=False, return_dict=True).logits
            outputs.append(outs)
    outs = torch.cat(outputs, dim=0)
    logprobs = torch.log_softmax(outs, dim=-1)
    lls = logprobs[:, -(sample_len+1):-1, :].gather(-1, samples.unsqueeze(-1))
    lls = lls.squeeze(-1).sum(-1)
    return lls


def score_evabyte(model, input_ids, samples, device='cuda', batch_size=5):
    # Score using EvaByte's independent heads
    assert len(input_ids.shape) == len(samples.shape) == 2
    num_samples, sample_len = samples.shape
    assert sample_len == 8
    # Prepare context
    context = input_ids.repeat(samples.shape[0], 1)
    inputs = torch.cat((context, samples), dim=1)
    outputs = []
    with torch.no_grad(), autocast(device_type=device, dtype=torch.bfloat16) as ctx:
        for i in range(0, num_samples, batch_size):
            outs = model(input_ids=inputs[i:min(i+batch_size, num_samples)], use_cache=False, return_dict=True, return_all_pred_logits=True).logits
            outputs.append(outs)
    outs = torch.cat(outputs, dim=0)
    logprobs = torch.log_softmax(outs, dim=-1)
    lls = logprobs[:, -(sample_len+1), :, :].gather(-1, samples.unsqueeze(-1))
    lls = lls.squeeze(-1).sum(-1)
    return lls


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to load model on')
    parser.add_argument('--seed', type=int, default=13, help='Random seed')
    parser.add_argument('--num-samples', type=int, default=20, help='Num samples to take from circuit')
    parser.add_argument('--prompt', type=str, required=True, help='Text to compare samples on')
    parser.add_argument('--draft-top-p', type=float, default=1., help='Cumulative distribution to truncate'
    ' probability (1. for no truncation, 0. corresponds to approx argmax prediction)')
    args = parser.parse_args()

    device = args.device
    checkpoint = args.checkpoint
    prompt = args.prompt
    assert 0 <= args.draft_top_p <= 1

    set_deterministic(args.seed)

    os.environ['DEVICE'] = device

    ckp = Checkpoint.load(checkpoint)
    print('Results for %s' % ckp.expname)
    lm = ckp.model
    lm.eval()

    # vanilla_lm = AutoModelForCausalLM.from_pretrained(lm.lm.from_huggingface, torch_dtype=torch.bfloat16, trust_remote_code=True, quantization_config=BitsAndBytesConfig(load_in_4bit=True))
    vanilla_lm = AutoModelForCausalLM.from_pretrained(lm.lm.from_huggingface, torch_dtype=torch.bfloat16, trust_remote_code=True)
    vanilla_lm.cuda()
    vanilla_lm.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(lm.lm.from_huggingface, use_fast=True, trust_remote_code=True)
    tokens = tokenizer(prompt, return_tensors='pt')
    if device == 'cuda':
        tokens['input_ids'] = tokens['input_ids'].cuda()
    with torch.no_grad(), autocast(device_type=device, dtype=torch.bfloat16):
        out = lm.generate(inputs=tokens['input_ids'], draft_top_p=args.draft_top_p)

    NS = args.num_samples
    out = lm.circuit.sample(NS)
    scores = lm.circuit(out)
    ar_scores = score_autoregressive(vanilla_lm, tokens['input_ids'], out, 'cuda')
    eva_scores = score_evabyte(vanilla_lm, tokens['input_ids'], out, 'cuda')
    order = scores.argsort()

    print('        Circuit     EvaByte NTP   EvaByte MTP    Sample')
    for i in range(NS):
        print(f'LL: {scores[order[i]]:12.2f}', f'{ar_scores[order[i]]:12.2f}', f'{eva_scores[order[i]]:12.2f}   ', '"%s"' % tokenizer.decode(out[order[i]]).replace('\n', '_'))
