import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DynamicCache


from mtp.models.lora_split_lm import LoRASplitLM
from mtp.utils.checkpoint import load_model_with_overrides
from mtp.models.evabyte.eva_cache import EvaStaticCacheForTriton
from mtp.models.evabyte.multibyte_decoding_evabyte import multi_byte_pred_prepare_attn_mask


def split_forward(model, inputs):
    assert isinstance(model, LoRASplitLM)


def load_evabyte_model(model_name, checkpoint=None, dtype=None):
    """Load EvaByte model and tokenizer"""

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    dtype = dtype or torch.bfloat16

    if checkpoint is None:
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            # attn_implementation="flash_attention_2",
            device_map="cuda",  # Automatically place on available device
        )
    else:
        model, cfg = load_model_with_overrides(checkpoint, ["lm.model.encoder_only=false"])

    model.eval()  # Set to evaluation mode

    return model, tokenizer


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Your script description")

    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use cache (default: False)"
    )

    parser.add_argument(
        "--num-tokens",
        type=int,
        default=10,
        help="Number of tokens (default: 10)"
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="EvaByte/EvaByte-SFT",
        help="Model name (default: EvaByte/EvaByte-SFT)"
    )

    parser.add_argument(
        '--dtype',
        type=str,
        choices=['float32', 'bfloat16', 'float16'],
        default='bfloat16',
        help='Data type for model computation'
    )


    args = parser.parse_args()


    dtype_map = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16
    }

    model, tokenizer = load_evabyte_model(args.model_name, dtype=dtype_map[args.dtype])
    model.cuda()

    inp = "my example text which grows pretty fast is really good for testing if a given model's activations differ in autoregressive vs batch mode"
    input_ids = tokenizer(inp, return_tensors="pt")["input_ids"].cuda()
    input_ids = input_ids[:, :args.num_tokens]


    with torch.no_grad():
        multi_out = model(input_ids, return_dict=True, use_cache=False, position_ids=torch.arange(input_ids[:, :].shape[1]).reshape(1, -1).cuda())

    if args.use_cache:
        # With kv-cache
        outs = []
        if 'EvaByte' not in args.model_name:
            pkv = DynamicCache()
        else:
            pkv = None
        with torch.no_grad():
            for i in range(input_ids.shape[1]):
                single_input = input_ids[:, [i]]
                # The Llama model needs that past_input_ids
                if args.model_name == "benjamin/Llama3-2-3B-IT-Byte":
                    single_out = model(single_input, past_input_ids=input_ids[:, :i+1], return_dict=True, use_cache=True, past_key_values=pkv, position_ids=torch.arange(i, i+1).reshape(1, -1).cuda())
                else:
                    single_out = model(single_input, return_dict=True, use_cache=True, past_key_values=pkv, position_ids=torch.arange(i, i+1).reshape(1, -1).cuda())
                pkv = single_out["past_key_values"]
                outs.append(single_out['logits'])

        for i in range(input_ids.shape[1]):
            diff = (multi_out['logits'][:, i] - outs[i][:, 0]).abs().sum()
            print(diff)

    else:
        # Without kv-cache
        outs = []
        with torch.no_grad():
            for i in range(input_ids.shape[1]):
                single_input = input_ids[:, :i+1]
                single_out = model(single_input, return_dict=True, use_cache=False, position_ids=torch.arange(i+1).reshape(1, -1).cuda())
                outs.append(single_out['logits'][:, [i]])

        for i in range(input_ids.shape[1]):
            diff = (multi_out['logits'][:, i] - outs[i][:, 0]).abs().sum()
            print(diff)

    print(f"First token diff: {(multi_out['logits'][:, 0] - outs[0][:, 0]).abs().sum()}")
