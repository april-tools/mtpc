import torch
import transformers

from datasets import load_dataset
from .hf_dataloader import HFDistributedDataLoader

from mtp.models.loss import IGNORE_TOKEN_ID


# https://github.com/FasterDecoding/Medusa/blob/e2a5d20c048a9b0a4092e6933c34313687422518/create_data.py#L25
def fix_source(source):
    # The sharegpt dataset has a different template compared to that
    # expected by the tokenizer chat template
    if source and source[0]["from"] == "gpt":
        # Skip if GPT is first to talk
        source = source[1:]
    new_source = []
    for item in source:
        role = "assistant" if item["from"] == "gpt" else "user"
        content = item["value"]
        new_source.append({"role": role, "content": content})
    return new_source


# Taken from the Medusa codebase
# https://github.com/FasterDecoding/Medusa/blob/e2a5d20c048a9b0a4092e6933c34313687422518/medusa/train/train_legacy.py#L163C1-L219C6
def process_sharegpt(sources, tokenizer: transformers.PreTrainedTokenizer) -> dict:
    """
    Preprocesses conversation data and tokenizes it for model input.

    Args:
        sources: A list of conversation sources.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenization.

    Returns:
        Dict: A dictionary containing tokenized inputs, labels, and attention mask.
    """

    # https://huggingface.co/lmsys/vicuna-13b-v1.5/discussions/12
    chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\\'s questions.' %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 %}{{ system_message }}{% endif %}{% if message['role'] == 'user' %}{{ ' USER: ' + message['content'].strip() }}{% elif message['role'] == 'assistant' %}{{ ' ASSISTANT: ' + message['content'].strip() + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ ' ASSISTANT:' }}{% endif %}"

    # Apply prompt templates
    conversations = []
    prompts = []

    conversation = sources["conversations"]
    conversation = fix_source(conversation)
    # Fix the chat
    # Chat template was removed, so add here
    prompt = tokenizer.apply_chat_template(
        conversation, tokenize=False, chat_template=chat_template
    )
    prompts.append(prompt)
    conversations.append(conversation)

    # Tokenize conversations
    encoding = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
    )
    # Set everything to be ignored, except the assistant part
    targets = torch.full_like(encoding.input_ids, IGNORE_TOKEN_ID)
    input_ids = encoding.input_ids

    # Mask targets. Only compute loss on the assistant outputs.
    for conv_index, (conversation, target, prompt) in enumerate(
        zip(conversations, targets, prompts)
    ):

        for turn in conversation:
            if turn["role"] == "assistant":
                content = turn["content"]
                # Unfortunate strip() necessary because chat templates are doing the same.
                start = prompt.index(content.strip())
                stop = start + len(content)
                indices = []
                for tok_index, (tok_start, tok_stop) in enumerate(
                    encoding.offset_mapping[conv_index]
                ):
                    if tok_stop >= start or tok_start < tok_stop:
                        indices.append(tok_index)
                target[indices] = encoding.input_ids[conv_index][indices]

    # NOTE: in our implementation we expect label @ i to be target for input_id @ i
    input_ids = input_ids[:, :-1]
    labels = targets[:, 1:]

    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class ShareGPTDataLoader(HFDistributedDataLoader):
    def __init__(
        self,
        hf_dataset: str,
        hf_model: str,
        B: int,
        T: int,
        process_rank: int,
        num_processes: int,
        device: str = "cuda",
        split="train",
    ):
        super().__init__(
            hf_dataset, hf_model, B, T, process_rank, num_processes, device, split
        )

    def process(self, x):
        return process_sharegpt(x, self.tokenizer)

    def load_dataset(self):
        if self.split == "train":
            return load_dataset(
                "Aeala/ShareGPT_Vicuna_unfiltered",
                data_files=["ShareGPT_V4.3_unfiltered_cleaned_split.json"],
                split=self.split,
            )
        else:
            raise ValueError("ShareGPT dataset has no %s split" % self.split)
