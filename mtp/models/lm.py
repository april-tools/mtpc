from collections import OrderedDict
from contextlib import contextmanager

import torch
import torch.nn.functional as F
import peft
from peft import PeftModel

from torch import nn, Tensor
from transformers import AutoModelForCausalLM

from mtp.utils.distributed import get_local_device
from mtp.utils.checkpoint import Checkpoint


class LM(nn.Module):
    """Wrapper to make a Language Model (LM) compatible with MTP."""

    def __init__(
        self,
        lm: nn.Module = None,
        from_checkpoint: str = None,
        from_huggingface: str = None,
        adaptor_kwargs: dict = None,
        ref_enc: str = "model",
        ref_head: str = "lm_head",
        encoder_only: bool = True,
        freeze: bool = True,
    ):
        super().__init__()

        self.from_checkpoint = from_checkpoint
        self.from_huggingface = from_huggingface
        # What lm attribute to find the encoder under
        self.ref_enc = ref_enc
        # What lm attribute to find the head under
        self.ref_head = ref_head
        # Whether to include the head or not
        self.encoder_only = encoder_only
        # Whether to freeze the lm or not
        self.freeze = freeze

        noneness = (
            lm is None,
            from_checkpoint is None,
            from_huggingface is None,
        )
        assert noneness in set(
            [(False, True, True), (True, False, True), (True, True, False)]
        )

        # Set the LLM, and freeze its weights if required
        if lm is None:
            lm = self._load_lm()
        if self.freeze:
            for p in lm.parameters():
                p.requires_grad = False

        # Use peft for lora
        # Note that the lora weights are not freezed, even if freeze=True above
        self.adaptor_kwargs = adaptor_kwargs
        if self.adaptor_kwargs is not None:
            assert not isinstance(lm, PeftModel)
            peft_config = peft.LoraConfig(
                task_type="CAUSAL_LM",
                **self.adaptor_kwargs
            )
            self._lm = peft.get_peft_model(lm, peft_config)
        else:
            self._lm = lm

        # Set the parameters key to include in the state_dict of the model
        # NOTE: here we are assuming that the requires_grad of the LM model will NOT be changed elsewhere
        #       after the completion of this __init__()
        state_dict = super().state_dict(keep_vars=True)
        if self.from_checkpoint is not None or self.from_huggingface is not None:
            # We are loading a checkpoint from either disk or from hugginface
            # As such, we retain only the keys of weights that require gradients
            self._filter_state_dict_keys: set = {k for k, v in state_dict.items() if not v.requires_grad}
            # For instance, if we loaded a model from huggingface, freezed it, and the applied peft on it;
            # then _state_dict_keys will contain only the keys of e.g., lora weights
        else:
            # An LM model has been initialized somewhere and passed to __init__()
            # As such, we retain all the keys of the weights, as we do not know if the given model is stored permantently somewhere else
            # (i.e., we are conservative)
            self._filter_state_dict_keys: set = {}

        # Keep lm head weights in case we want to use them during init
        self._lm_head_weights = self.head.weight.detach().clone().data
        # If encoder only, drop the head
        if self.encoder_only:
            setattr(self.lm_model, self.ref_head, None)

    def _load_lm(self):
        lm = None
        if self.from_checkpoint is not None:
            # Assume that if we can find the conf, we saved the checkpoint
            try:
                cp = Checkpoint.load(self.from_checkpoint)
                lm = cp.model.lm._lm
            # otherwise try loading as default pt
            except Exception:
                lm = torch.load(
                    self.from_checkpoint,
                    weights_only=False,
                    map_location=get_local_device(),
                )._lm
        elif self.from_huggingface is not None:
            if 'EvaByte' in self.from_huggingface:
                kwargs = {'trust_remote_code': True}
            else:
                kwargs = {'attn_implementation': "flash_attention_2"}
            lm = AutoModelForCausalLM.from_pretrained(
                self.from_huggingface,
                torch_dtype=torch.bfloat16,
                **kwargs
            )
            if 'EvaByte' in self.from_huggingface:
                # By default, we do not use caching in EvaByte, and always return dictionaries
                lm.config.use_cache = False
                lm.config.return_dict = True
        return lm

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        # Retain only the required keys
        # I.e., set to None those tensors that do not need to be serialized
        prefix = kwargs.pop("prefix")
        overriden_state = {f"{prefix}{k}": None for k in self._filter_state_dict_keys}
        sd.update(overriden_state)
        return sd

    @property
    def lm_head_weights(self):
        return self._lm_head_weights

    def drop_lm_head_weights(self):
        self._lm_head_weights = None

    @property
    def lm_model(self):
        if self.has_adapter:
            return self._lm.base_model.model
        return self._lm

    @property
    def encoder(self):
        return getattr(self.lm_model, self.ref_enc)

    @property
    def head(self):
        return getattr(self.lm_model, self.ref_head)

    @property
    def has_adapter(self) -> bool:
        return isinstance(self._lm, PeftModel)

    @contextmanager
    def disable_adapter(self):
        assert self.has_adapter
        # Forward the disable adapter context manager of the Peft-managed LM
        with self._lm.disable_adapter():
            yield

    @contextmanager
    def disable_adapter_if_any(self):
        if self.has_adapter:
            # Forward the disable adapter context manager of the Peft-managed LM,
            # only if the lm is Peft-managed
            with self._lm.disable_adapter():
                yield
        else:
            yield

    def forward(
        self,
        xx: Tensor,
        yy: Tensor | None = None,
        return_logits: bool = True,
    ) -> dict:
        assert (
            self.head is not None
        ), "The forward of GPT can only be called if encoder_only=False"

        # forward the encoder
        xx = self.encoder(xx)[
            "last_hidden_state"
        ]  # token embeddings of shape (b, t, n_embd)

        if yy is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.head(xx)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), yy.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.head(
                xx[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return dict(logits=logits, loss=loss)

    @torch.no_grad()
    def generate(
        self,
        inputs: torch.Tensor,
        use_argmax: bool = False,
        mode: str = "stp",
        use_cache: bool = True,
        past_key_values: Tensor = None,
    ) -> Tensor:
        self.eval()
        if mode != "stp":
            raise ValueError("Only single token generation is supported")
        if use_cache:
            # We only pass in the unseen inputs, because we are using cache
            seen_tokens = 0
            if past_key_values is not None:
                seen_tokens = past_key_values.get_seq_length()
            outputs = self.encoder(
                input_ids=inputs[:, seen_tokens:],
                use_cache=use_cache,
                past_key_values=past_key_values,
            )
            # token embeddings of shape (b, t, n_embd)
            xx = outputs["last_hidden_state"]
            past_key_values = outputs["past_key_values"]
        else:
            xx = self.encoder(inputs)["last_hidden_state"]

        logits = self.head(
            xx[:, [-1], :]
        )  # note: using list [-1] to preserve the time dim
        if use_argmax:
            tokens = torch.argmax(logits, dim=2)
        else:
            probs = torch.softmax(logits, dim=2)
            tokens = torch.multinomial(probs.squeeze(dim=1), num_samples=1)
        return dict(tokens=tokens, past_key_values=past_key_values)
