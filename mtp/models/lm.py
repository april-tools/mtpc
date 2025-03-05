import torch
import torch.nn.functional as F

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

        # We only save weights if a) the model is not frozen
        # or b) if we initialised a model that does not have a checkpoint
        self.save_weights = (lm is not None) or self.freeze is False

        self.lm = lm or self._load_lm()

        # Keep track of the keys which we set to None if save_weights=False
        # we need to do this before we drop the head
        self.none_keys = set("lm.%s" % k for k in self.lm.state_dict().keys())

        # Keep lm head weights in case we want to use them during init
        # We delete this in MultiTokenLM when we do not need it
        self.lm_head_weights = self.head.weight.detach().clone().data

        # If encoder only, drop the head
        if self.encoder_only:
            setattr(self.lm, self.ref_head, None)

        if self.freeze:
            for p in self.lm.parameters():
                p.requires_grad = False

    def _load_lm(self):
        lm = None
        if self.from_checkpoint is not None:
            # Assume that if we can find the conf, we saved the checkpoint
            try:
                cp = Checkpoint.load(self.from_checkpoint)
                lm = cp.model.lm
            # otherwise try loading as default pt
            except Exception:
                lm = torch.load(
                    self.from_checkpoint,
                    weights_only=False,
                    map_location=get_local_device(),
                ).lm
        elif self.from_huggingface is not None:
            lm = AutoModelForCausalLM.from_pretrained(
                self.from_huggingface,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
            )
        return lm

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        # If we have loaded from checkpoint and the weights are frozen
        # do not store the weights, we will load them again from the checkpoint
        if not self.save_weights:
            # NOTE: The complication here is that when we created none_keys
            # we were not prefixing with the current modules prefix
            prefix = kwargs.pop("prefix")
            new_state = {("%s%s" % (prefix, k)): None for k in self.none_keys}
            state.update(new_state)
        return state

    @property
    def encoder(self):
        return getattr(self.lm, self.ref_enc)

    @property
    def head(self):
        return getattr(self.lm, self.ref_head)

    def forward(
        self,
        xx: Tensor,
        yy: Tensor | None = None,
        return_logits: bool = True,
    ) -> tuple[Tensor | None, None]:
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
