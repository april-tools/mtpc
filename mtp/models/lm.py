import torch
from torch import nn, Tensor

from mtp.utils.distributed import get_local_device


class LM(nn.Module):
    """Wrapper to make a Language Model (LM) compatible with MTP."""

    def __init__(
        self,
        lm: nn.Module = None,
        from_checkpoint: str = None,
        ref_enc: str = "model",
        ref_head: str = "lm_head",
        encoder_only: bool = True,
        freeze: bool = True,
    ):
        super().__init__()

        if from_checkpoint is not None:
            assert lm is None
            self.lm = torch.load(from_checkpoint,
                                 weights_only=False,
                                 map_location=get_local_device())
        else:
            assert lm is not None
            self.lm = lm

        self.from_checkpoint = from_checkpoint
        # What lm attribute to find the encoder under
        self.ref_enc = ref_enc
        # What lm attribute to find the head under
        self.ref_head = ref_head
        # Whether to include the head or not
        self.encoder_only = encoder_only
        # Whether to freeze the lm or not
        self.freeze = freeze

        if self.encoder_only:
            setattr(self.lm, self.ref_head, None)

        if self.freeze:
            for p in self.lm.parameters():
                p.requires_grad = False

    @property
    def encoder(self):
        return getattr(self.lm, self.ref_enc)

    @property
    def head(self):
        return getattr(self.lm, self.ref_head)

    def forward(
        self,
        idx: Tensor,
        targets: Tensor | None = None,
        return_logits: bool = True,
        return_stp_loss: bool = True,
    ) -> tuple[Tensor | None, None]:
        assert (
            self.head is not None
        ), "The forward of GPT can only be called if encoder_only=False"
        assert (
            return_stp_loss
        ), "The forward of GPT always computes the single-token loss"

        # forward the GPT model itself
        x = self.encoder(idx)  # token embeddings of shape (b, t, n_embd)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return dict(logits=logits, loss=loss, stp_loss=loss)

    @torch.no_grad()
    def generate(
        self, inputs: torch.Tensor, use_argmax: bool = False, mode: str = "stp"
    ) -> Tensor:
        if mode != "stp":
            raise ValueError("Only single token generation is supported")
        results = self.forward(inputs, return_logits=True)
        logits = results["logits"]
        if use_argmax:
            toks = torch.argmax(logits, dim=2)
        else:
            probs = torch.softmax(logits, dim=2)
            toks = torch.multinomial(probs.squeeze(dim=1), num_samples=1)
        return toks
