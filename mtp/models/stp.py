import torch

from torch import Tensor

from .lm import LM


class SingleTokenLM(torch.nn.Module):
    """A SingleTokenLM is a wrapper around a LM that gives us the same
    interface as a MultiTokenLM.

    1. A LM encoder, which can be the encoder (i.e. arch without lm_head)
    of any pretrained LLM. The encoder provides contextual embeddings for
    tokens.

    """

    def __init__(self,
                 lm: LM):
        super().__init__()
        self.lm = lm

    def forward(
            self,
            xx: torch.Tensor,               # (B, S) input ids
            yy: torch.Tensor,               # (B, S) target ids
            return_logits: bool = False
            ) -> dict:
        return self.lm(xx=xx,
                       yy=yy,
                       return_logits=return_logits)

    @torch.no_grad()
    def generate(
        self, inputs: torch.Tensor, use_argmax: bool = False, mode: str = "stp"
    ) -> Tensor:
        return self.lm.generate(inputs, use_argmax=use_argmax, mode=mode)
