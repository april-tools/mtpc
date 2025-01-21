import torch

from torch import Tensor
from cirkit.utils.scope import Scope
from cirkit.backend.torch.queries import SamplingQuery, IntegrateQuery

from nanogpt.models.circuit import CircuitCP, MultiTokenHead
from nanogpt.models.gpt import GPT
from nanogpt.models.layers import TorchBatchedCategoricalLayer, TorchBatchedSumLayer


class MultiTokenLM(torch.nn.Module):
    """A MultiTokenLM comprises three parts:

    1. A LM encoder, which can be the encoder (i.e. arch without lm_head)
    of any pretrained LLM. The encoder provides contextual embeddings for
    tokens.

    2. A mt_head, which expands the contextual embeddings into parameters
    for the (circuit) output layer.

    3. A circuit which models the output tokens and encodes their dependencies.
    """

    def __init__(self, gpt: GPT, mt_head: MultiTokenHead, circuit: CircuitCP):
        super().__init__()
        self.gpt = gpt
        self.mt_head = mt_head
        self.circuit = circuit

        # Retrieve the circuit layers to parameterize
        layers = list(self.circuit.circuit.topological_ordering())
        self._cat_layer = next(l for l in layers if isinstance(l, TorchBatchedCategoricalLayer))
        self._sum_layer = next(l for l in layers if isinstance(l, TorchBatchedSumLayer))
        self.sampler = SamplingQuery(self.circuit.circuit)
        self.marginalizer = IntegrateQuery(self.circuit.circuit)

    def forward(self, xx: Tensor, yy: Tensor, return_log_probs: bool = False) -> tuple[Tensor | None, Tensor]:
        # Compute the loss, i.e., the multi-token average negated log-likelihood

        # xx: (B, S, D)
        xx = self.gpt.encoder(xx)

        # At training time, we want to learn to predict the next H tokens
        # xx: (B, S', D), where S' = S - H + 1
        xx = xx[:, : xx.shape[1] - self.mt_head.n_token + 1]

        # Parameterize the circuit
        self.parameterize_circuit(xx)

        # Compute sliding windows indices
        # from yy: (B, S) to yy: (B, S', H)
        # where S' = S - H + 1
        yy = yy.unfold(dimension=1, size=self.mt_head.n_token, step=1)
        # Note that we unsqueeze a channel dimension, as required by cirkit
        # yy: (B, S', H) -> (B * S', 1, H)
        yy = yy.reshape(-1, 1, yy.shape[2])

        # Compute the conditional log-likelihoods
        # yy: (B * S', 1, H)
        # log_probs: (B * S', 1, 1)
        log_probs = self.circuit(yy)

        # The loss is the negated average conditional log-likelihood
        loss = -log_probs.mean()
        if not return_log_probs:
            log_probs = False
        return log_probs, loss

    def parameterize_circuit(self, xx: Tensor):
        # Obtain dictionary of circuit parameters
        circuit_params = self.mt_head(xx)
        # cat_logits: (H, B, S', R, V)
        cat_log_probs = circuit_params["cat_log_probs"]
        # sum_weight: (B, S', 1, R)
        sum_weight = circuit_params["sum_weight"]

        # cat_log_probs: (H, B * S', R, V)
        cat_log_probs = cat_log_probs.view(cat_log_probs.shape[0], -1, cat_log_probs.shape[3], cat_log_probs.shape[4])
        # sum_weight: (1, B * S', 1, R)
        sum_weight = sum_weight.view(1, -1, 1, sum_weight.shape[3])

        # Set the parameters of the circuit
        self._cat_layer.log_probs = cat_log_probs
        self._sum_layer.weight = sum_weight

    @torch.no_grad()
    def generate(self, inputs: Tensor):
        # Calling forward with no targets simply sets the parameters to the circuit
        # xx: (B, S, D)
        xx = self.gpt.encoder(inputs)

        # At generation time, we want to do future token prediction
        # so we only condition on last output
        # xx: (B, S', D), where S' = 1
        xx = xx[:, [-1]]

        # Parameterize the circuit
        self.parameterize_circuit(xx)

        # Sample the next tokens
        tokens, _ = self.sampler(num_samples=1)
        # Remove extraneous channel dimension
        tokens = tokens.squeeze(dim=1)
        return tokens

    @torch.no_grad()
    def self_speculative_generate(self, seq: Tensor) -> Tensor:
        if len(seq.shape) != 2 or seq.shape[0] != 1:
            raise NotImplementedError("Multi-batch self-speculative decoding not implemented yet")
            # seq: (B, S), with B = 1 and also possibly S = 1

        # Compute the embeddings
        # xx: (B, S, D) -> (B, 1, D)
        xx = self.gpt.encoder(seq)
        xx = xx[:, [-1]]

        # Set the circuit parameters, based on the last embeddings
        self.parameterize_circuit(xx)

        # Sample the next H tokens
        # tokens: (B=1, 1, H) -> (B=1, H)
        tokens, _ = self.sampler(num_samples=1)
        tokens = tokens.squeeze(dim=1)

        # Concatenate the tokens with the current sequence,
        # which gives the candidate next sequence
        # gen_seq: (B, S + H)
        gen_seq = torch.cat([seq, tokens], dim=1)

        # Compute the next-token probabilities in parallel
        # zz: (B, S + H, D) -> (B, H + 1, D)
        zz = self.gpt.encoder(gen_seq)
        zz = zz[:, -tokens.shape[1] - 1 :]
        # logits: (B, H + 1, V)
        logits = self.gpt.head(zz)

        # Determine the number of accepted tokens,
        # by iteratively computing conditional probabilities with the circuit
        #
        # To do so, we first compute context-conditioned marginals in parallel
        # q(x_{t+1} \mid x_{\leq t})
        # q(x_{t+1}, x_{t+2} \mid x_{\leq t})
        # ...
        # q(x_{t+1}, ..., x_{t+n} \mid x_{\leq t})
        #
        # log_marginal_probs: (H, 1, 1) -> (B=1, H, 1)
        log_marginal_probs = self.marginalizer(
            tokens.expand(size=(tokens.shape[1], -1)).unsqueeze(dim=1),
            integrate_vars=list(
                reversed([Scope(tokens.shape[1] - i - 1 for i in range(t)) for t in range(tokens.shape[1])])
            ),
        )
        log_marginal_probs = log_marginal_probs.squeeze(dim=1).unsqueeze(dim=0)
        #
        # Sample H uniform noise values in [0,1), and take their log
        # log_noise: (B=1, H)
        log_noise = torch.log(
            torch.rand(
                size=(tokens.shape[0], tokens.shape[1]),
                device=log_marginal_probs.device,
                dtype=log_marginal_probs.dtype,
            )
        )
        #
        # Compute the number of tokens to accept
        num_accepted_tokens = 0
        for j in range(tokens.shape[1]):
            # Check whether noise > ratio of conditional univariate probabilities,
            # i.e., we should stop accepting tokens
            # In the log space, this becomes log noise > difference of some log probabilities,
            # which avoids many floating point divisions and is more numerically stable
            gpt_next_token_log_probs = torch.log_softmax(logits[:, j], dim=1)
            # gpt_jth_token_log_prob: (B, 1)
            gpt_jth_token_log_prob = torch.gather(gpt_next_token_log_probs, dim=1, index=tokens[:, [j]])
            # Compute log conditional probabilities, conditioned on the context
            if j == 0:
                # q(x_{t+1}\mid x_{\leq t})
                # mtp_jth_token_log_prob: (B, 1)
                mtp_jth_token_log_prob = log_marginal_probs[:, 0]
            else:
                # q(x_{t+j}\mid x_{\leq t}, x_{t+1}, ..., x_{t+j-1}) = \
                #     q(x_{t+1}, ..., x_{t+j}\mid x_{\leq t}) / q(x_{t+1}, ..., x_{t+j-1}\mid x_{\leq t})
                # mtp_jth_token_log_prob: (B, 1)
                mtp_jth_token_log_prob = log_marginal_probs[:, j] - log_marginal_probs[:, j - 1]
            # Check noise > \
            #     (p(x_{t+j}\mid x_{\leq t}, x_{t+1}, ..., x_{t+j-1}) / q(x_{t+j}\mid x_{\leq t}, x_{t+1}, ..., x_{t+j-1}))
            if log_noise[:, j] > (gpt_jth_token_log_prob - mtp_jth_token_log_prob):
                break
            num_accepted_tokens += 1

        if num_accepted_tokens == tokens.shape[1]:
            # We are so lucky! We accept all the H tokens
            # Let's index the probabilities to sample the H+1-th one
            gpt_last_probs = torch.softmax(logits[:, -1], dim=1)
            # Sample the last token
            last_token = torch.multinomial(gpt_last_probs, num_samples=1)
        else:  # num_accepted_tokens < tokens.shape[1]
            # We accepted H' < H tokens
            # Let's adjust the probabilities to sample the H'+1-th one
            # gpt_last_logits: (B, V)
            gpt_last_probs = torch.softmax(logits[:, num_accepted_tokens], dim=1)
            # Let j be the number of accepted tokens, then
            # max(0, p(x_{t+j+1}\mid x_{\leq t+j}) - q(x_{t+j+1}\mid x_{\leq t+j}))
            # under the consideration that
            # q(x_{t+j+1}\mid x_{\leq t+j}) = \
            #     q(x_{t+1}, ..., x_{t+j+1}\mid x_{\leq t}) / q(x_{t+1}, ..., x_{t+j}\mid x_{\leq t})
            expanded_tokens = tokens[:, :num_accepted_tokens].unsqueeze(dim=1).expand(-1, self.mt_head.vocab_size, -1)
            jp1th_token_assignments = (
                torch.arange(self.mt_head.vocab_size, device=tokens.device, dtype=tokens.dtype)
                .unsqueeze(dim=1)
                .unsqueeze(dim=0)
            )
            # mtp_jp1th_tokens: (B, V, H' + 1) -> (B * V, 1, H' + 1)
            mtp_jp1th_tokens = (
                torch.cat([expanded_tokens, jp1th_token_assignments], dim=2)
                .flatten(start_dim=0, end_dim=1)
                .unsqueeze(dim=1)
            )
            if mtp_jp1th_tokens.shape[2] == tokens.shape[1]:
                # mtp_jp1th_token_log_probs: (B * V, 1, 1)
                mtp_jp1th_token_log_probs = self.circuit(mtp_jp1th_tokens)
            else:
                # mtp_jp1th_token_log_probs: (B * V, 1, 1)
                mtp_jp1th_token_log_probs = self.marginalizer(
                    torch.nn.functional.pad(mtp_jp1th_tokens, pad=(0, tokens.shape[1] - num_accepted_tokens - 1)),
                    integrate_vars=Scope(range(num_accepted_tokens + 1, tokens.shape[1])),
                )
            # mtp_jp1th_token_log_probs: (B * V, 1, 1) -> (B, V)
            mtp_jp1th_token_log_probs = mtp_jp1th_token_log_probs.view(tokens.shape[0], self.mt_head.vocab_size)
            # mtp_last_log_probs: (B, V)
            mtp_last_log_probs = mtp_jp1th_token_log_probs - log_marginal_probs[:, num_accepted_tokens - 1]
            adj_last_probs = torch.relu(gpt_last_probs - torch.exp(mtp_last_log_probs)) + 1e-15
            adj_last_probs = adj_last_probs / torch.sum(adj_last_probs, dim=1, keepdim=True)
            # Sample the last token
            last_token = torch.multinomial(adj_last_probs, num_samples=1)

        # Retrieve the accepted tokens, plus the last one
        tokens = torch.cat([tokens[:, :num_accepted_tokens], last_token], dim=1)
        return tokens
