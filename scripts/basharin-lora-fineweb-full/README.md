In this set of experiments we make the LM backbone of MTP more expressive by adding LoRA.
We abandon KL for the time being and use cross-entropy as that worked better.
We also use discounting by default, as we found it to be useful.
We set the discount value to .8 like in Medusa as the default.

## Use LoRA
In preliminary experiments where we only train the MTP heads, we find that acceptance rates plateau early during training.
This is probably due to the fact that the LM features are frozen and the linear heads are expressive enough to fit the distribution of future tokens well.

While we could increase the expressivity opt for adding transformers in the head, as Gloecke did, there are several benefits to fine-tuning LoRA like the Medusa paper did:

1. The teacher and draft LLMs are evaluated in parallel, so there is no slowdown of decoding.
2. If the feature representations are not ideal for future prediction, with LoRA they can be adapted to be more useful for MTP. On the other hand, the feature representation for each output head is still shared, so there would still be some benefit to adding a transformer layer in each head.

### LoRA setup

We start from the Medusa setup and choose r=32 and alpha=16.
We apply LoRA to all linear layers of the LLM, but in contrast to Medusa, we do not apply LoRA to the unembedding matrix.
We use vanilla LoRA and not QLoRA.

### LoRA training

For the time being we also do not follow the Medusa advice of scaling the head learning rate by 4 or breaking up training into 2 stages where in the first we train only the heads.
