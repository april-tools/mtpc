
The scripts in this folder contain the commands used to train the models.

## Scripts
* debug.sh
* ablation.sh
* basharin-reproduce.sh

### debug.sh

A script to quickly check that everything works

### ablation.sh

The first runs run by Yu on 2 H100s on EIDF around 28/02/2025.
The models were trained using the ablation LM backbone from Penedo et al for 15k steps on the 10 first files of finewebedu.
We used n=4, r=1 and a sequence length of 1024.
The code at this point only supported the cross-entropy loss without discounting.

```bash
git checkout -b "lm-not-freeze" 15ff31eba7dc4d37d41671466de4b48e05eea404
```

### basharin-reproduce.sh

This includes two runs, where we vary r:

1. [n=2, r=1](https://wandb.ai/circuit-mtp/mtp/runs/peuuj25u), was run on 3 L40 GPUs on Murgia and took approx a day.
2. [n=2, r=2](https://wandb.ai/circuit-mtp/mtp/runs/hxoe9y1q), was run on 2 A100 GPUs on Wintermute and took approx two days.

These runs incorporate the weighted combination of .9 * KL + .1 * CE loss, and discounts future tokens using gamma=.8.
For the KL we needed to expand logits for each conditional, and this made the model much more memory hungry.
In order to get the n=2, r=2 model to fit in memory with a device batch size of 16, such that it can train in a bit over two days, we had to scale down n from 4 to 2 and seq length from 1024 to 256.

### basharin-ablation.sh

In the ablation, we take the simplest version of the Basharin model and ablate individual differences.
For the simplest version we pick no discounting (gamma=1) and use just the CE loss (beta=0).


* gamma = .8
* beta = 1 (KL only)
* token head = MLP
* token head transformer layers = 1
* sum head transformer layers = 1
* freeze LM = false
* freeze unembedding = false
