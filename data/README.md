# Datasets

This codebase was built off of nanogpt - and nanogpt did not have variable size sequences.
As a result, it did not implement attention masks or worry about padding.
However, for most evaluations, we need variable sized sequences - so we are adopting the HF API, padding, converting the datasets and serialising to the previous format.

## Next Steps
For our current datasets, let's just set the attention mask based on whether the input_ids are padding or not.

For future datasets, let's write a script per HF dataset and model which takes the dataset and serializes it in numpy form, as previously.
The script tokenises the text using the target tokenizer and pads to max sequence length, as in the [OLMO script](https://github.com/allenai/OLMo/blob/a87c459d038c049045b09a05c4987fdddb01393e/scripts/prepare_tulu_data.py).
The script should specify EOS and PAD token ids.

## Label Mask
For TULU, there is the slight complication that we need to save a label_mask too.
I believe the label mask means: Do not predict these tokens, but use them as context - i.e. attention mask is True for those tokens.

## Dealing with Padding
In OLMO they just [specify tokens they do not want to predict with a -100 placeholder in input_ids](https://github.com/allenai/OLMo/blob/a87c459d038c049045b09a05c4987fdddb01393e/olmo/train.py#L724), and then specify that number in the pt loss function.
We can probably do the same + we need to marginalise out the random variables that have the -100 s.


### TODOS:

* Change current dataloaders to also construct an attention mask.
* What do we do with the char-level dataset? Maybe find a char-level transformer that is HF compliant?
