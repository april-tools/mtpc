# Datasets

This codebase was built off of nanogpt - and nanogpt did not have variable size sequences.
As a result, it did not implement attention masks or worry about padding.
However, for most evaluations, we need variable sized sequences - so we are adopting the HF API, padding, converting the datasets and serialising to the previous format.

## Making datasets HF compliant
For our current datasets, we set the attention mask based on whether the input_ids are padding or not.

For future datasets, we will write a loader per HF dataset which takes advantage of all the implemented interfaces in HF (e.g. lets stream the datasets).
This can be easily done by extending `HFDistributedDataLoader(object)`, see `mtp/data/__init__.py`.

## Label Mask
For TULU, there is the slight complication that we need to process a label_mask too.
I believe the label mask means: Do not predict these tokens, but use them as context - i.e. attention mask is True for those tokens.

## Dealing with Padding
In OLMO they just [specify tokens they do not want to predict with a -100 placeholder in input_ids](https://github.com/allenai/OLMo/blob/a87c459d038c049045b09a05c4987fdddb01393e/olmo/train.py#L724), and then specify that number in the pt loss function.
We can probably do the same + we need to marginalise out the random variables that have the -100 s.
