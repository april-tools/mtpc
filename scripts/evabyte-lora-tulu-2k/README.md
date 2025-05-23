## Experiments

These experiments were carried out after fixing:

a) a batching bug where the loss changed a lot depending on device batch size
b) logits -> float32 and corrected copy of original head weights
c) splitting Tulu 3 into train and valid test and limit to examples shorter than 8192 bytes, see the dataset [here](https://huggingface.co/datasets/agrv/tulu-v3-sft-evabyte-seq-len-8196).


### Train time

We limit to 2k steps as this should take 2-4 days on 2 A100s


### Evaluation
 
It was important to stop at <eos> as models were noisy when predicting subsequent tokens.
After evaluating on 250 prompts, performance on tulu-train and tulu-valid are not particularly different, so we can stick to tulu-valid.


### Main results

* Tulu-train vs Tulu-valid: no diff
* For vanilla speculative decoding, we are improving mean acceptance rate 5. -> 5.3.
* For greedy speculative decoding, we cannot beat argmax decoding from the indep model; the indep model is really good (~7 mean acceptance rate).
