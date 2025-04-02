These are the experiments checking out approx KL on the full Finewebedu 10B.

## Results are cached

The commands in `reproduce.sh`:

1. Create new jsonl files with results for validation and throughput
2. Create the plots - but based on the cached versions of those files (see filepaths)

The produced plots should be the same as those [in the presentation](https://app.excalidraw.com/s/E7xnKEQRM1/6TtD00xreRu).


## Limitations

Note that some models hadn't finished training as they got NaNs due to approx KL issue.
