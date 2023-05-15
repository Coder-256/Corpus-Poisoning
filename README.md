# CSE 325 Research Project: Ayon Bhowmick and Jacob Greenfield

Please make sure you have Git LFS installed in order to clone the sample data.

Please see `CorpusPoison.ipynb` for an implementation of the greedy algorithm to generate Delta-hat, including code to parse the output from GloVe, a JIT-compiled CUDA kernel to calculate the distributional similarity scores, and the corpus placement algorithm to generate first-order and second-order sequences which can be be inserted into the corpus.

The notebook uses the GloVe benchmark parameters from section VIII of the paper. It outputs a file named e.g. `Delta_foo_bar.txt` (for source word "foo" and target word "bar"), which can be appended to the training corpus. To calculate the pre- and post-retraining similarity, see `QuickSimilarity.ipynb`; note that you will need to manually run GloVe on the modified corpus.

Sample output before poisoning, for source word "officio" and target word "leverett" using `text8` from GloVe's `demo.sh`:

```
SIM1 1.4838137370542808
SIM2 1.006991579258536
SIM1+2 1.2454026581564084
```

Sample output after retraining with second-order sequences:

```
SIM1 5.572523183980332
SIM2 7.695977689508876
SIM1+2 6.634250436744604
```

Sample output after retraining with first- and second-order sequences:

```
SIM1 16.25654683606917
SIM2 8.187946910791613
SIM1+2 12.222246873430391
```

By adding only second-order sequences (~7100 words) to a corpus of ~17 million words, the similarity score significantly increases when using the poisoned corpus, even though the target word "leverett" doesn't appear even once in any of the added sequences.

## Setup

First, you will need to download and build [GloVe](https://github.com/stanfordnlp/GloVe). Next, `cd` to the root of the GloVe repository, and call `run-glove.sh`. For example:

```
$ cd ~/GloVe
$ ~/cse325-research-project/run-glove.sh tutorial ~/cse325-research-project /path/to/output/dir
```

(Note that a few errors are expected; a handful of the original files in the Wikipedia dump seem to be corrupted). Once you've trained the embeddings, open `CorpusPoison.py` and update the paths in the first cell as necessary.

Next, simply run the notebook. It will automatically load the GloVe data and choose a few random pairs of words, as suggested by Section VIII, Benchmarks in the original paper. For each word pair, it will generate "Dhat", ie. Delta hat, the coocurrence change vector for the source word that maximizes the objective function. Then, it will output a few example word sequences from Delta for the last pair. The idea is to run the notebook as an attacker using the "Sub-Wikipedia" configuration (ie. 10% of English Wikipedia with GloVe-`tutorial` hyperparameters), then retrain embeddings with the augmented corpus using the "Wikipedia" configuration (ie. 100% of English Wikipedia with the GloVe-`paper` hyperparameters).