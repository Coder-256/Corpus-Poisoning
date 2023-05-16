# CSE 325 Research Project: Ayon Bhowmick and Jacob Greenfield

This repository contains sample code implementing the techniques described in *Humpty Dumpty: Controlling Word Meanings via Corpus Poisoning* by Schuster et al.

## Setup

Please make sure you have Git LFS installed in order to clone the sample data.

First, you will need to download and build [GloVe](https://github.com/stanfordnlp/GloVe). Next, `cd` to the root of the GloVe repository, and call `run-glove.sh`. For example:

```
$ cd ~/GloVe
$ ~/cse325-research-project/run-glove.sh tutorial ~/cse325-research-project /path/to/output/dir
```

(Note that a few errors are expected; a handful of the original files in the Wikipedia dump seem to be corrupted). Once you've trained the embeddings, open `CorpusPoison.py` and update the paths in the first cell as necessary.

Next, simply run the notebook. It will automatically load the GloVe data and choose a few random pairs of words, as suggested by Section VIII, Benchmarks in the original paper. For each word pair, it will generate "Dhat", ie. Delta hat, the coocurrence change vector for the source word that maximizes the objective function. Then, it will output a few example word sequences from Delta for the last pair. The idea is to run the notebook as an attacker using the "Sub-Wikipedia" configuration (ie. 10% of English Wikipedia with GloVe-`tutorial` hyperparameters), then retrain embeddings with the augmented corpus using the "Wikipedia" configuration (ie. 100% of English Wikipedia with the GloVe-`paper` hyperparameters).

The sequences will be output to `sequences.txt`. Once you have the additional sequences, they can be appended to the training corpus and new embeddings trained using GloVe.

Finally, the results of the retrained embeddings can be compared to the original embeddings using `CheckResults.ipynb`. This outputs a number of useful statistics about how the similarity of each of the test pairs has changed between the original and new embeddings.

## Layout

`cat_wikipedia.py` is used to process and tokenize the output of `WikiExtractor` into a format suitable for processing by GloVe.

`run-glove.sh` is a script to train word embeddings using GloVe with several different combinations of hyperparameters.

`CorpusPoison.ipynb` contains an implementation of the greedy algorithm to generate Delta-hat, including code to parse the output from GloVe, a JIT-compiled CUDA kernel to calculate the distributional similarity scores, and the corpus placement algorithm to generate first-order and second-order sequences which can be be inserted into the corpus.

`filt-cooccur.{c,sh}` is used to filter the cooccurrence matrix for faster processing in Python within `CheckResults.ipynb`.

`CheckResults.ipynb` contains code to compare two different word embeddings on a number of sample word pairs.

## Sample Results


| setting       | max Delta | median rank | avg. increase in proximity | rank < 10 |
|:------------- | ---------:| -----------:| --------------------------:| ---------:|
| GloVe         | -         |      179059 |                          - |         0 |
| GloVe-`paper` | 1250      |           1 |                   0.556851 |        83 |

