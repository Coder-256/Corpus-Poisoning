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

The algorithm doesn't seem to perform exactly as expected due to a bug. Since the equations are extremely complex, poorly documented in the original paper, and no reference implementation exists, we have slightly fell short of the expected results. However, this demonstration shows that corpus poisoning is possible and achievable, with performance only marginally worse than the original paper.