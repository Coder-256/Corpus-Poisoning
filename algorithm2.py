import random
from math import ceil

Gamma = 5  # TODO
lambda_ = 5  # TODO


def gamma(d):
  # TODO: Is this the right definition??
  return 1/d if d <= Gamma else 0


def place_additions(dhat, s, POS):
  """ Algorithm 2, Placement into corpus: finding the change set ∆ """
  delta = []

  for t in POS:
    delta += [f"{s} {t}"] * ceil(dhat[t]/gamma(1))

  change_map = dhat.copy()
  for u in POS:
    if u in change_map:
      del change_map[u]

  sum_coocur = sum(gamma(d) for d in range(1, lambda_+1))
  min_sequences_required = ceil(sum(change_map.values())/sum_coocur)
  default_seq = [None]*lambda_ + [s] + [None]*lambda_
  live = [default_seq[:] for _ in range(min_sequences_required)]
  indices = list(range(lambda_)) + list(range(lambda_+1, 2*lambda_+1))

  for u in change_map:
    while change_map[u] > 0:
      best_seq_i = best_i = None
      best = float("inf")
      for seq_i, seq in enumerate(live):
        cm_u = change_map[u]
        for i in indices:
          if seq[i] is None:
            score = abs(gamma(abs(lambda_-i))-cm_u)
            if score < best:
              best_seq_i, best_i = seq_i, i
              best = score
      seq, i = live[best_seq_i], best_i

      seq[i] = u
      change_map[u] -= gamma(abs(lambda_-i))
      if all(seq[i] is not None for i in indices):
        delta.append(seq)
        # remove seq from live
        if best_seq_i < len(live)-1:
          live[best_seq_i] = live.pop()
        else:
          live.pop()
      if not live:
        live = [default_seq[:]]

  cm_keys = list(change_map.keys())
  for seq in live:
    for i in indices:
      if seq[i] is None:
        seq[i] = random.choice(cm_keys)
    delta.append(seq)
  
  return delta
