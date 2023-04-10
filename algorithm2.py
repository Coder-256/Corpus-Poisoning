import random
from math import ceil

Gamma = 5  # TODO
lambda_ = 5  # TODO
dictionary = ["foo", "bar"]  # TODO


def gamma(d: int) -> float:
  # TODO: Is this the right definition??
  return 1/d if d <= Gamma else 0


def place_additions(Dhat: dict[int, int], s: int, POS: list[int]) -> list[int]:
  """ Algorithm 2, Placement into corpus: finding the change set ∆ """
  Delta: list[str] = []

  for t in POS:
    Delta += [f"{dictionary[s]} {dictionary[t]}"] * ceil(Dhat[t]/gamma(1))

  change_map = dict(Dhat)
  for u in POS:
    if u in change_map:
      del change_map[u]

  sum_coocur = sum(gamma(d) for d in range(1, lambda_+1))
  min_sequences_required = ceil(sum(change_map.values())/sum_coocur)
  default_seq = [-1]*lambda_ + [s] + [-1]*lambda_
  live = [default_seq[:] for _ in range(min_sequences_required)]
  indices = list(range(lambda_)) + list(range(lambda_+1, 2*lambda_+1))

  cm_keys = list(change_map.keys())
  for u in cm_keys:
    while change_map[u] > 0:
      best_seq_i = best_i = -1
      best = float("inf")
      for seq_i, seq in enumerate(live):
        cm_u = change_map[u]
        for i in indices:
          if seq[i] < 0:
            score = abs(gamma(abs(lambda_-i))-cm_u)
            if score < best:
              best_seq_i, best_i = seq_i, i
              best = score
      seq, i = live[best_seq_i], best_i

      seq[i] = u
      change_map[u] -= gamma(abs(lambda_-i))
      if all(seq[i] > 0 for i in indices):
        Delta.append(seq)
        # remove seq from live
        if best_seq_i < len(live)-1:
          live[best_seq_i] = live.pop()
        else:
          live.pop()
      if not live:
        live = [default_seq[:]]

  for seq in live:
    for i in indices:
      if seq[i] < 0:
        seq[i] = random.choice(cm_keys)
    Delta.append(seq)

  return Delta
