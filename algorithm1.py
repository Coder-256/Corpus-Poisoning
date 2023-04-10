import torch
from math import exp, log, sqrt
from numba import guvectorize

dictionary = ["foo", "bar"]  # TODO
C = {(0, 1): 0.8}            # TODO: map idx pairs to coocurrence
B = {0: 0.2, 1: 0.7}         # TODO
omega = [0]*len(dictionary)  # TODO


def comp_Jhat(s: int, NEG: list[int], POS: list[int], Dhat: list[int]) -> float:
  # TODO
  pass


class CompDiffState:
  def __init__(self, Jhat: float, Csum: list[float], M_norms: dict[int, float], t_dots: dict[int, float], Delta_size: int):
    self.Jhat = Jhat
    self.Csum = Csum
    self.M_norms = M_norms
    self.t_dots = t_dots
    self.Delta_size = Delta_size


class TensorPrime:
  def __init__(self, tensor: torch.Tensor, overrides: dict = {}):
    self.tensor = tensor
    self.overrides = overrides

  def __getitem__(self, key):
    return self.overrides[key] if key in self.overrides else self.tensor[key]

  def __setitem__(self, key, value):
    self.overrides[key] = value


def model_f(u: int, v: int, c: float, epsilon: float, B: dict[int, float]) -> float:
  return max(log(c)-B[u]-B[v], epsilon)


def solve_greedy(s: int, POS: list[int], NEG: list[int], t_rank: float, alpha: float, max_delta: float, M: torch.Tensor) -> list[int]:
  D = len(dictionary)
  T = POS + NEG
  Dhat = [0.]*D
  Cs = C[s]

  def comp_diff_naive(i: int, delta: float, state: CompDiffState) -> CompDiffState:
    # TODO: We assume in several places that C and M are symmetric. Is this true?
    Csumid = TensorPrime(state.Csum)
    # TODO: Do we need to update Csumid[] for any other words?
    # The paper implies that we may need to, but I don't see why.
    Csumid[s] += delta
    if i != s:
      Csumid[i] += delta
    # TODO: Update the bias if i is a fake word. See "Estimating biases", pp. 17-18
    Bpid = TensorPrime(B)

    fp_e60 = {(u, t): model_f(s, t, Csumid[u], exp(-60), Bpid)
              for u in T+[s] for t in T}
    fp_0 = {t: model_f(s, t, Cs[t]+Dhat[t]+delta, 0, Bpid) for t in T}

    old_Mp_si = model_f(s, i, Cs[i]+Dhat[i], 0, Bpid)
    new_Mp_si = model_f(s, i, Cs[i]+Dhat[i]+delta, 0, Bpid)
    d_Mp_si = new_Mp_si - old_Mp_si
    t_dotsid = TensorPrime(state.t_dots)
    for t in state.t_dots:
      t_dotsid[t] += d_Mp_si * M[t][i]
      if i == t:
        # This if block is prompted by the line "if i \in POS \cup NEG, we also add a similar term..." on p. 17
        # As I understand it, this whole thing relies on Mp[s][s] having a meaningful value; is that on purpose?
        # old_Mp_is = model_f(i, s, C[i][s]+Dhat[i], 0, Bpid)
        # new_Mp_is = model_f(i, s, C[i][s]+Dhat[i]+delta, 0, Bpid)
        # # Dhat[s] should be 0 I think, so Mp[s][s] = M[s][s]
        # t_dotsid[i] += (new_Mp_is - old_Mp_is) * M[s][s]s]
        t_dotsid[i] += (new_Mp_si - old_Mp_si) * M[s][s]

    M_normsid = TensorPrime(state.M_norms)
    d_Mp_si2 = new_Mp_si*new_Mp_si - old_Mp_si*old_Mp_si
    M_normsid[s] += d_Mp_si2
    if i in T:
      M_normsid[i] += d_Mp_si2

    dsim1 = {}
    for t in T:
      p1 = fp_0[t]/sqrt(fp_e60[(s, t)]*fp_e60[(t, t)])
      fs = model_f(s, t, state.Csum[s], exp(-60), Bpid)
      ft = model_f(s, t, state.Csum[t], exp(-60), Bpid)
      p2 = model_f(s, t, Cs[t]+Dhat[t], 0, Bpid)/sqrt(fs*ft)
      dsim1[t] = p1 - p2

    dsim2 = {}
    for t in T:
      p1 = t_dotsid[t]/sqrt(M_normsid[s]*M_normsid[t])
      p2 = state.t_dots[t]/sqrt(state.M_norms[s]*state.M_norms[t])
      dsim2[t] = p2 - p1

    dsim12 = {t: (dsim1[t]+dsim2[t])/2 for t in T}
    dJhat_numer = sum(dsim12[t] for t in POS) - sum(dsim12[t] for t in NEG)
    dJhat = dJhat_numer/(len(POS)+len(NEG))

    # TODO: return CompDiffState(dJhat, ...)
  
  t = 123
  Mt = [model_f(t, i, C[t][i], 0) for i in range(D)]
  res = [] # TODO: np.zeros() or something uninitialized
  sim2_vec_opt(delta, state.M_norms[s], B[s], C[s]+Dhat, Mt, state.t_dots[t], B, res)
  res /= Mt.norm()
      
  
  # use naive if: i in T or i is a made-up word (in which case, bias depends on delta)
  # always need: M_norms, t_dots, Dhat, omega, Csum, etc...
  # per-item: M[t][i] for all t in T, Cs[i], Dhat[i]


  # TODO: for simple cases (ie. cases that don't need naive), sim1 doesn't even depend on delta

  A = POS + NEG + [s]
  Csum = [C[i].sum() for i in range(D)]
  M_norms = {u: M[u].norm().item() ** 2 for u in A}
  t_dots = {t: M[s].dot(M[t]).item() for t in A}
  jp = comp_Jhat(s, NEG, POS, Dhat)
  state = CompDiffState(jp, Csum, M_norms, t_dots, 0)
  while state.Jhat < t_rank + alpha and state.Delta_size < max_delta:
    dmap: dict[tuple[int, int], CompDiffState] = {}
    # TODO: Parallelize
    for i in range(D):
      if i == s:
        # TODO: Should we actually handle this?
        continue
      for l in range(1, 31):
        delta = l/5
        dmap[(i, delta)] = comp_diff_naive(i, delta, state)
        dmap[(i, delta)].Delta_size = delta/omega[i]
    i_star, delta_star = -1, -1
    best = -1
    for i, delta in dmap:
      cd_state = dmap[(i, delta)]
      score = cd_state.Jhat / cd_state.Delta_size
      if score > best:
        best = score
        i_star, delta_star = i, delta
    cd_star = dmap[(i_star, delta_star)]
    # TODO: Is "+=" the right operator for everything here??
    state.Jhat += cd_star.Jhat
    # TODO: Why is the pseudocode missing the stars here?
    state.Csum += cd_star.Csum
    state.M_norms += cd_star.M_norms
    state.t_dots += cd_star.t_dots
    # TODO: Why isn't this in the pseudocode?:
    Dhat[i_star] += delta_star

  return Dhat
