import torch
from math import exp, sqrt

dictionary = ["foo", "bar"]  # TODO
C = {(0, 1): 0.8}            # TODO: map idx pairs to coocurrence
omega = [0]*len(dictionary)  # TODO


def comp_Jhat(s: int, NEG: list[int], POS: list[int], Dhat: list[int]) -> float:
  # TODO
  pass


class CompDiffState:
  def __init__(self, Jhat: float, cooccurs: list[float], M_norms: dict[int, float], t_dots: dict[int, float], Delta_size: int):
    self.Jhat = Jhat
    self.cooccurs = cooccurs
    self.M_norms = M_norms
    self.t_dots = t_dots
    self.Delta_size = Delta_size


def model_f(u: int, v: int, c: float, epsilon: float, B: dict[int, float]) -> float:
  # TODO
  pass


def solve_greedy(s: int, POS: list[int], NEG: list[int], t_rank: float, alpha: float, max_delta: float, M: torch.Tensor) -> list[int]:
  D = len(dictionary)

  def comp_diff(i: int, delta: float, state: CompDiffState) -> CompDiffState:
    # TODO: implement and find correct type for state
    # returns (jhat, cooccurs, M_norms, t_dots, Delta_size)}

    Cp = {}  # TODO: updated cooccurrence
    Bp = {}  # TODO: updated bias
    Mp = torch.tensor()  # TODO: updated distributional matrix
    D = len(dictionary)
    targets = POS + NEG  # TODO ???
    u_todo = [s] + targets

    fp_e60 = {(u, t): model_f(s, t, Cp[u].sum(), exp(-60), Bp)
              for u in u_todo for t in targets}
    fp_0 = {t: model_f(s, t, Cp[s][t], 0, Bp) for t in targets}

    d_Mp_si = torch.tensor()  # TODO
    d_Mp_ts = 123 if i in POS or i in NEG else 0  # TODO
    new_t_dots = dict(state.t_dots)
    for t in new_t_dots:
      new_t_dots[t] += d_Mp_si.dot(M[t][i]) + d_Mp_ts

    d_Mp_si2 = 0  # TODO
    new_Mp_s_norm = state.M_norms[s] + d_Mp_si2
    # TODO: if i in POS or i in NEG, get new_Mp_i_norm
    new_Mp_t_norm = 0  # TODO

    dsim1 = {}
    for t in targets:
      p1 = fp_0[t]/sqrt(fp_e60[(s, t)]*fp_e60[(t, t)])
      fs = model_f(s, t, Cp[s].sum(), exp(-60), Bp)
      ft = model_f(s, t, Cp[t].sum(), exp(-60), Bp)
      p2 = model_f(s, t, Cp[s][t], 0, Bp)/sqrt(fs*ft)
      dsim1[t] = p1 - p2

    dsim2 = {}
    for t in targets:
      p1 = new_t_dots[t]/sqrt(new_Mp_s_norm*new_Mp_t_norm)
      p2 = Mp[s].dot(Mp[t])/sqrt(Mp[s].norm()*Mp[t].norm())
      dsim2[t] = p2 - p1

    dsim12 = {(dsim1[t]+dsim2[t])/2 for t in targets}
    dJhat_numer = sum(dsim12[t] for t in POS) - sum(dsim12[t] for t in NEG)
    dJhat = dJhat_numer/(len(POS)+len(NEG))

    # TODO: return CompDiffState(dJhat, ...)

  Dhat = [0]*D
  A = POS + NEG + [s]
  cooccurs = [C[i].sum() for i in range(D)]
  M_norms = {u: M[u].norm().item() ** 2 for u in A}
  t_dots = {t: M[s].dot(M[t]).item() for t in A}
  jp = comp_Jhat(s, NEG, POS, Dhat)
  state = CompDiffState(jp, cooccurs, M_norms, t_dots, 0)
  while state.Jhat < t_rank + alpha and state.Delta_size < max_delta:
    dmap: dict[tuple[int, int], CompDiffState] = {}
    for i in range(D):
      for l in range(1, 31):
        delta = l/5
        dmap[(i, delta)] = comp_diff(i, delta, state)
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
      state.cooccurs += cd_star.cooccurs
      state.M_norms += cd_star.M_norms
      state.t_dots += cd_star.t_dots
      # TODO: Why isn't this in the pseudocode?:
      Dhat[i] += delta

  return Dhat
