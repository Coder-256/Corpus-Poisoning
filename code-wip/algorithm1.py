import torch
import numpy as np
import cupy as cp
from math import exp, log, sqrt
from numba import cuda

np.seterr(all='raise')

rng = np.random.default_rng()

dictionary = ["foo", "bar"]  # TODO
D = len(dictionary)
C = rng.random((D, D), dtype=np.float32)*1000  # TODO
B = rng.random(D, dtype=np.float32)/1000  # TODO
omega = np.ones(D)                        # TODO


class CompDiffState:
  def __init__(self, Jhat: float, Csum, M_norms, t_dots, Delta_size: int):
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

  def apply(self, other):
    for key in self.overrides:
      other[key] = self.overrides[key]


def model_f(u: int, v: int, c: float, epsilon: float, B) -> float:
  return max(log(c)-B[u]-B[v], epsilon)


def solve_greedy(s: int, POS, NEG, t_rank: float, alpha: float, max_delta: float):
  D = len(dictionary)
  T = POS + NEG
  ntargets = len(T)
  Dhat = [0.]*D
  Cs = C[s]

  @cuda.jit(device=True)
  def cuda_model_f(u, v, c, epsilon, B):
    return max(log(c)-B[u]-B[v], epsilon)

  # note: ntargets is captured
  @cuda.jit
  def sim2_kernel(s, delta, Cps, Mdots_t, Ms_norm, M_t, M_t_norm, B, T_wgt, res):
    i = cuda.grid(1)
    sim2 = 0.
    for t in range(ntargets):
      old_M_si = cuda_model_f(s, i, Cps[i], 0, B)
      new_M_si = cuda_model_f(s, i, Cps[i]+delta, 0, B)
      dot_id = Mdots_t[t] + (new_M_si-old_M_si)*M_t[t, i]
      Ms_normid = Ms_norm + new_M_si*new_M_si - old_M_si*old_M_si
      Mt_normid = M_t_norm[t]
      sim2 += T_wgt[t]*dot_id/(Ms_normid*Mt_normid)
    res[i] = sim2

  def comp_diff_naive(i: int, delta: float, state: CompDiffState) -> CompDiffState:
    Csumid = TensorPrime(state.Csum)
    Csumid[s] += delta
    if i != s:
      Csumid[i] += delta
    fp_e60 = {(u, t): model_f(s, t, Csumid[u], exp(-60), B)
              for u in T+[s] for t in T}
    fp_0 = {t: model_f(s, t, Cs[t]+Dhat[t]+delta, 0, B) for t in T}

    old_Mp_si = model_f(s, i, Cs[i]+Dhat[i], 0, B)
    new_Mp_si = model_f(s, i, Cs[i]+Dhat[i]+delta, 0, B)
    d_Mp_si = new_Mp_si - old_Mp_si
    t_dotsid = TensorPrime(state.t_dots)
    for t in state.t_dots:
      t_dotsid[t] += d_Mp_si * model_f(t, i, C[t][i], 0, B)
      if i == t:
        t_dotsid[i] += (new_Mp_si - old_Mp_si) * model_f(s, s, C[s][s], 0, B)

    M_normsid = TensorPrime(state.M_norms)
    d_Mp_si2 = new_Mp_si*new_Mp_si - old_Mp_si*old_Mp_si
    M_normsid[s] += d_Mp_si2
    if i in T:
      M_normsid[i] += d_Mp_si2

    dsim1 = {}
    for t in T:
      p1 = fp_0[t]/sqrt(fp_e60[(s, t)]*fp_e60[(t, t)])
      fs = model_f(s, t, state.Csum[s], exp(-60), B)
      ft = model_f(s, t, state.Csum[t], exp(-60), B)
      p2 = model_f(s, t, Cs[t]+Dhat[t], 0, B)/sqrt(fs*ft)
      dsim1[t] = p1 - p2

    dsim2 = {}
    for t in T:
      p1 = t_dotsid[t]/sqrt(M_normsid[s]*M_normsid[t])
      p2 = state.t_dots[t]/sqrt(state.M_norms[s]*state.M_norms[t])
      dsim2[t] = p2 - p1

    dsim12 = {t: (dsim1[t]+dsim2[t])/2 for t in T}
    dJhat_numer = sum(dsim12[t] for t in POS) - sum(dsim12[t] for t in NEG)
    dJhatid = dJhat_numer/(len(POS)+len(NEG))

    return CompDiffState(dJhatid, Csumid, M_normsid, t_dotsid, delta/omega[i])

  # Mt = [model_f(t, i, C[t][i], 0) for i in range(D)]
  # res = []  # TODO: np.zeros() or something uninitialized
  # sim2_vec_opt(delta, state.M_norms[s], B[s],
  #              C[s]+Dhat, Mt, state.t_dots[t], B, res)
  # res /= Mt.norm()

  # use naive if: i in T or i is a made-up word (in which case, bias depends on delta)
  # always need: M_norms, t_dots, Dhat, omega, Csum, etc...
  # per-item: M[t][i] for all t in T, Cs[i], Dhat[i]

  # TODO: for simple cases (ie. cases that don't need naive), sim1 doesn't even depend on delta

  A = POS + NEG + [s]
  Csum = [C[i].sum() for i in range(D)]

  def get_row(i):
    return np.array([model_f(i, j, C[i][j], 0, B) for j in range(D)], dtype=np.float32)

  M_norms = {u: np.linalg.norm(get_row(u)).item() ** 2 for u in A}
  t_dots = {t: get_row(s).dot(get_row(t)).item() for t in A}
  T_wgt = np.array([1]*len(POS) + [-1]*len(NEG), dtype=np.float32)

  start_sim1 = start_sim2 = 0.
  for t, w in zip(T, T_wgt):
    fs = model_f(s, t, Csum[s], exp(-60), B)
    ft = model_f(s, t, Csum[t], exp(-60), B)
    start_sim1 += w*model_f(s, t, Cs[t]+Dhat[t], 0, B)/sqrt(fs*ft)
    start_sim2 += w*t_dots[t]/sqrt(M_norms[s]*M_norms[t])
  start_sim12 = (start_sim1+start_sim2)/2

  state = CompDiffState(start_sim12, Csum, M_norms, t_dots, 0)

  # s, delta, Cps, Mdots_t, Ms_norm, M_t, M_t_norm, B, T_wgt, res
  Ms = np.array([model_f(s, i, Cs[i], 0, B)
                for i in range(D)], dtype=np.float32)
  Ms_norm = np.linalg.norm(Ms)
  Dstride = (4*D+31) & (~31)  # TODO: should this be larger?
  pad = Dstride//4 - D
  Mdots_t = np.array([C[s].dot(C[t])
                     for t in range(ntargets)], dtype=np.float32)
  M_t = np.array([[model_f(t, i, C[t][i], 0, B)
                 for i in range(D)]+[0]*pad for t in T], dtype=np.float32)

  sim2_Cps = cuda.to_device(C[s])
  sim2_Mdots_t = cuda.to_device(Mdots_t)
  sim2_M_t = cuda.to_device(M_t)
  sim2_M_t_norm = cuda.to_device(np.linalg.norm(M_t, axis=1))
  sim2_B = cuda.to_device(B)
  sim2_T_wgt = cuda.to_device(T_wgt)
  sim2_res = cuda.device_array(D, dtype=np.float32)

  threadsperblock = 32
  blockspergrid = (D + (threadsperblock - 1)) // threadsperblock

  print("will start...")
  while state.Jhat < t_rank + alpha and state.Delta_size < max_delta:
    dmap = {}

    for l in range(1, 31):
      delta = l/5
      sim1_numer = np.array([model_f(s, t, Cs[t]+delta, 0, B)
                            for t in T], dtype=np.float32)
      sim1_fsum_s = np.array(
          [model_f(s, t, state.Csum[s]+delta, exp(-60), B) for t in T], dtype=np.float32)
      sim1_fsum_t = np.array(
          [model_f(s, t, state.Csum[t], exp(-60), B) for t in T], dtype=np.float32)
      sim1 = ((sim1_numer/sqrt(sim1_fsum_s*sim1_fsum_t))*T_wgt).sum()

      sim2_kernel[blockspergrid, threadsperblock](
          s, delta, sim2_Cps, sim2_Mdots_t, Ms_norm, sim2_M_t, sim2_M_t_norm, sim2_B, sim2_T_wgt, sim2_res)

      sim2_res_cp = cp.asarray(sim2_res)
      sim12 = (sim1+sim2_res_cp)/2

      # get rid of all j for j in T so we don't pick it up during argmax
      for j in T:
        sim12[j] = -100
      i = sim12.argmax().item()

      # fill in dmap for argmax and all i in POS or NEG
      for j in T + [i]:
        dmap[(j, delta)] = comp_diff_naive(j, delta, state)

    i_star, delta_star = -1, -1
    best = -1
    for i, delta in dmap:
      cd_state = dmap[(i, delta)]
      score = cd_state.Jhat / cd_state.Delta_size
      if score > best:
        best = score
        i_star, delta_star = i, delta
    cd_star = dmap[(i_star, delta_star)]
    state.Jhat += cd_star.Jhat
    cd_star.Csum.apply(state.Csum)
    cd_star.M_norms.apply(state.M_norms)
    cd_star.t_dots.apply(state.t_dots)
    Dhat[i_star] += delta_star

    print("Iterated!", i_star, delta, state.Jhat)

  return Dhat


solve_greedy(0, [1], [], 1000, 0.01, 10)
