import torch
import numpy as np
import cupy as cp
from numba import cuda
from math import exp, log, sqrt, ceil
from bisect import bisect_left


@cuda.jit(device=True)
def cuda_model_f(u, v, c, epsilon, B):
  return max(log(c)-B[u]-B[v], epsilon)

# note: ntargets is captured


@cuda.jit
def sim2_kernel(s, delta, Cps, Mdots_t, Ms_norm, M_t, M_t_norm, B, T_wgt, res):
  i = cuda.grid(1)
  if i < Cps.shape[0]:
    sim2 = 0.
    for t in range(M_t.shape[0]):
      old_M_si = cuda_model_f(s, i, Cps[i], 0, B)
      new_M_si = cuda_model_f(s, i, Cps[i]+delta, 0, B)
      dot_id = Mdots_t[t] + (new_M_si-old_M_si)*M_t[t, i]
      Ms_normid = Ms_norm + new_M_si*new_M_si - old_M_si*old_M_si
      Mt_normid = M_t_norm[t]
      sim2 += T_wgt[t]*dot_id/(Ms_normid*Mt_normid)
    res[i] = sim2


class CorpusPoison:
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

  def __init__(self, dictionary, cooccur, bias, omega):
    self.dictionary = dictionary
    self.C = cooccur
    self.B = bias
    self.omega = omega

  def model_f(self, u: int, v: int, c: float, epsilon: float, B) -> float:
    return max(log(c)-B[u]-B[v], epsilon)

  def comp_diff_naive(self, i: int, delta: float, state: CompDiffState) -> CompDiffState:
    s = self.s
    C = self.C
    B = self.B
    T = self.T
    POS = self.POS
    NEG = self.NEG
    Dhat = self.Dhat
    Csumid = self.TensorPrime(state.Csum)
    Csumid[self.s] += delta
    if i != self.s:
      Csumid[i] += delta
    fp_e60 = {(u, t): self.model_f(s, t, Csumid[u], exp(-60), B)
              for u in T+[s] for t in T}
    fp_0 = {t: self.model_f(s, t, C[s, t]+Dhat[t]+delta, 0, B) for t in T}

    old_Mp_si = self.model_f(s, i, C[s, i]+Dhat[i], 0, B)
    new_Mp_si = self.model_f(s, i, C[s, i]+Dhat[i]+delta, 0, B)
    d_Mp_si = new_Mp_si - old_Mp_si
    t_dotsid = self.TensorPrime(state.t_dots)
    for t in state.t_dots:
      t_dotsid[t] += d_Mp_si * self.model_f(t, i, C[t, i], 0, B)
      if i == t:
        t_dotsid[i] += (new_Mp_si - old_Mp_si) * \
            self.model_f(s, s, C[s, s], 0, B)

    M_normsid = self.TensorPrime(state.M_norms)
    d_Mp_si2 = new_Mp_si*new_Mp_si - old_Mp_si*old_Mp_si
    M_normsid[s] += d_Mp_si2
    if i in T:
      M_normsid[i] += d_Mp_si2

    dsim1 = {}
    for t in T:
      p1 = fp_0[t]/sqrt(fp_e60[(s, t)]*fp_e60[(t, t)])
      fs = self.model_f(s, t, state.Csum[s], exp(-60), B)
      ft = self.model_f(s, t, state.Csum[t], exp(-60), B)
      p2 = self.model_f(s, t, C[s, t]+Dhat[t], 0, B)/sqrt(fs*ft)
      dsim1[t] = p1 - p2

    dsim2 = {}
    for t in T:
      p1 = t_dotsid[t]/sqrt(M_normsid[s]*M_normsid[t])
      p2 = state.t_dots[t]/sqrt(state.M_norms[s]*state.M_norms[t])
      dsim2[t] = p2 - p1

    dsim12 = {t: (dsim1[t]+dsim2[t])/2 for t in T}
    dJhat_numer = sum(dsim12[t] for t in POS) - sum(dsim12[t] for t in NEG)
    dJhatid = dJhat_numer/(len(POS)+len(NEG))

    return self.CompDiffState(dJhatid, Csumid, M_normsid, t_dotsid, delta/self.omega[i])

  def solve_greedy(self, s: int, POS, NEG, t_rank: float, alpha: float, max_delta: float):
    self.s = s
    self.POS = POS
    self.NEG = NEG
    self.t_rank = t_rank
    self.alpha = alpha
    self.max_delta = max_delta
    self.T = T = POS + NEG
    A = POS + NEG + [s]
    keep = np.fromiter(
        (j for j in range(len(self.dictionary))
         if j in A or any(self.model_f(u, j, C[u, j], 0, B) > 0 for u in A)),
        dtype=np.int_
    )
    K = len(keep)
    self.Dhat = Dhat = [0.]*K
    C = self.C
    B = self.B
    ntargets = len(T)
    Csum = {j: C[j].sum() for j in keep}

    def C_row(i):
      return np.array([self.model_f(i, j, C[i, j], 0, B) for j in keep], dtype=np.float32)

    M_norms = {u: np.linalg.norm(C_row(u)).item() ** 2 for u in A}
    t_dots = {t: C_row(s).dot(C_row(t)).item() for t in A}
    T_wgt = np.array([1]*len(POS) + [-1]*len(NEG), dtype=np.float32)

    start_sim1 = start_sim2 = 0.
    for t, w in zip(T, T_wgt):
      fs = self.model_f(s, t, Csum[s], exp(-60), B)
      ft = self.model_f(s, t, Csum[t], exp(-60), B)
      start_sim1 += w*self.model_f(s, t, C[s, t]+Dhat[t], 0, B)/sqrt(fs*ft)
      start_sim2 += w*t_dots[t]/sqrt(M_norms[s]*M_norms[t])
    start_sim12 = (start_sim1+start_sim2)/2

    state = self.CompDiffState(start_sim12, Csum, M_norms, t_dots, 0)

    # s, delta, Cps, Mdots_t, Ms_norm, M_t, M_t_norm, B, T_wgt, res
    Ms_norm = np.linalg.norm(
        np.array([self.model_f(s, j, C[s, j], 0, B) for j in keep], dtype=np.float32))
    Kstride = (4*K+31) & (~31)  # TODO: should this be larger?
    pad = Kstride//4 - K
    Mdots_t = np.array([C[s].dot(C[t])
                        for t in range(ntargets)], dtype=np.float32)
    M_t = np.array([[self.model_f(t, j, C[t, j], 0, B)
                     for j in keep]+[0]*pad for t in T], dtype=np.float32)

    sim2_Cps = cuda.to_device(C[s, keep])
    sim2_Mdots_t = cuda.to_device(Mdots_t)
    sim2_M_t = cuda.to_device(M_t)
    sim2_M_t_norm = cuda.to_device(np.linalg.norm(M_t, axis=1))
    sim2_B = cuda.to_device(B[keep])
    sim2_T_wgt = cuda.to_device(T_wgt)
    sim2_res = cuda.device_array(K, dtype=np.float32)

    threadsperblock = 32
    blockspergrid = (K + (threadsperblock - 1)) // threadsperblock

    print("will start...")
    while state.Jhat < t_rank + alpha and state.Delta_size < max_delta:
      dmap = {}

      for l in range(1, 31):
        delta = l/5
        sim1_numer = np.array([self.model_f(s, t, C[s, t]+delta, 0, B)
                              for t in T], dtype=np.float32)
        sim1_fsum_s = np.array(
            [self.model_f(s, t, state.Csum[s]+delta, exp(-60), B) for t in T], dtype=np.float32)
        sim1_fsum_t = np.array(
            [self.model_f(s, t, state.Csum[t], exp(-60), B) for t in T], dtype=np.float32)
        sim1 = ((sim1_numer/sqrt(sim1_fsum_s*sim1_fsum_t))*T_wgt).sum()

        # call the kernel
        sim2_kernel[blockspergrid, threadsperblock](
            s, delta, sim2_Cps, sim2_Mdots_t, Ms_norm, sim2_M_t, sim2_M_t_norm, sim2_B, sim2_T_wgt, sim2_res)

        # kernel complete! create cupy array view and use it to find sim12 vector
        sim2_res_cp = cp.asarray(sim2_res)
        sim12 = (sim1+sim2_res_cp)/2

        # we want to argmax only the correct values (everything not in A)
        for j in A:
          sim12[j] = -100
        i = keep[sim12.argmax().item()]

        # calculate naively for i and everything in A
        for j in A + [i]:
          dmap[(j, delta)] = self.comp_diff_naive(j, delta, state)

      i_star, delta_star = -1, -1
      best = -1
      for i, delta in dmap:
        cd_state = dmap[(i, delta)]
        score = cd_state.Jhat / cd_state.Delta_size
        if score > best:
          best = score
          i_star, delta_star = i, delta
      cd_star = dmap[(i_star, delta_star)]

      # Update naive state
      state.Jhat += cd_star.Jhat
      cd_star.Csum.apply(state.Csum)
      cd_star.M_norms.apply(state.M_norms)
      cd_star.t_dots.apply(state.t_dots)
      Dhat[i_star] += delta_star

      # Update CUDA state
      for t, x in cd_star.M_norms.overrides.items():
        if t in T:
          sim2_M_t_norm[T.index(t)] = x
      for t, x in cd_star.t_dots.overrides.items():
        if t in T:
          sim2_Mdots_t[T.index(t)] = x
      i_star_keep = bisect_left(keep, i_star)
      if i_star_keep >= K or keep[i_star_keep] != i_star:
        print("UNABLE TO FIND i_star in keep!")
        return None
      sim2_Cps[i_star_keep] += delta_star

      print("Iterated!", i_star, delta, state.Jhat)

    return Dhat


# solve_greedy(0, [1], [], 1000, 0.01, 10)
