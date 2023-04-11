from numba import cuda

# # TODO: JIT model_f
# # input: i,delta
# # output: list of sim12 values, one per target
# # @guvectorize(['void(float32,float32[:,:],float32[:],float32[:],float32,float32,float32[:],float32[:,:,:])'],
# #              '(),(T,D),(D),(D),(),(),(D)->(D,L,T)', target='cuda')
# def comp_diff_opt(delta, Mts, Cs, Dhat, base_sim1, base_sim2, B, res):
#   for i in range(Cs.shape[0]):
#     old_Mp_si = model_f(s, i, Cs[i]+Dhat[i], 0, B)
#     new_Mp_si = model_f(s, i, Cs[i]+Dhat[i]+delta, 0, B)
#     for t in range(Mts.shape[0]):
#       numer1 = model_f(Cs[t])

block_dim = 256
num_targets = 10
target_groups_per_block = block_dim // num_targets

# we stuff as many num_targets as we can, into threads_per_block

# TODO: JIT model_f
# input: i,delta
# output: list of sim12 values, one per target
# TODO: the result from here still needs to be divided by norm(M_t)

@cuda.jit(device=True)
def model_f(u, v, c, epsilon, B):
  return cuda.libdevice.fmaxf(cuda.libdevice.logf(c)-B[u]-B[v], epsilon)

ntargets = 0 # TODO

# TODO: put Mdot in constant memory

# s: int, delta: float, Cps: float[D], Mdot: float[T], Ms_norm: float, M_t: float[T, D], M_t_norm: float[T], B: float[D], T_wgt: float[T], res: float[D]
# NOTE: Cps = C[s] + Dhat
@cuda.jit
def sim2_vec_opt(s, delta, Cps, Mdot, Ms_norm, M_t, M_t_norm, B, T_wgt, res):
  i = cuda.grid(1)
  sim = 0.
  for t in range(ntargets):
    old_M_si = model_f(s, i, Cps[i], 0, B)
    new_M_si = model_f(s, i, Cps[i]+delta, 0, B)
    dot_id = Mdot[t] + (new_M_si-old_M_si)*M_t[t, i]
    Ms_normid = Ms_norm + new_M_si*new_M_si - old_M_si*old_M_si
    Mt_normid = M_t_norm[t]
    sim += T_wgt[t]*dot_id/(Ms_normid*Mt_normid)
  res[i] = sim


threadsperblock = 32
blockspergrid = (D + (threadsperblock - 1)) // threadsperblock
sim2_vec_opt[blockspergrid, threadsperblock](s, delta, )