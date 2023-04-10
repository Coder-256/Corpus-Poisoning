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

  # TODO: JIT model_f
  # input: i,delta
  # output: list of sim12 values, one per target
  # TODO: the result from here still needs to be divided by norm(M_t)
  @guvectorize(['void(float32,float32,float32,float32,float32[:],float32[:],float32[:],float32[:])'],
               '(),(),(),(),(n),(n),(n)->(n),', target='cuda')
  def sim2_vec_opt(delta, Ms_norm, Bs, Mdot, Cps, Mt, B, res):
    for i in range(Cs.shape[0]):
      # NOTE: Cps = C[s] + Dhat
      old_M_si = max(log(Cps[i])-B[i]-Bs, 0)
      new_M_si = max(log(Cps[i]+delta)-B[i]-Bs, 0)
      dot_id = Mdot + (new_M_si-old_M_si)*Mt[i]
      Ms_normid = Ms_norm + new_M_si*new_M_si - old_M_si*old_M_si
      res[i] = dot_id/Ms_normid