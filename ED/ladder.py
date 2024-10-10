import jax
import netket as nk
import numpy as np
from netket.operator.spin import sigmax,sigmaz, sigmap, sigmam
import time
from scipy.sparse.linalg import eigsh
import jax.numpy as jnp
import matplotlib.pyplot as plt
from ladder_tool import *

L = 8
periodic = False
hi = nk.hilbert.Spin(s=1 / 2, N =  L*2)

Jv_ = [0.2, 0.4, 0.6]
Jx_ = [0.2, 0.4, 0.6]


for Jv in  Jv_ :
    for Jx in Jx_:
        H = sum([2*(sigmap(hi, 2*y)*sigmam(hi, 2*(y+1))+sigmam(hi, 2*y)*sigmap(hi, 2*(y+1))+sigmap(hi, 2*y+1)*sigmam(hi, 2*(y+1)+1)+sigmam(hi, 2*y+1)*sigmap(hi, 2*(y+1)+1))+sigmaz(hi, 2*y)*sigmaz(hi, 2*(y+1))+sigmaz(hi, 2*y+1)*sigmaz(hi, 2*(y+1)+1) for y in range(L-1) ])  #J-parallel

        H += Jv*sum([2*(sigmap(hi, 2*y)*sigmam(hi, 2*y+1)+sigmam(hi, 2*y)*sigmap(hi, 2*y+1))+sigmaz(hi, 2*y)*sigmaz(hi, 2*y+1) for y in range(L)]) # J-vertical

        H += Jx*sum([2*(sigmap(hi, 2*y)*sigmam(hi, 2*(y+1)+1)+sigmam(hi, 2*y)*sigmap(hi, 2*(y+1)+1)+sigmap(hi, 2*y+1)*sigmam(hi, 2*(y+1))+sigmam(hi, 2*y+1)*sigmap(hi, 2*(y+1)))+sigmaz(hi, 2*y)*sigmaz(hi, 2*(y+1)+1)+sigmaz(hi, 2*y+1)*sigmaz(hi, 2*(y+1)) for y in range(L-1)])  #J-cross

        if (periodic == True):
        #periodic boundary conditions
            H+= 2*sigmap(hi, 0)*sigmam(hi, 2*L-2)+2*sigmam(hi, 0)*sigmap(hi, 2*L-2)+2*sigmap(hi, 1)*sigmam(hi, 2*L-1)+2*sigmam(hi, 1)*sigmap(hi, 2*L-1)+sigmaz(hi, 0)*sigmaz(hi, 2*L-2)+sigmaz(hi, 1)*sigmaz(hi, 2*L-1) #J-parallel

            H+= Jx*(2*sigmap(hi, 0)*sigmam(hi, 2*L-1)+2*sigmam(hi, 0)*sigmap(hi, 2*L-1)+2*sigmap(hi, 1)*sigmam(hi, 2*L-2)+2*sigmam(hi, 1)*sigmap(hi, 2*L-2)+sigmaz(hi, 0)*sigmaz(hi, 2*L-1)+sigmaz(hi, 1)*sigmaz(hi, 2*L-2)) #J-cross
        sp_h = H.to_sparse()
        eig_vals, eig_vecs = eigsh(sp_h, k=2, which="SA")
        print("eigenvalues with scipy sparse Jv="+str(Jv)+ "_Jx="+ str(Jx) +":", eig_vals)

        prob_exact = eig_vecs[:,0]**2
        shape = (2,) * (2*L)
        prob_exact = prob_exact.reshape(*shape)

        if periodic == True:
            mean_corr_dimer = correlation_periodic_dimer(prob_exact, L)
            mean_corr_single = correlation_periodic_single(prob_exact, L)
            cmi_dimer = cmi_periodic(prob_exact, L)

        else:
            mean_corr, var_corr = correlation_all(prob_exact, L)
            cmi_dimer = cmi_(prob_exact, L, "dimer")

        cmi_single = cmi_(prob_exact, L, "single")
        cmi_traceout = cmi_traceout_dimer(prob_exact, L)

        np.save("result/ladder/gap_ladder_L"+str(L)+"_Jv="+str(Jv)+"_Jx="+str(Jx)+"periodic_"+str(periodic)+".npy", np.array(eig_vals[1]-eig_vals[0]))
        np.save("result/ladder/cmi_ladder_dimer_L"+str(L)+"_Jv="+str(Jv)+"_Jx="+str(Jx)+"periodic_"+str(periodic)+".npy", cmi_dimer)
        np.save("result/ladder/cmi_ladder_single_L"+str(L)+"_Jv="+str(Jv)+"_Jx="+str(Jx)+"periodic_"+str(periodic)+".npy", cmi_single)
        np.save("result/ladder/cmi_ladder_traceout_L"+str(L)+"_Jv="+str(Jv)+"_Jx="+str(Jx)+"periodic_"+str(periodic)+".npy", cmi_traceout)

        if periodic == False:
            np.save("result/ladder/mean_corr_ladder_dimer_L"+str(L)+"_Jv="+str(Jv)+"_Jx="+str(Jx)+"periodic_"+str(periodic)+".npy", mean_corr)
            np.save("result/ladder/var_corr_ladder_dimer_L"+str(L)+"_Jv="+str(Jv)+"_Jx="+str(Jx)+"periodic_"+str(periodic)+".npy", var_corr)
        elif periodic == True:
            np.save("result/ladder/mean_corr_ladder_dimer_L"+str(L)+"_Jv="+str(Jv)+"_Jx="+str(Jx)+"periodic_"+str(periodic)+".npy", mean_corr_dimer)
            np.save("result/ladder/mean_corr_ladder_single_L"+str(L)+"_Jv="+str(Jv)+"_Jx="+str(Jx)+"periodic_"+str(periodic)+".npy", mean_corr_single)

