import quimb as qu
import quimb.tensor as qtn
import itertools
import numpy as np
import netket as nk
import matplotlib.pyplot as plt
import scipy
import scipy.sparse.linalg
import scipy.linalg
import scipy.sparse
from twoD_tool import *

L = 4
mean_ = [0.0, 0.02, 0.05, 0.1, 0.2]
bond_dim_ = [2, 5, 10]
dtype = "float64"   #dtype of the peps, float64 or complex128
iter_ = 5
for mean in mean_:
    for bond_dim in bond_dim_:
        for i in range(iter_):
            peps = qtn.PEPS.rand(Lx = L, Ly = L, dtype=dtype, bond_dim=bond_dim, seed=i)
            for tensor in peps.tensors:
                data = tensor.data
                data += mean
            prob = peps.contract(all, optimize='auto-hq')
            print("contraction_finished", "dtype:", prob.dtype, "bond_dim:", bond_dim, "mean:", mean, "iter:", i)
            prob_exact = np.real(prob.data.conj()*prob.data)
            norm = prob_exact.sum()
            prob_exact = prob_exact/norm
            mean_corr_, var_corr_ = correlation_all(prob_exact, L)
            cmi = cmi_(prob_exact, L)
            cmi_all = cmi_traceout(prob_exact, L)

            np.save("result/random_peps/cmi_random_peps_L"+str(L)+"_mean_"+str(mean)+"_bond_dim_"+str(bond_dim)+"_iter="+str(i)+"_dtype="+str(dtype)+".npy", np.array(cmi))
            np.save("result/random_peps/mean_corr_random_peps_L"+str(L)+"_mean_"+str(mean)+"_bond_dim_"+str(bond_dim)+"_iter="+str(i)+"_dtype="+str(dtype)+".npy", np.array(mean_corr_))
            np.save("result/random_peps/var_corr_random_peps_L"+str(L)+"_mean_"+str(mean)+"_bond_dim_"+str(bond_dim)+"_iter="+str(i)+"_dtype="+str(dtype)+".npy", np.array(var_corr_))
            np.save("result/random_peps/cmi_traceout_random_peps_L"+str(L)+"_mean_"+str(mean)+"_bond_dim_"+str(bond_dim)+"_iter="+str(i)+"_dtype="+str(dtype)+".npy", np.array(cmi_all))

