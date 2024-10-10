import jax
import netket as nk
import numpy as np
from netket.operator.spin import sigmax,sigmaz, sigmap, sigmam
import time
from scipy.sparse.linalg import eigsh
import jax.numpy as jnp
import matplotlib.pyplot as plt
def ABC(L):
    B = {}
    A = {0}
    C = {}
    for i in range (1, L):
        B[i] = [i+j*(L-1) for j in range(i+1)]
    for i in range (L, 2*L-2):
        B[i] = [(i-L+1)*L+j*(L-1) for j in range(1, 2*L-i)]
    for i in B:
        if i!=1:
            B[i]+=B[i-1]
            B[i].sort()
    for i in B:
        C[i] = set(range(L**2))-set(B[i])-A
    return A, B, C

L = 4
N = L*L
periodic = False
hi = nk.hilbert.Spin(s=1 / 2, N =  N)

B_ = [1.0, 2.0, 2.5, 2.8, 2.9, 2.95, 3.0, 3.05, 3.2, 4.]



for B_field in B_ :
    H = -sum([sigmaz(hi, y*L+x)*sigmaz(hi, (y+1)*L+x) for y in range(L-1) for x in range(L)])  #up-down
    H -= sum([sigmaz(hi, y*L+x)*sigmaz(hi, y*L+x+1) for y in range(L) for x in range(L-1)]) #left-right
    H -= B_field*sum([sigmax(hi, y*L+x) for y in range(L) for x in range(L)]) # B
    if (periodic == True):
    #periodic boundary conditions
        H-= sum([sigmaz(hi, x)*sigmaz(hi, (L-1)*L+x) for x in range(L)]) # last row - first row
        H-= sum([sigmaz(hi, y*L)*sigmaz(hi, y*L+L-1) for y in range(L)]) # last column - first column

    sp_h = H.to_sparse()
    eig_vals, eig_vecs = eigsh(sp_h, k=3, which="SA")
    print("eigenvalues with scipy sparse B="+str(B_field) +":", eig_vals)
    np.save("eigvals_2DTFIM_L"+str(L)+"_B_"+str(B_field)+".npy", np.array(eig_vals))
    prob_exact = eig_vecs[:,0]**2
    shape = (2,) * (L**2)
    prob_exact = prob_exact.reshape(*shape)
    A,B,C = ABC(L)
    cmi = []
    p_ab = prob_exact.sum(axis=tuple(set(range(L**2))-A))+1e-30
    p_bc = prob_exact.sum(axis=tuple(A))+1e-30
    cmi.append(-np.sum(p_ab*np.log(p_ab))-np.sum(p_bc*np.log(p_bc))+np.sum(prob_exact*np.log(prob_exact+1e-30)))
    for i in B:
        tmp = 0
        p_ab = prob_exact.sum(axis=tuple(C[i]))
        tmp += np.sum(p_ab*np.log(p_ab))+1e-30
        p_bc = prob_exact.sum(axis=tuple(A))
        tmp += np.sum(p_bc*np.log(p_bc))+1e-30
        tmp -= np.sum(prob_exact*np.log(prob_exact+1e-30))
        p_b = prob_exact.sum(axis=tuple(C[i].union(A)))+1e-30
        tmp -= np.sum(p_b*np.log(p_b))
        cmi.append(-tmp)
    np.save("cmi_2DTFIM_L"+str(L)+"_B_"+str(B_field)+".npy", np.array(cmi))
    plt.plot(np.log(cmi), label = "B = "+str(B_field))
    plt.legend()
    plt.xlabel("grid_distance")
    plt.ylabel("log_cmi")