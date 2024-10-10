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
J1 = 1.0
J2_ = [0., 0.2, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.8, 1.0]



for J2 in J2_ :
    H = J1*sum([2*(sigmap(hi, y*L+x)*sigmam(hi, (y+1)*L+x)+sigmam(hi, y*L+x)*sigmap(hi, (y+1)*L+x))+sigmaz(hi, y*L+x)*sigmaz(hi, (y+1)*L+x) for y in range(L-1) for x in range(L)])  #up-down J1
    H += J1*sum([2*(sigmap(hi, y*L+x)*sigmam(hi, y*L+x+1)+sigmam(hi, y*L+x)*sigmap(hi, y*L+x+1))+sigmaz(hi, y*L+x)*sigmaz(hi, y*L+x+1) for y in range(L) for x in range(L-1)]) #left-right J1
    H += J2*sum([2*(sigmap(hi, y*L+x)*sigmam(hi, (y+1)*L+x+1)+sigmam(hi, y*L+x)*sigmap(hi, (y+1)*L+x+1))+sigmaz(hi, y*L+x)*sigmaz(hi, (y+1)*L+x+1) for y in range(L-1) for x in range(L-1)])  #right-down J2
    H += J2*sum([2*(sigmap(hi, y*L+x+1)*sigmam(hi, (y+1)*L+x)+sigmam(hi, y*L+x+1)*sigmap(hi, (y+1)*L+x))+sigmaz(hi, y*L+x+1)*sigmaz(hi, (y+1)*L+x) for y in range(L-1) for x in range(L-1)])  #left-down J2
    if (periodic == True):
    #periodic boundary conditions
        H+= J1*sum([2*(sigmap(hi, x)*sigmam(hi, (L-1)*L+x)+sigmam(hi, x)*sigmap(hi, (L-1)*L+x))+sigmaz(hi, x)*sigmaz(hi, (L-1)*L+x) for x in range(L)]) # last row - first row J1
        H+= J1*sum([2*(sigmap(hi, y*L)*sigmam(hi, y*L+L-1)+sigmam(hi, y*L)*sigmap(hi, y*L+L-1))+sigmaz(hi, y*L)*sigmaz(hi, y*L+L-1) for y in range(L)]) # last column - first column J1
        H+= J2*sum([2*(sigmap(hi, y*L+L-1)*sigmam(hi, (y+1)*L)+sigmam(hi, y*L+L-1)*sigmap(hi, (y+1)*L))+sigmaz(hi, y*L+L-1)*sigmaz(hi, (y+1)*L) for y in range(L-1)]) # last column - first column J2 (right down)
        H+= J2*sum([2*(sigmap(hi, y*L)*sigmam(hi, (y+2)*L-1)+sigmam(hi, y*L)*sigmap(hi, (y+2)*L-1))+sigmaz(hi, y*L)*sigmaz(hi, (y+2)*L-1) for y in range(L-1)]) #  last column - first column J2 (left down)
        H+= J2*sum([2*(sigmap(hi, x+1)*sigmam(hi, (L-1)*L+x)+sigmam(hi, x+1)*sigmap(hi, (L-1)*L+x))+sigmaz(hi, x+1)*sigmaz(hi, (L-1)*L+x) for x in range(L-1)]) # last row - first row J2 (right down)
        H+= J2*sum([2*(sigmap(hi, x)*sigmam(hi, (L-1)*L+x+1)+sigmam(hi, x)*sigmap(hi, (L-1)*L+x+1))+sigmaz(hi, x)*sigmaz(hi, (L-1)*L+x+1) for x in range(L-1)]) # last row - first row J2 (left down)
        H+= J2*(2*(sigmap(hi, L*L-1)*sigmam(hi, 0)+sigmam(hi, L*L-1)*sigmap(hi, 0))+sigmaz(hi, L*L-1)*sigmaz(hi, 0)) # right down corner J2
        H+= J2*(2*(sigmap(hi, L*(L-1))*sigmam(hi, L-1)+sigmam(hi, L*(L-1))*sigmap(hi, L-1))+sigmaz(hi, L*(L-1))*sigmaz(hi, L-1)) # left down corner J2
    sp_h = H.to_sparse()
    eig_vals, eig_vecs = eigsh(sp_h, k=3, which="SA")
    print("eigenvalues with scipy sparse J2="+str(J2) +":", eig_vals)
    np.save("eigvals_2DJ1J2_L"+str(L)+"_J2_"+str(J2)+".npy", np.array(eig_vals))
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
        p_ab = prob_exact.sum(axis=tuple(C[i]))+1e-30
        tmp += np.sum(p_ab*np.log(p_ab))
        p_bc = prob_exact.sum(axis=tuple(A))+1e-30
        tmp += np.sum(p_bc*np.log(p_bc))
        tmp -= np.sum(prob_exact*np.log(prob_exact+1e-30))
        p_b = prob_exact.sum(axis=tuple(C[i].union(A)))+1e-30
        tmp -= np.sum(p_b*np.log(p_b))
        cmi.append(-tmp)
    np.save("cmi_2DJ1J2_L"+str(L)+"_J2_"+str(J2)+".npy", np.array(cmi))
    plt.plot(np.log(cmi), label = "J2 = "+str(J2))
    plt.legend()
    plt.xlabel("grid_distance")
    plt.ylabel("log_cmi")