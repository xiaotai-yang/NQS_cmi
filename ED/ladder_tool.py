import jax
import netket as nk
import numpy as np
from netket.operator.spin import sigmax,sigmaz, sigmap, sigmam
import time
from scipy.sparse.linalg import eigsh
import jax.numpy as jnp
import matplotlib.pyplot as plt

def ABC_dimer(L, a):
    B = {}
    A = {2*a, 2*a+1}
    C = {}
    for i in range (a+1, L-1):
        B[i] = [2*i, 2*i+1]

    for i in B:
        if i!=(a+1):
            B[i]+=B[i-1]
            B[i].sort()
    for i in B:
        C[i] = set(range(2*L))-set(B[i])-A
    return A, B, C

def ABC_single(L, a):
    B = {}
    A = {a}
    C = {}
    for i in range (2*a+1, L-1):
        B[i] = [2*i+1, 2*(i+1)]
    for i in B:
        if i!=2*a+1:
            B[i]+=B[i-1]
            B[i].sort()
    for i in B:
        C[i] = set(range(2*a, 2*L))-set(B[i])-A
    return A, B, C

def ABC_periodic(L):
    B = {}
    A = {}
    C = {}
    for i in range (int(L/2)):
        A[i] = [2*i, 2*i+1]
        C[i] = [2*i+L, 2*i+1+L]

    for i in A:
        if i!=0:
            A[i]+=A[i-1]
            C[i]+=C[i-1]
            A[i].sort()
            C[i].sort()
        B[i] = list(set(range(L))-set(A[i])-set(C[i]))
    return A, B, C

def cmi_(prob_exact,L, pattern):
    if pattern == "dimer":
        A,B,C = ABC_dimer(L, 0)
    elif pattern == "single":
        A,B,C = ABC_single(L, 0)
    cmi = []
    p_ab = prob_exact.sum(axis=tuple(set(range(2*L))-A))+1e-30
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
    return np.array(cmi)

def cmi_periodic (prob_exact, L):
    A,B,C = ABC_periodic(L)
    cmi = []
    for i in B:
        tmp = 0
        p_ab = prob_exact.sum(axis=tuple(C[i]))
        tmp += np.sum(p_ab*np.log(p_ab))+1e-30
        p_bc = prob_exact.sum(axis=tuple(A[i]))
        tmp += np.sum(p_bc*np.log(p_bc))+1e-30
        tmp -= np.sum(prob_exact*np.log(prob_exact+1e-30))
        if i != int(L/2)-1:
            p_b = prob_exact.sum(axis=tuple(list(set(C[i]).union(set(A[i])))))+1e-30
            tmp -= np.sum(p_b*np.log(p_b))
        cmi.append(-tmp)
    return cmi

def cmi_traceout_dimer(prob_exact, L):
    I = []
    cmi = [[] for i in range(L-5)]
    for i in range(L-5):
        if i!=0:
            I.append(2*(i-1))
            I.append(2*(i-1)+1)
        A, B, C = ABC_dimer(L, i)
        p_ab = prob_exact.sum(axis=tuple(set(range(2*L))-A))+1e-30
        p_bc = prob_exact.sum(axis=tuple(A.union(set(I))))+1e-30
        prob_exact_ = prob_exact.sum(axis=tuple(set(I)))+1e-30
        cmi[i].append(-np.sum(p_ab*np.log(p_ab))-np.sum(p_bc*np.log(p_bc))+np.sum(prob_exact_*np.log(prob_exact_+1e-30)))
        for j in B:
            tmp = 0
            p_ab = prob_exact.sum(axis=tuple(C[j].union(set(I))))
            tmp += np.sum(p_ab*np.log(p_ab))+1e-30
            p_bc = prob_exact.sum(axis=tuple(A.union(set(I))))
            tmp += np.sum(p_bc*np.log(p_bc))+1e-30
            tmp -= np.sum(prob_exact_*np.log(prob_exact_+1e-30))
            p_b = prob_exact.sum(axis=tuple(C[j].union(A).union(set(I))))+1e-30
            tmp -= np.sum(p_b*np.log(p_b))
            cmi[i].append(-tmp)
    return cmi

def correlation_all(prob_exact, L):
    corr = [[] for i in range(L)]
    mean_corr = [[] for i in range(L)]
    var_corr = [[] for i in range(L)]
    for i in range(L):
        for j in range(i+1, L):
            # the observable is sigma_z(2*i)*sigma_z(2*i+1)
            p_spin = prob_exact.sum(axis=tuple(set(range(2*L))-set([2*i, 2*i+1, 2*j, 2*j+1])))+1e-30
            p_spin0 = p_spin.sum(axis=(2, 3))
            p_spin1 = p_spin.sum(axis=(0, 1))
            cor = (p_spin[0, 0, 0, 0]-p_spin[0,0,0,1]-p_spin[0,0,1,0]-p_spin[0,1,0,0]-p_spin[1,0,0,0]+p_spin[0,0,1,1]+p_spin[0,1,0,1]+p_spin[0,1,1,0]+p_spin[1,0,0,1]+ p_spin[1,0,1,0]+p_spin[1,1,0,0]-p_spin[1,1,1,0]-p_spin[1,1,0,1]-p_spin[1,0,1,1]-p_spin[0,1,1,1]+p_spin[1,1,1,1])-(p_spin0[0, 0]-p_spin0[0, 1]-p_spin0[1, 0]+p_spin0[1, 1])*(p_spin1[0, 0]-p_spin1[0, 1]-p_spin1[1, 0]+p_spin1[1, 1])
            corr[np.abs(j-i)].append(cor)
        p_spin = prob_exact.sum(axis=tuple(set(range(2*L))-set([i,i+1])))+1e-30
        cor = (p_spin**2).sum()-(p_spin[0,0]-p_spin[0,1]-p_spin[1,0]+p_spin[1,1])**2
        corr[0].append(cor)
    for i in range(len(corr)):
        mean_corr[i] = np.array(corr[i]).mean()
        var_corr[i] = np.array(corr[i]).var()
    return np.array(mean_corr), np.array(var_corr)

def correlation_one(prob_exact, L):
    corr = [[] for i in range(L)]
    mean_corr = [[] for i in range(L)]
    var_corr = [[] for i in range(L)]
    for i in range(1, L):
        p_spin = prob_exact.sum(axis=tuple(set(range(2*L))-set([0, 1, 2*i, 2*i+1])))+1e-30
        p_spin0 = p_spin.sum(axis=(2, 3))
        p_spin1 = p_spin.sum(axis=(0, 1))
        cor = (p_spin[0, 0, 0, 0]-p_spin[0,0,0,1]-p_spin[0,0,1,0]-p_spin[0,1,0,0]-p_spin[1,0,0,0]+p_spin[0,0,1,1]+p_spin[0,1,0,1]+p_spin[0,1,1,0]+p_spin[1,0,0,1]+ p_spin[1,0,1,0]+p_spin[1,1,0,0]-p_spin[1,1,1,0]-p_spin[1,1,0,1]-p_spin[1,0,1,1]-p_spin[0,1,1,1]+p_spin[1,1,1,1])-(p_spin0[0, 0]-p_spin0[0, 1]-p_spin0[1, 0]+p_spin0[1, 1])*(p_spin1[0, 0]-p_spin1[0, 1]-p_spin1[1, 0]+p_spin1[1, 1])
        corr[np.abs(i)].append(cor)
    p_spin = prob_exact.sum(axis=tuple(set(range(2*L))-set([0, 1])))+1e-30
    cor = (p_spin**2).sum()-(p_spin[0,0]-p_spin[0,1]-p_spin[1,0]+p_spin[1,1])**2
    corr[0].append(cor)
    for i in range(len(corr)):
        mean_corr[i] = np.array(corr[i]).mean()
        var_corr[i] = np.array(corr[i]).var()
    return np.array(mean_corr), np.array(var_corr)

def correlation_periodic_dimer(prob_exact, L):
    corr = [[] for i in range(int(L/2))]
    mean_corr = [[] for i in range(int(L/2))]
    for i in range(1, int(L/2)):
        p_spin = prob_exact.sum(axis=tuple(set(range(2*L))-set([0, 1, 2*i, 2*i+1])))+1e-30
        p_spin0 = p_spin.sum(axis=(2, 3))
        p_spin1 = p_spin.sum(axis=(0, 1))
        cor = (p_spin[0, 0, 0, 0]-p_spin[0,0,0,1]-p_spin[0,0,1,0]-p_spin[0,1,0,0]-p_spin[1,0,0,0]+p_spin[0,0,1,1]+p_spin[0,1,0,1]+p_spin[0,1,1,0]+p_spin[1,0,0,1]+ p_spin[1,0,1,0]+p_spin[1,1,0,0]-p_spin[1,1,1,0]-p_spin[1,1,0,1]-p_spin[1,0,1,1]-p_spin[0,1,1,1]+p_spin[1,1,1,1])-(p_spin0[0, 0]-p_spin0[0, 1]-p_spin0[1, 0]+p_spin0[1, 1])*(p_spin1[0, 0]-p_spin1[0, 1]-p_spin1[1, 0]+p_spin1[1, 1])
        corr[np.abs(i)].append(cor)
    p_spin = prob_exact.sum(axis=tuple(set(range(2*L))-set([0, 1])))+1e-30
    cor = (p_spin**2).sum()-(p_spin[0,0]-p_spin[0,1]-p_spin[1,0]+p_spin[1,1])**2
    corr[0].append(cor)
    for i in range(len(corr)):
        mean_corr[i] = np.array(corr[i]).mean()
    return np.array(mean_corr)

def correlation_periodic_single(prob_exact, L):
    corr = [[] for i in range(int(L/2))]
    mean_corr = [[] for i in range(int(L/2))]
    for i in range(1, int(L/2)):
        p_spin = prob_exact.sum(axis=tuple(set(range(2*L))-set([0, 2*i])))+1e-30
        p_spin0 = p_spin.sum(axis=1)
        p_spin1 = p_spin.sum(axis=0)
        cor = (p_spin[0, 0]- p_spin[0,1]- p_spin[1,0]+ p_spin[1,1])-(p_spin0[0]-p_spin0[1])*(p_spin1[0]-p_spin1[1])
        corr[np.abs(i)].append(cor)

    p_spin = prob_exact.sum(axis=tuple(set(range(2*L))-set([0])))+1e-30
    cor = (p_spin**2).sum()-(p_spin[0]-p_spin[1])**2
    corr[0].append(cor)
    for i in range(len(corr)):
        mean_corr[i] = np.array(corr[i]).mean()
    return np.array(mean_corr)