
import netket as nk
import jax
from netket.operator.spin import sigmax,sigmaz, sigmap, sigmam
from scipy.sparse.linalg import eigsh
import numpy as np
import matplotlib.pyplot as plt
import time
def ABC(L, a):
    B = {}
    A = {a}
    C = {}
    for i in range (a+1, L-1):
        B[i] = [i]
    for i in B:
        if i!=a+1:
            B[i]+=B[i-1]
            B[i].sort()
    for i in B:
        C[i] = set(range(L))-set(B[i])-A
    return A, B, C

def ABC_periodic(L):
    B = {}
    A = {}
    C = {}
    for i in range (int(L/2)):
        A[i] = [i]
        C[i] = [i+int(L/2)]

    for i in A:
        if i!=0:
            A[i]+=A[i-1]
            C[i]+=C[i-1]
            A[i].sort()
            C[i].sort()
        B[i] = list(set(range(L))-set(A[i])-set(C[i]))
    return A, B, C

def cmi_(prob_exact, L):
    A,B,C = ABC(L, 0)
    cmi = []
    p_ab = prob_exact.sum(axis=tuple(set(range(L))-A))+1e-30
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

def cmi_periodic(prob_exact, L):
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

def cmi_traceout(prob_exact, L):
    I = []
    cmi = [[] for i in range(L-5)]
    for i in range(L-5):
        if i!=0:
            I.append(i-1)
        A,B,C = ABC(L, i)
        p_ab = prob_exact.sum(axis=tuple(set(range(L))-A))+1e-30
        p_bc = prob_exact.sum(axis=tuple(A.union(set(I))))+1e-30
        prob_exact_ = prob_exact.sum(axis=tuple(set(I)))
        cmi[i].append(-np.sum(p_ab*np.log(p_ab))-np.sum(p_bc*np.log(p_bc))+np.sum(prob_exact_*np.log(prob_exact_+1e-30)))

        for j in B:
            tmp = 0
            p_ab = prob_exact.sum(axis=tuple(C[j].union(set(I))))
            tmp += np.sum(p_ab*np.log(p_ab))
            p_bc = prob_exact.sum(axis=tuple(A.union(set(I))))
            tmp += np.sum(p_bc*np.log(p_bc))
            tmp -= np.sum(prob_exact_*np.log(prob_exact_+1e-30))
            p_b = prob_exact.sum(axis=tuple(C[j].union(A).union(set(I))))
            tmp -= np.sum(p_b*np.log(p_b))
            print(tmp)
            cmi[i].append(-tmp)
    return np.array(cmi, dtype=object)

def cmi_traceout_periodic(prob_exact, L):
    I = []
    cmi = [[] for i in range(L-5)]
    for i in range(L-5):
        if i!=0:
            I.append(i-1)
        A,B,C = ABC(L, i)
        p_ab = prob_exact.sum(axis=tuple(set(range(L))-A))+1e-30
        p_bc = prob_exact.sum(axis=tuple(A.union(set(I))))+1e-30
        prob_exact_ = prob_exact.sum(axis=tuple(set(I)))+1e-30
        cmi[i].append(-np.sum(p_ab*np.log(p_ab))-np.sum(p_bc*np.log(p_bc))+np.sum(prob_exact_*np.log(prob_exact_+1e-30)))
        for j in B:
            tmp = 0
            p_ab = prob_exact.sum(axis=tuple(C[j].union(set(I))))
            tmp += np.sum(p_ab*np.log(p_ab))
            p_bc = prob_exact.sum(axis=tuple(A.union(set(I))))
            tmp += np.sum(p_bc*np.log(p_bc))
            tmp -= np.sum(prob_exact_*np.log(prob_exact_+1e-30))
            p_b = prob_exact.sum(axis=tuple(C[j].union(A).union(set(I))))
            tmp -= np.sum(p_b*np.log(p_b))
            cmi[i].append(-tmp)
    return np.array(cmi, dtype = object)

def spin_correlation_all(prob_exact, L):
    corr = [[] for i in range(L)]
    mean_corr = [[] for i in range(L)]
    var_corr = [[] for i in range(L)]
    for i in range(L):
        for j in range(i+1, L):
            p_spin = prob_exact.sum(axis=tuple(set(range(L))-set([i, j])))+1e-30
            p_spin0 = p_spin.sum(axis=0)
            p_spin1 = p_spin.sum(axis=1)
            cor = (p_spin[0, 0]+p_spin[1, 1]-p_spin[0, 1]-p_spin[1, 0])-(p_spin0[0]-p_spin0[1])*(p_spin1[0]-p_spin1[1])
            corr[np.abs(j-i)].append(cor)
        p_spin = prob_exact.sum(axis=tuple(set(range(L))-set([i])))+1e-30
        cor = (p_spin**2).sum()-(p_spin[0]-p_spin[1])**2
        corr[0].append(cor)
    for i in range(len(corr)):
        mean_corr[i] = np.abs(corr[i]).mean()
        var_corr[i] = np.abs(corr[i]).var()
    return np.abs(mean_corr), np.array(var_corr)

def spin_correlation_periodic(prob_exact, L):
    corr = [[] for i in range(int(L/2)+1)]
    mean_corr = [[] for i in range(int(L/2)+1)]

    for i in range(1, int(L/2)+1):
        p_spin = prob_exact.sum(axis=tuple(set(range(L))-set([0, i])))+1e-30
        p_spin0 = p_spin.sum(axis=0)
        p_spin1 = p_spin.sum(axis=1)
        cor = (p_spin[0, 0]+p_spin[1, 1]-p_spin[0, 1]-p_spin[1, 0])-(p_spin0[0]-p_spin0[1])*(p_spin1[0]-p_spin1[1])
        corr[np.abs(i)].append(cor)
        p_spin = prob_exact.sum(axis=tuple(set(range(L))-set([0])))+1e-30
        cor = (p_spin**2).sum()-(p_spin[0]-p_spin[1])**2
        corr[0].append(cor)
    for i in range(len(corr)):
        mean_corr[i] = np.array(corr[i]).mean()
    return np.abs(mean_corr)
def count_diff_ones_zeros(n):
    # Generating all combinations from 0^n to 1^n
    combinations = np.array([np.binary_repr(i, width=n) for i in range(2**n)])

    # Counting the difference between the number of 1s and 0s in each combination
    diff_counts = np.array([combination.count('1') - combination.count('0') for combination in combinations])

    return diff_counts