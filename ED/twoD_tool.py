
import numpy as np
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

def ABC_complete(L, pos_A):
    B = {}
    A = {pos_A}
    C = {}
    y0 = tuple(A)[0]//L
    x0 = tuple(A)[0]%L

    for y in range (L):
        for x in range (L):
            if y*L+x> tuple(A)[0]:
                d = np.abs(y-y0)+np.abs(x-x0)
                if d not in B:
                    B[d] = [y*L+x]
                else:
                    B[d].append(y*L+x)
    B = dict(sorted(B.items()))
    C = dict(sorted(C.items()))
    for i in B:
        if i!=1:
            B[i] += B[i-1]
        B[i].sort()
    B.popitem()     #remove the last element
    for i in B:
        C[i] = set(range(tuple(A)[0]+1, L**2))-set(B[i])
    B = dict(sorted(B.items()))
    C = dict(sorted(C.items()))
    return A,B,C

def cmi_traceout(prob_exact, L):
    cmi = [[] for i in range(L*(L-2))]
    for i in range (L*(L-2)):
        A,B,C = ABC_complete(L, i)
        p_ab = prob_exact.sum(axis=tuple(set(range(L**2))-A))+1e-30
        p_bc = prob_exact.sum(axis=tuple(set(range(i)).union(A)))+1e-30
        p_abc = prob_exact.sum(axis=tuple(set(range(i))))+1e-30
        cmi[i].append(-np.sum(p_ab*np.log(p_ab))-np.sum(p_bc*np.log(p_bc))+np.sum(p_abc*np.log(p_abc)))
        for j in B:
            tmp = 0
            p_ab = prob_exact.sum(axis=tuple(set(range(i)).union(C[j])))
            tmp += np.sum(p_ab*np.log(p_ab))+1e-30
            p_bc = prob_exact.sum(axis=tuple(set(range(i)).union(A)))
            tmp += np.sum(p_bc*np.log(p_bc))+1e-30
            tmp -= np.sum(p_abc*np.log(p_abc))
            p_b = prob_exact.sum(axis=tuple(set(range(i)).union(C[j].union(A))))+1e-30
            tmp -= np.sum(p_b*np.log(p_b))
            cmi[i].append(-tmp)

    return cmi

def count_diff_ones_zeros(n):
    # Generating all combinations from 0^n to 1^n
    combinations = np.array([np.binary_repr(i, width=n) for i in range(2**n)])

    # Counting the difference between the number of 1s and 0s in each combination
    diff_counts = np.array([combination.count('1') - combination.count('0') for combination in combinations])

    return diff_counts

def cmi_(prob_exact, L):  #cmi for the first spin as |A|
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
    return np.array(cmi)
def correlation_all(prob_exact, L):   #correlation function respect to all spins and then taking average
    corr = [[] for i in range(2*(L-1)+1)]
    mean_corr = [[] for i in range(2*(L-1)+1)]
    var_corr = [[] for i in range(2*(L-1)+1)]
    for i in range(L**2):
        for j in range(i+1, L**2):
            y0 = i//L
            x0 = i%L
            y1 = j//L
            x1 = j%L
            p_spin = prob_exact.sum(axis=tuple(set(range(L**2))-set([i, j])))+1e-30
            p_spin0 = p_spin.sum(axis=0)
            p_spin1 = p_spin.sum(axis=1)
            cor = (p_spin[0, 0]+p_spin[1, 1]-p_spin[0, 1]-p_spin[1, 0])-(p_spin0[0]-p_spin0[1])*(p_spin1[0]-p_spin1[1])
            corr[np.abs(y1-y0)+np.abs(x1-x0)].append(cor)
        p_spin = prob_exact.sum(axis=tuple(set(range(L**2))-set([i])))+1e-30
        cor = (p_spin**2).sum()-(p_spin[0]-p_spin[1])**2
        corr[0].append(cor)

    for i in range(len(corr)):
        mean_corr[i] = np.abs(corr[i]).mean()
        var_corr[i] = np.abs(corr[i]).var()
    return np.array(mean_corr), np.array(var_corr)

def correlation_one(prob_exact, L):    # correlation function respect to [0, 0]
    corr = [[] for i in range(2*(L-1)+1)]
    mean_corr = [[] for i in range(2*(L-1)+1)]
    var_corr = [[] for i in range(2*(L-1)+1)]
    for i in range(1, L**2):
        y0 = 0
        x0 = 0
        y1 = i//L
        x1 = i%L
        p_spin = prob_exact.sum(axis=tuple(set(range(L**2))-set([0, i])))+1e-30
        p_spin0 = p_spin.sum(axis=0)
        p_spin1 = p_spin.sum(axis=1)
        cor = (p_spin[0, 0]+p_spin[1, 1]-p_spin[0, 1]-p_spin[1, 0])-(p_spin0[0]-p_spin0[1])*(p_spin1[0]-p_spin1[1])
        corr[np.abs(y1-y0)+np.abs(x1-x0)].append(cor)
    p_spin = prob_exact.sum(axis=tuple(set(range(L**2))-set([0])))+1e-30
    cor = (p_spin**2).sum()-(p_spin[0]-p_spin[1])**2
    corr[0].append(cor)

    for i in range(len(corr)):
        mean_corr[i] = np.abs(corr[i]).mean()
        var_corr[i] = np.abs(corr[i]).var()
    return np.array(mean_corr), np.array(var_corr)


def create_alternating_matrix(n):
    # Create an n*n matrix where each element is 1
    matrix = np.ones((n, n), dtype=int)

    # Multiply by -1 at every other index
    matrix[1::2, ::2] = -1  # Change every other row starting from the second row
    matrix[::2, 1::2] = -1  # Change every other column starting from the second column

    return matrix