import netket as nk
import jax
from netket.operator.spin import sigmax, sigmaz, sigmap, sigmam
from scipy.sparse.linalg import eigsh
import numpy as np
import matplotlib.pyplot as plt
import time
from oneD_tool import *

L = 16
hi = nk.hilbert.Spin(s=1 / 2, N=L)
periodic = True
delta_ = [1.2, 1.05, 1.01, 1., 0.99, 0.8]

h = 0.0

for delta in delta_:
    H = sum([2*(sigmap(hi,i)*sigmam(hi,i+1)+sigmam(hi,i)*sigmap(hi,i+1))+delta*sigmaz(hi,i)*sigmaz(hi,i+1) for i in range(L-1)])
    if periodic:
        H += 2*(sigmap(hi,0)*sigmam(hi,1)+sigmam(hi,0)*sigmap(hi,1))+delta*sigmaz(hi,0)*sigmaz(hi,L-1)

    sp_h = H.to_sparse()
    eig_vals, eig_vecs = eigsh(sp_h, k=2, which="SA")

    print("eigenvalues with scipy sparse:", eig_vals)
    print("spectral gap:", eig_vals[1]-eig_vals[0])

    probs_exact = np.abs(eig_vecs[:, 0]) ** 2
    mag = np.sum(probs_exact * np.array(count_diff_ones_zeros(L)))
    shape = (2,) * (L)
    probs_exact = probs_exact.reshape(*shape)

    if (periodic == False):
        cmi = cmi_(probs_exact, L)
        mean_corr, var_corr = spin_correlation_all(probs_exact, L)
        cmi_all = cmi_traceout(probs_exact, L)

    else:
        cmi = cmi_periodic(probs_exact, L)
        mean_corr = spin_correlation_periodic(probs_exact, L)
        cmi_all = cmi_traceout_periodic(probs_exact, L)


    np.save("result/1DXXZ/cmi_1DXXZ_L" + str(L) + "_delta_" + str(delta) + "periodic_" + str(periodic) + ".npy", cmi)
    np.save("result/1DXXZ/mean_corr_1DXXZ_L" + str(L) + "_delta_" + str(delta) + "periodic_" + str(periodic) + ".npy",
            mean_corr)
    if periodic == False:
        np.save(
            "result/1DXXZ/var_corr_1DXXZ_L" + str(L) + "_delta_" + str(delta) + "periodic_" + str(periodic) + ".npy",
            var_corr)
    np.save(
        "result/1DXXZ/cmi_traceout_1DXXZ_L" + str(L) + "_delta_" + str(delta) + "periodic_" + str(periodic) + ".npy",
        cmi_all)
    np.save("result/1DXXZ/mag_1DXXZ_L" + str(L) + "_delta_" + str(delta) + "periodic_" + str(periodic) + ".npy", mag)
    np.save("result/1DXXZ/gap_1DXXZ_L" + str(L) + "_delta_" + str(delta) + "periodic_" + str(periodic) + ".npy",
            np.array(eig_vals[1] - eig_vals[0]))

cmi = {}
mean_corr = {}
var_corr = {}
gap = {}
cmi_traceout = {}
mag = {}
# Looping through the values and loading each file

for b in delta_:
    filename = f"result/1DXXZ/cmi_1DXXZ_L{L}_delta_{b}periodic_{periodic}.npy"
    cmi[b] = np.load(filename)
    if periodic == True:
        cmi[b] = np.flip(cmi[b])
    filename = f"result/1DXXZ/mean_corr_1DXXZ_L{L}_delta_{b}periodic_{periodic}.npy"
    mean_corr[b] = np.load(filename)
    if periodic == False:
        filename = f"result/1DXXZ/var_corr_1DXXZ_L{L}_delta_{b}periodic_{periodic}.npy"
        var_corr[b] = np.load(filename)
    filename = f"result/1DXXZ/gap_1DXXZ_L{L}_delta_{b}periodic_{periodic}.npy"
    gap[b] = np.load(filename)
    filename = f"result/1DXXZ/cmi_traceout_1DXXZ_L{L}_delta_{b}periodic_{periodic}.npy"
    cmi_traceout[b] = np.load(filename, allow_pickle=True)
    filename = f"result/1DXXZ/mag_1DXXZ_L{L}_delta_{b}periodic_{periodic}.npy"
    mag[b] = np.load(filename)

cmi_length = []
err_cmi_length = []
corre_length = []
err_corre_length = []
cmi_length_traceout = [[] for i in range(len(delta_))]
err_cmi_length_traceout = [[] for i in range(len(delta_))]
for i in delta_:
    cmi[i] = np.abs(cmi[i])
    for j in range(len(cmi_traceout[i])):
        cmi_traceout[i][j] = np.abs(np.array(cmi_traceout[i][j]))
for i in delta_:
    df = len(cmi[i]) - 2
    (cmi_len, b1), residuals, _, _, _ = np.polyfit(np.arange(len(cmi[i])), -np.log(cmi[i]), 1, full=True)
    mean_x = np.mean(np.arange(len(cmi[i])))
    err = np.sqrt(residuals / df) / np.sqrt(np.sum((np.arange(len(cmi[i])) - mean_x) ** 2))
    cmi_length.append(cmi_len)
    err_cmi_length.append(err)

    (corr_len, b2), residuals, _, _, _ = np.polyfit(np.arange(len(mean_corr[i])), -np.log(mean_corr[i]), 1,
                                                    full=True)
    mean_x = np.mean(np.arange(len(mean_corr[i])))
    err = np.sqrt(residuals / df) / np.sqrt(np.sum((np.arange(len(mean_corr[i])) - mean_x) ** 2))
    corre_length.append(corr_len)
    err_corre_length.append(err)

    for j in range(len(cmi_traceout[i])):
        df = len(cmi_traceout[i][j]) - 2
        (cmi_len, b1), residuals, _, _, _ = np.polyfit(np.arange(len(cmi_traceout[i][j])),
                                                       -np.log(cmi_traceout[i][j] + 1e-32), 1, full=True)
        mean_x = np.mean(np.arange(len(cmi_traceout[i][j])))
        err = np.sqrt(residuals / df) / np.sqrt(np.sum((np.arange(len(cmi_traceout[i][j])) - mean_x) ** 2))
        cmi_length_traceout[delta_.index(i)].append(cmi_len)
        err_cmi_length_traceout[delta_.index(i)].append(err)

for i in delta_:
    plt.plot(np.log(cmi[i]), label="delta=" + str(i))
plt.xlabel("distance")
plt.ylabel("log_CMI")
plt.legend()
plt.savefig("figure/1DXXZ/cmi_1DXXZ_L" + str(L) + "_delta_" + str(delta) + "periodic_" + str(periodic) + ".png")
plt.show()
plt.clf()

for i in delta_:
    plt.plot(np.arange(len(mean_corr[i]) - 1) + 1, np.log(mean_corr[i][1:]), label="delta=" + str(i))
    if periodic == False:
        plt.fill_between(np.arange(len(mean_corr[i]) - 1) + 1,
                         np.log(mean_corr[i][1:]) - 1 / mean_corr[i][1:] * np.sqrt(var_corr[i][1:]),
                         np.log(mean_corr[i][1:]) + 1 / mean_corr[i][1:] * np.sqrt(var_corr[i][1:]), alpha=0.2)
plt.xlabel("distance")
plt.ylabel("log_correlation")
plt.legend()
plt.savefig(
    "figure/1DXXZ/correlation_1DXXZ_L" + str(L) + "_delta_" + str(delta) + "periodic_" + str(periodic) + ".png")
plt.show()
plt.clf()

B_ = [-delta_[i] for i in range(len(delta_))]
plt.scatter(B_, gap.values(), s=10)
plt.xlabel("delta")
plt.ylabel("gap")
plt.legend()
plt.savefig("figure/1DXXZ/gap_1DXXZ_L" + str(L) + "_delta_" + str(delta) + "periodic_" + str(periodic) + ".png")
plt.show()
plt.clf()

for i in delta_:
    plt.plot(np.log(mean_corr[i][1:]), np.log(cmi[i]), label="delta=" + str(i))
    if periodic == False:
        plt.fill_betweenx(np.log(cmi[i]),
                          np.log(mean_corr[i][1:]) - 1 / mean_corr[i][1:] * np.sqrt(var_corr[i][1:]),
                          np.log(mean_corr[i][1:]) + 1 / mean_corr[i][1:] * np.sqrt(var_corr[i][1:]), alpha=0.2)
plt.xlabel("log_correlation")
plt.ylabel("log_CMI")
plt.legend()
plt.savefig(
    "figure/1DXXZ/correlation_cmi_1DXXZ_L" + str(L) + "_delta_" + str(delta) + "periodic_" + str(periodic) + ".png")
plt.show()
plt.clf()

gap_list = []
for i in delta_:
    gap_list.append(gap[i])

plt.scatter(gap_list, cmi_length, label="cmi", s=10)
plt.errorbar(gap_list, cmi_length, yerr=err_cmi_length[:][0], fmt='o', ecolor='g', capthick=1, capsize=10)
plt.xlabel("gap")
plt.ylabel("cmi_length")
plt.legend()
plt.savefig(
    "figure/1DXXZ/cmi_length_gap_1DXXZ_L" + str(L) + "_delta_" + str(delta) + "periodic_" + str(periodic) + ".png")
plt.show()
plt.clf()

plt.scatter(gap_list, corre_length, label="correlation", s=10)
plt.errorbar(gap_list, corre_length, yerr=err_corre_length[:][0], fmt='o', ecolor='g', capthick=1, capsize=10)
plt.xlabel("gap")
plt.ylabel("correlation_length")
plt.legend()
plt.savefig("figure/1DXXZ/correlation_length_gap_1DXXZ_L" + str(L) + "_delta_" + str(delta) + "periodic_" + str(
    periodic) + ".png")
plt.show()
plt.clf()

for i in range(len(delta_)):
    plt.scatter(np.arange(len(cmi_length_traceout[i])), cmi_length_traceout[i], label="delta = " + str(delta_[i]))
    plt.errorbar(np.arange(len(cmi_length_traceout[i])), cmi_length_traceout[i],
                 yerr=np.array(err_cmi_length_traceout[i]).ravel(), fmt='o', ecolor='g', capthick=1, capsize=10)
    plt.xlabel("trace_size")
    plt.ylabel("cmi_length")
    plt.legend()
    plt.savefig(
        "figure/1DXXZ/traceout/cmi_length_traceout_1DXXZ_L" + str(L) + "_delta_" + str(delta_[i]) + "periodic_" + str(
            periodic) + ".png")
    plt.show()
    plt.clf()

