import jax
import netket as nk
import numpy as np
from netket.operator.spin import sigmax, sigmaz, sigmap, sigmam, identity
import time
from scipy.sparse.linalg import eigsh
import jax.numpy as jnp
import matplotlib.pyplot as plt
from twoD_tool import *

Lx = 4
Ly = 4
N = Lx * Ly
params = [0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 4.4]
Omega = 1.0
Rb = 1.2 ** 6
hi = nk.hilbert.Spin(s=1 / 2, N=N)
for param in params:
    ni = []
    H = Omega / 2 * sum([sigmax(hi, y * Lx + x) for y in range(Ly) for x in range(Lx)])  # X
    H -= param / 2 * sum([(identity(hi) - sigmaz(hi, y * Lx + x)) for y in range(Ly) for x in range(Lx)])
    H += Omega * Rb / 4 * sum(
        [((identity(hi) - sigmaz(hi, y1 * Lx + x1)) * (identity(hi) - sigmaz(hi, y1 * Lx + x2))) / ((x1 - x2) ** 2) ** 3 \
         for y1 in range(Ly) for x1 in range(Lx) for x2 in range(x1 + 1, Lx)])
    H += Omega * Rb / 4 * sum([((identity(hi) - sigmaz(hi, y1 * Lx + x1)) * (
                identity(hi) - sigmaz(hi, y2 * Lx + x2))) / (((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 3) \
                               for y1 in range(Ly) for x1 in range(Lx) for y2 in range(y1 + 1, Ly) for x2 in range(Lx)])
    sp_h = H.to_sparse()
    eig_vals, eig_vecs = eigsh(sp_h, k=2, which="SA")

    stagger_H = sum([sigmaz(hi, y * Lx + x) * (-1) ** (y * Lx + x) for y in range(Ly) for x in range(Lx)])
    for y in range(Ly):
        for x in range(Lx):
            h_ni = (identity(hi) - sigmaz(hi, y * Lx + x)) / 2
            ni.append(eig_vecs[:, 0].conj() @ h_ni.to_sparse() @ eig_vecs[:, 0])
    stagger_H_sp = stagger_H.to_sparse()
    stagger_mag0 = np.abs(eig_vecs[:, 0].conj() @ stagger_H_sp @ eig_vecs[:, 0])
    stagger_mag1 = np.abs(eig_vecs[:, 0].conj() @ stagger_H_sp @ eig_vecs[:, 1])
    stagger_mag2 = np.abs(eig_vecs[:, 1].conj() @ stagger_H_sp @ eig_vecs[:, 1])
    print(stagger_mag0)
    print(stagger_mag1)
    print(stagger_mag2)
    print(eig_vals[1] - eig_vals[0])
    print(ni)