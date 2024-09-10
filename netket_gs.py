import jax.random
import matplotlib.pyplot as plt
import jax.numpy as jnp
import netket as nk
from scipy.sparse.linalg import eigsh
import numpy as np
from netket.operator.spin import sigmax,sigmaz
import itertools
import optax
import time


# Parameters
ang_ = [0.0, 0.157, 0.314, 0.471, 0.628, 0.785, 0.942, 1.1, 1.257, 1.414, 1.571]
L = 8
N = L**2
numsamples = 4096
hi = nk.hilbert.Spin(s=1 / 2, N=N)
g = nk.graph.Hypercube(length=N, n_dim=1, pbc=False)
a = 0
for angle in jnp.linspace(0.0, jnp.pi/2, 11):
    # First set of terms
    h = - np.cos(angle) ** 3 * sigmaz(hi, 0) @  sigmax(hi, 1) * sigmax(hi, L)
    h +=  np.cos(angle) ** 2 * np.sin(angle) *  sigmax(hi, 0) @  sigmax(hi, 1) * sigmax(hi, L)
    h -=  np.cos(angle) ** 2 * np.sin(angle) *  sigmaz(hi, 0) @  sigmaz(hi, 1) * sigmax(hi, L)
    h -=  np.cos(angle) ** 2 * np.sin(angle) *  sigmaz(hi, 0) @  sigmax(hi, 1) * sigmaz(hi, L)
    h +=  np.cos(angle) * np.sin(angle) ** 2 *  sigmax(hi, 0) @  sigmaz(hi, 1) * sigmax(hi, L)
    h +=  np.cos(angle) * np.sin(angle) ** 2 *  sigmax(hi, 0) @  sigmax(hi, 1) * sigmaz(hi, L)
    h -=  np.cos(angle) * np.sin(angle) ** 2 *  sigmaz(hi, 0) @  sigmaz(hi, 1) * sigmaz(hi, L)
    h +=  np.sin(angle) ** 3 * sigmax(hi, 0) @  sigmaz(hi, 1) * sigmaz(hi, L)

    h -= np.cos(angle) ** 3 * sigmaz(hi, L-1) @ sigmax(hi, L-2) @ sigmax(hi, 2*L-1)
    h += np.cos(angle) ** 2 * np.sin(angle) * sigmax(hi, L-1) @ sigmax(hi, L-2) @ sigmax(hi, 2*L-1)
    h -= np.cos(angle) ** 2 * np.sin(angle) * sigmaz(hi, L-1) @ sigmaz(hi, L-2) @ sigmax(hi, 2*L-1)
    h -= np.cos(angle) ** 2 * np.sin(angle) * sigmaz(hi, L-1) @ sigmax(hi, L-2) @ sigmaz(hi, 2*L-1)
    h += np.cos(angle) * np.sin(angle) ** 2 * sigmax(hi, L-1) @ sigmaz(hi, L-2) @ sigmax(hi, 2*L-1)
    h += np.cos(angle) * np.sin(angle) ** 2 * sigmax(hi, L-1) @ sigmax(hi, L-2) @ sigmaz(hi, 2*L-1)
    h -= np.cos(angle) * np.sin(angle) ** 2 * sigmaz(hi, L-1) @ sigmaz(hi, L-2) @ sigmaz(hi, 2*L-1)
    h += np.sin(angle) ** 3 * sigmax(hi, L-1) @ sigmaz(hi, L-2) @ sigmaz(hi, 2*L-1)

    h -= np.cos(angle) ** 3 * sigmaz(hi, L*(L-1)) @ sigmax(hi, L*(L-2)) @ sigmax(hi, L*(L-1)+1)
    h += np.cos(angle) ** 2 * np.sin(angle) * sigmax(hi, L*(L-1)) @ sigmax(hi, L*(L-2)) @ sigmax(hi, L*(L-1)+1)
    h -= np.cos(angle) ** 2 * np.sin(angle) * sigmaz(hi, L*(L-1)) @ sigmaz(hi, L*(L-2)) @ sigmax(hi, L*(L-1)+1)
    h -= np.cos(angle) ** 2 * np.sin(angle) * sigmaz(hi, L*(L-1)) @ sigmax(hi, L*(L-2)) @ sigmaz(hi, L*(L-1)+1)
    h += np.cos(angle) * np.sin(angle) ** 2 * sigmax(hi, L*(L-1)) @ sigmaz(hi, L*(L-2)) @ sigmax(hi, L*(L-1)+1)
    h += np.cos(angle) * np.sin(angle) ** 2 * sigmax(hi, L*(L-1)) @ sigmax(hi, L*(L-2)) @ sigmaz(hi, L*(L-1)+1)
    h -= np.cos(angle) * np.sin(angle) ** 2 * sigmaz(hi, L*(L-1)) @ sigmaz(hi, L*(L-2)) @ sigmaz(hi, L*(L-1)+1)
    h += np.sin(angle) ** 3 * sigmax(hi, L*(L-1)) @ sigmaz(hi, L*(L-2)) @ sigmaz(hi, L*(L-1)+1)


    h -= np.cos(angle) ** 3 * sigmaz(hi, L*L-1) @ sigmax(hi, L*L-2) @ sigmax(hi, L*(L-1)-1)
    h += np.cos(angle) ** 2 * np.sin(angle) * sigmax(hi, L*L-1) @ sigmax(hi, L*L-2) @ sigmax(hi, L*(L-1)-1)
    h -= np.cos(angle) ** 2 * np.sin(angle) * sigmaz(hi, L*L-1) @ sigmaz(hi, L*L-2) @ sigmax(hi, L*(L-1)-1)
    h -= np.cos(angle) ** 2 * np.sin(angle) * sigmaz(hi, L*L-1) @ sigmax(hi, L*L-2) @ sigmaz(hi, L*(L-1)-1)
    h += np.cos(angle) * np.sin(angle) ** 2 * sigmax(hi, L*L-1) @ sigmaz(hi, L*L-2) @ sigmax(hi, L*(L-1)-1)
    h += np.cos(angle) * np.sin(angle) ** 2 * sigmax(hi, L*L-1) @ sigmax(hi, L*L-2) @ sigmaz(hi, L*(L-1)-1)
    h -= np.cos(angle) * np.sin(angle) ** 2 * sigmaz(hi, L*L-1) @ sigmaz(hi, L*L-2) @ sigmaz(hi, L*(L-1)-1)
    h += np.sin(angle) ** 3 * sigmax(hi, L*L-1) @ sigmaz(hi, L*L-2) @ sigmaz(hi, L*(L-1)-1)


    for i in range(1, L-1):

        h -= np.cos(angle) ** 4 * sigmaz(hi, i) @ sigmax(hi, i-1) @ sigmax(hi, i+1) @ sigmax(hi, i+L)
        h += np.cos(angle) ** 3 * np.sin(angle) * sigmax(hi, i) @ sigmax(hi, i-1) @ sigmax(hi, i+1) @ sigmax(hi, i+L)
        h -= np.cos(angle) ** 3 * np.sin(angle) * sigmaz(hi, i) @ sigmaz(hi, i-1) @ sigmax(hi, i+1) @ sigmax(hi, i+L)
        h -= np.cos(angle) ** 3 * np.sin(angle) * sigmaz(hi, i) @ sigmax(hi, i-1) @ sigmaz(hi, i+1) @ sigmax(hi, i+L)
        h -= np.cos(angle) ** 3 * np.sin(angle) * sigmaz(hi, i) @ sigmax(hi, i-1) @ sigmax(hi, i+1) @ sigmaz(hi, i+L)
        h += np.cos(angle) ** 2 * np.sin(angle) ** 2 * sigmax(hi, i) @ sigmaz(hi, i-1) @ sigmax(hi, i+1) @ sigmax(hi, i+L)
        h += np.cos(angle) ** 2 * np.sin(angle) ** 2 * sigmax(hi, i) @ sigmax(hi, i-1) @ sigmaz(hi, i+1) @ sigmax(hi, i+L)
        h += np.cos(angle) ** 2 * np.sin(angle) ** 2 * sigmax(hi, i) @ sigmax(hi, i-1) @ sigmax(hi, i+1) @ sigmaz(hi, i+L)
        h -= np.cos(angle) ** 2 * np.sin(angle) ** 2 * sigmaz(hi, i) @ sigmaz(hi, i-1) @ sigmaz(hi, i+1) @ sigmax(hi, i+L)
        h -= np.cos(angle) ** 2 * np.sin(angle) ** 2 * sigmaz(hi, i) @ sigmaz(hi, i-1) @ sigmax(hi, i+1) @ sigmaz(hi, i+L)
        h -= np.cos(angle) ** 2 * np.sin(angle) ** 2 * sigmaz(hi, i) @ sigmax(hi, i-1) @ sigmaz(hi, i+1) @ sigmaz(hi, i+L)
        h += np.cos(angle) * np.sin(angle) ** 3 * sigmax(hi, i) @ sigmax(hi, i-1) @ sigmaz(hi, i+1) @ sigmaz(hi, i+L)
        h += np.cos(angle) * np.sin(angle) ** 3 * sigmax(hi, i) @ sigmaz(hi, i-1) @ sigmax(hi, i+1) @ sigmaz(hi, i+L)
        h += np.cos(angle) * np.sin(angle) ** 3 * sigmax(hi, i) @ sigmaz(hi, i-1) @ sigmaz(hi, i+1) @ sigmax(hi, i+L)
        h -= np.cos(angle) * np.sin(angle) ** 3 * sigmaz(hi, i) @ sigmaz(hi, i-1) @ sigmaz(hi, i+1) @ sigmaz(hi, i+L)
        h += np.sin(angle) ** 4 * sigmax(hi, i) @ sigmaz(hi, i-1) @ sigmaz(hi, i+1) @ sigmaz(hi, i+L)

        j = i*L
        h -= np.cos(angle) ** 4 * sigmaz(hi, j) @ sigmax(hi, j-L) @ sigmax(hi, j+1) @ sigmax(hi, j+L)
        h += np.cos(angle) ** 3 * np.sin(angle) * sigmax(hi, j) @ sigmax(hi, j-L) @ sigmax(hi, j+1) @ sigmax(hi, j+L)
        h -= np.cos(angle) ** 3 * np.sin(angle) * sigmaz(hi, j) @ sigmaz(hi, j-L) @ sigmax(hi, j+1) @ sigmax(hi, j+L)
        h -= np.cos(angle) ** 3 * np.sin(angle) * sigmaz(hi, j) @ sigmax(hi, j-L) @ sigmaz(hi, j+1) @ sigmax(hi, j+L)
        h -= np.cos(angle) ** 3 * np.sin(angle) * sigmaz(hi, j) @ sigmax(hi, j-L) @ sigmax(hi, j+1) @ sigmaz(hi, j+L)
        h += np.cos(angle) ** 2 * np.sin(angle) ** 2 * sigmax(hi, j) @ sigmaz(hi, j-L) @ sigmax(hi, j+1) @ sigmax(hi, j+L)
        h += np.cos(angle) ** 2 * np.sin(angle) ** 2 * sigmax(hi, j) @ sigmax(hi, j-L) @ sigmaz(hi, j+1) @ sigmax(hi, j+L)
        h += np.cos(angle) ** 2 * np.sin(angle) ** 2 * sigmax(hi, j) @ sigmax(hi, j-L) @ sigmax(hi, j+1) @ sigmaz(hi, j+L)
        h -= np.cos(angle) ** 2 * np.sin(angle) ** 2 * sigmaz(hi, j) @ sigmaz(hi, j-L) @ sigmaz(hi, j+1) @ sigmax(hi, j+L)
        h -= np.cos(angle) ** 2 * np.sin(angle) ** 2 * sigmaz(hi, j) @ sigmaz(hi, j-L) @ sigmax(hi, j+1) @ sigmaz(hi, j+L)
        h -= np.cos(angle) ** 2 * np.sin(angle) ** 2 * sigmaz(hi, j) @ sigmax(hi, j-L) @ sigmaz(hi, j+1) @ sigmaz(hi, j+L)
        h += np.cos(angle) * np.sin(angle) ** 3 * sigmax(hi, j) @ sigmax(hi, j-L) @ sigmaz(hi, j+1) @ sigmaz(hi, j+L)
        h += np.cos(angle) * np.sin(angle) ** 3 * sigmax(hi, j) @ sigmaz(hi, j-L) @ sigmax(hi, j+1) @ sigmaz(hi, j+L)
        h += np.cos(angle) * np.sin(angle) ** 3 * sigmax(hi, j) @ sigmaz(hi, j-L) @ sigmaz(hi, j+1) @ sigmax(hi, j+L)
        h -= np.cos(angle) * np.sin(angle) ** 3 * sigmaz(hi, j) @ sigmaz(hi, j-L) @ sigmaz(hi, j+1) @ sigmaz(hi, j+L)
        h += np.sin(angle) ** 4 * sigmax(hi, j) @ sigmaz(hi, j-L) @ sigmaz(hi, j+1) @ sigmaz(hi, j+L)

        j += L-1

        h -= np.cos(angle) ** 4 * sigmaz(hi, j) @ sigmax(hi, j-L) @ sigmax(hi, j-1) @ sigmax(hi, j+L)
        h += np.cos(angle) ** 3 * np.sin(angle) * sigmax(hi, j) @ sigmax(hi, j-L) @ sigmax(hi, j-1) @ sigmax(hi, j+L)
        h -= np.cos(angle) ** 3 * np.sin(angle) * sigmaz(hi, j) @ sigmaz(hi, j-L) @ sigmax(hi, j-1) @ sigmax(hi, j+L)
        h -= np.cos(angle) ** 3 * np.sin(angle) * sigmaz(hi, j) @ sigmax(hi, j-L) @ sigmaz(hi, j-1) @ sigmax(hi, j+L)
        h -= np.cos(angle) ** 3 * np.sin(angle) * sigmaz(hi, j) @ sigmax(hi, j-L) @ sigmax(hi, j-1) @ sigmaz(hi, j+L)
        h += np.cos(angle) ** 2 * np.sin(angle) ** 2 * sigmax(hi, j) @ sigmaz(hi, j-L) @ sigmax(hi, j-1) @ sigmax(hi, j+L)
        h += np.cos(angle) ** 2 * np.sin(angle) ** 2 * sigmax(hi, j) @ sigmax(hi, j-L) @ sigmaz(hi, j-1) @ sigmax(hi, j+L)
        h += np.cos(angle) ** 2 * np.sin(angle) ** 2 * sigmax(hi, j) @ sigmax(hi, j-L) @ sigmax(hi, j-1) @ sigmaz(hi, j+L)
        h -= np.cos(angle) ** 2 * np.sin(angle) ** 2 * sigmaz(hi, j) @ sigmaz(hi, j-L) @ sigmaz(hi, j-1) @ sigmax(hi, j+L)
        h -= np.cos(angle) ** 2 * np.sin(angle) ** 2 * sigmaz(hi, j) @ sigmaz(hi, j-L) @ sigmax(hi, j-1) @ sigmaz(hi, j+L)
        h -= np.cos(angle) ** 2 * np.sin(angle) ** 2 * sigmaz(hi, j) @ sigmax(hi, j-L) @ sigmaz(hi, j-1) @ sigmaz(hi, j+L)
        h += np.cos(angle) * np.sin(angle) ** 3 * sigmax(hi, j) @ sigmax(hi, j-L) @ sigmaz(hi, j-1) @ sigmaz(hi, j+L)
        h += np.cos(angle) * np.sin(angle) ** 3 * sigmax(hi, j) @ sigmaz(hi, j-L) @ sigmax(hi, j-1) @ sigmaz(hi, j+L)
        h += np.cos(angle) * np.sin(angle) ** 3 * sigmax(hi, j) @ sigmaz(hi, j-L) @ sigmaz(hi, j-1) @ sigmax(hi, j+L)
        h -= np.cos(angle) * np.sin(angle) ** 3 * sigmaz(hi, j) @ sigmaz(hi, j-L) @ sigmaz(hi, j-1) @ sigmaz(hi, j+L)
        h += np.sin(angle) ** 4 * sigmax(hi, j) @ sigmaz(hi, j-L) @ sigmaz(hi, j-1) @ sigmaz(hi, j+L)

        i += L*(L-1)
        h -= np.cos(angle) ** 4 * sigmaz(hi, i) @ sigmax(hi, i-1) @ sigmax(hi, i+1) @ sigmax(hi, i-L)
        h += np.cos(angle) ** 3 * np.sin(angle) * sigmax(hi, i) @ sigmax(hi, i-1) @ sigmax(hi, i+1) @ sigmax(hi, i-L)
        h -= np.cos(angle) ** 3 * np.sin(angle) * sigmaz(hi, i) @ sigmaz(hi, i-1) @ sigmax(hi, i+1) @ sigmax(hi, i-L)
        h -= np.cos(angle) ** 3 * np.sin(angle) * sigmaz(hi, i) @ sigmax(hi, i-1) @ sigmaz(hi, i+1) @ sigmax(hi, i-L)
        h -= np.cos(angle) ** 3 * np.sin(angle) * sigmaz(hi, i) @ sigmax(hi, i-1) @ sigmax(hi, i+1) @ sigmaz(hi, i-L)
        h += np.cos(angle) ** 2 * np.sin(angle) ** 2 * sigmax(hi, i) @ sigmaz(hi, i-1) @ sigmax(hi, i+1) @ sigmax(hi, i-L)
        h += np.cos(angle) ** 2 * np.sin(angle) ** 2 * sigmax(hi, i) @ sigmax(hi, i-1) @ sigmaz(hi, i+1) @ sigmax(hi, i-L)
        h += np.cos(angle) ** 2 * np.sin(angle) ** 2 * sigmax(hi, i) @ sigmax(hi, i-1) @ sigmax(hi, i+1) @ sigmaz(hi, i-L)
        h -= np.cos(angle) ** 2 * np.sin(angle) ** 2 * sigmaz(hi, i) @ sigmaz(hi, i-1) @ sigmaz(hi, i+1) @ sigmax(hi, i-L)
        h -= np.cos(angle) ** 2 * np.sin(angle) ** 2 * sigmaz(hi, i) @ sigmaz(hi, i-1) @ sigmax(hi, i+1) @ sigmaz(hi, i-L)
        h -= np.cos(angle) ** 2 * np.sin(angle) ** 2 * sigmaz(hi, i) @ sigmax(hi, i-1) @ sigmaz(hi, i+1) @ sigmaz(hi, i-L)
        h += np.cos(angle) * np.sin(angle) ** 3 * sigmax(hi, i) @ sigmax(hi, i-1) @ sigmaz(hi, i+1) @ sigmaz(hi, i-L)
        h += np.cos(angle) * np.sin(angle) ** 3 * sigmax(hi, i) @ sigmaz(hi, i-1) @ sigmax(hi, i+1) @ sigmaz(hi, i-L)
        h += np.cos(angle) * np.sin(angle) ** 3 * sigmax(hi, i) @ sigmaz(hi, i-1) @ sigmaz(hi, i+1) @ sigmax(hi, i-L)
        h -= np.cos(angle) * np.sin(angle) ** 3 * sigmaz(hi, i) @ sigmaz(hi, i-1) @ sigmaz(hi, i+1) @ sigmaz(hi, i-L)
        h += np.sin(angle) ** 4 * sigmax(hi, i) @ sigmaz(hi, i-1) @ sigmaz(hi, i+1) @ sigmaz(hi, i-L)

    for i in range ((L-2)**2):
        y = i // (L-2)
        x = i % (L-2)
        i = (y+1)*L+x+1
        h -= np.cos(angle) ** 5 * sigmaz(hi, i) @ sigmax(hi, i-1) @ sigmax(hi, i+1) @ sigmax(hi, i-L) @ sigmax(hi, i+L)
        h += np.cos(angle) ** 4 * np.sin(angle) * sigmax(hi, i) @ sigmax(hi, i-1) @ sigmax(hi, i+1) @ sigmax(hi, i-L) @ sigmax(hi, i+L)
        h -= np.cos(angle) ** 4 * np.sin(angle) * sigmaz(hi, i) @ sigmaz(hi, i-1) @ sigmax(hi, i+1) @ sigmax(hi, i-L) @ sigmax(hi, i+L)
        h -= np.cos(angle) ** 4 * np.sin(angle) * sigmaz(hi, i) @ sigmax(hi, i-1) @ sigmaz(hi, i+1) @ sigmax(hi, i-L) @ sigmax(hi, i+L)
        h -= np.cos(angle) ** 4 * np.sin(angle) * sigmaz(hi, i) @ sigmax(hi, i-1) @ sigmax(hi, i+1) @ sigmaz(hi, i-L) @ sigmax(hi, i+L)
        h -= np.cos(angle) ** 4 * np.sin(angle) * sigmaz(hi, i) @ sigmax(hi, i-1) @ sigmax(hi, i+1) @ sigmax(hi, i-L) @ sigmaz(hi, i+L)

        h += np.cos(angle) ** 3 * np.sin(angle) ** 2 * sigmax(hi, i) @ sigmaz(hi, i-1) @ sigmax(hi, i+1) @ sigmax(hi, i-L) @ sigmax(hi, i+L)
        h += np.cos(angle) ** 3 * np.sin(angle) ** 2 * sigmax(hi, i) @ sigmax(hi, i-1) @ sigmaz(hi, i+1) @ sigmax(hi, i-L) @ sigmax(hi, i+L)
        h += np.cos(angle) ** 3 * np.sin(angle) ** 2 * sigmax(hi, i) @ sigmax(hi, i-1) @ sigmax(hi, i+1) @ sigmaz(hi, i-L) @ sigmax(hi, i+L)
        h += np.cos(angle) ** 3 * np.sin(angle) ** 2 * sigmax(hi, i) @ sigmax(hi, i-1) @ sigmax(hi, i+1) @ sigmax(hi, i-L) @ sigmaz(hi, i+L)
        h -= np.cos(angle) ** 3 * np.sin(angle) ** 2 * sigmaz(hi, i) @ sigmaz(hi, i-1) @ sigmaz(hi, i+1) @ sigmax(hi, i-L) @ sigmax(hi, i+L)
        h -= np.cos(angle) ** 3 * np.sin(angle) ** 2 * sigmaz(hi, i) @ sigmaz(hi, i-1) @ sigmax(hi, i+1) @ sigmaz(hi, i-L) @ sigmax(hi, i+L)
        h -= np.cos(angle) ** 3 * np.sin(angle) ** 2 * sigmaz(hi, i) @ sigmaz(hi, i-1) @ sigmax(hi, i+1) @ sigmax(hi, i-L) @ sigmaz(hi, i+L)
        h -= np.cos(angle) ** 3 * np.sin(angle) ** 2 * sigmaz(hi, i) @ sigmax(hi, i-1) @ sigmaz(hi, i+1) @ sigmaz(hi, i-L) @ sigmax(hi, i+L)
        h -= np.cos(angle) ** 3 * np.sin(angle) ** 2 * sigmaz(hi, i) @ sigmax(hi, i-1) @ sigmaz(hi, i+1) @ sigmax(hi, i-L) @ sigmaz(hi, i+L)
        h -= np.cos(angle) ** 3 * np.sin(angle) ** 2 * sigmaz(hi, i) @ sigmax(hi, i-1) @ sigmax(hi, i+1) @ sigmaz(hi, i-L) @ sigmaz(hi, i+L)

        h += np.cos(angle) ** 2 * np.sin(angle) ** 3 * sigmax(hi, i) @ sigmaz(hi, i-1) @ sigmaz(hi, i+1) @ sigmax(hi, i-L) @ sigmax(hi, i+L)
        h += np.cos(angle) ** 2 * np.sin(angle) ** 3 * sigmax(hi, i) @ sigmaz(hi, i-1) @ sigmax(hi, i+1) @ sigmaz(hi, i-L) @ sigmax(hi, i+L)
        h += np.cos(angle) ** 2 * np.sin(angle) ** 3 * sigmax(hi, i) @ sigmaz(hi, i-1) @ sigmax(hi, i+1) @ sigmax(hi, i-L) @ sigmaz(hi, i+L)
        h += np.cos(angle) ** 2 * np.sin(angle) ** 3 * sigmax(hi, i) @ sigmax(hi, i-1) @ sigmaz(hi, i+1) @ sigmaz(hi, i-L) @ sigmax(hi, i+L)
        h += np.cos(angle) ** 2 * np.sin(angle) ** 3 * sigmax(hi, i) @ sigmax(hi, i-1) @ sigmaz(hi, i+1) @ sigmax(hi, i-L) @ sigmaz(hi, i+L)
        h += np.cos(angle) ** 2 * np.sin(angle) ** 3 * sigmax(hi, i) @ sigmax(hi, i-1) @ sigmax(hi, i+1) @ sigmaz(hi, i-L) @ sigmaz(hi, i+L)
        h -= np.cos(angle) ** 2 * np.sin(angle) ** 3 * sigmaz(hi, i) @ sigmax(hi, i-1) @ sigmaz(hi, i+1) @ sigmaz(hi, i-L) @ sigmaz(hi, i+L)
        h -= np.cos(angle) ** 2 * np.sin(angle) ** 3 * sigmaz(hi, i) @ sigmaz(hi, i-1) @ sigmax(hi, i+1) @ sigmaz(hi, i-L) @ sigmaz(hi, i+L)
        h -= np.cos(angle) ** 2 * np.sin(angle) ** 3 * sigmaz(hi, i) @ sigmaz(hi, i-1) @ sigmaz(hi, i+1) @ sigmax(hi, i-L) @ sigmaz(hi, i+L)
        h -= np.cos(angle) ** 2 * np.sin(angle) ** 3 * sigmaz(hi, i) @ sigmaz(hi, i-1) @ sigmaz(hi, i+1) @ sigmaz(hi, i-L) @ sigmax(hi, i+L)

        h -= np.cos(angle) * np.sin(angle) ** 4 * sigmaz(hi, i) @ sigmaz(hi, i-1) @ sigmaz(hi, i+1) @ sigmaz(hi, i-L) @ sigmaz(hi, i+L)
        h += np.cos(angle) * np.sin(angle) ** 4 * sigmax(hi, i) @ sigmax(hi, i-1) @ sigmaz(hi, i+1) @ sigmaz(hi, i-L) @ sigmaz(hi, i+L)
        h += np.cos(angle) * np.sin(angle) ** 4 * sigmax(hi, i) @ sigmaz(hi, i-1) @ sigmax(hi, i+1) @ sigmaz(hi, i-L) @ sigmaz(hi, i+L)
        h += np.cos(angle) * np.sin(angle) ** 4 * sigmax(hi, i) @ sigmaz(hi, i-1) @ sigmaz(hi, i+1) @ sigmax(hi, i-L) @ sigmaz(hi, i+L)
        h += np.cos(angle) * np.sin(angle) ** 4 * sigmax(hi, i) @ sigmaz(hi, i-1) @ sigmaz(hi, i+1) @ sigmaz(hi, i-L) @ sigmax(hi, i+L)
        h += np.sin(angle) ** 5 * sigmax(hi, i) @ sigmaz(hi, i-1) @ sigmaz(hi, i+1) @ sigmaz(hi, i-L) @ sigmaz(hi, i+L)

    ma = nk.models.RBM(alpha=4, param_dtype=complex)
    sa = nk.sampler.MetropolisLocal(hilbert=hi, n_chains=16)
    schedule = optax.warmup_cosine_decay_schedule(0.001, peak_value=0.01, warmup_steps=100, decay_steps=5000, end_value=2e-4)
    # Optimizer
    op = nk.optimizer.Sgd(learning_rate=schedule)
    # Stochastic Reconfiguration
    sr = nk.optimizer.SR(diag_shift=optax.linear_schedule(0.03, 0.005, 1000))
    # The variational state
    vs = nk.vqs.MCState(sa, ma, n_samples=numsamples)
    gs = nk.VMC(
        hamiltonian=h,
        optimizer=op,
        preconditioner=sr,
        variational_state=vs)

    start = time.time()
    gs.run(out='RBM' + str(a) + "L=" + str(N) + "_numsample=" + str(numsamples), n_iter=5000)

    if N <= 20:
        combinations = np.array(list(itertools.product([-1, 1], repeat=N)))
        np.save("RBM" + str(a) + "L=" + str(N) + "_numsample4096_amp.npy", vs.log_value(combinations))

    end = time.time()
    a += 1
    print('### RBM calculation')
    print('Has', vs.n_parameters, 'parameters')
    print('The RBM calculation took', end - start, 'seconds')