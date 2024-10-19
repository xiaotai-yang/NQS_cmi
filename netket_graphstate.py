import jax.random
import matplotlib.pyplot as plt
import jax.numpy as jnp
import netket as nk
import numpy as np
from netket.operator.spin import sigmax,sigmaz
import optax
import time
import argparse
import pickle
# Parameters

parser = argparse.ArgumentParser()
parser.add_argument('--L', type = int, default = 4)
parser.add_argument('--p', type = int, default=1)
parser.add_argument('--numsamples', type = int, default = 4096)
parser.add_argument('--alpha', type = int, default=16)
parser.add_argument('--nchain_per_rank', type = int, default=16)
parser.add_argument('--numsteps', type = int, default=15000)
parser.add_argument('--chunk_size', type = int, default=131072)
parser.add_argument('--angle', type = float, default=0.0*jnp.pi)
parser.add_argument('--previous_training', type = bool, default=False)

args = parser.parse_args()
L = args.L
numsamples = args.numsamples
alpha = args.alpha
nchain_per_rank = args.nchain_per_rank
numsteps = args.numsteps
chunk_size = args.chunk_size
angle = args.angle
previous_training = args.previous_training

N = L**2
hi = nk.hilbert.Spin(s=1 / 2, N=N)
g = nk.graph.Hypercube(length=N, n_dim=1, pbc=False)
ang = round(angle, 3)

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

ma = nk.models.RBM(alpha=alpha, param_dtype=complex)
sa = nk.sampler.MetropolisLocal(hilbert=hi,
                                n_chains_per_rank = nchain_per_rank)
#learning rate schedule
schedule = optax.warmup_cosine_decay_schedule(init_value=2e-4,
                                              peak_value=2e-3,
                                              warmup_steps = 500,
                                              decay_steps = 2500,
                                              end_value = 2.5e-4)
# Optimizer
if previous_training == False:
    op = nk.optimizer.Sgd(learning_rate=schedule)
    sr = nk.optimizer.SR(diag_shift = optax.linear_schedule(init_value = 0.03,
                                                            end_value = 0.002,
                                                            transition_steps = 1000))
else:
    op = nk.optimizer.Sgd(learning_rate=2.5e-4)
    sr = nk.optimizer.SR(diag_shift=0.002)

# The variational state
vs = nk.vqs.MCState(sa, ma, n_samples=numsamples, chunk_size = chunk_size)
if previous_training == True:
    with open(f"params/params_model2D_RBM_graphstate_L{L}_units{alpha}_batch{numsamples}_angle{ang}.pkl", "rb") as f:
        params = pickle.load(f)
    vs.parameters = params

gs = nk.VMC(
    hamiltonian=h,
    optimizer=op,
    preconditioner=sr,
    variational_state=vs)

start = time.time()
gs.run(out='result/RBM/RBM_gs_angle=' + str(ang) + "L=" + str(N) + "_numsample=" + str(numsamples), n_iter=numsteps)
end = time.time()
print('### RBM calculation')
print('Has', vs.n_parameters, 'parameters')
print('The RBM calculation took', end - start, 'seconds')