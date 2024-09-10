import jax
import netket as nk
import numpy as np
from netket.operator.spin import sigmax, sigmaz
import time
import netket.nn as nknn
import flax.linen as nn
import jax.numpy as jnp
from scipy.sparse.linalg import eigsh
import argparse
import optax
import itertools

jax.config.update("jax_enable_x64", False)
parser = argparse.ArgumentParser()
parser.add_argument('--L', type = int, default = 64)
parser.add_argument('--numsamples', type = int, default=256)
parser.add_argument('--alpha', type = int, default=4)

args = parser.parse_args()
L = args.L
N = L
numsamples = args.numsamples
alpha = args.alpha

hi = nk.hilbert.Spin(s=1 / 2, N=N)
g = nk.graph.Hypercube(length=N, n_dim=1, pbc=False)
angle_list = [0, 0.05 * jnp.pi, 0.10 * jnp.pi, 0.15 * jnp.pi, 0.2 * jnp.pi, 0.25 * jnp.pi, 0.3 * jnp.pi, 0.35 * jnp.pi,
              0.4 * jnp.pi, 0.45 * jnp.pi, 0.5 * jnp.pi]

a = 0

for angle in angle_list:

    h = - np.cos(angle) ** 2 * sigmax(hi, 0) @ sigmaz(hi, 1)
    h -= np.cos(angle) * np.sin(angle) * sigmaz(hi, 0) @ sigmaz(hi, 1)
    h += np.cos(angle) * np.sin(angle) * sigmax(hi, 0) @ sigmax(hi, 1)
    h += np.sin(angle) ** 2 * sigmaz(hi, 0) @ sigmax(hi, 1)

    # Middle set of terms (for j = 1 to N-3)
    for j in range(1, L - 1):
        h -= np.cos(angle) ** 3 * sigmaz(hi, j - 1) @ sigmax(hi, j) @ sigmaz(hi, j + 1)
        h += np.cos(angle) ** 2 * np.sin(angle) * sigmax(hi, j - 1) @ sigmax(hi, j) @ sigmaz(hi, j + 1)
        h -= np.cos(angle) ** 2 * np.sin(angle) * sigmaz(hi, j - 1) @ sigmaz(hi, j) @ sigmaz(hi, j + 1)
        h += np.cos(angle) ** 2 * np.sin(angle) * sigmaz(hi, j - 1) @ sigmax(hi, j) @ sigmax(hi, j + 1)
        h += np.cos(angle) * np.sin(angle) ** 2 * sigmax(hi, j - 1) @ sigmaz(hi, j) @ sigmaz(hi, j + 1)
        h -= np.cos(angle) * np.sin(angle) ** 2 * sigmax(hi, j - 1) @ sigmax(hi, j) @ sigmax(hi, j + 1)
        h += np.cos(angle) * np.sin(angle) ** 2 * sigmaz(hi, j - 1) @ sigmaz(hi, j) @ sigmax(hi, j + 1)
        h -= np.sin(angle) ** 3 * sigmax(hi, j - 1) @ sigmaz(hi, j) @ sigmax(hi, j + 1)

    h -= np.cos(angle) ** 2 * sigmaz(hi, j) @ sigmax(hi, j+1)
    h += np.cos(angle) * np.sin(angle) * sigmax(hi, j) @ sigmax(hi, j+1)
    h -= np.cos(angle) * np.sin(angle) * sigmaz(hi, j) @ sigmaz(hi, j+1)
    h += np.sin(angle) ** 2  * sigmax(hi, j) @ sigmaz(hi, j+1)
    H = h.to_sparse()
    eigvec, eigval = eigsh(H, k=1, which='SA')
    print(eigval)
    # RBM ansatz with alpha=1
    ma = nk.models.RBM(alpha=4, param_dtype=complex)
    sa = nk.sampler.MetropolisLocal(hilbert=hi, n_chains=16)
    schedule = optax.warmup_cosine_decay_schedule(0.001, peak_value=0.01, warmup_steps = 100, decay_steps = 5000, end_value = 2e-4)
    # Optimizer
    op = nk.optimizer.Sgd(learning_rate=schedule)

    # Stochastic Reconfiguration
    sr = nk.optimizer.SR(diag_shift = optax.linear_schedule(0.03, 0.005, 1000))

    # The variational state
    vs = nk.vqs.MCState(sa, ma, n_samples=numsamples)
    # The ground-state optimization loop
    gs = nk.VMC(
        hamiltonian=h,
        optimizer=op,
        preconditioner=sr,
        variational_state=vs)

    start = time.time()
    gs.run(out='RBM' + str(a)+ "L=" +str(N)+"_numsample="+str(numsamples), n_iter=5000)

    if N<= 20:
        combinations = np.array(list(itertools.product([-1, 1], repeat=N)))
        np.save("RBM" + str(a)+"L="+str(N) + "_numsample4096_amp.npy", vs.log_value(combinations))

    end = time.time()
    a += 1
    print('### RBM calculation')
    print('Has', vs.n_parameters, 'parameters')
    print('The RBM calculation took', end - start, 'seconds')
