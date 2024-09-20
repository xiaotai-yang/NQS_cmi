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
parser.add_argument('--numsamples', type = int, default=8192)
parser.add_argument('--alpha', type = int, default=16)
parser.add_argument('--nchain_per_rank', type = int, default=512)


args = parser.parse_args()
L = args.L
N = L
numsamples = args.numsamples
alpha = args.alpha
nchain_per_rank = args.nchain_per_rank

hi = nk.hilbert.Spin(s=1 / 2, N=N)
g = nk.graph.Hypercube(length=N, n_dim=1, pbc=False)
angle_list = [0.0 * jnp.pi, 0.05 * jnp.pi, 0.10 * jnp.pi, 0.15 * jnp.pi, 0.2 * jnp.pi, 0.25 * jnp.pi, 0.3 * jnp.pi, 0.35 * jnp.pi,
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
    # RBM ansatz with alpha=1
    ma = nk.models.RBM(alpha=alpha, param_dtype=complex)
    sa = nk.sampler.MetropolisLocal(hilbert=hi,
                                    n_chains_per_rank = nchain_per_rank)
    schedule = optax.warmup_cosine_decay_schedule(init_value=5e-4,
                                                  peak_value=5e-3,
                                                  warmup_steps=500,
                                                  decay_steps=4500,
                                                  end_value=2e-4)    # Optimizer
    op = nk.optimizer.Sgd(learning_rate=schedule)

    # Stochastic Reconfiguration
    sr = nk.optimizer.SR(diag_shift=optax.linear_schedule(init_value=0.03,
                                                          end_value=0.001,
                                                          transition_steps=1000))
    # The variational state
    vs = nk.vqs.MCState(sa, ma, n_samples=numsamples, chunk_size = 4096)
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
