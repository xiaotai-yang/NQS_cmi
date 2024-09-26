import jax
import netket as nk
import numpy as np
from jax import vmap
from netket.operator.spin import sigmax, sigmaz
import time
from jax import lax
from functools import partial
from model.model_utlis import *
import netket.nn as nknn
import flax.linen as nn
import jax.numpy as jnp
from scipy.sparse.linalg import eigsh
import argparse
import optax
import itertools
from jax.random import PRNGKey, split
import pickle

jax.config.update("jax_enable_x64", False)

parser = argparse.ArgumentParser()
parser.add_argument('--L', type = int, default = 64)
parser.add_argument('--p', type = int, default=1)
parser.add_argument('--numsamples', type = int, default = 8192)
parser.add_argument('--alpha', type = int, default=16)
parser.add_argument('--nchain_per_rank', type = int, default=512)
parser.add_argument('--numsteps', type = int, default=5000)
parser.add_argument('--dmrg', type = bool, default=False)
parser.add_argument('--H_type', type = str, default="cluster")
parser.add_argument('--angle', type = float, default=0.0)
parser.add_argument('--previous_training', type = bool, default=False)
args = parser.parse_args()

L = args.L
p = args.p
N = L
numsamples = args.numsamples
alpha = args.alpha
nchain_per_rank = args.nchain_per_rank
numsteps = args.numsteps
dmrg = args.dmrg
H_type = args.H_type
previous_training = args.previous_training
angle = args.angle
ang = round(angle, 3)

if dmrg == True:
    if H_type == "ES":
        M0 = jnp.load("DMRG/mps_tensors/ES_tensor_init_" + str(L * p) + "_angle_" + str(ang) + ".npy")
        M = jnp.load("DMRG/mps_tensors/ES_tensor_" + str(L * p) + "_angle_" + str(ang) + ".npy")
        Mlast = jnp.load("DMRG/mps_tensors/ES_tensor_last_" + str(L * p) + "_angle_" + str(ang) + ".npy")
    else:
        M0 = jnp.load("DMRG/mps_tensors/cluster_tensor_init_" + str(L * p) + "_angle_" + str(ang) + ".npy")
        M = jnp.load("DMRG/mps_tensors/cluster_tensor_" + str(L * p) + "_angle_" + str(ang) + ".npy")
        Mlast = jnp.load("DMRG/mps_tensors/cluster_tensor_last_" + str(L * p) + "_angle_" + str(ang) + ".npy")
    batch_log_phase_dmrg = jax.jit(vmap(partial(log_phase_dmrg, M0=M0, M=M, Mlast=Mlast, netket= True), 0))


    def uniform_init(scale):
        def init(key, shape, dtype=jnp.float32):
            return jax.random.uniform(key, shape, dtype=dtype, minval=-scale, maxval=scale)

        return init
    class RBM_dmrg_model(nn.Module):
        def setup(self):
            self.dense = nn.Dense(
            features=alpha * L,
            param_dtype=jnp.float32,
            kernel_init= uniform_init(1/jnp.sqrt(alpha * L)),
            bias_init= uniform_init(1/jnp.sqrt(alpha * L))
            )
            self.ai = jnp.zeros(L, dtype=jnp.float32)

        def __call__(self, x):
            y = self.dense(x)  # x shape: (batch_size, input_dim)
            y = nk.nn.activation.log_cosh(y)
            y_sum = jnp.sum(y, axis=-1).astype(jnp.complex64)
            # Apply batch_log_phase_dmrg to x
            phase_corrections = batch_log_phase_dmrg(x)
            y_sum += phase_corrections
            y_sum += jnp.dot(x, self.ai)
            return y_sum
    ma = RBM_dmrg_model()
else:
    ma = nk.models.RBM(alpha=alpha)

hi = nk.hilbert.Spin(s=1 / 2, N=N)
g = nk.graph.Hypercube(length=N, n_dim=1, pbc=False)

if H_type == "ES":
    h = - np.cos(angle) ** 2 * sigmaz(hi, 0) @ sigmax(hi, 1)
    h -= np.cos(angle) * np.sin(angle) * sigmaz(hi, 0) @ sigmaz(hi, 1)
    h += np.sin(angle) ** 2 * sigmax(hi, 0) @ sigmaz(hi, 1)
    h += np.cos(angle) * np.sin(angle) * sigmax(hi, 0) @ sigmax(hi, 1)

    # Last set of terms
    h -= np.cos(angle) ** 2 * sigmax(hi, L - 2) @ sigmax(hi, L - 1)
    h -= np.cos(angle) * np.sin(angle) * sigmax(hi, L - 2) @ sigmaz(hi, L - 1)
    h -= np.sin(angle) ** 2 * sigmaz(hi, L - 2) @ sigmaz(hi, L - 1)
    h -= np.cos(angle) * np.sin(angle) * sigmaz(hi, L - 2) @ sigmax(hi, L - 1)

    # Middle set of terms (for j = 1 to N-3)
    for j in range(1, L - 2):
        h -= np.cos(angle) ** 3 * sigmax(hi, j - 1) @ sigmaz(hi, j) @ sigmax(hi, j + 1)
        h -= np.cos(angle) ** 2 * np.sin(angle) * sigmaz(hi, j - 1) @ sigmaz(hi, j) @ sigmax(hi, j + 1)
        h += np.cos(angle) ** 2 * np.sin(angle) * sigmax(hi, j - 1) @ sigmax(hi, j) @ sigmax(hi, j + 1)
        h -= np.cos(angle) ** 2 * np.sin(angle) * sigmax(hi, j - 1) @ sigmaz(hi, j) @ sigmaz(hi, j + 1)
        h += np.cos(angle) * np.sin(angle) ** 2 * sigmaz(hi, j - 1) @ sigmax(hi, j) @ sigmax(hi, j + 1)
        h += np.cos(angle) * np.sin(angle) ** 2 * sigmax(hi, j - 1) @ sigmax(hi, j) @ sigmaz(hi, j + 1)
        h -= np.cos(angle) * np.sin(angle) ** 2 * sigmaz(hi, j - 1) @ sigmaz(hi, j) @ sigmaz(hi, j + 1)
        h += np.sin(angle) ** 3 * sigmaz(hi, j - 1) @ sigmax(hi, j) @ sigmaz(hi, j + 1)
    j += 1

    h -= np.cos(angle) ** 3 * sigmax(hi, j - 1) @ sigmaz(hi, j) @ sigmaz(hi, j + 1)
    h -= np.cos(angle) ** 2 * np.sin(angle) * sigmaz(hi, j - 1) @ sigmaz(hi, j) @ sigmaz(hi, j + 1)
    h += np.cos(angle) ** 2 * np.sin(angle) * sigmax(hi, j - 1) @ sigmax(hi, j) @ sigmaz(hi, j + 1)
    h += np.cos(angle) ** 2 * np.sin(angle) * sigmax(hi, j - 1) @ sigmaz(hi, j) @ sigmax(hi, j + 1)
    h += np.cos(angle) * np.sin(angle) ** 2 * sigmaz(hi, j - 1) @ sigmax(hi, j) @ sigmaz(hi, j + 1)
    h += np.cos(angle) * np.sin(angle) ** 2 * sigmaz(hi, j - 1) @ sigmaz(hi, j) @ sigmax(hi, j + 1)
    h -= np.cos(angle) * np.sin(angle) ** 2 * sigmax(hi, j - 1) @ sigmax(hi, j) @ sigmax(hi, j + 1)
    h -= np.sin(angle) ** 3 * sigmaz(hi, j - 1) @ sigmax(hi, j) @ sigmax(hi, j + 1)

elif H_type == "cluster":
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

    h -= np.cos(angle) ** 2 * sigmaz(hi, j) @ sigmax(hi, j + 1)
    h += np.cos(angle) * np.sin(angle) * sigmax(hi, j) @ sigmax(hi, j + 1)
    h -= np.cos(angle) * np.sin(angle) * sigmaz(hi, j) @ sigmaz(hi, j + 1)
    h += np.sin(angle) ** 2 * sigmax(hi, j) @ sigmaz(hi, j + 1)

#sampler
sa = nk.sampler.MetropolisLocal(hilbert=hi,
                                n_chains_per_rank = nchain_per_rank)
#learning rate schedule
schedule = optax.warmup_cosine_decay_schedule(init_value=5e-4,
                                              peak_value=5e-3,
                                              warmup_steps = 500,
                                              decay_steps = 4500,
                                              end_value = 5e-4)
# Optimizer

# Stochastic Reconfiguration
if previous_training == False:
    op = nk.optimizer.Sgd(learning_rate=schedule)
    sr = nk.optimizer.SR(diag_shift = optax.linear_schedule(init_value = 0.03,
                                                            end_value = 0.001,
                                                            transition_steps = 1000))
else:
    op = nk.optimizer.Sgd(learning_rate=5e-4)
    sr = nk.optimizer.SR(diag_shift=0.001)

# The variational state

vs = nk.vqs.MCState(sa, ma, n_samples=numsamples, chunk_size = 4096)
if previous_training == True:
    with open(f"params/params_model1D_RBM_Htype{H_type}_L{L}_units{alpha}_batch{numsamples}_dmrg{dmrg}_angle{ang}.pkl", "rb") as f:
        params = pickle.load(f)
    vs.parameters = params

# The ground-state optimization loop
gs = nk.VMC(
    hamiltonian=h,
    optimizer=op,
    preconditioner=sr,
    variational_state=vs)

start = time.time()
gs.run(out='RBM' + "_angle=" + str(ang)+ "_L=" +str(N)+"_numsample="+str(numsamples), n_iter=numsteps)

with open(f"params/params_model1D_RBM_Htype{H_type}_L{L}_units{alpha}_batch{numsamples}_dmrg{dmrg}_angle{ang}.pkl", "wb") as f:
    pickle.dump(vs.parameters, f)

if N<= 20:
    combinations = np.array(list(itertools.product([-1, 1], repeat=N)))
    np.save("RBM" + "angle_=" + str(ang) +"_L="+str(N) + "_numsample"+str(numsamples)+"_amp.npy", vs.log_value(combinations))

end = time.time()
print('### RBM calculation')
print('Has', vs.n_parameters, 'parameters')
print('The RBM calculation took', end - start, 'seconds')