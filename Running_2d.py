import argparse
import itertools
import os
import optax
import jax
from jax import lax
from tools.Helperfunction import *
from tools.Hamiltonian_2dgf import *
from model.params_initializations import *
from model.twoDRNN import *
from model.twoDRWKV import *
from model.twoDTQS import *
import pickle

jax.config.update("jax_enable_x64", False)

parser = argparse.ArgumentParser()
parser.add_argument('--L', type = int, default = 4)
parser.add_argument('--p', type = int, default = 1)
parser.add_argument('--numunits', type = int, default=32)
parser.add_argument('--lr', type = float, default=2e-4)
parser.add_argument('--gradient_clip', type = bool, default=True)
parser.add_argument('--gradient_clipvalue', type = float, default=10.0)
parser.add_argument('--numsteps', type = int, default=100)
parser.add_argument('--numsamples', type = int, default=64)
parser.add_argument('--testing_sample', type = int, default=2**15)
parser.add_argument('--seed', type = int, default=3)
parser.add_argument('--model_type', type = str, default="RWKV")
parser.add_argument('--basis_rotation', type = bool, default=True)
parser.add_argument('--previous_training', type = bool, default = False)
parser.add_argument('--RWKV_layer', type = int, default=2)
parser.add_argument('--RWKV_emb', type = int, default = 6)
parser.add_argument('--RWKV_hidden', type = int, default=32)
parser.add_argument('--RWKV_ff', type = int, default=512)
parser.add_argument('--TQS_layer', type = int, default=2)
parser.add_argument('--TQS_ff', type = int, default=512)
parser.add_argument('--TQS_units', type = int, default=64)
parser.add_argument('--TQS_head', type = int, default=4)

args = parser.parse_args()
L = args.L
units = args.numunits
numsamples = args.numsamples
p = args.p
px = p
py = p
Nx = L
Ny = L
N = Nx*Ny
lr = args.lr
gradient_clip = args.gradient_clip
gradient_clipvalue = args.gradient_clipvalue
numsteps = args.numsteps
testing_sample = args.testing_sample
model_type = args.model_type
basis_rotation = args.basis_rotation
previous_training = args.previous_training
RWKV_layer = args.RWKV_layer
RWKV_hidden = args.RWKV_hidden
RWKV_ff = args.RWKV_ff
RWKV_emb = args.RWKV_emb
TQS_layer = args.TQS_layer
TQS_ff = args.TQS_ff
TQS_units = args.TQS_units
TQS_head = args.TQS_head
eval_steps = int(testing_sample/numsamples)
input_size = 2 ** (px*py)
key = PRNGKey(args.seed)
meanEnergy=[]
varEnergy=[]
eval_meanEnergy=[]
eval_varEnergy=[]
angle_list = [0.0*jnp.pi, 0.05*jnp.pi, 0.1*jnp.pi, 0.15*jnp.pi, 0.20*jnp.pi, 0.25*jnp.pi, 0.3*jnp.pi, 0.35*jnp.pi, 0.4*jnp.pi, 0.45*jnp.pi, 0.5*jnp.pi]
a = 0
if (model_type == "tensor_gru"):
    if previous_training == True:
        meanEnergy = jnp.load(f"result/meanE_2DRNN_L{L}_patch{p}_units{units}_batch{numsamples}_seed{args.seed}.npy").reshape(len(angle_list), -1).tolist()
        varEnergy = jnp.load(f"result/varE_2DRNN_L{L}_patch{p}_units{units}_batch{numsamples}_seed{args.seed}.npy").reshape(len(angle_list), -1).tolist()
    else:
        meanEnergy = [[] for i in range(len(angle_list))]
        varEnergy = [[] for i in range(len(angle_list))]
elif (model_type == "RWKV"):
    if previous_training == True:
        meanEnergy = jnp.load(f"result/meanE_2DRWKV_L{L}_patch{p}_emb{RWKV_emb}_layer{RWKV_layer}_hidden{RWKV_hidden}_ff{RWKV_ff}_batch{numsamples}_seed{args.seed}.npy").reshape(len(angle_list), -1).tolist()
        varEnergy = jnp.load(f"result/varE_2DRWKV_L{L}_patch{p}_emb{RWKV_emb}_layer{RWKV_layer}_hidden{RWKV_hidden}_ff{RWKV_ff}_batch{numsamples}_seed{args.seed}.npy").reshape(len(angle_list), -1).tolist()
    else:
        meanEnergy = [[] for i in range(len(angle_list))]
        varEnergy = [[] for i in range(len(angle_list))]
elif (model_type == "TQS"):
    if previous_training == True:
        meanEnergy = jnp.load("result/meanE_1DTQS"+"_L" + str(L) + "_patch" + str(p) + "_layer" + str(TQS_layer) + "_ff" + str(TQS_ff) + "_units" + str(TQS_units) + "_head" + str(TQS_head) + "_batch" + str(numsamples) +"_dmrg"+str(dmrg)+ "_seed" + str(args.seed) + ".npy").reshape(len(angle_list), -1).tolist()
        varEnergy = jnp.load("result/varE_1DTQS"+"_L" + str(L) + "_patch" + str(p) + "_layer" + str(TQS_layer) + "_ff" + str(TQS_ff) + "_units" + str(TQS_units) + "_head" + str(TQS_head) + "_batch" + str(numsamples) +"_dmrg"+str(dmrg)+ "_seed" + str(args.seed) + ".npy").reshape(len(angle_list), -1).tolist()
    else:
        meanEnergy = [[] for i in range(len(angle_list))]
        varEnergy = [[] for i in range(len(angle_list))]

for angle in (angle_list):
    evalmeanE = 0
    evalvarE = 0
    x, y = jnp.cos(angle), jnp.sin(angle)
    key, subkey = split(key, 2)
    if (model_type == "tensor_gru"):
        if previous_training == True:
            with open(f"params/params_model1D{model_type}_L{L}_patch{p}_units{units}_batch{numsamples}_angle{angle}_seed{args.seed}.pkl", "rb") as f:
                params = pickle.load(f)
        else:
            params = init_2dtensor_gru_params(input_size, units, Ny, Nx, key)
        fixed_params = Ny, Nx, py, px, units
        batch_sample_prob = jax.jit(vmap(sample_prob, (None, None, 0)), static_argnames=['fixed_params'])
        batch_log_amp = jax.jit(vmap(log_amp, (0, None, None)), static_argnames=['fixed_params'])
    elif (model_type == "RWKV"):
        if previous_training == True:
            with open(f"params/params_model2D{model_type}_L{L}_patch{p}_emb{RWKV_emb}_layer{RWKV_layer}_hidden{RWKV_hidden}_ff{RWKV_ff}_batch{numsamples}_angle{angle}_seed{args.seed}.pkl", "rb") as f:
                params = pickle.load(f)
        else:
            params = init_2DRWKV_params(input_size, RWKV_emb, RWKV_hidden, RWKV_layer, RWKV_ff, Ny, Nx, key)
        fixed_params = Ny, Nx, py, px, RWKV_layer
        batch_sample_prob = jax.jit(vmap(sample_prob_2DRWKV, (None, None, 0)), static_argnames=['fixed_params'])
        batch_log_amp = jax.jit(vmap(log_amp_2DRWKV, (0, None, None)), static_argnames=['fixed_params'])
    elif (model_type == "TQS"):
        if previous_training == True :
            with open( f"params/params_model2D{model_type}_layer{TQS_layer}_units{TQS_units}_head{TQS_head}_batch{numsamples}_angle{angle}_seed{args.seed}.pkl", "rb") as f:
                params = pickle.load(f)
        else:
            params = init_2DTQS_params(input_size, TQS_layer, TQS_ff, TQS_units, TQS_head, key)
        fixed_params = Ny, Nx, py, px, TQS_layer
        batch_sample_prob = jax.jit(vmap(sample_prob_TQS, (None, None, 0, None)), static_argnames=['fixed_params'])
        batch_log_amp = jax.jit(vmap(log_amp_TQS, (0, None, None, None)), static_argnames=['fixed_params'])

    optimizer = optax.adam(learning_rate=lr)
    optimizer_state = optimizer.init(params)
    batch_total_samples_2d = vmap(total_samples_2d, (0, None), 0)
    batch_new_coe_2d = vmap(new_coe_2d, (0, None, None, None, None))
    batch_diag_coe = vmap(diag_coe, (0, None, None, None, None, None, None))

    # Assuming params are your model's parameters:

    (xy_loc_bulk, xy_loc_edge, xy_loc_corner, yloc_bulk, yloc_edge, yloc_corner, zloc_bulk, zloc_edge,
     zloc_corner, off_diag_bulk_coe, off_diag_edge_coe, off_diag_corner_coe, zloc_bulk_diag, zloc_edge_diag,
     zloc_corner_diag, coe_bulk_diag, coe_edge_diag, coe_corner_diag) = vmc_off_diag_es(Ny, Nx, px ,py, angle, basis_rotation)
    total_samples = lax.cond( basis_rotation, lambda: numsamples * ((Nx * px - 2) * (Ny * py - 2) * (2 ** (5) - 1) + (Nx * px - 2) * 2 * (2 ** (4) - 1) + (Ny * py - 2) * 2 * (2 ** (4) - 1) + 4 * (2 ** (3) - 1))
    ,lambda: numsamples*Nx*px*Ny*py)


    @partial(jax.jit, static_argnames=['fixed_parameters',])
    def compute_cost(parameters, fixed_parameters, samples, Eloc):
        samples = jax.lax.stop_gradient(samples)
        Eloc = jax.lax.stop_gradient(Eloc)
        log_amps_tensor = batch_log_amp(samples, parameters, fixed_parameters)
        cost = 2 * jnp.real(jnp.mean(log_amps_tensor.conjugate() * (Eloc - jnp.mean(Eloc))))
        return cost

    grad_f = jax.jit(jax.grad(compute_cost), static_argnums=(1, ))

    for it in range(0, numsteps+eval_steps):

        key, subkey = split(key ,2)
        sample_key = split(key, numsamples)
        samples, sample_log_amp = batch_sample_prob(params, fixed_params, sample_key)
        samples = jnp.transpose(samples.reshape(numsamples, Ny, Nx, py, px), (0, 1, 3, 2, 4)).reshape(numsamples, Ny*py, Nx*px)

        sigmas = jnp.concatenate((batch_total_samples_2d(samples, xy_loc_bulk),
                                 batch_total_samples_2d(samples, xy_loc_edge),
                                 batch_total_samples_2d(samples, xy_loc_corner)), axis=1).reshape(-1, Ny*py, Nx*px)
        matrixelements = jnp.concatenate((batch_new_coe_2d(samples, off_diag_bulk_coe, yloc_bulk, zloc_bulk, basis_rotation),
                                         batch_new_coe_2d(samples, off_diag_edge_coe, yloc_edge, zloc_edge, basis_rotation),
                                         batch_new_coe_2d(samples, off_diag_corner_coe, yloc_corner, zloc_corner, basis_rotation)), axis=1).reshape(numsamples, -1)

        sigmas = jnp.transpose(sigmas.reshape(total_samples, Ny, py, Nx, px), (0, 1, 3, 2, 4))
        log_all_amp = batch_log_amp(sigmas, params, fixed_params)
        log_diag_amp = jnp.repeat(sample_log_amp, (jnp.ones(numsamples)*(matrixelements.shape[1])).astype(int), axis=0)
        amp = jnp.exp(log_all_amp.ravel()-log_diag_amp).reshape(numsamples, -1)
        Eloc = jnp.sum((amp*matrixelements), axis=1) + batch_diag_coe(samples, zloc_bulk_diag, zloc_edge_diag, zloc_corner_diag, coe_bulk_diag, coe_edge_diag, coe_corner_diag)
        meanE, varE = jnp.mean(Eloc), jnp.var(Eloc)

        samples = jnp.transpose(samples.reshape(numsamples, Ny, py, Nx, px), (0, 1, 3, 2, 4))
        meanE, varE = jnp.mean(Eloc), jnp.var(Eloc)

        if it<numsteps:
            meanEnergy[a].append(meanE)
            varEnergy[a].append(varE)

            if (it + 1) % 50 == 0 or it == 0:
                print("learning_rate =", lr)
                print("Magnetization =", jnp.mean(jnp.sum(2 * samples - 1, axis=(1))))
                print('mean(E): {0}, varE: {1}, #samples {2}, #Step {3} \n\n'.format(meanE, varE, numsamples, it + 1))

            grads = grad_f(params, fixed_params, samples, Eloc)
            if gradient_clip == True:
                grads = jax.tree.map(clip_grad, grads)

            # Update the optimizer state and the parameters
            updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
            params = optax.apply_updates(params, updates)

            if not os.path.exists('./params/'):
                os.mkdir('./params/')
            if (it % 500 == 0):
                if (model_type == "tensor_gru"):
                    with open(
                            f"params/params_model2D{model_type}_L{L}_patch{p}_units{units}_batch{numsamples}_angle{angle}_seed{args.seed}.pkl",
                            "wb") as f:
                        pickle.dump(params, f)
                if (model_type == "RWKV"):
                    with open(
                            f"params/params_model2D{model_type}_L{L}_patch{p}_emb{RWKV_emb}_layer{RWKV_layer}_hidden{RWKV_hidden}_ff{RWKV_ff}_batch{numsamples}_angle{angle}_seed{args.seed}.pkl",
                            "wb") as f:
                        pickle.dump(params, f)
                if (model_type == "TQS"):
                    with open(
                            f"params/params_model2D{model_type}_layer{TQS_layer}_units{TQS_units}_head{TQS_head}_batch{numsamples}_angle{angle}_seed{args.seed}.pkl",
                            "wb") as f:
                        pickle.dump(params, f)
        else:
            evalmeanE += meanE / eval_steps
            evalvarE += varE / eval_steps ** 2
    eval_meanEnergy.append(evalmeanE)
    eval_varEnergy.append(evalvarE)
    a += 1
if not os.path.exists('./result/'):
    os.mkdir('./result/')
if model_type == "tensor_gru":
    jnp.save("result/meanE_2DRNN" + "_L" + str(L) + "_patch" + str(p) + "_units" + str(units) + "_batch" + str(numsamples) + "_seed" + str(args.seed) + ".npy", jnp.array(meanEnergy))
    jnp.save("result/varE_2DRNN"  + "_L" + str(L) + "_patch" + str(p) + "_units" + str(units) + "_batch" + str(numsamples) + "_seed" + str(args.seed) + ".npy", jnp.array(varEnergy))
    jnp.save("result/evalmeanE_1DRNN"  + "_L" + str(L) + "_patch" + str(p) + "_units" + str(units) + "_batch" + str(numsamples) + "_seed" + str(args.seed) + ".npy", jnp.array(eval_meanEnergy))
    jnp.save("result/evalvarE_1DRNN"  + "_L" + str(L) + "_patch" + str(p) + "_units" + str(units) + "_batch" + str(numsamples) + "_seed" + str(args.seed) + ".npy", jnp.array(eval_varEnergy))
elif model_type == "RWKV":
    jnp.save("result/meanE_2DRWKV"  + "_L" + str(L) + "_patch" + str(p) + "_emb" + str(RWKV_emb) + "_layer" + str(RWKV_layer) + "_hidden" + str(RWKV_hidden) + "_ff" + str(RWKV_ff) + "_batch" + str(numsamples) + "_seed" + str(args.seed) + ".npy", jnp.array(meanEnergy))
    jnp.save("result/varE_2DRWKV" + "_L" + str(L) + "_patch" + str(p) + "_emb" + str(RWKV_emb) + "_layer" + str(RWKV_layer) + "_hidden" + str(RWKV_hidden) + "_ff" + str(RWKV_ff) + "_batch" + str(numsamples) + "_seed" + str(args.seed) + ".npy", jnp.array(varEnergy))
    jnp.save("result/evalmeanE_1DRWKV"+"_L" + str(L) + "_patch" + str(p) + "_emb" + str(RWKV_emb) + "_layer"+ str(RWKV_layer) +"_hidden" + str(RWKV_hidden) + "_ff" + str(RWKV_ff) + "_batch" + str(numsamples) + "_seed" + str(args.seed) + ".npy", jnp.array(eval_meanEnergy))
    jnp.save("result/evalvarE_1DRWKV" +"_L" + str(L) + "_patch" + str(p) + "_emb" + str(RWKV_emb) + "_layer" + str(RWKV_layer) + "_hidden" + str(RWKV_hidden) + "_ff" + str(RWKV_ff) + "_batch" + str(numsamples) + "_seed" + str(args.seed) + ".npy", jnp.array(eval_varEnergy))

elif model_type == "TQS":
    jnp.save("result/meanE_2DTQS"  + "_L" + str(L) + "_patch" + str(p) + "_layer" + str(TQS_layer) + "_ff" + str(TQS_ff) + "_units" + str(TQS_units) + "_head" + str(TQS_head) + "_batch" + str(numsamples) + "_seed" + str(args.seed) + ".npy", jnp.array(meanEnergy))
    jnp.save("result/varE_2DTQS" + "_L" + str(L) + "_patch" + str(p) + "_layer" + str(TQS_layer) + "_ff" + str(TQS_ff) + "_units" + str(TQS_units) + "_head" + str(TQS_head) + "_batch" + str(numsamples) + "_seed" + str(args.seed) + ".npy", jnp.array(varEnergy))
    jnp.save("result/evalmeanE_2DTQS"+"_L" + str(L) + "_patch" + str(p) + "_layer" + str(TQS_layer) + "_ff" + str(TQS_ff) + "_units" + str(TQS_units) + "_head" + str(TQS_head) + "_batch" + str(numsamples) + "_seed" + str(args.seed) + ".npy", jnp.array(eval_meanEnergy))
    jnp.save("result/evalvarE_2DTQS"+"_L" + str(L) + "_patch" + str(p) + "_layer" + str(TQS_layer) + "_ff" + str(TQS_ff) + "_units" + str(TQS_units) + "_head" + str(TQS_head) + "_batch" + str(numsamples) + "_seed" + str(args.seed) + ".npy", jnp.array(eval_varEnergy))
