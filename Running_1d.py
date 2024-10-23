import argparse
import itertools
import os
import optax
import jax
from tools.Helperfunction import *
from model.params_initializations import *
from model.oneDRNN import *
from model.oneDRWKV import *
from model.oneDTQS import *
import pickle

if not os.path.exists('./result/'):
    os.mkdir('./result/')
parser = argparse.ArgumentParser()
parser.add_argument('--L', type = int, default = 16)
parser.add_argument('--p', type = int, default = 4)
parser.add_argument('--numunits', type = int, default=64)
parser.add_argument('--lr', type = float, default=2e-4)
parser.add_argument('--gradient_clip', type = bool, default=True)
parser.add_argument('--gradient_clipvalue', type = float, default=10.0)
parser.add_argument('--numsteps', type = int, default=100000)
parser.add_argument('--numsamples', type = int, default=256)
parser.add_argument('--testing_sample', type = int, default=2**15)
parser.add_argument('--seed', type = int, default=3)
parser.add_argument('--model_type', type = str, default="tensor_gru")
parser.add_argument('--H_type', type = str, default="ES")
parser.add_argument('--basis_rotation', type = bool, default=True)
parser.add_argument('--previous_training', type = bool, default=False)
parser.add_argument('--dmrg', type = bool, default=False)
parser.add_argument('--RWKV_layer', type = int, default=3)
parser.add_argument('--RWKV_emb', type = int, default = 32)
parser.add_argument('--RWKV_hidden', type = int, default=128)
parser.add_argument('--RWKV_head', type = int, default=4)
parser.add_argument('--RWKV_ff', type = int, default=128)
parser.add_argument('--TQS_layer', type = int, default=2)
parser.add_argument('--TQS_ff', type = int, default=512)
parser.add_argument('--TQS_units', type = int, default=64)
parser.add_argument('--TQS_head', type = int, default=4)
parser.add_argument('--angle', type = float, default=jnp.pi*0.5)

args = parser.parse_args()
L = args.L
units = args.numunits
numsamples = args.numsamples
p = args.p
lr = args.lr
gradient_clip = args.gradient_clip
gradient_clipvalue = args.gradient_clipvalue
numsteps = args.numsteps
testing_sample = args.testing_sample
model_type = args.model_type
H_type = args.H_type
if H_type == "ES":
    from tools.Hamiltonian_1des import *
elif H_type == "cluster":
    from tools.Hamiltonian_1dcluster import *
basis_rotation = args.basis_rotation
previous_training = args.previous_training
dmrg = args.dmrg
RWKV_layer = args.RWKV_layer
RWKV_head = args.RWKV_head
RWKV_hidden = args.RWKV_hidden
RWKV_ff = args.RWKV_ff
RWKV_emb = args.RWKV_emb
TQS_layer = args.TQS_layer
TQS_ff = args.TQS_ff
TQS_units = args.TQS_units
TQS_head = args.TQS_head
n_indices = jnp.arange(L)
eval_steps = int(testing_sample/numsamples)
input_size = 2 ** p
key = PRNGKey(args.seed)
eval_meanEnergy=[]
eval_varEnergy=[]
angle = args.angle
ang = round(angle, 3)
evalmeanE = 0
evalvarE = 0
x, y = jnp.cos(angle), jnp.sin(angle)
key, subkey = split(key, 2)

boundaries_and_scales = {
    8000: 0.5,  # After 8000 steps, scale the learning rate by 0.5
    24000: 0.4  # After 24000 steps, scale the learning rate by 0.5
}
lr_schedule = optax.piecewise_constant_schedule(
    init_value=lr,
    boundaries_and_scales=boundaries_and_scales
)
optimizer = optax.adam(learning_rate=lr_schedule)

if (model_type == "tensor_gru"):
    if previous_training == True:
        meanEnergy = jnp.load(f"result/meanE_1DRNN"+"_Htype"+str(H_type)+"_L"+str(L)+"_patch"+str(p)+"_units"+str(units)+"_batch"+str(numsamples)+"_dmrg"+str(dmrg)+"_seed"+str(args.seed)+"_angle"+str(round(angle, 3))+".npy").tolist()
        varEnergy = jnp.load(f"result/varE_1DRNN"+"_Htype"+str(H_type)+"_L"+str(L)+"_patch"+str(p)+"_units"+str(units)+"_batch"+str(numsamples)+"_dmrg"+str(dmrg)+"_seed"+str(args.seed)+"_angle"+str(round(angle, 3))+".npy").tolist()
    else:
        meanEnergy=[]
        varEnergy=[]

elif (model_type == "RWKV"):
    if previous_training == True:
        jnp.load("result/meanE_1DRWKV" + "_Htype" + str(H_type) + "_L" + str(L) + "_patch" + str(p) + "_emb" + str(
            RWKV_emb) + "_layer" + str(RWKV_layer) + "_hidden" + str(RWKV_hidden) + "_ff" + str(
            RWKV_ff) + "_batch" + str(numsamples) + "_dmrg" + str(dmrg) + "_seed" + str(args.seed) + "_angle" + str(
            round(angle, 3)) + ".npy").tolist()
        jnp.load("result/varE_1DRWKV" + "_Htype" + str(H_type) + "_L" + str(L) + "_patch" + str(p) + "_emb" + str(
            RWKV_emb) + "_layer" + str(RWKV_layer) + "_hidden" + str(RWKV_hidden) + "_ff" + str(
            RWKV_ff) + "_batch" + str(numsamples) + "_dmrg" + str(dmrg) + "_seed" + str(args.seed) + "_angle" + str(
            round(angle, 3)) + ".npy").tolist()
    else:
        meanEnergy = []
        varEnergy = []

elif (model_type == "TQS"):
    setting = (dmrg, n_indices, TQS_layer)
    if previous_training == True:
        meanEnergy = jnp.save("result/meanE_1DTQS" + "_Htype" + str(H_type) + "_L" + str(L) + "_patch" + str(p) + "_layer" + str(
            TQS_layer) + "_ff" + str(TQS_ff) + "_units" + str(TQS_units) + "_head" + str(TQS_head) + "_batch" + str(
            numsamples) + "_dmrg" + str(dmrg) + "_seed" + str(args.seed) + "_angle" + str(round(angle, 3)) + ".npy").tolist()
        varEnergy = jnp.save("result/varE_1DTQS" + "_Htype" + str(H_type) + "_L" + str(L) + "_patch" + str(p) + "_layer" + str(
            TQS_layer) + "_ff" + str(TQS_ff) + "_units" + str(TQS_units) + "_head" + str(TQS_head) + "_batch" + str(
            numsamples) + "_dmrg" + str(dmrg) + "_seed" + str(args.seed) + "_angle" + str(round(angle, 3)) + ".npy").tolist()
    else:
        meanEnergy = []
        varEnergy = []

if (model_type == "tensor_gru"):
    if previous_training == True:
        with open(f"params/params_model1D{model_type}_Htype{H_type}_L{L}_patch{p}_units{units}_batch{numsamples}_dmrg{dmrg}_angle{round(angle,3)}_seed{args.seed}.pkl", "rb") as f:
            checkpoint = pickle.load(f)
        params = checkpoint['params']
        optimizer_state = checkpoint['optimizer_state']
    else:
        params = init_tensor_gru_params(input_size, units, L, key)
        optimizer_state = optimizer.init(params)
    fixed_params = L, p, units
    setting = (dmrg, n_indices)
    batch_sample_prob = jax.jit(vmap(sample_prob, (None, None, 0, None)), static_argnames=['fixed_params'])
    batch_log_amp = jax.jit(vmap(log_amp, (0, None, None, None)), static_argnames=['fixed_params'])

elif (model_type == "RWKV"):
    if previous_training == True:
        with open(f"params/params_model1D{model_type}_Htype{H_type}_L{L}_patch{p}_emb{RWKV_emb}_layer{RWKV_layer}_hidden{RWKV_hidden}_head{RWKV_head}_ff{RWKV_ff}_batch{numsamples}_dmrg{dmrg}_angle{round(angle,3)}_seed{args.seed}.pkl", "rb") as f:
            checkpoint = pickle.load(f)
        params = checkpoint['params']
        optimizer_state = checkpoint['optimizer_state']

    else:
        params = init_RWKV_params(input_size, RWKV_emb, RWKV_hidden, RWKV_head, RWKV_ff, RWKV_layer, key)
        optimizer_state = optimizer.init(params)
    setting = (dmrg, n_indices, jnp.arange(RWKV_layer))
    fixed_params = L, p, RWKV_head
    batch_sample_prob = jax.jit(vmap(sample_prob_RWKV, (None, None, 0, None)), static_argnames=['fixed_params'])
    batch_log_amp = jax.jit(vmap(log_amp_RWKV, (0, None, None, None)), static_argnames=['fixed_params'])

elif (model_type == "TQS"):
    if previous_training == True:
        with open(f"params/params_model1D{model_type}_Htype{H_type}_L{L}_patch{p}_layer{TQS_layer}_units{TQS_units}_head{TQS_head}_batch{numsamples}_dmrg{dmrg}_angle{round(angle, 3)}_seed{args.seed}.pkl","rb") as f:
            checkpoint = pickle.load(f)
        params = checkpoint['params']
        optimizer_state = checkpoint['optimizer_state']
    else:
        params = init_1DTQS_params(input_size, TQS_layer, TQS_ff, TQS_units, TQS_head, key)
        optimizer_state = optimizer.init(params)
    setting = (dmrg, n_indices, TQS_layer)
    fixed_params = L, p
    batch_sample_prob = jax.jit(vmap(sample_prob_TQS, (None, None, 0, None)), static_argnames=['fixed_params'])
    batch_log_amp = jax.jit(vmap(log_amp_TQS, (0, None, None, None)), static_argnames=['fixed_params'])


if dmrg == True:
    if H_type == "ES":
        M0 = jnp.load("DMRG/mps_tensors/ES_tensor_init_" + str(L * p) + "_angle_" + str(ang) + ".npy")
        M = jnp.load("DMRG/mps_tensors/ES_tensor_" + str(L * p) + "_angle_" + str(ang) + ".npy")
        Mlast = jnp.load("DMRG/mps_tensors/ES_tensor_last_" + str(L * p) + "_angle_" + str(ang) + ".npy")
    else:
        M0 = jnp.load("DMRG/mps_tensors/cluster_tensor_init_" + str(L * p) + "_angle_" + str(ang) + ".npy")
        M = jnp.load("DMRG/mps_tensors/cluster_tensor_" + str(L * p) + "_angle_" + str(ang) + ".npy")
        Mlast = jnp.load("DMRG/mps_tensors/cluster_tensor_last_" + str(L * p) + "_angle_" + str(ang) + ".npy")
    batch_log_phase_dmrg = jax.jit(vmap(log_phase_dmrg, (0, None, None, None)))
else:
    M0, M, Mlast = None, None, None
if H_type == "ES":
    (xy_loc_bulk, xy_loc_fl, xy_loc_xzz, yloc_bulk, yloc_fl, yloc_xzz, zloc_bulk, zloc_fl,
    zloc_xzz, off_diag_bulk_coe, off_diag_fl_coe, off_diag_xzz_coe, zloc_bulk_diag, zloc_fl_diag,
    zloc_xzz_diag, coe_bulk_diag, coe_fl_diag, coe_xzz_diag) = vmc_off_diag_es(L, p, angle, basis_rotation)
    batch_diag_coe = vmap(diag_coe, (0, None, None, None, None, None, None))
elif H_type == "cluster":
    (xy_loc_bulk, xy_loc_fl, yloc_bulk, yloc_fl, zloc_bulk, zloc_fl, off_diag_bulk_coe, off_diag_fl_coe,
     zloc_bulk_diag, zloc_fl_diag, coe_bulk_diag, coe_fl_diag) = vmc_off_diag_es(L, p, angle, basis_rotation)
    batch_diag_coe = vmap(diag_coe, (0, None, None, None, None))

batch_total_samples_1d = vmap(total_samples_1d, (0, None), 0)
batch_new_coe_1d = vmap(new_coe_1d, (0, None, None, None, None))


@partial(jax.jit, static_argnames=['fixed_parameters', 'dmrg'])
def compute_cost(parameters, fixed_parameters, samples, Eloc, dmrg, M0_, M_, Mlast_):
    samples = jax.lax.stop_gradient(samples)
    Eloc = jax.lax.stop_gradient(Eloc)
    log_amps_tensor = batch_log_amp(samples, parameters, fixed_parameters, setting)
    if dmrg == True:
        log_amps_tensor += batch_log_phase_dmrg(samples.reshape(samples.shape[0], -1), M0_, M_, Mlast_)
    cost = 2 * jnp.real(jnp.mean(log_amps_tensor.conjugate() * (Eloc - jnp.mean(Eloc))))
    return cost

grad_f = jax.jit(jax.grad(compute_cost), static_argnums=(1, 4))
it = len(meanEnergy)

while(it<numsteps+eval_steps):
    key, subkey = split(subkey, 2)
    key_ = split(key, numsamples)
    samples, sample_log_amp = batch_sample_prob(params, fixed_params, key_, setting)
    if dmrg == True:
        sample_log_amp += batch_log_phase_dmrg(samples, M0, M, Mlast)
    samples_grad = samples.reshape(-1, L, p)

    if H_type == "ES":
        sigmas = jnp.concatenate((batch_total_samples_1d(samples, xy_loc_bulk),
                             batch_total_samples_1d(samples, xy_loc_fl),
                             batch_total_samples_1d(samples, xy_loc_xzz)), axis=1).reshape(-1, L, p)
        matrixelements = (jnp.concatenate((batch_new_coe_1d(samples, off_diag_bulk_coe, yloc_bulk, zloc_bulk, basis_rotation),
                                     batch_new_coe_1d(samples, off_diag_fl_coe, yloc_fl, zloc_fl, basis_rotation),
                                     batch_new_coe_1d(samples, off_diag_xzz_coe, yloc_xzz, zloc_xzz, basis_rotation)), axis=1).reshape(numsamples, -1))
        diag_E = batch_diag_coe(samples, zloc_bulk_diag, zloc_fl_diag, zloc_xzz_diag, coe_bulk_diag, coe_fl_diag, coe_xzz_diag)

    else:
        sigmas = jnp.concatenate((batch_total_samples_1d(samples, xy_loc_bulk),
                                  batch_total_samples_1d(samples, xy_loc_fl)), axis=1).reshape(-1, L, p)
        matrixelements = (jnp.concatenate((batch_new_coe_1d(samples, off_diag_bulk_coe, yloc_bulk, zloc_bulk, basis_rotation),
                        batch_new_coe_1d(samples, off_diag_fl_coe, yloc_fl, zloc_fl, basis_rotation)),axis=1).reshape(numsamples, -1))
        diag_E = batch_diag_coe(samples, zloc_bulk_diag, zloc_fl_diag, coe_bulk_diag, coe_fl_diag)

    log_all_amp = batch_log_amp(sigmas, params, fixed_params, setting)
    if dmrg == True:
        log_all_amp += batch_log_phase_dmrg(sigmas.reshape(-1, L*p), M0, M, Mlast)
    log_diag_amp = jnp.repeat(sample_log_amp, (jnp.ones(numsamples)*(matrixelements.shape[1])).astype(int), axis=0)
    amp = jnp.exp(log_all_amp.ravel()-log_diag_amp).reshape(numsamples, -1)
    Eloc = jnp.sum((amp*matrixelements), axis=1) + diag_E
    meanE,  varE = jnp.mean(Eloc), jnp.var(Eloc)

    if it<numsteps:
        meanEnergy.append(complex(meanE))
        varEnergy.append(float(varE))
        grads = grad_f(params, fixed_params, samples_grad, Eloc, dmrg, M0, M, Mlast)
        if gradient_clip == True:
            grads = jax.tree.map(clip_grad, grads)
        if (it+1)%100==0 or it==0:
            print("learning_rate =", lr)
            print("Magnetization =", jnp.mean(jnp.sum(2*samples-1, axis = (1))))
            print('mean(E): {0}, varE: {1}, #samples {2}, #Step {3} \n\n'.format(meanE,varE,numsamples, it+1))

        # Update the optimizer state and the parameters
        updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
        params = optax.apply_updates(params, updates)
        if not os.path.exists('./params/'):
            os.mkdir('./params/')
        if ((it+1)%1000 == 0):
            if (model_type == "tensor_gru"):
                with open(f"params/params_model1D{model_type}_Htype{H_type}_L{L}_patch{p}_units{units}_batch{numsamples}_dmrg{dmrg}_angle{round(angle, 3)}_seed{args.seed}.pkl", "wb") as f:
                    checkpoint = {'params': params, 'optimizer_state': optimizer_state}
                    pickle.dump(checkpoint, f)
            if (model_type == "RWKV"):
                with open(f"params/params_model1D{model_type}_Htype{H_type}_L{L}_patch{p}_emb{RWKV_emb}_layer{RWKV_layer}_hidden{RWKV_hidden}_ff{RWKV_ff}_batch{numsamples}_dmrg{dmrg}_angle{round(angle, 3)}_seed{args.seed}.pkl", "wb") as f:
                    checkpoint = {'params': params, 'optimizer_state': optimizer_state}
                    pickle.dump(checkpoint, f)
            if (model_type == "TQS"):
                with open(f"params/params_model1D{model_type}_Htype{H_type}_L{L}_patch{p}_layer{TQS_layer}_units{TQS_units}_head{TQS_head}_batch{numsamples}_dmrg{dmrg}_angle{round(angle, 3)}_seed{args.seed}.pkl", "wb") as f:
                    checkpoint = {'params': params, 'optimizer_state': optimizer_state}
                    pickle.dump(checkpoint, f)

            if model_type == "tensor_gru":
                jnp.save(
                    "result/meanE_1DRNN" + "_Htype" + str(H_type) + "_L" + str(L) + "_patch" + str(p) + "_units" + str(
                        units) + "_batch" + str(numsamples) + "_dmrg" + str(dmrg) + "_seed" + str(
                        args.seed) + "_angle" + str(round(angle, 3)) + ".npy", jnp.array(meanEnergy).ravel())
                jnp.save(
                    "result/varE_1DRNN" + "_Htype" + str(H_type) + "_L" + str(L) + "_patch" + str(p) + "_units" + str(
                        units) + "_batch" + str(numsamples) + "_dmrg" + str(dmrg) + "_seed" + str(
                        args.seed) + "_angle" + str(round(angle, 3)) + ".npy", jnp.array(varEnergy).ravel())
            elif model_type == "RWKV":
                jnp.save(
                    "result/meanE_1DRWKV" + "_Htype" + str(H_type) + "_L" + str(L) + "_patch" + str(p) + "_emb" + str(
                        RWKV_emb) + "_layer" + str(RWKV_layer) + "_hidden" + str(RWKV_hidden) + "_ff" + str(
                        RWKV_ff) + "_batch" + str(numsamples) + "_dmrg" + str(dmrg) + "_seed" + str(
                        args.seed) + "_angle" + str(round(angle, 3)) + ".npy", jnp.array(meanEnergy).ravel())
                jnp.save(
                    "result/varE_1DRWKV" + "_Htype" + str(H_type) + "_L" + str(L) + "_patch" + str(p) + "_emb" + str(
                        RWKV_emb) + "_layer" + str(RWKV_layer) + "_hidden" + str(RWKV_hidden) + "_ff" + str(
                        RWKV_ff) + "_batch" + str(numsamples) + "_dmrg" + str(dmrg) + "_seed" + str(
                        args.seed) + "_angle" + str(round(angle, 3)) + ".npy", jnp.array(varEnergy).ravel())

            elif model_type == "TQS":
                jnp.save(
                    "result/meanE_1DTQS" + "_Htype" + str(H_type) + "_L" + str(L) + "_patch" + str(p) + "_layer" + str(
                        TQS_layer) + "_ff" + str(TQS_ff) + "_units" + str(TQS_units) + "_head" + str(
                        TQS_head) + "_batch" + str(numsamples) + "_dmrg" + str(dmrg) + "_seed" + str(
                        args.seed) + "_angle" + str(round(angle, 3)) + ".npy", jnp.array(meanEnergy).ravel())
                jnp.save(
                    "result/varE_1DTQS" + "_Htype" + str(H_type) + "_L" + str(L) + "_patch" + str(p) + "_layer" + str(
                        TQS_layer) + "_ff" + str(TQS_ff) + "_units" + str(TQS_units) + "_head" + str(
                        TQS_head) + "_batch" + str(numsamples) + "_dmrg" + str(dmrg) + "_seed" + str(
                        args.seed) + "_angle" + str(round(angle, 3)) + ".npy", jnp.array(varEnergy).ravel())

    else:
        evalmeanE += meanE/eval_steps
        evalvarE += varE/eval_steps**2
    it += 1
eval_meanEnergy.append(evalmeanE)
eval_varEnergy.append(evalvarE)

if model_type == "tensor_gru":
    jnp.save("result/evalmeanE_1DRNN"+"_Htype"+str(H_type)+"_L"+str(L)+"_patch"+str(p)+"_units"+str(units)+"_batch"+str(numsamples)+"_dmrg"+str(dmrg)+"_seed"+str(args.seed)+"_angle"+str(round(angle, 3))+".npy", jnp.array(eval_meanEnergy))
    jnp.save("result/evalvarE_1DRNN"+"_Htype"+str(H_type)+"_L"+str(L)+"_patch"+str(p)+"_units"+str(units)+"_batch"+str(numsamples)+"_dmrg"+str(dmrg)+"_seed"+str(args.seed)+"_angle"+str(round(angle, 3))+".npy", jnp.array(eval_varEnergy))
elif model_type == "RWKV":
    jnp.save("result/evalmeanE_1DRWKV"+"_Htype"+str(H_type)+"_L" + str(L) + "_patch" + str(p) + "_emb" + str(RWKV_emb) + "_layer"+ str(RWKV_layer) +"_hidden" + str(RWKV_hidden) + "_ff" + str(RWKV_ff) + "_batch" + str(numsamples) +"_dmrg"+str(dmrg)+ "_seed" + str(args.seed) +"_angle"+str(round(angle, 3))+ ".npy", jnp.array(eval_meanEnergy))
    jnp.save("result/evalvarE_1DRWKV" + "_Htype" + str(H_type) + "_L" + str(L) + "_patch" + str(p) + "_emb" + str(RWKV_emb) + "_layer" + str(RWKV_layer) + "_hidden" + str(RWKV_hidden) + "_ff" + str(RWKV_ff) + "_batch" + str(numsamples) +"_dmrg"+str(dmrg)+ "_seed" + str(args.seed) +"_angle"+str(round(angle, 3)) + ".npy", jnp.array(eval_varEnergy))

elif model_type == "TQS":
    jnp.save("result/evalmeanE_1DTQS"+"_Htype"+str(H_type)+"_L" + str(L) + "_patch" + str(p) + "_layer" + str(TQS_layer) + "_ff" + str(TQS_ff) + "_units" + str(TQS_units) + "_head" + str(TQS_head) + "_batch" + str(numsamples) +"_dmrg"+str(dmrg)+ "_seed" + str(args.seed)+"_angle"+str(round(angle, 3)) + ".npy", jnp.array(eval_meanEnergy))
    jnp.save("result/evalvarE_1DTQS"+"_Htype"+str(H_type)+"_L" + str(L) + "_patch" + str(p) + "_layer" + str(TQS_layer) + "_ff" + str(TQS_ff) + "_units" + str(TQS_units) + "_head" + str(TQS_head) + "_batch" + str(numsamples)+"_dmrg"+str(dmrg) + "_seed" + str(args.seed) +"_angle"+str(round(angle, 3))+ ".npy", jnp.array(eval_varEnergy))
