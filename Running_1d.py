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

jax.config.update("jax_enable_x64", False)

parser = argparse.ArgumentParser()
parser.add_argument('--L', type = int, default = 8)
parser.add_argument('--p', type = int, default = 2)
parser.add_argument('--numunits', type = int, default=32)
parser.add_argument('--lr', type = float, default=2e-4)
parser.add_argument('--gradient_clip', type = bool, default=True)
parser.add_argument('--gradient_clipvalue', type = float, default=10.0)
parser.add_argument('--numsteps', type = int, default=101)
parser.add_argument('--numsamples', type = int, default=256)
parser.add_argument('--testing_sample', type = int, default=2**15)
parser.add_argument('--seed', type = int, default=3)
parser.add_argument('--model_type', type = str, default="TQS")
parser.add_argument('--H_type', type = str, default="cluster")
parser.add_argument('--basis_rotation', type = bool, default=True)
parser.add_argument('--previous_training', type = bool, default=True)
parser.add_argument('--dmrg', type = bool, default=False)
parser.add_argument('--RWKV_layer', type = int, default=2)
parser.add_argument('--RWKV_emb', type = int, default = 16)
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
RWKV_hidden = args.RWKV_hidden
RWKV_ff = args.RWKV_ff
RWKV_emb = args.RWKV_emb
TQS_layer = args.TQS_layer
TQS_ff = args.TQS_ff
TQS_units = args.TQS_units
TQS_head = args.TQS_head
eval_steps = int(testing_sample/numsamples)
input_size = 2 ** p
key = PRNGKey(args.seed)

eval_meanEnergy=[]
eval_varEnergy=[]
N = L
angle_list = [0., 0.05*jnp.pi, 0.1*jnp.pi, 0.15*jnp.pi, 0.20*jnp.pi, 0.25*jnp.pi, 0.3*jnp.pi, 0.35*jnp.pi, 0.4*jnp.pi, 0.45*jnp.pi, 0.5*jnp.pi]
a = 0
if (model_type == "tensor_gru"):
    if previous_training == True:
        meanEnergy = jnp.load(f"result/meanE_1DRNN_Htype{H_type}_L{L}_patch{p}_units{units}_batch{numsamples}_dmrg{dmrg}_seed{args.seed}.npy").reshape(len(angle_list), -1).tolist()
        varEnergy = jnp.load(f"result/varE_1DRNN_Htype{H_type}_L{L}_patch{p}_units{units}_batch{numsamples}_dmrg{dmrg}_seed{args.seed}.npy").reshape(len(angle_list), -1).tolist()
    else:
        meanEnergy = [[] for i in range(len(angle_list))]
        varEnergy = [[] for i in range(len(angle_list))]
elif (model_type == "RWKV"):
    if previous_training == True:
        meanEnergy = jnp.load(f"result/meanE_1DRWKV_Htype{H_type}_L{L}_patch{p}_emb{RWKV_emb}_layer{RWKV_layer}_hidden{RWKV_hidden}_ff{RWKV_ff}_batch{numsamples}_dmrg{dmrg}_seed{args.seed}.npy").reshape(len(angle_list), -1).tolist()
        varEnergy = jnp.load(f"result/varE_1DRWKV_Htype{H_type}_L{L}_patch{p}_emb{RWKV_emb}_layer{RWKV_layer}_hidden{RWKV_hidden}_ff{RWKV_ff}_batch{numsamples}_dmrg{dmrg}_seed{args.seed}.npy").reshape(len(angle_list), -1).tolist()
    else:
        meanEnergy = [[] for i in range(len(angle_list))]
        varEnergy = [[] for i in range(len(angle_list))]
elif (model_type == "TQS"):
    if previous_training == True:
        meanEnergy = jnp.load("result/meanE_1DTQS"+"_Htype"+str(H_type)+"_L" + str(L) + "_patch" + str(p) + "_layer" + str(TQS_layer) + "_ff" + str(TQS_ff) + "_units" + str(TQS_units) + "_head" + str(TQS_head) + "_batch" + str(numsamples) +"_dmrg"+str(dmrg)+ "_seed" + str(args.seed) + ".npy").reshape(len(angle_list), -1).tolist()
        varEnergy = jnp.load("result/varE_1DTQS"+"_Htype"+str(H_type)+"_L" + str(L) + "_patch" + str(p) + "_layer" + str(TQS_layer) + "_ff" + str(TQS_ff) + "_units" + str(TQS_units) + "_head" + str(TQS_head) + "_batch" + str(numsamples) +"_dmrg"+str(dmrg)+ "_seed" + str(args.seed) + ".npy").reshape(len(angle_list), -1).tolist()
    else:
        meanEnergy = [[] for i in range(len(angle_list))]
        varEnergy = [[] for i in range(len(angle_list))]

for angle in (angle_list):
    evalmeanE = 0
    evalvarE = 0
    # x and y are the cosine and sine of the rotation angle
    x, y = jnp.cos(angle), jnp.sin(angle)
    key, subkey = split(key, 2)
    if (model_type == "tensor_gru"):
        if previous_training == True:
            with open(f"params/params_model1D{model_type}_Htype{H_type}_L{L}_patch{p}_units{units}_batch{numsamples}_dmrg{dmrg}_angle{angle}_seed{args.seed}.pkl", "rb") as f:
                params = pickle.load(f)
        else:
            params = init_tensor_gru_params(input_size, units, L, key)
        fixed_params = N, p, units
        batch_sample_prob = jax.jit(vmap(sample_prob, (None, None, 0, None)), static_argnames=['fixed_params'])
        batch_log_amp = jax.jit(vmap(log_amp, (0, None, None, None)), static_argnames=['fixed_params'])

    elif (model_type == "RWKV"):
        if previous_training == True:
            with open(f"params/params_model1D{model_type}_Htype{H_type}_L{L}_patch{p}_emb{RWKV_emb}_layer{RWKV_layer}_hidden{RWKV_hidden}_ff{RWKV_ff}_batch{numsamples}_dmrg{dmrg}_angle{angle}_seed{args.seed}.pkl", "rb") as f:
                params = pickle.load(f)

        else:
            params = init_RWKV_params(RWKV_emb, RWKV_hidden, RWKV_layer, RWKV_ff, input_size, L, key)

        fixed_params = N, p, RWKV_hidden, RWKV_layer, RWKV_emb
        batch_sample_prob = jax.jit(vmap(sample_prob_RWKV, (None, None, 0, None)), static_argnames=['fixed_params'])
        batch_log_amp = jax.jit(vmap(log_amp_RWKV, (0, None, None, None)), static_argnames=['fixed_params'])

    elif (model_type == "TQS"):
        if previous_training == True:
            with open(f"params/params_model1D{model_type}_Htype{H_type}_L{L}_patch{p}_layer{TQS_layer}_units{TQS_units}_head{TQS_head}_batch{numsamples}_dmrg{dmrg}_angle{angle}_seed{args.seed}.pkl","rb") as f:
                params = pickle.load(f)
        else:
            params = init_TQS_params(input_size, TQS_layer, TQS_ff, TQS_units, TQS_head, key)

        fixed_params = N, p, TQS_layer
        batch_sample_prob = jax.jit(vmap(sample_prob_TQS, (None, None, 0, None)), static_argnames=['fixed_params'])
        batch_log_amp = jax.jit(vmap(log_amp_TQS, (0, None, None, None)), static_argnames=['fixed_params'])

    optimizer = optax.adam(learning_rate=lr)
    optimizer_state = optimizer.init(params)

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
        batch_log_phase_dmrg = jax.jit(vmap(log_phase_dmrg, (0, None, None, None)))
    else:
        M0, M, Mlast = None, None, None
    if H_type == "ES":
        (xy_loc_bulk, xy_loc_fl, xy_loc_xzz, yloc_bulk, yloc_fl, yloc_xzz, zloc_bulk, zloc_fl,
        zloc_xzz, off_diag_bulk_coe, off_diag_fl_coe, off_diag_xzz_coe, zloc_bulk_diag, zloc_fl_diag,
        zloc_xzz_diag, coe_bulk_diag, coe_fl_diag, coe_xzz_diag) = vmc_off_diag_es(N, p, angle, basis_rotation)
        batch_diag_coe = vmap(diag_coe, (0, None, None, None, None, None, None))
    elif H_type == "cluster":
        (xy_loc_bulk, xy_loc_fl, yloc_bulk, yloc_fl, zloc_bulk, zloc_fl, off_diag_bulk_coe, off_diag_fl_coe,
         zloc_bulk_diag, zloc_fl_diag, coe_bulk_diag, coe_fl_diag) = vmc_off_diag_es(N, p, angle, basis_rotation)
        batch_diag_coe = vmap(diag_coe, (0, None, None, None, None))

    batch_total_samples_1d = vmap(total_samples_1d, (0, None), 0)
    batch_new_coe_1d = vmap(new_coe_1d, (0, None, None, None, None))


    @partial(jax.jit, static_argnames=['fixed_parameters', 'dmrg'])
    def compute_cost(parameters, fixed_parameters, samples, Eloc, dmrg, M0_, M_, Mlast_):
        samples = jax.lax.stop_gradient(samples)
        Eloc = jax.lax.stop_gradient(Eloc)
        log_amps_tensor = batch_log_amp(samples, parameters, fixed_parameters, dmrg)
        if dmrg == True:
            log_amps_tensor += batch_log_phase_dmrg(samples.reshape(samples.shape[0], -1), M0_, M_, Mlast_)
        cost = 2 * jnp.real(jnp.mean(log_amps_tensor.conjugate() * (Eloc - jnp.mean(Eloc))))
        return cost

    grad_f = jax.jit(jax.grad(compute_cost), static_argnums=(1, 4))

    for it in range(numsteps+eval_steps):
        key, subkey = split(subkey, 2)
        key_ = split(key, numsamples)
        samples, sample_log_amp = batch_sample_prob(params, fixed_params, key_, dmrg)
        samples = samples.reshape(-1, N * p)
        if dmrg == True:
            sample_log_amp += batch_log_phase_dmrg(samples, M0, M, Mlast)
        samples_grad = samples.reshape(-1, N, p)

        if H_type == "ES":
            sigmas = jnp.concatenate((batch_total_samples_1d(samples, xy_loc_bulk),
                                 batch_total_samples_1d(samples, xy_loc_fl),
                                 batch_total_samples_1d(samples, xy_loc_xzz)), axis=1).reshape(-1, N, p)
            matrixelements = (jnp.concatenate((batch_new_coe_1d(samples, off_diag_bulk_coe, yloc_bulk, zloc_bulk, basis_rotation),
                                         batch_new_coe_1d(samples, off_diag_fl_coe, yloc_fl, zloc_fl, basis_rotation),
                                         batch_new_coe_1d(samples, off_diag_xzz_coe, yloc_xzz, zloc_xzz, basis_rotation)), axis=1).reshape(numsamples, -1))
            diag_E = batch_diag_coe(samples, zloc_bulk_diag, zloc_fl_diag, zloc_xzz_diag, coe_bulk_diag, coe_fl_diag, coe_xzz_diag)

        else:
            sigmas = jnp.concatenate((batch_total_samples_1d(samples, xy_loc_bulk),
                                      batch_total_samples_1d(samples, xy_loc_fl)), axis=1).reshape(-1, N, p)
            matrixelements = (jnp.concatenate((batch_new_coe_1d(samples, off_diag_bulk_coe, yloc_bulk, zloc_bulk, basis_rotation),
                            batch_new_coe_1d(samples, off_diag_fl_coe, yloc_fl, zloc_fl, basis_rotation)),axis=1).reshape(numsamples, -1))
            diag_E = batch_diag_coe(samples, zloc_bulk_diag, zloc_fl_diag, coe_bulk_diag, coe_fl_diag)

        log_all_amp = batch_log_amp(sigmas, params, fixed_params, dmrg)
        if dmrg == True:
            log_all_amp += batch_log_phase_dmrg(sigmas.reshape(-1, L*p), M0, M, Mlast)
        log_diag_amp = jnp.repeat(sample_log_amp, (jnp.ones(numsamples)*(matrixelements.shape[1])).astype(int), axis=0)
        amp = jnp.exp(log_all_amp.ravel()-log_diag_amp).reshape(numsamples, -1)
        Eloc = jnp.sum((amp*matrixelements), axis=1) + diag_E
        meanE,  varE = jnp.mean(Eloc), jnp.var(Eloc)

        if it<numsteps:
            meanEnergy[a].append(complex(meanE))
            varEnergy[a].append(float(varE))
            if (it+1)%50==0 or it==0:
                print("learning_rate =", lr)
                print("Magnetization =", jnp.mean(jnp.sum(2*samples-1, axis = (1))))
                print('mean(E): {0}, varE: {1}, #samples {2}, #Step {3} \n\n'.format(meanE,varE,numsamples, it+1))

            grads = grad_f(params, fixed_params, samples_grad, Eloc, dmrg, M0, M, Mlast)
            if gradient_clip == True:
                grads = jax.tree.map(clip_grad, grads)

            # Update the optimizer state and the parameters
            updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
            params = optax.apply_updates(params, updates)
            if not os.path.exists('./params/'):
                os.mkdir('./params/')
            if (it%100 == 0):
                params_dict = jax.tree_util.tree_leaves(params)
                if (model_type == "tensor_gru"):
                    with open(f"params/params_model1D{model_type}_Htype{H_type}_L{L}_patch{p}_units{units}_batch{numsamples}_dmrg{dmrg}_angle{angle}_seed{args.seed}.pkl", "wb") as f:
                        pickle.dump(params_dict, f)
                if (model_type == "RWKV"):
                    with open(f"params/params_model1D{model_type}_Htype{H_type}_L{L}_patch{p}_emb{RWKV_emb}_layer{RWKV_layer}_hidden{RWKV_hidden}_ff{RWKV_ff}_batch{numsamples}_dmrg{dmrg}_angle{angle}_seed{args.seed}.pkl", "wb") as f:
                        pickle.dump(params_dict, f)
                if (model_type == "TQS"):
                    with open(f"params/params_model1D{model_type}_Htype{H_type}_L{L}_patch{p}_layer{TQS_layer}_units{TQS_units}_head{TQS_head}_batch{numsamples}_dmrg{dmrg}_angle{angle}_seed{args.seed}.pkl", "wb") as f:
                        pickle.dump(params_dict, f)
        else:
            evalmeanE += meanE/eval_steps
            evalvarE += varE/eval_steps**2
    eval_meanEnergy.append(evalmeanE)
    eval_varEnergy.append(evalvarE)
    a += 1

if not os.path.exists('./result/'):
    os.mkdir('./result/')
if model_type == "tensor_gru":
    jnp.save("result/meanE_1DRNN"+"_Htype"+str(H_type)+"_L"+str(L)+"_patch"+str(p)+"_units"+str(units)+"_batch"+str(numsamples)+"_dmrg"+str(dmrg)+"_seed"+str(args.seed)+".npy", jnp.array(meanEnergy).ravel())
    jnp.save("result/varE_1DRNN"+"_Htype"+str(H_type)+"_L"+str(L)+"_patch"+str(p)+"_units"+str(units)+"_batch"+str(numsamples)+"_dmrg"+str(dmrg)+"_seed"+str(args.seed)+".npy", jnp.array(varEnergy).ravel())
    jnp.save("result/evalmeanE_1DRNN"+"_Htype"+str(H_type)+"_L"+str(L)+"_patch"+str(p)+"_units"+str(units)+"_batch"+str(numsamples)+"_dmrg"+str(dmrg)+"_seed"+str(args.seed)+".npy", jnp.array(eval_meanEnergy))
    jnp.save("result/evalvarE_1DRNN"+"_Htype"+str(H_type)+"_L"+str(L)+"_patch"+str(p)+"_units"+str(units)+"_batch"+str(numsamples)+"_dmrg"+str(dmrg)+"_seed"+str(args.seed)+".npy", jnp.array(eval_varEnergy))
elif model_type == "RWKV":
    jnp.save("result/meanE_1DRWKV"+"_Htype"+str(H_type)+"_L" + str(L) + "_patch" + str(p) + "_emb" + str(RWKV_emb) + "_layer"+ str(RWKV_layer) +"_hidden" + str(RWKV_hidden) + "_ff" + str(RWKV_ff) + "_batch" + str(numsamples) +"_dmrg"+str(dmrg)+ "_seed" + str(args.seed) + ".npy", jnp.array(meanEnergy).ravel())
    jnp.save("result/varE_1DRWKV"+"_Htype"+str(H_type)+"_L" + str(L) + "_patch" + str(p) + "_emb" + str(RWKV_emb) + "_layer"+ str(RWKV_layer) +"_hidden" + str(RWKV_hidden) + "_ff" + str(RWKV_ff) + "_batch" + str(numsamples) +"_dmrg"+str(dmrg)+ "_seed" + str(args.seed) + ".npy", jnp.array(varEnergy).ravel())
    jnp.save("result/evalmeanE_1DRWKV"+"_Htype"+str(H_type)+"_L" + str(L) + "_patch" + str(p) + "_emb" + str(RWKV_emb) + "_layer"+ str(RWKV_layer) +"_hidden" + str(RWKV_hidden) + "_ff" + str(RWKV_ff) + "_batch" + str(numsamples) +"_dmrg"+str(dmrg)+ "_seed" + str(args.seed) + ".npy", jnp.array(eval_meanEnergy))
    jnp.save("result/evalvarE_1DRWKV" + "_Htype" + str(H_type) + "_L" + str(L) + "_patch" + str(p) + "_emb" + str(RWKV_emb) + "_layer" + str(RWKV_layer) + "_hidden" + str(RWKV_hidden) + "_ff" + str(RWKV_ff) + "_batch" + str(numsamples) +"_dmrg"+str(dmrg)+ "_seed" + str(args.seed) + ".npy", jnp.array(eval_varEnergy))

elif model_type == "TQS":
    jnp.save("result/meanE_1DTQS"+"_Htype"+str(H_type)+"_L" + str(L) + "_patch" + str(p) + "_layer" + str(TQS_layer) + "_ff" + str(TQS_ff) + "_units" + str(TQS_units) + "_head" + str(TQS_head) + "_batch" + str(numsamples) +"_dmrg"+str(dmrg)+ "_seed" + str(args.seed) + ".npy", jnp.array(meanEnergy).ravel())
    jnp.save("result/varE_1DTQS"+"_Htype"+str(H_type)+"_L" + str(L) + "_patch" + str(p) + "_layer" + str(TQS_layer) + "_ff" + str(TQS_ff) + "_units" + str(TQS_units) + "_head" + str(TQS_head) + "_batch" + str(numsamples) +"_dmrg"+str(dmrg)+ "_seed" + str(args.seed) + ".npy", jnp.array(varEnergy).ravel())
    jnp.save("result/evalmeanE_1DTQS"+"_Htype"+str(H_type)+"_L" + str(L) + "_patch" + str(p) + "_layer" + str(TQS_layer) + "_ff" + str(TQS_ff) + "_units" + str(TQS_units) + "_head" + str(TQS_head) + "_batch" + str(numsamples) +"_dmrg"+str(dmrg)+ "_seed" + str(args.seed) + ".npy", jnp.array(eval_meanEnergy))
    jnp.save("result/evalvarE_1DTQS"+"_Htype"+str(H_type)+"_L" + str(L) + "_patch" + str(p) + "_layer" + str(TQS_layer) + "_ff" + str(TQS_ff) + "_units" + str(TQS_units) + "_head" + str(TQS_head) + "_batch" + str(numsamples)+"_dmrg"+str(dmrg) + "_seed" + str(args.seed) + ".npy", jnp.array(eval_varEnergy))