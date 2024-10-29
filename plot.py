import json
import jax.numpy as jnp
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt

L = 64
p = 1
units = 128
numsamples = 256
dmrg = False
seed = 3
angle_list = [0.0,  0.157, 0.314, 0.471, 0.628, 0.785, 0.942, 1.1, 1.257, 1.414, 1.571]
def moving_average(data, window_size = 50):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

meanE_ES = []
meanE_ES_p2 = []
meanE_ES_p4 = []
meanE_ES_dmrg = []
meanE_ES_dmrg_p2 = []
meanE_ES_dmrg_p4 = []
meanE_cluster = []
meanE_cluster_p2 = []
meanE_cluster_p4 = []
evalE_ES = []
evalE_ES_p2 = []
evalE_ES_p4 = []
evalE_ES_dmrg = []
evalE_ES_dmrg_p2 = []
evalE_ES_dmrg_p4 = []
evalE_cluster = []
evalE_cluster_p2 = []
evalE_cluster_p4 = []
meanE_ES_TQS = []
meanE_ES_TQS_p2 = []
meanE_ES_TQS_p4 = []
meanE_ES_TQS_dmrg = []
meanE_ES_TQS_dmrg_p2 = []
meanE_ES_TQS_dmrg_p4 = []
meanE_cluster_TQS = []
meanE_cluster_TQS_p2 = []
meanE_cluster_TQS_p4 = []
evalE_ES_TQS = []
evalE_ES_TQS_p2 = []
evalE_ES_TQS_p4 = []
evalE_ES_TQS_dmrg = []
evalE_ES_TQS_dmrg_p2 = []
evalE_ES_TQS_dmrg_p4 = []
evalE_cluster_TQS = []
evalE_cluster_TQS_p2 = []
evalE_cluster_TQS_p4 = []
meanE_gf =[]
meanE_gf_p2 = []
varE_gf = []
varE_gf_p2 = []
evalE_gf = []
evalE_gf_p2 = []
meanE_gf_tqs_p2 = []

for i in angle_list:
    meanE_ES.append(np.load("result/oneDRNN/meanE_1DRNN_HtypeES_L64_patch1_units128_batch256_dmrgFalse_seed3_angle" + str(i) + ".npy").ravel().tolist())
    meanE_ES_p2.append(np.load("result/oneDRNN/meanE_1DRNN_HtypeES_L32_patch2_units64_batch256_dmrgFalse_seed3_angle" + str(i) + ".npy").ravel().tolist())
    meanE_ES_p4.append(np.load("result/oneDRNN/meanE_1DRNN_HtypeES_L16_patch4_units64_batch256_dmrgFalse_seed3_angle" + str(i) + ".npy").ravel().tolist())
    meanE_ES_dmrg.append(np.load( "result/oneDRNN/meanE_1DRNN_HtypeES_L64_patch1_units128_batch256_dmrgTrue_seed3_angle" + str(i) + ".npy").ravel().tolist())
    meanE_ES_dmrg_p2.append(np.load("result/oneDRNN/meanE_1DRNN_HtypeES_L32_patch2_units64_batch256_dmrgTrue_seed3_angle" + str(i) + ".npy").ravel().tolist())
    meanE_ES_dmrg_p4.append(np.load("result/oneDRNN/meanE_1DRNN_HtypeES_L16_patch4_units64_batch256_dmrgTrue_seed3_angle" + str(i) + ".npy").ravel().tolist())
    meanE_cluster.append(np.load("result/oneDRNN/meanE_1DRNN_Htypecluster_L64_patch1_units128_batch256_dmrgFalse_seed3_angle" + str(i) + ".npy").ravel().tolist())
    meanE_cluster_p2.append(np.load("result/oneDRNN/meanE_1DRNN_Htypecluster_L32_patch2_units64_batch256_dmrgFalse_seed3_angle" + str(i) + ".npy").ravel().tolist())
    meanE_cluster_p4.append(np.load("result/oneDRNN/meanE_1DRNN_Htypecluster_L16_patch4_units64_batch256_dmrgFalse_seed3_angle" + str(i) + ".npy").ravel().tolist())
    evalE_ES.append(np.load("result/oneDRNN/evalmeanE_1DRNN_HtypeES_L64_patch1_units128_batch256_dmrgFalse_seed3_angle" + str(i) + ".npy").ravel().tolist())
    evalE_ES_p2.append(np.load("result/oneDRNN/evalmeanE_1DRNN_HtypeES_L32_patch2_units64_batch256_dmrgFalse_seed3_angle" + str(i) + ".npy").ravel().tolist())
    evalE_ES_p4.append(np.load("result/oneDRNN/evalmeanE_1DRNN_HtypeES_L16_patch4_units64_batch256_dmrgFalse_seed3_angle" + str(i) + ".npy").ravel().tolist())
    evalE_ES_dmrg.append(np.load("result/oneDRNN/evalmeanE_1DRNN_HtypeES_L64_patch1_units128_batch256_dmrgTrue_seed3_angle" + str(i) + ".npy").ravel().tolist())
    evalE_ES_dmrg_p2.append(np.load( "result/oneDRNN/evalmeanE_1DRNN_HtypeES_L32_patch2_units64_batch256_dmrgTrue_seed3_angle" + str(i) + ".npy").ravel().tolist())
    evalE_ES_dmrg_p4.append(np.load("result/oneDRNN/evalmeanE_1DRNN_HtypeES_L16_patch4_units64_batch256_dmrgTrue_seed3_angle" + str(i) + ".npy").ravel().tolist())
    evalE_cluster.append(np.load("result/oneDRNN/evalmeanE_1DRNN_Htypecluster_L64_patch1_units128_batch256_dmrgFalse_seed3_angle" + str(i) + ".npy").ravel().tolist())
    evalE_cluster_p2.append(np.load("result/oneDRNN/evalmeanE_1DRNN_Htypecluster_L32_patch2_units64_batch256_dmrgFalse_seed3_angle" + str(i) + ".npy").ravel().tolist())
    evalE_cluster_p4.append(np.load("result/oneDRNN/evalmeanE_1DRNN_Htypecluster_L16_patch4_units64_batch256_dmrgFalse_seed3_angle" + str(i) + ".npy").ravel().tolist())
    meanE_ES_TQS.append(np.load(f"result/oneDTQS/meanE_1DTQS_HtypeES_L64_patch1_layer2_ff512_units32_head4_batch256_dmrgFalse_seed3_angle{i}.npy").tolist())
    meanE_ES_TQS_p2.append(np.load(f"result/oneDTQS/meanE_1DTQS_HtypeES_L32_patch2_layer2_ff1024_units128_head4_batch256_dmrgFalse_seed3_angle{i}.npy").tolist())
    meanE_ES_TQS_p4.append(np.load(f"result/oneDTQS/meanE_1DTQS_HtypeES_L16_patch4_layer2_ff1024_units128_head4_batch256_dmrgFalse_seed3_angle{i}.npy").tolist())
    meanE_ES_TQS_dmrg.append(np.load(f"result/oneDTQS/meanE_1DTQS_HtypeES_L64_patch1_layer2_ff512_units32_head4_batch256_dmrgTrue_seed3_angle{i}.npy").tolist())
    meanE_ES_TQS_dmrg_p2.append(np.load(f"result/oneDTQS/meanE_1DTQS_HtypeES_L32_patch2_layer2_ff1024_units128_head4_batch256_dmrgTrue_seed3_angle{i}.npy").tolist())
    meanE_ES_TQS_dmrg_p4.append(np.load(f"result/oneDTQS/meanE_1DTQS_HtypeES_L16_patch4_layer2_ff1024_units128_head4_batch256_dmrgTrue_seed3_angle{i}.npy").tolist())
    meanE_cluster_TQS.append(np.load(f"result/oneDTQS/meanE_1DTQS_Htypecluster_L64_patch1_layer2_ff512_units32_head4_batch256_dmrgFalse_seed3_angle{i}.npy").tolist())
    meanE_cluster_TQS_p2.append(np.load(f"result/oneDTQS/meanE_1DTQS_Htypecluster_L32_patch2_layer2_ff1024_units128_head4_batch256_dmrgFalse_seed3_angle{i}.npy").tolist())
    meanE_cluster_TQS_p4.append(np.load(f"result/oneDTQS/meanE_1DTQS_Htypecluster_L16_patch4_layer2_ff1024_units128_head4_batch256_dmrgFalse_seed3_angle{i}.npy")[:36000].tolist())
    evalE_ES_TQS.append(np.load(f"result/oneDTQS/evalmeanE_1DTQS_HtypeES_L64_patch1_layer2_ff512_units32_head4_batch256_dmrgFalse_seed3_angle{i}.npy").tolist())
    evalE_ES_TQS_p2.append(np.load(f"result/oneDTQS/evalmeanE_1DTQS_HtypeES_L32_patch2_layer2_ff1024_units128_head4_batch256_dmrgFalse_seed3_angle{i}.npy").tolist())
    evalE_ES_TQS_p4.append(np.load(f"result/oneDTQS/evalmeanE_1DTQS_HtypeES_L16_patch4_layer2_ff1024_units128_head4_batch256_dmrgFalse_seed3_angle{i}.npy").tolist())
    evalE_ES_TQS_dmrg.append(np.load(f"result/oneDTQS/evalmeanE_1DTQS_HtypeES_L64_patch1_layer2_ff512_units32_head4_batch256_dmrgTrue_seed3_angle{i}.npy").tolist())
    evalE_ES_TQS_dmrg_p2.append(np.load(f"result/oneDTQS/evalmeanE_1DTQS_HtypeES_L32_patch2_layer2_ff1024_units128_head4_batch256_dmrgTrue_seed3_angle{i}.npy").tolist())
    evalE_ES_TQS_dmrg_p4.append(np.load(f"result/oneDTQS/evalmeanE_1DTQS_HtypeES_L16_patch4_layer2_ff1024_units128_head4_batch256_dmrgTrue_seed3_angle{i}.npy").tolist())
    evalE_cluster_TQS.append(np.load(f"result/oneDTQS/evalmeanE_1DTQS_Htypecluster_L64_patch1_layer2_ff512_units32_head4_batch256_dmrgFalse_seed3_angle{i}.npy").tolist())
    evalE_cluster_TQS_p2.append(np.load(f"result/oneDTQS/evalmeanE_1DTQS_Htypecluster_L32_patch2_layer2_ff1024_units128_head4_batch256_dmrgFalse_seed3_angle{i}.npy").tolist())
    evalE_cluster_TQS_p4.append(np.load(f"result/oneDTQS/evalmeanE_1DTQS_Htypecluster_L16_patch4_layer2_ff1024_units128_head4_batch256_dmrgFalse_seed3_angle{i}.npy").tolist())
    meanE_gf.append(np.load("result/twoD/meanE_2DRNN_L8_patch1_angle"+str(i)+"_units64_batch256_seed3.npy").ravel().tolist())
    varE_gf.append(np.load("result/twoD/varE_2DRNN_L8_patch1_angle"+str(i)+"_units64_batch256_seed3.npy").ravel().tolist())
    meanE_gf_p2.append(np.load("result/twoD/meanE_2DRNN_L4_patch2_angle"+str(i)+"_units64_batch256_seed3.npy").ravel().tolist())
    varE_gf_p2.append(np.load("result/twoD/varE_2DRNN_L4_patch2_angle"+str(i)+"_units64_batch256_seed3.npy").ravel().tolist())
    meanE_gf_tqs_p2.append(np.load("result/twoD/meanE_2DTQS_L4_patch2_angle"+str(i)+"_layer2_ff1024_units128_head4_batch256_seed3.npy").ravel().tolist())

# Training curves tensor-gru patch 1
plt.figure(figsize=(24, 8))
plt.plot(np.arange(len(np.array(meanE_ES).ravel())), np.array(meanE_ES).ravel().real, label = "ES")
plt.plot(np.arange(len(np.array(meanE_ES_dmrg).ravel())), np.array(meanE_ES_dmrg).ravel().real, label = "ES_dmrg")
plt.plot(np.arange(len(np.array(meanE_cluster).ravel())), np.flip(np.array(meanE_cluster).reshape(12, -1), axis = 0) .ravel().real, label = "cluster")
for i in range(1, 12):
    plt.axvline(x=i*150000, color="gray", linestyle="--")
plt.ylim(-64, -48)
plt.ylabel("Energy")
plt.xlabel("Training Step")
plt.title("Energy vs Training Step by tensor_gru (L = 64, units = 128, batch = 256)")
plt.legend()
plt.show()
plt.clf()
# Training curves tensor-gru patch 2
plt.figure(figsize=(24, 8))
plt.plot(np.arange(len(np.array(meanE_ES_p2).ravel())), moving_average(np.array(meanE_ES_p2).ravel().real), label = "ES")
plt.plot(np.arange(len(np.array(meanE_ES_dmrg_p2).ravel())), moving_average(np.array(meanE_ES_dmrg_p2).ravel().real), label = "ES_dmrg")
plt.plot(np.arange(len(np.array(meanE_cluster_p2).ravel())), moving_average(np.flip(np.array(meanE_cluster_p2).reshape(11, -1), axis = 0).ravel().real), label = "cluster")
for i in range(0, 12):
    plt.axvline(x=i*100000, color="gray", linestyle="--")
plt.ylim(-64, -40)
plt.ylabel("Energy")
plt.xlabel("Training Step")
plt.title("Energy vs Steps by tensor_gru (moving_avg = 50, L = 32, units = 64, batch = 256, patch = 2)")
plt.legend()
plt.show()
plt.clf()
# Training curves 1Dtensor-gru patch 4
plt.figure(figsize=(24, 8))
plt.plot(np.arange(len(np.array(meanE_ES_p4).ravel())), moving_average(np.array(meanE_ES_p4).ravel().real), label = "ES")
plt.plot(np.arange(len(np.array(meanE_ES_dmrg_p4).ravel())), moving_average(np.array(meanE_ES_dmrg_p4).ravel().real), label = "ES_dmrg")
plt.plot(np.arange(len(np.array(meanE_cluster_p4).ravel())), moving_average(np.flip(np.array(meanE_cluster_p4).reshape(11, -1), axis = 0).ravel().real), label = "cluster")
for i in range(0, 12):
    plt.axvline(x=i*100000, color="gray", linestyle="--")
plt.ylim(-64, -40)
plt.ylabel("Energy")
plt.xlabel("Training Step")
plt.title("Energy vs Steps by tensor_gru (moving_avg = 50, L = 16, units = 64, batch = 256, patch = 4)")
plt.legend()
plt.show()
plt.clf()
# Training curves 2Dtensor-gru patch 1*1
plt.figure(figsize=(24, 8))
plt.plot(np.arange(len(np.array(meanE_gf).ravel())), moving_average(np.array(meanE_gf).ravel().real, 50), label = "patch=1")
for i in range(0, 12):
    plt.axvline(x=i*80000, color="gray", linestyle="--")
plt.ylim(-64, -40)
plt.ylabel("Energy")
plt.xlabel("Training Step")
plt.title("Energy vs Steps by 2DRNN (moving_avg = 50, L = 8, units = 64, batch = 256, patch = 1)")
plt.legend()
plt.show()
plt.clf()
# Training curves 2Dtensor-gru patch 2*2
plt.figure(figsize=(24, 8))
plt.plot(np.arange(len(np.array(meanE_gf_p2).ravel())), moving_average(np.array(meanE_gf_p2).ravel().real, 50), label = "2DRNN")
for i in range(0, 12):
    plt.axvline(x=i*60000, color="gray", linestyle="--")
plt.ylim(-64, -40)
plt.ylabel("Energy")
plt.xlabel("Training Step")
plt.title("Energy vs Steps by 2DRNN (moving_avg = 50, L = 4, units = 64, batch = 256, patch = 2)")
plt.legend()
plt.show()
plt.clf()
# Training curves 1DTQS patch 1
plt.figure(figsize=(24, 8))
plt.plot(np.arange(len(np.array(meanE_ES_TQS).ravel())), moving_average(np.array(meanE_ES_TQS).ravel().real), label = "ES")
plt.plot(np.arange(len(np.array(meanE_ES_TQS_dmrg).ravel())), moving_average(np.array(meanE_ES_TQS_dmrg).ravel().real), label = "ES_dmrg")
plt.plot(np.arange(len(np.array(meanE_cluster_TQS).ravel())), moving_average(np.flip(np.array(meanE_cluster_TQS), 0).ravel().real), label = "cluster")
for i in range(0, 12):
    plt.axvline(x=i*12000, color="gray", linestyle="--")
plt.ylim(-64, -30)
plt.ylabel("Energy")
plt.xlabel("Training Step")
plt.title("Energy vs Steps by 1DTQS (moving_avg = 50, L = 64, batch = 256, patch = 1)")
plt.legend()
plt.show()
plt.clf()
# Training curves 1DTQS patch 2
plt.figure(figsize=(24, 8))
plt.plot(np.arange(len(np.array(meanE_ES_TQS_p2).ravel())), moving_average(np.array(meanE_ES_TQS_p2).ravel().real), label = "ES")
plt.plot(np.arange(len(np.array(meanE_ES_TQS_dmrg_p2).ravel())), moving_average(np.array(meanE_ES_TQS_dmrg_p2).ravel().real), label = "ES_dmrg")
plt.plot(np.arange(len(np.array(meanE_cluster_TQS_p2).ravel())), moving_average(np.flip(np.array(meanE_cluster_TQS_p2), 0).ravel().real), label = "cluster")
for i in range(0, 12):
    plt.axvline(x=i*36000, color="gray", linestyle="--")
plt.ylim(-64, -30)
plt.ylabel("Energy")
plt.xlabel("Training Step")
plt.title("Energy vs Steps by 1DTQS (moving_avg = 50, L = 32,  batch = 256, patch = 2)")
plt.legend()
plt.show()
plt.clf()
# Training curves 1DTQS patch 4
plt.figure(figsize=(24, 8))
plt.plot(np.arange(len(np.array(meanE_ES_TQS_p4).ravel())), moving_average(np.array(meanE_ES_TQS_p4).ravel().real), label = "ES")
plt.plot(np.arange(len(np.array(meanE_ES_TQS_dmrg_p4).ravel())), moving_average(np.array(meanE_ES_TQS_dmrg_p4).ravel().real), label = "ES_dmrg")
plt.plot(np.arange(len(np.array(meanE_cluster_TQS_p4).ravel())), moving_average(np.flip(np.array(meanE_cluster_TQS_p4), 0).ravel().real), label = "cluster")
for i in range(0, 12):
    plt.axvline(x=i*36000, color="gray", linestyle="--")
plt.ylim(-64, -30)
plt.ylabel("Energy")
plt.xlabel("Training Step")
plt.title("Energy vs Steps by 1DTQS (moving_avg = 50, L = 16, batch = 256, patch = 4)")
plt.legend()
plt.show()
plt.clf()

# Error plot 1Dtensor-gru patch 1
plt.scatter(np.array(angle_list), ((np.array(evalE_ES)+64).real/64*100).ravel(), label = "ES")
plt.scatter(np.array(angle_list), ((np.array(evalE_ES_dmrg)+64).real/64*100).ravel(), label = "ES_dmrg")
plt.scatter(np.array(angle_list), ((np.flip(np.array(evalE_cluster))+64).real/64*100).ravel(), label = "cluster")
plt.xlabel("rotation angle")
plt.ylabel("Energy Error (%)")
plt.title("Energy Error (%) vs Rotation Angle by tensor_gru (L = 64, units = 128, batch = 256 by eval)")
plt.yscale("log")
plt.ylim(1e-5, 5e1)
plt.legend()
plt.savefig("figure/energy_error_tensor_gru_L64_units64_batch256_eval.png")
plt.show()
plt.clf()

plt.scatter(np.array(angle_list), ((np.min(moving_average(np.array(meanE_ES).ravel().real).reshape(11, -1), axis=1)+64)/64*100).ravel(), label = "ES")
plt.scatter(np.array(angle_list), ((np.min(moving_average(np.array(meanE_ES_dmrg).ravel().real).reshape(11, -1), axis=1)+64)/64*100).ravel(),  label = "ES_dmrg")
plt.scatter(np.array(angle_list), (np.flip(np.min(moving_average(np.array(meanE_cluster).ravel().real).reshape(11, -1), axis=1)+64)/64*100).ravel(),  label = "cluster")
plt.xlabel("rotation angle")
plt.ylabel("Energy Error (%)")
plt.title("Energy Error (%) vs Rotation Angle by tensor_gru (L = 64, units = 64, batch = 256 by mvag)")
plt.yscale("log")
plt.ylim(1e-5, 5e1)
plt.legend()
plt.savefig("figure/energy_error_tensor_gru_L64_units64_batch256_mvag.png")
plt.show()
plt.clf()
# Error plot 1Dtensor-gru patch 2
plt.scatter(np.array(angle_list), ((np.array(evalE_ES_p2)+64).real/64*100).ravel(), label = "ES")
plt.scatter(np.array(angle_list), ((np.array(evalE_ES_dmrg_p2)+64).real/64*100).ravel(), label = "ES_dmrg")
plt.scatter(np.array(angle_list), ((np.flip(np.array(evalE_cluster_p2))+64).real/64*100).ravel(), label = "cluster")
plt.xlabel("rotation angle")
plt.ylabel("Energy Error (%)")
plt.title("Energy Error (%) vs Rotation Angle by tensor_gru (L = 32, units = 128, batch = 256, patch = 2 by eval)")
plt.yscale("log")
plt.ylim(1e-5, 5e1)
plt.legend()
plt.savefig("figure/energy_error_tensor_gru_L32_units128_batch256_patch2_eval.png")
plt.show()
plt.clf()

plt.scatter(np.array(angle_list), ((np.min(moving_average(np.array(meanE_ES_p2).ravel().real).reshape(11, -1), axis=1)+64)/64*100).ravel(), label = "ES")
plt.scatter(np.array(angle_list), ((np.min(moving_average(np.array(meanE_ES_dmrg_p2).ravel().real).reshape(11, -1), axis=1)+64)/64*100).ravel(),  label = "ES_dmrg")
plt.scatter(np.array(angle_list), (np.flip(np.min(moving_average(np.array(meanE_cluster_p2).ravel().real).reshape(11, -1), axis=1)+64)/64*100).ravel(),  label = "cluster")
plt.xlabel("rotation angle")
plt.ylabel("Energy Error (%)")
plt.title("Energy Error (%) vs Rotation Angle by tensor_gru (L = 32, units = 64, batch = 256, patch = 2 by mvag)")
plt.yscale("log")
plt.ylim(1e-5, 5e1)
plt.legend()
plt.savefig("figure/energy_error_tensor_gru_L32_units64_batch256_patch2_mvag.png")
plt.show()
plt.clf()
# Error plot 1Dtensor-gru patch 4
plt.scatter(np.array(angle_list), ((np.array(evalE_ES_p4)+64).real/64*100).ravel(), label = "ES")
plt.scatter(np.array(angle_list), ((np.array(evalE_ES_dmrg_p4)+64).real/64*100).ravel(), label = "ES_dmrg")
plt.scatter(np.array(angle_list), ((np.flip(np.array(evalE_cluster_p4))+64).real/64*100).ravel(), label = "cluster")
plt.xlabel("rotation angle")
plt.ylabel("Energy Error (%)")
plt.title("Energy Error (%) vs Rotation Angle by tensor_gru (L = 16, units = 128, batch = 256, patch = 4 by eval)")
plt.yscale("log")
plt.ylim(1e-5, 5e1)
plt.legend()
plt.savefig("figure/energy_error_tensor_gru_L16_units128_batch256_patch4_eval.png")
plt.show()
plt.clf()

plt.scatter(np.array(angle_list), ((np.min(moving_average(np.array(meanE_ES_p4).ravel().real).reshape(11, -1), axis=1)+64)/64*100).ravel(),  label = "ES")
plt.scatter(np.array(angle_list), ((np.min(moving_average(np.array(meanE_ES_dmrg_p4).ravel().real).reshape(11, -1), axis=1)+64)/64*100).ravel(), label = "ES_dmrg")
plt.scatter(np.array(angle_list), (np.flip(np.min(moving_average(np.array(meanE_cluster_p4).ravel().real).reshape(11, -1), axis=1)+64)/64*100).ravel(), label = "cluster")
plt.xlabel("rotation angle")
plt.ylabel("Energy Error (%)")
plt.title("Energy Error (%) vs Rotation Angle by tensor_gru (L = 16, units = 64, batch = 256, patch = 4 by mvag)")
plt.yscale("log")
plt.ylim(1e-5, 5e1)
plt.legend()
plt.savefig("figure/energy_error_tensor_gru_L16_units64_batch256_patch4_mvag.png")
plt.show()
plt.clf()
# Error plot 1DTQS patch 1
plt.scatter(np.array(angle_list), ((np.array(evalE_ES_TQS)+64).real/64*100).ravel(), label = "ES")
plt.scatter(np.array(angle_list), ((np.array(evalE_ES_TQS_dmrg)+64).real/64*100).ravel(), label = "ES_dmrg")
plt.scatter(np.array(angle_list), ((np.flip(np.array(evalE_cluster_TQS))+64).real/64*100).ravel(), label = "cluster")
plt.xlabel("rotation angle")
plt.ylabel("Energy Error (%)")
plt.title("Energy Error (%) vs Rotation Angle by 1DTQS (L = 64, units = 128, batch = 256, patch = 1 by eval)")
plt.yscale("log")
plt.ylim(1e-5, 1e2)
plt.legend()
plt.savefig("figure/energy_error_1DTQS_L64_units128_batch256_patch1_eval.png")
plt.show()
plt.clf()

plt.scatter(np.array(angle_list), ((np.min(moving_average(np.array(meanE_ES_TQS).ravel().real).reshape(11, -1), axis=1)+64)/64*100).ravel(),  label = "ES")
plt.scatter(np.array(angle_list), ((np.min(moving_average(np.array(meanE_ES_TQS_dmrg).ravel().real).reshape(11, -1), axis=1)+64)/64*100).ravel(), label = "ES_dmrg")
plt.scatter(np.array(angle_list), (np.flip(np.min(moving_average(np.array(meanE_cluster_TQS).ravel().real).reshape(11, -1), axis=1)+64)/64*100).ravel(), label = "cluster")
plt.xlabel("rotation angle")
plt.ylabel("Energy Error (%)")
plt.title("Energy Error (%) vs Rotation Angle by 1DTQS (L = 64, units = 64, batch = 256, patch = 1 by mvag)")
plt.yscale("log")
plt.ylim(1e-5, 1e2)
plt.legend()
plt.savefig("figure/energy_error_1DTQS_L64_units64_batch256_patch1_mvag.png")
plt.show()
plt.clf()
# Error plot 1DTQS patch 2
plt.scatter(np.array(angle_list), ((np.array(evalE_ES_TQS_p2)+64).real/64*100).ravel(), label = "ES")
plt.scatter(np.array(angle_list), ((np.array(evalE_ES_TQS_dmrg_p2)+64).real/64*100).ravel(), label = "ES_dmrg")
plt.scatter(np.array(angle_list), ((np.flip(np.array(evalE_cluster_TQS_p2))+64).real/64*100).ravel(), label = "cluster")
plt.xlabel("rotation angle")
plt.ylabel("Energy Error (%)")
plt.title("Energy Error (%) vs Rotation Angle by 1DTQS (L = 32, units = 128, batch = 256, patch = 2 by eval)")
plt.yscale("log")
plt.ylim(1e-5, 1e2)
plt.legend()
plt.savefig("figure/energy_error_1DTQS_L32_units128_batch256_patch2_eval.png")
plt.show()
plt.clf()

plt.scatter(np.array(angle_list), ((np.min(moving_average(np.array(meanE_ES_TQS_p2).ravel().real).reshape(11, -1), axis=1)+64)/64*100).ravel(),  label = "ES")
plt.scatter(np.array(angle_list), ((np.min(moving_average(np.array(meanE_ES_TQS_dmrg_p2).ravel().real).reshape(11, -1), axis=1)+64)/64*100).ravel(), label = "ES_dmrg")
plt.scatter(np.array(angle_list), (np.flip(np.min(moving_average(np.array(meanE_cluster_TQS_p2).ravel().real).reshape(11, -1), axis=1)+64)/64*100).ravel(), label = "cluster")
plt.xlabel("rotation angle")
plt.ylabel("Energy Error (%)")
plt.title("Energy Error (%) vs Rotation Angle by 1DTQS (L = 32, units = 64, batch = 256, patch = 2 by mvag)")
plt.yscale("log")
plt.ylim(1e-5, 1e2)
plt.legend()
plt.savefig("figure/energy_error_1DTQS_L32_units64_batch256_patch2_mvag.png")
plt.show()
plt.clf()
# Error plot 1DTQS patch 4
plt.scatter(np.array(angle_list), ((np.array(evalE_ES_TQS_p4)+64).real/64*100).ravel(), label = "ES")
plt.scatter(np.array(angle_list), ((np.array(evalE_ES_TQS_dmrg_p4)+64).real/64*100).ravel(), label = "ES_dmrg")
plt.scatter(np.array(angle_list), ((np.flip(np.array(evalE_cluster_TQS_p4))+64).real/64*100).ravel(), label = "cluster")
plt.xlabel("rotation angle")
plt.ylabel("Energy Error (%)")
plt.title("Energy Error (%) vs Rotation Angle by 1DTQS (L = 16, units = 128, batch = 256, patch = 4 by eval)")
plt.yscale("log")
plt.ylim(1e-5, 1e2)
plt.legend()
plt.savefig("figure/energy_error_1DTQS_L16_units128_batch256_patch4_eval.png")
plt.show()
plt.clf()

plt.scatter(np.array(angle_list), ((np.min(moving_average(np.array(meanE_ES_TQS_p4).ravel().real).reshape(11, -1), axis=1)+64)/64*100).ravel(),  label = "ES")
plt.scatter(np.array(angle_list), ((np.min(moving_average(np.array(meanE_ES_TQS_dmrg_p4).ravel().real).reshape(11, -1), axis=1)+64)/64*100).ravel(), label = "ES_dmrg")
plt.scatter(np.array(angle_list), (np.flip(np.min(moving_average(np.array(meanE_cluster_TQS_p4).ravel().real).reshape(11, -1), axis=1)+64)/64*100).ravel(), label = "cluster")
plt.xlabel("rotation angle")
plt.ylabel("Energy Error (%)")
plt.title("Energy Error (%) vs Rotation Angle by 1DTQS (L = 16, units = 64, batch = 256, patch = 4 by mvag)")
plt.yscale("log")
plt.ylim(1e-5, 1e2)
plt.legend()
plt.savefig("figure/energy_error_1DTQS_L16_units64_batch256_patch4_mvag.png")
plt.show()
plt.clf()

# Error plot 2Dtensor-gru
plt.scatter(np.array(angle_list), ((np.min(moving_average(np.array(meanE_gf).ravel().real).reshape(11, -1), axis=1)+64)/64*100).ravel(),  label = "patch = 1")
plt.scatter(np.array(angle_list), ((np.min(moving_average(np.array(meanE_gf_p2).ravel().real).reshape(11, -1), axis=1)+64)/64*100).ravel(), label = "patch = 2")

plt.xlabel("rotation angle")
plt.ylabel("Energy Error (%)")
plt.title("Energy Error (%) vs Rotation Angle by 2Dtensor_gru (L = 8, units = 64, batch = 256, patch = 1 by mvag)")
plt.yscale("log")
plt.ylim(1e-5, 1e2)
plt.legend()
plt.show()
plt.clf()

# Convergence time
avg_meanE_ES = moving_average(np.array(meanE_ES).ravel().real).reshape(11, -1)
avg_meanE_ES_dmrg = moving_average(np.array(meanE_ES_dmrg).ravel().real).reshape(11, -1)
avg_meanE_cluster = moving_average(np.flip(np.array(meanE_cluster).reshape(11, -1), axis=0).ravel().real).reshape(11, -1)
avg_meanE_ES_p2 = moving_average(np.array(meanE_ES_p2).ravel().real).reshape(11, -1)
avg_meanE_ES_dmrg_p2 = moving_average(np.array(meanE_ES_dmrg_p2).ravel().real).reshape(11, -1)
avg_meanE_cluster_p2 = moving_average(np.flip(np.array(meanE_cluster_p2).reshape(11, -1), axis=0).ravel().real).reshape(11, -1)
avg_meanE_ES_p4 = moving_average(np.array(meanE_ES_p4).ravel().real).reshape(11, -1)
avg_meanE_ES_dmrg_p4 = moving_average(np.array(meanE_ES_dmrg_p4).ravel().real).reshape(11, -1)
avg_meanE_cluster_p4 = moving_average(np.flip(np.array(meanE_cluster_p4).reshape(11, -1), axis=0).ravel().real).reshape(11, -1)
avg_meanE_TQS = moving_average(np.array(meanE_ES_TQS).ravel().real).reshape(11, -1)
avg_meanE_TQS_dmrg = moving_average(np.array(meanE_ES_TQS_dmrg).ravel().real).reshape(11, -1)
avg_meanE_cluster_TQS = moving_average(np.flip(np.array(meanE_cluster_TQS), 0).ravel().real).reshape(11, -1)
avg_meanE_TQS_p2 = moving_average(np.array(meanE_ES_TQS_p2).ravel().real).reshape(11, -1)
avg_meanE_TQS_dmrg_p2 = moving_average(np.array(meanE_ES_TQS_dmrg_p2).ravel().real).reshape(11, -1)
avg_meanE_cluster_TQS_p2 = moving_average(np.flip(np.array(meanE_cluster_TQS_p2), 0).ravel().real).reshape(11, -1)
avg_meanE_TQS_p4 = moving_average(np.array(meanE_ES_TQS_p4).ravel().real).reshape(11, -1)
avg_meanE_TQS_dmrg_p4 = moving_average(np.array(meanE_ES_TQS_dmrg_p4).ravel().real).reshape(11, -1)
avg_meanE_cluster_TQS_p4 = moving_average(np.flip(np.array(meanE_cluster_TQS_p4), 0).ravel().real).reshape(11, -1)
avg_meanE_gf = moving_average(np.array(meanE_gf).ravel().real).reshape(11, -1)
avg_meanE_gf_p2 = moving_average(np.array(meanE_gf_p2).ravel().real).reshape(11, -1)
ES_converge = []
ES_dmrg_converge = []
cluster_converge = []
ES_p2_converge = []
ES_dmrg_p2_converge = []
cluster_p2_converge = []
ES_p4_converge = []
ES_dmrg_p4_converge = []
cluster_p4_converge = []
ES_TQS_converge = []
ES_dmrg_TQS_converge = []
cluster_TQS_converge = []
ES_TQS_p2_converge = []
ES_dmrg_TQS_p2_converge = []
cluster_TQS_p2_converge = []
ES_TQS_p4_converge = []
ES_dmrg_TQS_p4_converge = []
cluster_TQS_p4_converge = []
ES_gf_converge = []
ES_gf_p2_converge = []

tarE = -63
for i in range(avg_meanE_ES.shape[0]):
    j = 0
    while (avg_meanE_ES[i, j] > tarE):
        j += 1
        if j == 10 ** 5:
            break
    ES_converge.append(j)
    j = 0
    while (avg_meanE_ES_dmrg[i, j] > tarE):
        j += 1
        if j == 10 ** 5:
            break
    ES_dmrg_converge.append(j)
    j = 0
    while (avg_meanE_cluster[i, j] > tarE):
        j += 1
        if j == 10 ** 5:
            break
    cluster_converge.append(j)
    j = 0
    while (avg_meanE_ES_p2[i, j] > tarE):
        j += 1
        if j == 10 ** 5:
            break
    ES_p2_converge.append(j)
    j = 0
    while (avg_meanE_ES_dmrg_p2[i, j] > tarE):
        j += 1
        if j == 10 ** 5:
            break
    ES_dmrg_p2_converge.append(j)
    j = 0
    while (avg_meanE_cluster_p2[i, j] > tarE):
        j += 1
        if j == 10 ** 5:
            break
    cluster_p2_converge.append(j)
    j = 0
    while (avg_meanE_ES_p4[i, j] > tarE):
        j += 1
        if j == 10 ** 5:
            break
    ES_p4_converge.append(j)
    j = 0
    while (avg_meanE_ES_dmrg_p4[i, j] > tarE):
        j += 1
        if j == 10 ** 5:
            break
    ES_dmrg_p4_converge.append(j)
    j = 0
    while (avg_meanE_cluster_p4[i, j] > tarE):
        j += 1
        if j == 10 ** 5:
            break
    cluster_p4_converge.append(j)
    j = 0
    while (avg_meanE_TQS[i, j] > tarE):
        j += 1
        if j == 12000:
            break
    ES_TQS_converge.append(j)
    j = 0
    while (avg_meanE_TQS_dmrg[i, j] > tarE):
        j += 1
        if j == 12000:
            break
    ES_dmrg_TQS_converge.append(j)
    j = 0
    while (avg_meanE_cluster_TQS[i, j] > tarE):
        j += 1
        if j == 12000:
            break
    cluster_TQS_converge.append(j)
    j = 0
    while (avg_meanE_TQS_p2[i, j] > tarE):
        j += 1
        if j == 12000:
            break
    ES_TQS_p2_converge.append(j)
    j = 0
    while (avg_meanE_TQS_dmrg_p2[i, j] > tarE):
        j += 1
        if j == 12000:
            break
    ES_dmrg_TQS_p2_converge.append(j)
    j = 0
    while (avg_meanE_cluster_TQS_p2[i, j] > tarE):
        j += 1
        if j == 12000:
            break
    cluster_TQS_p2_converge.append(j)
    j = 0
    while (avg_meanE_TQS_p4[i, j] > tarE):
        j += 1
        if j == 12000:
            break
    ES_TQS_p4_converge.append(j)
    j = 0
    while (avg_meanE_TQS_dmrg_p4[i, j] > tarE):
        j += 1
        if j == 12000:
            break
    ES_dmrg_TQS_p4_converge.append(j)
    j = 0
    while (avg_meanE_cluster_TQS_p4[i, j] > tarE):
        j += 1
        if j == 12000:
            break
    cluster_TQS_p4_converge.append(j)
    j = 0
    while (avg_meanE_gf[i, j] > tarE):
        j += 1
        if j == 60000:
            break
    ES_gf_converge.append(j)
    j = 0
    while (avg_meanE_gf_p2[i, j] > tarE):
        j += 1
        if j == 60000:
            break
    ES_gf_p2_converge.append(j)
# Convergence 1Dtensor-gru patch 1
plt.scatter(np.array(angle_list), ES_converge, alpha = 0.7, label = "ES")
plt.scatter(np.array(angle_list), ES_dmrg_converge, alpha = 0.7, label = "ES_dmrg")
plt.scatter(np.array(angle_list), cluster_converge, alpha = 0.7,label = "cluster")
plt.xlabel("rotation angle")
plt.ylabel("Convergence Steps to E = " + str(tarE))
plt.title("Convergence Steps to E = " + str(tarE) + " vs Rotation Angle by tensor_gru (L = 64, units = 64, batch = 256 patch = 1)")
plt.legend()
plt.yscale("log")
plt.savefig("figure/convergence_tensor_gru_L64_units64_batch256_patch1.png")
plt.show()
plt.clf()
# Convergence 1Dtensor-gru patch 2
plt.scatter(np.array(angle_list), ES_p2_converge, alpha = 0.7, label = "ES")
plt.scatter(np.array(angle_list), ES_dmrg_p2_converge, alpha = 0.7, label = "ES_dmrg")
plt.scatter(np.array(angle_list), cluster_p2_converge, alpha = 0.7,label = "cluster")
plt.xlabel("rotation angle")
plt.ylabel("Convergence Steps to E = " + str(tarE))
plt.title("Convergence Steps to E = " + str(tarE) + " vs Rotation Angle by tensor_gru (L = 32, units = 64, batch = 256 patch = 2)")
plt.legend()
plt.yscale("log")
plt.savefig("figure/convergence_tensor_gru_L32_units64_batch256_patch2.png")
plt.show()
plt.clf()
# Convergence 1Dtensor-gru patch 4
plt.scatter(np.array(angle_list), ES_p4_converge, alpha = 0.7, label = "ES")
plt.scatter(np.array(angle_list), ES_dmrg_p4_converge, alpha = 0.7, label = "ES_dmrg")
plt.scatter(np.array(angle_list), cluster_p4_converge, alpha = 0.7,label = "cluster")
plt.xlabel("rotation angle")
plt.ylabel("Convergence Steps to E = " + str(tarE))
plt.title("Convergence Steps to E = " + str(tarE) + " vs Rotation Angle by tensor_gru (L = 16, units = 64, batch = 256 patch = 4)")
plt.legend()
plt.yscale("log")
plt.savefig("figure/convergence_tensor_gru_L16_units128_batch256_patch4.png")
plt.show()
plt.clf()
# Convergence 1DTQS patch 1
plt.scatter(np.array(angle_list), ES_TQS_converge, alpha = 0.7, label = "ES")
plt.scatter(np.array(angle_list), ES_dmrg_TQS_converge, alpha = 0.7, label = "ES_dmrg")
plt.scatter(np.array(angle_list), cluster_TQS_converge, alpha = 0.7,label = "cluster")
plt.xlabel("rotation angle")
plt.ylabel("Convergence Steps to E = " + str(tarE))
plt.title("Convergence Steps to E = " + str(tarE) + " vs Rotation Angle by 1DTQS (L = 64, units = 128, batch = 256 patch = 1)")
plt.legend()
plt.yscale("log")
plt.savefig("figure/convergence_1DTQS_L64_units128_batch256_patch1.png")
plt.show()
plt.clf()
# Convergence 1DTQS patch 2
plt.scatter(np.array(angle_list), ES_TQS_p2_converge, alpha = 0.7, label = "ES")
plt.scatter(np.array(angle_list), ES_dmrg_TQS_p2_converge, alpha = 0.7, label = "ES_dmrg")
plt.scatter(np.array(angle_list), cluster_TQS_p2_converge, alpha = 0.7,label = "cluster")
plt.xlabel("rotation angle")
plt.ylabel("Convergence Steps to E = " + str(tarE))
plt.title("Convergence Steps to E = " + str(tarE) + " vs Rotation Angle by 1DTQS (L = 32, units = 128, batch = 256 patch = 2)")
plt.legend()
plt.yscale("log")
plt.savefig("figure/convergence_1DTQS_L32_units128_batch256_patch2.png")
plt.show()
plt.clf()
# Convergence 1DTQS patch 4
plt.scatter(np.array(angle_list), ES_TQS_p4_converge, alpha = 0.7, label = "ES")
plt.scatter(np.array(angle_list), ES_dmrg_TQS_p4_converge, alpha = 0.7, label = "ES_dmrg")
plt.scatter(np.array(angle_list), cluster_TQS_p4_converge, alpha = 0.7,label = "cluster")
plt.xlabel("rotation angle")
plt.ylabel("Convergence Steps to E = " + str(tarE))
plt.title("Convergence Steps to E = " + str(tarE) + " vs Rotation Angle by 1DTQS (L = 64, units = 128, batch = 256 patch = 1)")
plt.legend()
plt.yscale("log")
plt.savefig("figure/convergence_1DTQS_L16_units128_batch256_patch4.png")
plt.show()
plt.clf()
# Convergence 2Dtensor-gru
plt.scatter(np.array(angle_list), ES_gf_converge, alpha = 0.7, label = "patch = 1")
plt.scatter(np.array(angle_list), ES_gf_p2_converge, alpha = 0.7, label = "patch = 2")
plt.xlabel("rotation angle")
plt.ylabel("Convergence Steps to E = " + str(tarE))
plt.title("Convergence Steps to E = " + str(tarE) + " vs Rotation Angle by 2Dtensor_gru (L = 8, units = 64, batch = 256)")
plt.legend()
plt.yscale("log")
plt.savefig("figure/convergence_2Dtensor_gru_L8_units64_batch256.png")
plt.show()
plt.clf()

## RBM session
temp_ang = [0.0, 0.471, 0.785, 1.414, 1.571]
temp_ang_c = [0.157, 0.314, 0.628, 0.942, 1.1, 1.257]
E_cluster = []
E_c = []
E_es = []

for ang in angle_list:
    if ang in temp_ang:
        with open('result/RBM/first/RBM_default_Htypecluster_angle='+str(ang)+'_L=64_numsample=4096.log', 'r') as file:
            # Read the entire file
            log_data = json.load(file)
            E_cluster.append(log_data["Energy"]["Mean"]["real"])
        with open('result/RBM/second/RBM_default_Htypecluster_angle='+str(ang)+'_L=64_numsample=4096.log', 'r') as file:
            log_data = json.load(file)
            E_cluster.append(log_data["Energy"]["Mean"]["real"])
        with open('result/RBM/third/RBM_default_Htypecluster_angle='+str(ang)+'_L=64_numsample=4096.log', 'r') as file:
            log_data = json.load(file)
            E_cluster.append(log_data["Energy"]["Mean"]["real"])
        with open('result/RBM/fourth/RBM_default_Htypecluster_angle='+str(ang)+'_L=64_numsample=4096.log', 'r') as file:
            log_data = json.load(file)
            E_cluster.append(log_data["Energy"]["Mean"]["real"])
        with open('result/RBM/fifth/RBM_default_Htypecluster_angle='+str(ang)+'_L=64_numsample=4096.log', 'r') as file:
            log_data = json.load(file)
            E_cluster.append(log_data["Energy"]["Mean"]["real"])
    else:
        with open('result/RBM/first/RBM_default_Htypecluster_angle='+str(ang)+'_L=64_numsample=4096.log', 'r') as file:
            # Read the entire file
            log_data = json.load(file)
            E_cluster.append(log_data["Energy"]["Mean"]["real"])
            if ang!= 0.628 and ang !=1.257 :
                with open('result/RBM/second/RBM_default_Htypecluster_angle='+str(ang)+'_L=64_numsample=4096.log', 'r') as file:
                    # Read the entire file
                    log_data = json.load(file)
                    E_cluster.append(log_data["Energy"]["Mean"]["real"])
for ang in angle_list:

    with open('result/RBM/second/RBM_default_HtypeES_angle='+str(ang)+'_L=64_numsample=4096.log', 'r') as file:
        log_data = json.load(file)
        E_es.append(log_data["Energy"]["Mean"]["real"])
    with open('result/RBM/third/RBM_default_HtypeES_angle='+str(ang)+'_L=64_numsample=4096.log', 'r') as file:
        log_data = json.load(file)
        E_es.append(log_data["Energy"]["Mean"]["real"])
    with open('result/RBM/fourth/RBM_default_HtypeES_angle='+str(ang)+'_L=64_numsample=4096.log', 'r') as file:
        log_data = json.load(file)
        E_es.append(log_data["Energy"]["Mean"]["real"])
    with open('result/RBM/fifth/RBM_default_HtypeES_angle='+str(ang)+'_L=64_numsample=4096.log', 'r') as file:
        log_data = json.load(file)
        E_es.append(log_data["Energy"]["Mean"]["real"])

E_es = sum(E_es, [])
E_cluster = sum(E_cluster, [])

# Training curve 1DRBM
plt.figure(figsize=(20, 8))
plt.plot(np.arange(np.array(E_cluster).ravel().shape[0]), np.flip(np.array(E_cluster).reshape(11, -1), 0).ravel(),  label="cluster")
plt.plot(np.arange(np.array(E_es).ravel().shape[0]), np.array(E_es).ravel(),  label="ES")
plt.ylim(-64, 0)
plt.ylabel("Energy")
plt.xlabel("Training Step")
plt.title("Energy vs Training Step by RBM (L = 64, numsample = 4096)")
for i in range(11):
    plt.axvline(x=(i+1)*10000, color="gray", linestyle="--")
plt.legend()
plt.show()
plt.clf()

# Error plot 1DRBM
eval_E_cluster = np.flip(np.array(E_cluster).reshape(11, -1), axis = 0)[:, -1].ravel()
eval_E_es = np.array(E_es).reshape(11, -1)[:, -1].ravel()
eval_E_es[5] = np.array(E_es)[59570] #Explode after 9570 steps
plt.scatter(np.array(angle_list), ((eval_E_es+64).real/64*100).ravel(), label = "ES")
plt.scatter(np.array(angle_list), ((eval_E_cluster+64).real/64*100).ravel(), label = "cluster")
plt.xlabel("rotation angle")
plt.ylabel("Energy Error (%)")
plt.title("Energy Error (%) vs Rotation Angle by RBM (L = 64, numsample = 4096)")
plt.yscale("log")
plt.legend()
plt.show()
plt.clf()