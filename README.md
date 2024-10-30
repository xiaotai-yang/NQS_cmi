
## Introduction

The Neural Quantum State (NQS) is a variational ansatz for representing the wave function of quantum systems. 
It leverages the power of neural networks to approximate the quantum wave function and solve the Schrödinger equation 
using variational methods. 
However, different projection basis affects the performance of the NQS significantly. 
Here we show that we can quantify this difficulty by conditional mutual information 
with rotated cluster state and rotated graph state. (The url of the paper...)

## 1D Hamiltonian 

- H_type: "ES" or "cluster" which means entanglement swapping state(rotated cluster state) or a cluster state. Here we present the
corresponding Hamiltonian when the rotating angle is 0. The rotation is done by acting each qubit with $R_y(\theta)$ gate.

![Rotated cluster state](https://github.com/xiaotai-yang/NQS_cmi/blob/main/readme_eq/H_ES.png?raw=true)

![Cluster state](https://github.com/xiaotai-yang/NQS_cmi/blob/main/readme_eq/H_cluster.png?raw=true)
## 2D Hamiltonian

Only consider rotated graph state with open boundary condition so no variable here

![Graph state](https://github.com/xiaotai-yang/NQS_cmi/blob/main/readme_eq/H_graph.png?raw=true)
# 1D & 2D NQS model
- L: The number of blocks along the edge
- p: Side length of each block 

So the total qubit number is $(L\times p)$ or  $(L\times p)^2$ and each block has $p$ or $p^2$ qubits for 1D and 2D respectively.

- numsamples: The batch size of each iteration.
- testing_sample: The batch size for final evaluation of the ground state energy after the training. 
- dmrg (only for 1D): Whether to use dmrg ground truth to get the phases of the quantum state. Dmrg code is provided in "DMRG" directory.
- previous_training: Whether to continue to train the previous model. (set to False for the first training)

#### For 1D models, please run the DMRG/1D_dmrg.ipynb first for the model you want to test and then run the Running_1d.py file with corresponding parameters. 
## Recurrent neural network (RNN)
- numunits: Hidden layer width of the RNN.

The example command for running the 1D RNN model is:
```
python Running_1d.py --model tensor_gru --L 16 --p 1 --H_type ES --numunits 64 --numsamples 256 --testing_sample 8192 --angle 0.0 --numsteps 5000
```
## Receptance Weighted Key Value (RWKV)
- RWKV_emb: Width of the embedding input 
- RWKV_hidden: Hidden layer width of the RWKV
- RWKV_layer: How many layer of RWKV we apply
- RWKV_ff: The final feedforward network size in each RWKV layer

The example command for running the 1D RWKV model with dmrg informed phase is:
```
python Running_1d.py --model tensor_rwkv --L 16--p 1 --H_type cluster --RWKV_emb 32 --RWKV_hidden 64 --RWKV_layer 2 --RWKV_ff 64 --numsamples 256 --testing_sample 8192 --angle 0.0 --numsteps 5000 --dmrg
```
## Transformer Quantum state (TQS)
- TQS_layer: How many layer of Transformer we apply
- TQS_units : Hidden layer width (QKV) and embedding size for the transformer
- TQS_ff: The final feedforward network size in each layer
- TQS_head: The number of head in multihead attenetion. 

The example command for running the 1D TQS model is:
```
python Running_1d.py --model tensor_tqs --L 16 --p 1 --H_type ES --TQS_layer 2 --TQS_units 64 --TQS_ff 256 --TQS_head 4 --numsamples 256 --testing_sample 8192 --angle 0.0 --numsteps 5000
```
## Restriced Boltzmann Machine
There is only two models and no submodule so we just wrote them in two individual files. The variables are:
- alpha: The width of the hidden layer
- numsamples: The batch size of each training
- numsteps: The number of steps for training

The example command for running the 1D RBM model is: 
```
python netket_1d.py --model ES --L 16  --alpha 8 --numsamples 4096 --numsteps 5000 --angle 0.0 --numsteps 5000
```
# Exact Diagonalization Stduy

Inside directory ED, we explore the exact diagonalization for some common Hamiltonians and show that the ground state 
shows decay of conditional mutual information in the measurement basis.

## Dependencies

Our implmentation is based on Python (3.9.16), jax (0.4.30), optax(0.2.3), numpy (1.26.4), julia(1.10.4) and netket(3.13.0).

For further questions or inquiries, please feel free to send an email to ty39@illinois.edu.

---

