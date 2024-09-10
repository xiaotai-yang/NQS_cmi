
## Introduction

The Neural Quantum State (NQS) is a variational ansatz for representing the wave function of quantum systems. It leverages the power of neural networks to approximate the quantum wave function and solve the Schrödinger equation using variational methods. However, different projection basis affects the performance of the NQS significantly. Here we show that we can quantify this difficulty with conditional mutual information.

## 1D Hamiltonian 

- H_type: "ES" or "cluster" which means entanglement swapping state or a cluster state.

## 2D Hamiltonian

- Only graph state now so no variable here

# 1D & 2D autoregressive model
- numsamples: The batch size of each training
- testing_sample: The batch size for final evaluation of the ground state energy after the training. 
- dmrg (only for 1D): Whether to use dmrg ground truth to get the phases of the quantum state.
## Recurrent neural network (RNN)
- numunits: Hidden layer width of the RNN.
## Receptance Weighted Key Value (RWKV)
- RWKV_emb: THe size of the embedding input
- RWKV_hidden: The hidden layer length of the RWKV
- RWKV_layer: How many layer of RWKV we apply
- RWKV_ff: The final feedforward network size in each RWKV layer
## Transformer Quantum state (TQS)
- TQS_layer: How many layer of Transformer we apply
- TQS_units : Hidden layer width (QKV) for the transformer
- TQS_ff: The final feedforward network size in each layer
- TQS_head: The number of head in multihead attenetion.

# Restriced Boltzmann Machine
There is only two models and no submodule so I just wrote them in two individual files. The variables are:
- alpha: The width of the hidden layer
- numsamples: The batch size of each training
---
