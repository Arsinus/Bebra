import pennylane as qml
from pennylane import numpy as np
from const import *
import torch



dev = qml.device('lightning.qubit', wires=range(3 * (N_a + N_dim) - 1))

def single(start, end, w):
    # print('single: ', w)
    for weight_index, wire in enumerate(range(start, end + 1)):
        qml.RY(w[weight_index], wires=wire)

def Single(start, end):
    # print('single: ', w)
    for weight_index, wire in enumerate(range(start, end + 1)):
        qml.RY(0, wires=wire)


def dual(start, end, w):
    # print('dual: ', w)
    for weight_index, wire in enumerate(range(start, end)):
        qml.IsingXY(w[weight_index], wires=[wire, wire + 1])

def Dual(start, end):
    # print('dual: ', w)
    for weight_index, wire in enumerate(range(start, end)):
        qml.IsingXY(0, wires=[wire, wire + 1])


def entangle(start, end, w):
    # print('entangle: ', w)
    for weight_index, wire in enumerate(range(start, end)):
        qml.CRY(w[weight_index], wires=[wire, wire + 1])
    qml.CRY(w[-1], wires=[end, start])

def Entangle(start, end):
    # print('entangle: ', w)
    for weight_index, wire in enumerate(range(start, end)):
        qml.CRY(0, wires=[wire, wire + 1])
    qml.CRY(0, wires=[end, start])


def hadamard_column(start, end):
    for wire in range(start, end + 1):
        qml.Hadamard(wires=wire)


def input_data(data):
    single(2 * N_a + N_dim + 1, 2 * (N_a + N_dim), data)

def Input_data(data, n_a, n_dim):
    single(2 * n_a + n_dim + 1, 2 * (n_a + n_dim), data)

def embed(data):
    qml.AmplitudeEmbedding(features=data, wires=range(2 * N_a + N_dim + 1, 2 * (N_a + N_dim) + 1), normalize=True)

def swap_test():
    qml.Hadamard(wires=0)
    for i in range(N_dim):
        qml.CSWAP(wires=[0, N_a + 1 + i, 2 * N_a + N_dim + 1 + i])
    qml.Hadamard(wires=0)

def Swap_test(n_a, n_dim):
    qml.Hadamard(wires=0)
    for i in range(n_dim):
        qml.CSWAP(wires=[0, n_a + 1 + i, 2 * n_a + n_dim + 1 + i])
    qml.Hadamard(wires=0)


def layer(start, end, w):
    # print('layer: ', w)
    length = end - start + 1
    single(start, end, w[:length])
    dual(start, end, w[length:2 * length - 1])
    entangle(start, end, w[2 * length - 1:3 * length - 1])
    hadamard_column(start, end)

def Layer(start, end):
    # print('layer: ', w)
    length = end - start + 1
    Single(start, end)
    Dual(start, end)
    Entangle(start, end)
    hadamard_column(start, end)


def deep_layer(start, end, repeat, w):
    # print('deep_layer: ', w)
    for i in range(repeat):
        layer(start, end, w[num_layer_weights * i:num_layer_weights * (i + 1)])

def Deep_layer(start, end, repeat):
    # print('deep_layer: ', w)
    for i in range(repeat):
        Layer(start, end)

@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def gen_sample(disc_weights, gen_weights, noise):
    # input_data(noise)
    single(N_a + N_dim + 1, 2 * (N_a + N_dim), noise)
    # single(1, (N_a + N_dim), init_random_variables(N_dim + N_a, np.pi / 6))
    # embed(noise)
    deep_layer(1, N_a + N_dim, depth, disc_weights)
    deep_layer(N_a + N_dim + 1, 2 * (N_a + N_dim), depth, gen_weights)
    # swap_test()
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(2 * N_a + N_dim + 1, 2 * (N_a + N_dim) + 1)]


@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def train_on_fake(disc_weights, gen_weights, noise):
    single(N_a + N_dim + 1, 2 * (N_a + N_dim), noise)
    # single(1, (N_a + N_dim), init_random_variables(N_dim + N_a, np.pi / 6))
    # embed(noise)
    deep_layer(1, N_a + N_dim, depth, disc_weights)
    deep_layer(N_a + N_dim + 1, 2 * (N_a + N_dim), depth, gen_weights)
    swap_test()
    return qml.expval(qml.PauliZ(0))


@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def train_on_real(disc_weights, data):
    # single(1, (N_a + N_dim), init_random_variables(N_dim + N_a, np.pi / 6))
    input_data(data)
    # embed(data)
    deep_layer(1, N_a + N_dim, depth, disc_weights)
    swap_test()
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def Gen_sample(disc_weights, gen_weights, noise):
    # input_data(noise)
    # single(N_a + N_dim + 1, 2 * (N_a + N_dim), noise)
    # single(1, (N_a + N_dim), init_random_variables(N_dim + N_a, np.pi / 6))
    embed(noise)
    deep_layer(1, N_a + N_dim, depth, disc_weights)
    deep_layer(N_a + N_dim + 1, 2 * (N_a + N_dim), depth, gen_weights)
    # swap_test()
    return qml.probs(wires=range(2 * N_a + N_dim + 1, 2 * (N_a + N_dim) + 1))


@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def Train_on_fake(disc_weights, gen_weights, noise):
    # single(N_a + N_dim + 1, 2 * (N_a + N_dim), noise)
    # single(1, (N_a + N_dim), init_random_variables(N_dim + N_a, np.pi / 6))
    embed(noise)
    deep_layer(1, N_a + N_dim, depth, disc_weights)
    deep_layer(N_a + N_dim + 1, 2 * (N_a + N_dim), depth, gen_weights)
    swap_test()
    return qml.expval(qml.PauliZ(0))


@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def Train_on_real(disc_weights, data):
    # single(1, (N_a + N_dim), init_random_variables(N_dim + N_a, np.pi / 6))
    # input_data(data)
    embed(data)
    deep_layer(1, N_a + N_dim, depth, disc_weights)
    swap_test()
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def gn_smpl(n_a, n_dim, dep):
    Single(n_a + n_dim + 1, 2 * (n_a + n_dim))
    Deep_layer(1, n_a + n_dim, dep)
    Deep_layer(n_a + n_dim + 1, 2 * (n_a + n_dim), dep)
    return qml.probs(wires=range(2 * n_a + n_dim + 1, 2 * (n_a + n_dim) + 1))


@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def trn_fk(n_a, n_dim, dep):
    Single(n_a + n_dim + 1, 2 * (n_a + n_dim))
    Deep_layer(1, n_a + n_dim, dep)
    Deep_layer(n_a + n_dim + 1, 2 * (n_a + n_dim), dep)
    Swap_test(n_a, n_dim)
    return qml.expval(qml.PauliZ(0))


@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def trn_rl(n_a, n_dim, dep):
    data = np.zeros(n_dim)
    Input_data(data, n_a, n_dim)
    Deep_layer(1, n_a + n_dim, dep)
    Swap_test(n_a, n_dim)
    return qml.expval(qml.PauliZ(0))