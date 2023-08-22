from quantum_circuit import *
import torch


def batch_real_loss(disc_weights, batch):
    # print(batch.shape)
    cost = torch.mean(1 - torch.log(train_on_real(disc_weights, torch.transpose(batch, 0, 1))))
    # fig, ax = qml.draw_mpl(train_on_real)(disc_weights, batch)
    # fig.show()
    return cost

def Batch_real_loss(disc_weights, batch):
    # print(batch.shape)
    a = Train_on_real(disc_weights, batch)
    cost = torch.mean(torch.log(1 - a))
    # fig, ax = qml.draw_mpl(Train_on_real)(disc_weights, batch)
    # fig.show()
    print("Real: ", a.item())
    return cost


def batch_fake_loss(disc_weights, gen_weights, batch):
    cost = torch.mean(torch.log(train_on_fake(disc_weights, gen_weights, batch)))
    # fig, ax = qml.draw_mpl(train_on_fake)(disc_weights, gen_weights, batch)
    # fig.show()
    return cost

def Batch_fake_loss(disc_weights, gen_weights, batch):
    a = Train_on_fake(disc_weights, gen_weights, batch)
    cost = torch.mean(torch.log(a))
    # fig, ax = qml.draw_mpl(Train_on_fake)(disc_weights, gen_weights, batch)
    # fig.show()
    print("Fake: ",a.item())
    return cost


def batch_gen_loss(disc_weights, gen_weights, batch):
    cost = torch.mean(1 - torch.log(train_on_fake(disc_weights, gen_weights, batch)))
    # fig, ax = qml.draw_mpl(Train_on_fake)(disc_weights, gen_weights, batch)
    # fig.show()
    return cost

def Batch_gen_loss(disc_weights, gen_weights, batch):
    a = Train_on_fake(disc_weights, gen_weights, batch)
    cost = torch.mean(torch.log(1 - a))
#     # fig, ax = qml.draw_mpl(train_on_fake)(disc_weights, gen_weights, batch)
#     # fig.show()
    print("Gen: ",a.item())

    return cost


def disc_real_cost(disc_weights, data):
    b = train_on_real(disc_weights, data)
    # print(b)
    return b


def disc_fake_cost(disc_weights, gen_weights):
    b = train_on_fake(disc_weights, gen_weights, )
    # print(b)
    return b


def gen_cost(disc_weights, gen_weights):
    b = train_on_fake(disc_weights, gen_weights)
    # print(b)
    return b
