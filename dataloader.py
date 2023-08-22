import pennylane.numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as torch_utils
from sklearn.mixture import GaussianMixture
from sklearn.cluster import OPTICS
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from const import *

fig = plt.figure()
ax = fig.add_subplot(111)


def img2matrix(path, file):
    image = Image.open('{}/{}'.format(path, file))
    transform = transforms.ToTensor()
    tensor = torch.flip(transform(image), [1])
    gray = torch.mean(tensor, dim=0)
    return gray


def vec2matrix(vec):
    matrix = vec.reshape(64, 64)
    return matrix


def matrix2vec(matrix):
    vec = matrix.reshape(matrix.shape[0] * matrix.shape[1])
    return vec


def show_img(ep='A', bidx='A', img=None, matrix=None, vec=None):
    if img != None:
        matrix = img2matrix(img)
        plt.imshow(matrix, origin='lower', cmap='gray')
        plt.show()
        plt.clf()
    if matrix != None:
        plt.imshow(matrix, origin='lower', cmap='gray')
        plt.show()
        plt.clf()
    if vec.any() != None:
        matrix = vec2matrix(vec)
        plt.imshow(matrix, origin='lower', cmap='gray')
        plt.show()
        plt.clf()


def get_dataset(path):
    files = os.listdir(path)[:1000]
    dataset = torch.from_numpy(np.array([matrix2vec(img2matrix(path, file)) for file in files]))
    return dataset




def pca_ende(path, file, latent_dim):
    test_pca = PCA(n_components=latent_dim)
    test_pca.fit(dataset)
    ende = img2matrix(path, file)
    ende = matrix2vec(ende)
    ende = test_pca.inverse_transform(test_pca.transform(ende.reshape(1, -1)))
    ende = vec2matrix(ende)
    return ende


cxr_dataset = get_dataset('MedMnist/CXR')
hand_dataset = get_dataset('MedMnist/Hand')
head_dataset = get_dataset('MedMnist/HeadCT')
dataset = np.concatenate((cxr_dataset, hand_dataset, head_dataset), axis=0)

pca = PCA(n_components=LATENT_DIM)
pca.fit(dataset)
cxr_pca = pca.transform(cxr_dataset)
hand_pca = pca.transform(hand_dataset)
head_pca = pca.transform(head_dataset)
pca_data = pca.transform(dataset)

#
# plt.show()
# print(pca_data.shape)
# for i in range(10):
#     show_img(vec=pca.inverse_transform(pca_data[np.random.randint(len(pca_data))]))
# Scale = MinMaxScaler(feature_range=(0, np.sqrt(1 / LATENT_DIM)))
# Scale.fit(pca_data)
#
# def pca2hat(data):
#     norm_data = Scale.transform(data)
#     z_dim = np.sqrt(np.abs(1 - np.sum(np.power(norm_data, 2), axis=1)).reshape(data.shape[0], 1))
#     hat_data = np.concatenate((norm_data, z_dim), axis=1)
#     # print(hat_data.shape)
#     return hat_data
#
#
# def hat2pca(hat_data):
#     try:
#         norm_data = hat_data[:, :-1].detach().numpy()
#     except:
#         norm_data = hat_data[:-1].detach().numpy()
#
#     data = Scale.inverse_transform(norm_data.reshape(1,-1))
#     return data


# hat_dataset = pca2hat(pca_data)
# cxr_hat = pca2hat(cxr_pca)
# hand_hat = pca2hat(hand_pca)
# head_hat = pca2hat(head_pca)
# hat_clusters = list([cxr_hat, hand_hat, head_hat])

scale = MinMaxScaler(feature_range=(0, 1))
scale.fit(pca_data)
latent_data = scale.transform(pca_data)
cxr_latent = scale.transform(cxr_pca)
hand_latent = scale.transform(hand_pca)
head_latent = scale.transform(head_pca)
latent_clusters = list([cxr_latent, hand_latent, head_latent])

# for data in latent_clusters:
#     ax.scatter(data[:, 0], data[:, 1])
#
# ax.xaxis.label.set_color('red')
# ax.spines['bottom'].set_color('white')
# ax.spines['left'].set_color('white')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.tick_params(axis='x', colors='white')
# ax.tick_params(axis='y', colors='white')
# # ax.yaxis.label.set_color('white')
# plt.savefig('aboba/trans_pca', transparent=True)
