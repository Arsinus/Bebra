import pennylane.numpy as np
from const import *
from quantum_circuit import *
import torch
import torch.nn as nn
import torch.nn.utils as torch_utils
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots()
plt.rcParams['figure.dpi'] = 200


def make_noise(noise_scale=NOISE_SCALE, batch_size=BATCH_SIZE, dim=N_dim + N_a):
    return torch.from_numpy(
        2 * np.arcsin(np.sqrt(np.random.uniform(0.5 - noise_scale, 0.5 + noise_scale, [batch_size, dim]))))

def Make_noise(noise_scale=NOISE_SCALE, batch_size=BATCH_SIZE, dim=LATENT_DIM):
    return torch.from_numpy(
        np.random.uniform(np.sqrt(1/(2*LATENT_DIM)) - noise_scale, np.sqrt(1/(2*LATENT_DIM)) + noise_scale, [batch_size, dim]))


def get_samples(num_samples, disc_weights, gen_weights, noise):
    samples = np.zeros((num_samples, N_dim))
    # print(make_noise(batch_size=1).shape)
    for i in range(num_samples):
        res = torch.tensor(
            gen_sample(disc_weights, gen_weights, torch.from_numpy(
                2 * np.arcsin(np.sqrt(np.random.uniform(0.5 - noise, 0.5 + noise, N_dim + N_a))))))
        res = (1 - res) / 2
        samples[i] = res
    return samples




def show_lanent_space(noise, disc_weights, gen_weights, cls, img_size):
    disc_grid = np.zeros([img_size, img_size])

    for i in range(img_size):
        for j in range(img_size):
            point_angles = 2 * np.arcsin(np.sqrt(np.array([i, j]) / img_size + 1 / (2 * img_size)))
            disc_grid[j][i] = train_on_real(disc_weights, point_angles)
    samples = get_samples(NUM_SAMPLES, disc_weights, gen_weights, noise)
    sns.jointplot(x=samples[:, 0], y=samples[:, 1], kind='kde', cmap='autumn', label='Generator', levels=15)
    plt.scatter(cls[:, 0], cls[:, 1], s=1, color='red', label="Cluster: {}")
    plt.imshow(disc_grid, extent=[0, 1, 0, 1], origin='lower', label='Discriminator')
    plt.legend()
    plt.show()
    plt.clf()


def save_lanent_space(noise, disc_weights, gen_weights, cls, batch, img_size, num_cls, epoch, batch_idx, circ, iter):
    disc_grid = np.zeros([img_size, img_size])

    for i in range(img_size):
        for j in range(img_size):
            point_angles = 2 * np.arcsin(np.sqrt(np.array([i, j]) / img_size + 1 / (2 * img_size)))
            disc_grid[j][i] = train_on_real(disc_weights, point_angles)
    aboba = (np.sin(batch / 2)) ** 2
    samples = get_samples(NUM_SAMPLES, disc_weights, gen_weights, noise)
    sns.jointplot(x=samples[:, 0], y=samples[:, 1], kind='kde', cmap='autumn', label='Generator', levels=15)
    plt.scatter(cls[:, 0], cls[:, 1], s=1, color='red', label="Cluster: {}".format(num_cls))
    plt.scatter(aboba[:, 0], aboba[:, 1], s=10, color='green', label='Batch: {}'.format(batch_idx))
    plt.imshow(disc_grid, extent=[0, 1, 0, 1], origin='lower', label='Discriminator')
    plt.legend()
    plt.savefig("ancilla_image/image-{}-{}-{}-{}-{}".format(num_cls, epoch, batch_idx, circ, iter))
    plt.clf()


def save_lanent_space_foraboba(noise, disc_weights, gen_weights, cls, img_size, num_cls, num_samples):
    disc_grid = np.zeros([img_size, img_size])

    for i in range(img_size):
        for j in range(img_size):
            point_angles = 2 * np.arcsin(np.sqrt(np.array([i, j]) / img_size + 1 / (2 * img_size)))
            disc_grid[j][i] = train_on_real(disc_weights, point_angles)
    samples = get_samples(num_samples, disc_weights, gen_weights, noise)
    sns.jointplot(x=samples[:, 0], y=samples[:, 1], kind='kde', cmap='autumn', label='Generator', levels=15)
    plt.scatter(cls[:, 0], cls[:, 1], s=1, color='red', label="Cluster: {}".format(num_cls))
    plt.imshow(disc_grid, extent=[0, 1, 0, 1], origin='lower', label='Discriminator')
    plt.legend()
    plt.savefig("aboba/show_progress")
    plt.clf()


def save_loss_graphic(loss_data, cls, epoch, batch_dix, num_batches):
    plt.plot([i / num_batches for i in range(len(loss_data))], loss_data)
    ax.set_ylabel('Hellinger Distance, Cluster: {}'.format(cls))
    ax.set_xlabel('Epoch')
    plt.savefig("loss_graphics/image-{}-{}-{}".format(cls, epoch, batch_dix))
    ax.clear()
    plt.clf()


def make_distribution(data, num_columns):
    distr = np.zeros([num_columns, num_columns])
    for point in data:
        try:
            distr[int(point[1] * num_columns)][int(point[0] * num_columns)] += 1
        except:
            print('Вылезла, падла')
    distr /= len(data) * num_columns ** (-2)
    return distr


def hellinger_distance(real_data, fake_data, num_columns):
    real_distr = make_distribution(real_data, num_columns)
    fake_distr = make_distribution(fake_data, num_columns)
    distance = np.sqrt(np.sum((np.sqrt(real_distr) - np.sqrt(fake_distr)) ** 2 * num_columns ** (-2))) / np.sqrt(2)
    return distance


def save_weights(disc, gen, cls, epoch, batch_idx):
    with open('weights/model-{}-{}-{}.txt'.format(cls, epoch, batch_idx), 'w') as f:
        for value in disc:
            f.write('{} '.format(value))
        f.write('\n')
        for value in gen:
            f.write('{} '.format(value))


def init_random_variables(length, val, grad=True):
    par = torch.rand(length) * val
    par.requires_grad_(grad)
    return par
