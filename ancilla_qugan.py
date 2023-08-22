
from tqdm import tqdm
import statistics
import tkinter as Tk
from tkinter import ttk
import PySimpleGUI as sg
from dataloader import *
from loss_utils import *
from utils import *

disc_weights = init_random_variables(num_deep_layer_weights, np.pi / 2)
gen_weights = init_random_variables(num_deep_layer_weights, np.pi / 2)

criterion = nn.BCELoss()
disc_optimizer = torch.optim.AdamW([disc_weights], lr=learning_rate, betas=(0.5, 0.999))
gen_optimizer = torch.optim.AdamW([gen_weights], lr=learning_rate, betas=(0.5, 0.999))


def train_vanilla():
    for num_cls, cls in enumerate(latent_clusters):
        pca_data_rot = torch.from_numpy(2 * np.arcsin(np.sqrt(cls)))
        # hat_dataset = latent2hat(cls)
        train_data = torch.utils.data.DataLoader(pca_data_rot, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        hellinger_loss = list()
        hellinger_loss.append(
            hellinger_distance(cls, get_samples(NUM_SAMPLES_HELL, disc_weights, gen_weights, NOISE_SCALE), len(train_data)))

        for epoch in tqdm(range(NUM_EPOCHS), desc="Epoch: "):

            if hellinger_loss[-1] < 0.4:
                break

            for batch_idx, batch in enumerate(tqdm(train_data, desc="Batch: ", leave=False)):

                # print(batch)
                # sample = torch.sqrt(gen_sample(disc_weights, gen_weights,
                #                     torch.from_numpy(np.random.uniform(0, NOISE_SCALE, [1, LATENT_DIM + 1]))).reshape(1, -1)).detach().numpy()
                # sample = hat2latent(sample)
                # show_img(epoch, batch_idx, vec=pca.inverse_transform(scale.inverse_transform(sample)))

                if epoch >= 1:
                    last_loss = hellinger_loss[-len(train_data):]
                    last_loss = [(last_loss[i] - last_loss[i - 1]).item() for i in range(1, len(last_loss))]
                    # print(last_loss)
                    print(statistics.mean(last_loss))

                for iter in range(A):

                    if iter % 1 == 0:
                        save_lanent_space(NOISE_SCALE, disc_weights, gen_weights, cls, batch, image_size, num_cls, epoch, batch_idx, 'A',
                                          iter)

                    disc_optimizer.zero_grad()
                    loss = batch_real_loss(disc_weights, batch)
                    loss.backward()
                    torch_utils.clip_grad_norm_(disc_weights, max_norm=max_norm)
                    disc_optimizer.step()

                for iter in range(B):

                    if iter % 1 == 0:
                        save_lanent_space(NOISE_SCALE, disc_weights, gen_weights, cls, batch, image_size, num_cls, epoch, batch_idx, 'B',
                                          iter)

                    disc_optimizer.zero_grad()
                    loss = batch_fake_loss(disc_weights, gen_weights, make_noise())
                    loss.backward()
                    torch_utils.clip_grad_norm_(disc_weights, max_norm=max_norm)
                    disc_optimizer.step()

                for iter in range(C):
                    if iter % 1 == 0:
                        save_lanent_space(NOISE_SCALE, disc_weights, gen_weights, cls, batch, image_size, num_cls, epoch, batch_idx, 'C',
                                          iter)

                    gen_optimizer.zero_grad()
                    loss = batch_gen_loss(disc_weights, gen_weights, make_noise())
                    loss.backward()
                    # print(torch.gradient(gen_weights))
                    grad_norm = (torch.sqrt(torch.sum(torch.pow(gen_weights.grad, 2)))).item()
                    torch_utils.clip_grad_norm_(disc_weights, max_norm=max_norm)
                    gen_optimizer.step()

                hellinger_loss.append(
                    hellinger_distance(cls, get_samples(NUM_SAMPLES_HELL, disc_weights, gen_weights, NOISE_SCALE), len(train_data)))

                save_loss_graphic(hellinger_loss, num_cls, epoch, batch_idx, len(train_data))
                save_weights(disc_weights, gen_weights, num_cls, epoch, batch_idx)

train_vanilla()

# amogus = torch.utils.data.DataLoader(pca2hat(Make_noise(batch_size=2)), batch_size=1)
# for _, shtuka in amogus:
#     hat_noise = shtuka
# print(hat_noise)
# hat_noise = torch.tensor(pca2hat(Make_noise(batch_size=1)))
# print(hat_noise.shape)
# def train_amplitude():
#     for num_cls, cls in enumerate(hat_clusters):
#         if num_cls != 1:
#             continue
#         # hat_dataset = latent2hat(cls)
#         # print(cls.shape)
#         train_data = torch.utils.data.DataLoader(cls, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
#         for epoch in tqdm(range(NUM_EPOCHS), desc="Epoch: "):
#
#             for batch_idx, batch in enumerate(tqdm(train_data, desc="Batch: ", leave=False)):
#
#                 # print(batch)
#                 # sample = torch.sqrt(gen_sample(disc_weights, gen_weights,
#                 #                     torch.from_numpy(np.random.uniform(0, NOISE_SCALE, [1, LATENT_DIM + 1]))).reshape(1, -1)).detach().numpy()
#                 # sample = hat2latent(sample)
#                 # show_img(epoch, batch_idx, vec=pca.inverse_transform(scale.inverse_transform(sample)))
#                 # fig, ax = qml.draw_mpl(Gen_sample)(disc_weights, gen_weights, hat_noise)
#                 # fig.show()
#                 sample = torch.sqrt(Gen_sample(disc_weights, gen_weights, hat_noise))
#                 sample = hat2pca(sample)
#                 sample = pca.inverse_transform(sample.reshape(1, -1)).reshape(64,64)
#                 plt.imshow(sample, origin='lower', cmap='gray')
#                 plt.savefig("gen_med/image-{}-{}".format(epoch, batch_idx))
#                 plt.clf()
#
#                 for iter in range(A):
#
#                     disc_optimizer.zero_grad()
#                     loss = Batch_real_loss(disc_weights, batch)
#                     loss.backward()
#                     torch_utils.clip_grad_norm_(disc_weights, max_norm=max_norm)
#                     disc_optimizer.step()
#
#                 for iter in range(B):
#
#
#                     disc_optimizer.zero_grad()
#                     loss = Batch_fake_loss(disc_weights, gen_weights, hat_noise)
#                     loss.backward()
#                     torch_utils.clip_grad_norm_(disc_weights, max_norm=max_norm)
#                     disc_optimizer.step()
#
#                 for iter in range(C):
#
#
#                     gen_optimizer.zero_grad()
#                     loss = Batch_gen_loss(disc_weights, gen_weights, hat_noise)
#                     loss.backward()
#                     # print(torch.gradient(gen_weights))
#                     grad_norm = (torch.sqrt(torch.sum(torch.pow(gen_weights.grad, 2)))).item()
#                     torch_utils.clip_grad_norm_(disc_weights, max_norm=max_norm)
#                     gen_optimizer.step()

# train_amplitude()