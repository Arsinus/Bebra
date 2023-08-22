import numpy as np
import torch
from dataloader import *
from loss_utils import *
from utils import *
import os

# path = 'weights'
# files = os.listdir(path)
# files.sort()
# file_indexes = [list(map(int, file[6:-4].split('-'))) for file in files]
# num_cls = max(list(set(i[0] for i in file_indexes))) + 1
#
# best_weights = ['' for _ in range(num_cls)]

def load_weights(path):
    with open(path, 'r') as f:
        disc_weights, gen_weights = f.readlines()
        disc_weights = torch.tensor(list(map(float, disc_weights.split(' ')[:-1])))
        gen_weights = torch.tensor(list(map(float, gen_weights.split(' ')[:-1])))
    return disc_weights, gen_weights

# def gen_best_weights():
#     best_weights = list()
#     for num in range(num_cls):
#
#         cls_files = list()
#
#         for file in files:
#             path = 'weights/{}'.format(file)
#             if file[6] == num:
#                 # disc_weights, gen_weights = load_weights(path)
#                 cls_files.append(path)
#
#         loss_list = np.zeros(num_cls)

# def check_noise(disc_weights, gen_weights, res):
#     hellinger_loss = list()
#     for i in range(res):
#         i /= res
#         i *= 2
#         print(i)
#         hellinger_loss.append(
#             hellinger_distance(classes_data[1], get_samples(NUM_SAMPLES_HELL, disc_weights, gen_weights, i),
#                                20))
#         show_lanent_space(i, disc_weights, gen_weights, classes_data[1], image_size)
#     print(hellinger_loss)



# disc_weights, gen_weights = load_weights('weights/model-2-3.txt')
#
# check_noise(disc_weights, gen_weights, 20)