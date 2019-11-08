from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python import keras
# from keras import utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import time

train_images = pd.read_pickle('train_max_x')
test_images = pd.read_pickle('test_max_x')
train_labels = pd.read_csv('train_max_y.csv').to_numpy()[:, 1]
# train_images = shuffle(train_images)
# train_images_test = train_images[-10000:]  #reserved for validation
# train_images = train_images[:-10000]


# this one sets all black bits as 1 and other bits as 0, this is the vectorization that we use
def simple_process():
    x_train, x_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42, stratify=train_labels)
    x_train = x_train.reshape(x_train.shape[0], 128, 128, 1)
    x_val = x_val.reshape(x_val.shape[0], 128, 128, 1)
    x_train = x_train.astype('int')
    x_val = x_val.astype('int')
    x_train //= 255
    x_val //= 255
    y_train = keras.utils.to_categorical(y_train, 10)
    y_val = keras.utils.to_categorical(y_val, 10)

# def simple_process():
#     vectors = []
#     for i in range(len(train_images[0])):
#         for j in range(len(train_images[0])):
#             for k in range(len(train_images[0][0])):
#                 if not int(train_images[i][j][k]) == 255:
#                     train_images[i][j][k] = 0
#                 else:
#                     train_images[i][j][k] = 1
#         plt.imshow(train_images[0], interpolation='none')
#         plt.savefig(f'{str(time.time())}.png')
#         vectors.append(train_images[i].flatten())
#     return vectors


# this one keeps the rgb of each bit and normalizes the colour into a range[0,1], not used
def zero_process():
    x_train, x_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42, stratify=train_labels)
    x_train = x_train.reshape(x_train.shape[0], 128, 128, 1)
    x_val = x_val.reshape(x_val.shape[0], 128, 128, 1)
    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_train /= 255
    x_val /= 255
    y_train = keras.utils.to_categorical(y_train, 10)
    y_val = keras.utils.to_categorical(y_val, 10)


# def zero_process():
#     vectors = []
#     for i in range(len(train_images[0])):
#         for j in range(len(train_images[0])):
#             for k in range(len(train_images[0][0])):
#                 train_images[i][j][k] /= 255.0
#         plt.imshow(train_images[i], interpolation='none')
#         plt.savefig(f'{str(time.time())}.png')
#         vectors.append(train_images[i].flatten())
#     return vectors


# this one splits the image into three digits and keeps each matrix, not used
def complete_process():
    coord = []
    result = []
    # dif = 0
    for i in range(len(train_images[0])):
        res = []
        for j in range(len(train_images[0])):
            for k in range(len(train_images[0][0])):
                if not int(train_images[i][j][k]) == 255:
                    train_images[i][j][k] = 0
                else:
                    res.append([j, k])
        data = np.array(res)
        zero = []
        one = []
        two = []
        kmeans = KMeans(n_clusters=3, random_state=0).fit(data)
        for i in range(len(kmeans.labels_)):
            if kmeans.labels_[i] == 0:
                zero.append(data[i])
            elif kmeans.labels_[i] == 1:
                one.append(data[i])
            else:
                two.append(data[i])
        coord.append([zero, one, two])
    for i, c in enumerate(coord):
        for j, k in enumerate(c):
            coord[i][j].sort(key=lambda x: x[0])
            xmin = coord[i][j][0][0]
            # xmax = coord[i][j][-1][0]
            # if xmax - xmin > dif:
            #     dif = xmax - xmin
            coord[i][j].sort(key=lambda x: x[1])
            ymin = coord[i][j][0][1]
            # ymax = coord[i][j][-1][1]
            # if ymax - ymin > dif:
            #     dif = ymax - ymin
            for a, b in enumerate(k):
                coord[i][j][a][0] -= xmin
                coord[i][j][a][1] -= ymin
    for i, c in enumerate(coord):
        images = []
        for j, k in enumerate(c):
            image = np.full((128, 128), 0)
            for a, b in enumerate(k):
                image[coord[i][j][a][0]][coord[i][j][a][1]] = 255
            images.append(image)
            # plt.imshow(image, interpolation='none')
            # plt.savefig(f'{str(time.time())}.png')
            # plt.show()
        result.append(images)
    return result


# if __name__ == "__main__":
#     simple_process()
#     zero_process()
#     complete_process()
