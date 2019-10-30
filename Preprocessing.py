from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_images = pd.read_pickle('train_max_x')
test_images = pd.read_pickle('test_max_x')


# this is the vectorization that we use
def simple_process():
    vectors = []
    for i in range(len(train_images)):
        for j in range(len(train_images[0])):
            for k in range(len(train_images[0][0])):
                if not int(train_images[i][j][k]) == 255:
                    train_images[i][j][k] = 0
                else:
                    train_images[i][j][k] = 1
        print(train_images[i].shape)
        np.savetxt("result1.csv", train_images[i], delimiter=",")
        vectors.append(train_images[i].flatten())
        print(vectors[i].shape)
        np.savetxt("result2.csv", vectors[i], delimiter=",")
    return vectors


# this one splits the image into three digits, and is not used
def preprocess():
    coord = []
    result = []
    # dif = 0
    for i in range(len(train_images)):
        print(train_images[i].shape)
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
            # plt.show()
        result.append(images)
    return result


if __name__ == "__main__":
    simple_process()
