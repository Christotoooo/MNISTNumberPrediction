from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

train_images = pd.read_pickle('train_max_x')
test_images = pd.read_pickle('test_max_x')


def Preprocessing():
    coord = []
    for i in range(len(train_images)):
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
            elif kmeans.labels_[i] == 0:
                one.append(data[i])
            else:
                two.append(data[i])
        coord.append([zero, one, two])


if __name__ == "__main__":
    Preprocessing()
