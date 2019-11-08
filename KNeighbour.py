import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

train_images = pd.read_pickle('train_max_x')
test_images = pd.read_pickle('test_max_x')
train_labels = pd.read_csv('train_max_y.csv').to_numpy()[:, 1]


def zero_process():
    vectors = []
    for i in range(len(train_images)):
        for j in range(len(train_images[0])):
            for k in range(len(train_images[0][0])):
                train_images[i][j][k] /= 255.0
        vector = np.append(train_images[i].flatten(), train_labels[i])
        vectors.append(vector)
    return np.array(vectors)


def predict_and_test():
    train_data = zero_process()
    np.random.shuffle(train_data)
    x_all = train_data[:, :-1]
    y_all = train_data[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=0)
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(x_train, y_train)
    y_pred = neigh.predict(x_test)
    return evaluate(y_pred, y_test)


def evaluate(y_true, y_label):
    total = 0
    for i in range(y_true.shape[0]):
        if y_true[i] == y_label[i]:
            total = total + 1
    return total / y_true.shape[0]


if __name__ == "__main__":
    print(predict_and_test())
