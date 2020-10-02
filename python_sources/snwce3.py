import os
import sys
import pickle
import numpy as np
import pandas as pd
import PIL
from PIL import Image, ImageEnhance
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss, confusion_matrix
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

np.random.seed(100)


class SigmoidNeuron:
    def __init__(self):
        self.w = None

    def perceptron(self, x):
        return np.dot(self.w, x)

    def model(self, x):
        p = self.perceptron(x)
        return (1.0/(1.0 + np.exp(-p)))

    def predict(self, X):
        Y_pred = []
        x0 = np.ones(X.shape[0])
        X1 = np.c_[X, x0]
        for x in X1:
            Y_pred.append(self.model(x))
        return np.asarray(Y_pred)

    def grad(self, x, y):
        y_pred = self.model(x)
        return (y_pred - y) * x

    def loss(self, Y_pred, Y):
        l = 0
        for y_pred, y in zip(Y_pred, Y):
            l1 = 0
            if (y_pred != 0):
                l -= (y * np.log2(y_pred))
            if (y_pred != 1):
                l -= ((1-y) * np.log2(1-y_pred))
              
        return l

    def fit(self, X, Y, loss, epochs=2, lr=1):
        x0 = np.ones(X.shape[0])
        X1 = np.c_[X, x0]
        self.w = lr * np.ones(X1.shape[1])
        for i in range(epochs):
            dw = 0
            for x,y in zip(X1, Y):
                dw += self.grad(x, y)

            self.w -= (lr * dw)
            # To speed up, do not calculate loss and plot it
            #loss.append(self.loss(self.predict(X), Y))

        return loss


def read_all(folder_path, key_prefix=""):
    '''
    It returns a dictionary with 'file names' as keys and 'flattened image arrays' as values.
    '''
    print("Reading:")
    images = {}
    files = os.listdir(folder_path)
    for i, file_name in enumerate(files):
        file_path = os.path.join(folder_path, file_name)
        image_index = key_prefix + file_name[:-4]
        image = Image.open(file_path)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(15.0)
        image = image.convert("L")
        images[image_index] = np.array(image.copy()).flatten()
        image.close()
    return images


def load():
    languages = ['ta', 'hi', 'en']

    images_train = read_all("../input/level_3_train/level_3/background", key_prefix='bgr_')
    for language in languages:
        images_train.update(read_all("../input/level_3_train/level_3/"+language, key_prefix=language+"_" ))

    images_test = read_all("../input/level_3_test/kaggle_level_3/", key_prefix='') 

    X_train = []
    Y_train = []
    for key, value in images_train.items():
        X_train.append(value)
        if key[:4] == "bgr_":
            Y_train.append(0)
        else:
            Y_train.append(1)

    ID_test = []
    X_test = []
    for key, value in images_test.items():
      ID_test.append(int(key))
      X_test.append(value)
          
                
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)

    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train)
    X_scaled_test = scaler.transform(X_test)

    return X_scaled_train, Y_train, ID_test, X_scaled_test


X, Y, I, T = load()
loss = []
sn = SigmoidNeuron()
#loss = sn.fit(X, Y, loss, 10000, 0.0000018)
loss = sn.fit(X, Y, loss, 2500, 0.000005)

#plt.plot(loss)
#plt.show()

Y_pred = sn.predict(X)
Y_binarised = (Y_pred >= 0.5).astype(int)
print('Accuracy: ', accuracy_score(Y_binarised, Y))


Y_pred_test = sn.predict(T)
Y_pred_binarised_test = (Y_pred_test >= 0.5).astype(int)

submission = {}
submission['ImageId'] = I
submission['Class'] = Y_pred_binarised_test

submission = pd.DataFrame(submission)
submission = submission[['ImageId', 'Class']]
submission = submission.sort_values(['ImageId'])
submission.to_csv("submission.csv", index=False)

