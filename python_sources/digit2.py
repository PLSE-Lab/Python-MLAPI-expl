#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#https://www.kaggle.com/anirbanshaw24/convnet-digit-recognizer
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from numpy import newaxis
from keras.optimizers import RMSprop
from keras.optimizers import SGD
import skimage
from skimage import transform
from skimage import util
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D


# In[ ]:


train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
test.head()
#train.head()


# In[ ]:


x_train = train.iloc[:,1:]
y_train = train.iloc[:,0]
x_test = test.iloc[:,:] 


# In[ ]:


x_train.shape[0]


# In[ ]:


def rotate_image(X, degrees):
    X_flip = []
    for i in range(x_train.shape[0]):
        img = np.array(X.iloc[i,:]).reshape((28, 28))
        img = skimage.transform.rotate(img, degrees)
        X_flip.append(img.reshape((784)))
    X_trfr = np.array(X_flip)
    X = np.concatenate((X, X_flip))
    return X


# In[ ]:


def noise_image(X):
    X_flip = []
    for i in range(10000):
        img = np.array(X.iloc[i,:]).reshape((28, 28))
        img = skimage.util.random_noise(img, mode='pepper')
        X_flip.append(img.reshape((784)))
    X_trfr = np.array(X_flip)
    X = np.concatenate((X, X_flip))
    return X


# In[ ]:


def scale_up_image(X, scale):
    X_flip=[]
    for i in range(10000, 15000):
        img = np.array(X.iloc[i,:]).reshape((28,28))
        img = skimage.transform.rescale(img, scale, clip = True)
        img = skimage.util.crop(img, ((0,28),(0,28)))
        X_flip.append(img.reshape((784)))
    X_trfr = np.array(X_flip)
    X = np.concatenate((X, X_flip))
    return X


# In[ ]:


def translate_image(X, h, w):
    X_flip = []
    M = np.float32([[1, 0, h], [0, 1, w]])
    for i in range(10000,20000):
        img = np.array(X.iloc[i,:]).reshape((28, 28))
        img = img.astype(np.float32)
        img = cv2.warpAffine(img, M, (28, 28))
        X_flip.append(img.reshape((784)))
    X_trfr = np.array(X_flip)
    X = np.concatenate((X, X_flip))
    return X


# In[ ]:


def plot_digits(X, Y, shape):
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.tight_layout()
        plt.imshow(np.array(X.iloc[i,:]).reshape((28,28)), interpolation='none', cmap='gray')
        plt.title('Digit:{}'.format(Y[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


# In[ ]:


X = rotate_image(x_train, 30)


# In[ ]:


X.shape


# In[ ]:


y= np.concatenate((y_train, y_train))
type(y_train)


# In[ ]:


X = noise_image(pd.DataFrame(X))


# In[ ]:


X.shape


# In[ ]:


y= np.concatenate((y, y_train[:10000]))
y.shape


# In[ ]:


#X = translate_image(pd.DataFrame(X), 1.5, 1.5)


# In[ ]:


#y= np.concatenate((y, y_train[10000:20000]))
#y.shape


# In[ ]:


#X.shape


# In[ ]:


#noise added
plot_digits(pd.DataFrame(X[84000:]), y[84000:], 28)


# In[ ]:


#rotated 30 degrees
plot_digits(pd.DataFrame(X[42000:]), y[42000:], 28)


# In[ ]:


#translated image
#plot_digits(pd.DataFrame(X[94000:]), y[94000:], 28)


# # flipping the digits made it worse-- 6..9
# def flip_digits(X):
#     X_flip = []
#     for i in range(42000):
#         img = np.array(X.iloc[i,:]).reshape((28, 28))
#         img = np.fliplr(img)
#         X_flip.append(img.reshape((784)))
#     X_trfr = np.array(X_flip)
#     #X = np.concatenate((X, X_flip))
#     return X_trfr
# 
# X_flip = flip_digits(x_train)
# X = np.concatenate((X, X_flip))
# 
# y = np.concatenate((y_train, y_train))
# y= np.concatenate((y, y_train))
# type(y)

# In[ ]:


#label encoding my training ys
#from keras.utils.np_utils import to_categorical
#y1 = to_categorical(y, num_classes = 10)
#DIDNT WORK


# In[ ]:


#reshaping training set
bleh=[]
for i in range(X.shape[0]):
    pic1 = [X[i]]  #pic1 --> this is one row of x_train
    v = np.array(np.reshape(pic1,(28,28))) #v is the 2D matrix --> 28X28
    bleh.append(v)


# In[ ]:


arr = np.array(bleh)   #arr has my xtrains!!!
arr.shape


# In[ ]:


#reshaping my test set
bleh_test=[]
for j in range(x_test.shape[0]):
    pic1_test = [x_test.iloc[j,:]]  #pic1 --> this is one row of x_train
    v_test = np.array(np.reshape(pic1_test,(28,28))) #v is the 2D matrix --> 28X28
    bleh_test.append(v_test)


# In[ ]:


arr_test = np.array(bleh_test)   #arr_test has my x_tests!!!
arr_test.shape


# In[ ]:


arr = tf.keras.utils.normalize(arr, axis=1)
arr_test = tf.keras.utils.normalize(arr_test, axis=1)


# In[ ]:


arr_test.shape


# In[ ]:


X = np.array(arr).reshape(-1, 28, 28, 1)


# In[ ]:


# Set the CNN model 
# my CNN architechture is In -> [[Conv2D->relu]*2 
#-> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out
model = Sequential()


# In[ ]:


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


# In[ ]:


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


# In[ ]:


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


# In[ ]:


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.25))
model.add(Dense(512, activation = "relu"))
model.add(Dropout(0.25))
model.add(Dense(10, activation = "softmax"))


# In[ ]:


#model.add(Flatten())


# In[ ]:


sgd = SGD(lr=0.1, decay=0.0, nesterov=True)


# In[ ]:


model.compile(optimizer=sgd,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',factor=0.3, patience=3, min_lr=0.001)


# In[ ]:


#earlystopping=EarlyStopping(patience=3)


# In[ ]:


model.fit(X, y, epochs=40, callbacks=[learning_rate_reduction], validation_split=0.3)


# In[ ]:


y_test = np.array(arr_test).reshape(-1, 28, 28, 1)


# In[ ]:


predictions = model.predict(y_test)


# In[ ]:


print(predictions.shape)


# In[ ]:


pred=[]
for u in range(28000):
    pred.append(np.argmax(predictions[u]))


# In[ ]:


imageId = [l for l in range(1,28001)]


# In[ ]:


d = {'ImageId': imageId, 'Label': pred}
df1 = pd.DataFrame(d)
df1.to_csv('mycsvfile7.csv',index=False)

df1


# In[ ]:




