# Simple CNN with no regularization layer nor data augmentation. Gets top result in 5 epochs!
# The 'RFA' objective and prediction steps are from https://arxiv.org/abs/1904.10387

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from keras.utils import np_utils
from tensorflow.keras.layers import Dense, Activation, Dropout
import tensorflow as tf
from tensorflow.linalg import transpose, inv, trace
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, concatenate, Lambda

import pandas as pd
import numpy as np

# Read data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
y_train = np_utils.to_categorical(train["label"].values)
x_train = (train.drop(labels = ["label"], axis = 1).values).astype('float32')
x_test = (pd.read_csv('../input/test.csv').values).astype('float32')
del train
scale = np.max(x_train)
x_train = x_train.reshape(-1,28,28,1)/scale
x_test = x_test.reshape(-1,28,28,1)/scale

# bs = mini-batch size. Make sure 42000 % bs = 0,therwise, a small last batch 
# would cause stability issues
bs = 100

num_epochs = 5
num_run = 10

num_feat = 10

def create_model():
    # We need one network producing features for the first data type 
    # (the images in this example)
    in1 = Input(shape=(28,28,1)) 
    x = Conv2D(32, kernel_size=(3, 3), strides=(1,1), activation='relu')(in1)
    x = Conv2D(64, kernel_size=(4, 4), strides=(2,2), activation='relu')(x)
    x = Conv2D(64, kernel_size=(3, 3), strides=(1,1), activation='relu')(x)
    x = Conv2D(128, kernel_size=(4, 4), strides=(2,2), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)

    out1 = Dense(num_feat)(x)

    # and another network producing features for the second data type
    # In this example these are the labels, but the one-hot encoding already 
    # represents the optimal features, so we just pass it through
    in2 = Input(shape=(10,))
    out2 = Lambda(lambda x: x)(in2)

    # We will use each models separately for predictions...
    feat1 = Model(inputs=in1, outputs=out1)
    feat2 = Model(inputs=in2, outputs=out2)

    # ...but together for training as they have a joint loss function
    model = Model(inputs=[in1, in2], outputs=concatenate([out1, out2], axis=1))

    model.compile(optimizer=Adam(lr=0.001, decay=0.001), loss=RFA_Loss)

    return feat1, feat2, model


D = tf.constant(1e-8 * np.identity(num_feat), tf.float32)

def get_batch_size(X):
    return tf.cast(tf.shape(X)[0], tf.float32)

# computes the covariances between batches of features F of X and G of Y
def cov(F, G):
    n = get_batch_size(G) 
    K = transpose(F)/n @ F
    L = transpose(G)/n @ G
    A = transpose(F)/n @ G
    return K, L, A

# relevance of features given their covariances
def relevance(ker):
    K, L, A = ker
    return trace(inv(K + D) @ A @ inv(L + D) @ transpose(A))

# produces a matrix which maps a vector of features on X to the inferred (expected) value of Y
def inferY(ker, G, Y):
    n = get_batch_size(G)  
    K, L, A = ker
    return transpose(Y)/n @ G @ inv(L + D) @ transpose(A) @ inv(K + D)

# produces a matrix inferring X from Y
def inferX(ker, F, X):
    n = get_batch_size(F)
    K, L, A = ker
    return transpose(X)/n @ F @ inv(K + D) @ A @ inv(L + D)

def RFA_Loss(dummy, features):
    F, G = tf.split(features, 2, axis=1)
    return num_feat - relevance(cov(F, G)) 


# keras really wants us to have a target, although we don't.
dummy = np.zeros(y_train.shape[0])

for run_count in range(num_run):
    
    feat1, feat2, model = create_model()
    
    model.fit([x_train, y_train], dummy, epochs=num_epochs, batch_size=bs, shuffle=True, verbose=2) 

    sess = tf.Session()  # because we're not using tf 2.0

    # compute the features on the training images
    F = feat1.predict(x_train, batch_size=bs, verbose=0)
    G = feat2.predict(y_train, batch_size=bs, verbose=0)

    # produce a matrix mapping the output of the model to a prediction 
    # (average over the posterior)
    P = inferY(cov(F, G), G, y_train)

    # label predictions on test data
    tF = feat1.predict(x_test, batch_size=bs, verbose=0)
    y_pred = sess.run(tF @ transpose(P))

    sess.close()
    
    if run_count == 0: predictions = y_pred
    else: predictions += y_pred
        
results = pd.Series(np.argmax(predictions, axis=1), name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("MNIST-CNN-RFA.csv", index=False)