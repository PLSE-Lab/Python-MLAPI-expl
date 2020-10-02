#!/usr/bin/env python
# coding: utf-8

# This experiment compares MNIST classification example outlined in the **Tensor Flow tutorial**: https://www.tensorflow.org/tutorials/layers with **LeNet-5**, as described in *Gradient-Based Learning Applied to Document Recognition*  **Proc. of the IEEE, Nov 1998**. (See http://yann.lecun.com/exdb/lenet/.)
# 
# Why compare these two architectures, one almost 20 years old!? 
# 
# The first set of six feature maps extracted in LeNet-5 is (28x28), so that motivated me to compare these models -- our TF tutorial model input images are also of the size (28x28) . However, originally LeNet-5 paper shows inputs of size 32x32x1, while we shall still compare them only via (28x28x1) MNIST images. Some other differences:
# 
# LeNet-5 paper suggests an input normalization scheme: pixels vary from -0.1 in intensity to 1.175 to keep the variance at 1, mean at 0. This MNIST data is rescaled, to keep its variance at 1 and mean at 0. Its convolutional layers are followed by AveragePooling layer each, while contrastingly the TF Tutorial uses MaxPooling Layers.  'Sigmoid' activation for LeNet-5, while TF model uses 'Relu'. 'Relu' was adopted after 1998, but 'Sigmoid' is attractive for its no-bias,  and while the network is not deep. Appendix B discusses some details of Sigmoidal functionality. 
# 
# The original Tensorflow tutorial has two (5x5) convolutional layers (and a Maxpooling layer each), followed by a 1024 neuron dense layer -- followed by a softmax layer of 10 units. It yields over ~ 99% validation accuracy, while its input data is normalized. We compare here additionally a 128 neuron dense model, because both these sizes yield similar results (YMMV). SGD is used for both examples. Appendix C of LeNet-5 paper discusses the usefulness of Stochastic Gradient methodolgy over other -- afterall, the title of the document contains these words *Gradient-Based* . TF model uses Dropout, which is not used for LeNet-5.
# 
# **Note:** LeNet-5 paper is quite complex, and training difficulties are highlighted well. I didn't implement the LeNet-5 exactly as described in the paper, but much the same. Interestingly, LeNet-5 used a home-grown (object-oriented) dialect of Lisp, while the TF is a high level framework so much easier to use with its Python based APIs! My earlier experiment for LeNet-5 had wrong convolutional filter size  -- thanks to Chris Deotte for pointing that out. I also had a mix of 'relu' and 'sigmoid' activations there, for no good reason, and it is all sigmoidal now, but with larger number of epochs the validation_accuracy has gone up considerably. Earlier versions of this kernel were without any scaling. Scaling was introduced with higher learning rates for both models.
# 

# In[ ]:


import os
os.environ['KERAS_BACKEND']='tensorflow'
import pandas as pd
import numpy as np
from keras.layers import Convolution2D, Activation, AveragePooling2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import SGD
import matplotlib.pyplot as plt


# In[ ]:


def tf_tutorial_model(dropout=0.4, dense_size=1024, lr=0.0005, act='relu'):
    model = Sequential()
    # first convolutional layer:
    model.add(Convolution2D(32, (5,5), batch_input_shape=(None, 1, 28, 28),
        padding='same', data_format='channels_first'))
    model.add(Activation(act))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same', data_format='channels_first'))
    # second convolutional layer:
    model.add(Convolution2D(64, (5,5), padding='same', data_format='channels_first'))
    model.add(Activation(act))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same', data_format='channels_first'))
    model.add(Flatten())
    # first dense layer:
    if dense_size > 0:
        model.add(Dense(dense_size))
        model.add(Activation(act))
        if dropout > 0:
            model.add(Dropout(dropout))
    # final (dense) layer:
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    sgd = SGD(lr=lr)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[ ]:


def lenet_5_model(lr=0.1, act='sigmoid'):  # see Fig 2 in LeNet-5/Lecun.
    model = Sequential()
    # first convolutional layer:
    model.add(Convolution2D(6, (5,5), batch_input_shape=(None, 1, 28, 28),
        padding='same', data_format='channels_first'))
    model.add(Activation(act))
    model.add(AveragePooling2D(pool_size=2, strides=2, padding='same', data_format='channels_first'))
    # second convolutional layer:
    model.add(Convolution2D(16, (5,5), padding='valid', data_format='channels_first'))
    model.add(Activation(act))
    model.add(AveragePooling2D(pool_size=2, strides=2, padding='same', data_format='channels_first'))
    model.add(Flatten())
    # full connection i
    model.add(Dense(120))
    model.add(Activation(act))
    # full connection ii
    model.add(Dense(84))
    model.add(Activation(act))
    # final (dense) layer:
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    sgd = SGD(lr=lr)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[ ]:


# little plotter helper to see results. various dense layer data is plotted simultaneously against the epochs.
def plotxy(hist, title="tf-mnist & LeNet-5", ytitle="accuracy", xtitle="epoch", legend=()):
    fig = plt.figure(figsize=(15, 10)) #change dpi to get bigger resolution..
    for h in hist:
        plt.plot(h)
    plt.ylabel(ytitle)
    plt.xlabel(xtitle)
    plt.legend(legend, loc='lower right')
    plt.title(title)
    plt.show()
    plt.close(fig)


# In[ ]:


# this is just to pull data off the history object. Look at keras documentation for history. It gives back data in columns, and their labels.
def get_data_n_legend(hist, id):
    plot_data = []
    legend = []
    for d in sorted(hist.keys()):
        plot_data.append(hist[d].history[id])
        legend.append(str(d))  # legends e.g. d-128, d-256 denote the dense layer size.
    return((plot_data, legend))
        


# In[ ]:


def list_val_acc_fm_history(hist, column='Dense layer'):
    fmt_str='{0:11} :   {1:6}'
    print(fmt_str.format(column, id))
    for d in sorted(hist.keys()):
        print(fmt_str.format(str(d), round(hist[d].history[id][-1], 3)))


# In[ ]:


print(os.listdir("../input"))
train_data = pd.read_csv('../input/train.csv')
x_train = train_data.drop('label',axis=1).values.reshape(-1,1,28,28)
y_train = np_utils.to_categorical(train_data['label'])


# In[ ]:


print("Original Data-set, Variance:", round(np.var(x_train), 3), "  Mean:", round(np.mean(x_train), 3),   "  Min:", round(np.min(x_train), 3), "  Max:", round(np.max(x_train), 3))


# In[ ]:


n_train = x_train/ np.sqrt(np.var(x_train))
n_train = n_train - np.mean(n_train)


# In[ ]:


print("Rescaled Data-set, Variance:", round(np.var(n_train), 3), "  Mean:", round(np.mean(n_train), 3),   "  Min:", round(np.min(n_train), 3), "  Max:", round(np.max(n_train), 3))


# Let us train the CNN from 50% of the input csv file, and use the other 50% for testing/validation:

# In[ ]:


hist = {}


# In[ ]:


for dense_units in (128, 1024): 
    modld = tf_tutorial_model(dense_size=dense_units, lr=0.1)
    print("Training for dense layer of size:", dense_units)
    hist["TF-" + str(dense_units)+"-norm"] = modld.fit(n_train, y_train, epochs=64, batch_size=32, validation_split=0.5)


# In[ ]:


modl5 = lenet_5_model(lr=1.0) # see Fig 2 in LeNet-5/Lecun.
print("Training LeNet-5 model")
hist["LeNet-5-norm"] = modl5.fit(n_train, y_train, epochs=64, batch_size=32, validation_split=0.5)   # normalize 0--255 to -0.1--1.175


# In[ ]:


id = 'val_acc'
(pl_data,legnd) = get_data_n_legend(hist, id)
plotxy(pl_data, title="tf-mnist (d-1024 & d-128) and lenet-5 (d-120) validation-accuracy comparison:", xtitle='epoch', legend=legnd, ytitle=id)


# In[ ]:


list_val_acc_fm_history(hist)


# **Comparison: (Trainable parameters & Validation Accuracy) **
# 
# Seen in one test run. *Note: These results should vary somewhat amongst various test-runs*.
# 
# * LeNet-5: (~120K.  ~98.5 %). 
# * 128-neuron Dense layer TF model: (~450K.  ~ 99 %). Once again, results are very similar between TF-1024 and TF-128.
# * Normalizing the data as per the original paper (mean=0, var=1) helped both models increase validation accuracy by > 0.25%.
# * Another simpler way to Normalizing the original input range: just by divinding the original input (min=0, max=255) by 255 gave similar results.
# * Large number of epochs are used here, as Sigmoidal function converges slower. Relu would converge much faster of course.
# 
# 

# **Summary:**
# * It is quite exciting to compare a state of the art CNN model of 1998 with what can be so easily acheived today in 2018. 
# * This experiment made me better appreciate the progress this field has made! And we are still discussing the good old MNIST! 
# * Sigmoidal activation functions can be very useful, and yield great accuracy -- but with slower convergence.
