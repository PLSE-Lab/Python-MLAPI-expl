#!/usr/bin/env python
# coding: utf-8

# Convolutional neural networks have been used to find exoplanets (from time series data) with great results. A recent Google/Kepler collaboration has had significant press coverage; the paper is available online: [https://www.cfa.harvard.edu/~avanderb/kepler90i.pdf](https://www.cfa.harvard.edu/~avanderb/kepler90i.pdf).
# 
# Searching for exoplanets in time series data presents similar challenges to image recognition - another area where convolutional neural networks have experienced success. In this case, we are looking for features based on the relationship between a number of consecutive time points (most likely a 'U' shaped dip in brightness ) - analagously, in image recognition we are looking for relationships between a small number of adjacent pixels (i.e., vertical edges, crosses etc.). In both cases the patterns sought are translation invariant (a transit may occur anywhere in a time series, an object may be in different locations in an image) - convolutional neural networks train a similarly tranlation invariant 'mask' to recognise similar features anywhere in a dataset. 
# 
# The paper detailing the The Google/Kepler approach goes into significant detail - I will implement some basic ideas:
# 
# - The research uses TensorFlow to implement the neural network: Keras provides a high level, easy to use Python interface. 
# - Normalizing the time series data is particularly important.
# - Dropout layers are used to prevent overfitting (this is otherwise a significant problem).

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


train_df = pd.read_csv('../input/exoTrain.csv')
test_df = pd.read_csv('../input/exoTest.csv')

print(train_df.head())
print(train_df.info())


# The time series data are not normalized - this will be an important part of preprocessing. I'll also switch to one-hot encoding of the ground truth. 

# In[ ]:


train_df['LABEL'] = train_df['LABEL'].replace([1], [0])
train_df['LABEL'] = train_df['LABEL'].replace([2], [1])
test_df['LABEL'] = test_df['LABEL'].replace([1], [0])
test_df['LABEL'] = test_df['LABEL'].replace([2], [1])


# In[ ]:


train_pr = sum(train_df.LABEL == 1) / float(len(train_df.LABEL))
print('Train set positive rate', train_pr)

test_pr = sum(test_df.LABEL == 1) / float(len(test_df.LABEL))
print('Test set positive rate', test_pr)

total_pr = ((sum(train_df.LABEL == 1) + sum(test_df.LABEL == 1)) 
            /(float(len(train_df.LABEL)) + float(len(test_df.LABEL))))
print('Overall positive rate', total_pr)


# The classes are very unbalanced - less than 1% represent confirmed planets. This will make training more difficult (without using dropout, networks consistenly predict 'no planet').
# 
# Lets have a look at some times series. 

# In[ ]:


def plot_series(series, title):
    plt.plot(np.arange(len(series)), series)
    plt.title(title)
    plt.xlabel('Time step')
    plt.ylabel('Luminosity')
    plt.show()  


# In[ ]:


plot_series(train_df.iloc[6, 1:], 'Time series data: exoplanet confirmed')
plot_series(train_df.iloc[100, 1:], 'Time series data: no exoplanet confirmed')


# Lots of candidate transits on the first example; the second (no planet) shows why this is such a difficult task - a drop in luminosity can be caused by other phenomena. Taking a closer look:

# In[ ]:


plot_series(train_df.iloc[6, 200:500], 'Time series data: exoplanet confirmed')
plot_series(train_df.iloc[100, 2600:2800], 'Time series data: no exoplanet confirmed')


# The top figure appears to show a transit at around x=120. However, a sharp dip in brightness alone does not indicate a transit - as shown in Figure 2. The network will have to look for subtle features. 
# 
# As emphasized in the paper, feature normalization is important for the CNN to work properly.

# In[ ]:


def normalize_rows(input_array):
    normalized = np.zeros(input_array.shape)
    for index in range(input_array.shape[0]):
        series = input_array[index, :]
        normalized[index, :] = (series - np.mean(series)) / np.std(series)
    return normalized


# We can create more positive examples by time reversing the existing examples (Netwon's laws - and therefore the relevant physics for a planetary occultation - are symmetric in time)$. This still leaves a significant class imbalance - it is useful to have as many positive training examples, but it is not necessary to use all the thousands of negative examples. This will slow down the training process without a significant improvement in performance.
# 
# I will also split the training set into training and cross validation sets: with 37 positive examples, a split of 30:7 is approximately 80:20. While the training set can have a higher ratio of positive examples than the base rate, the rates in the CV set should reflect the base rate to enable meaningful model scoring. 
# 
# $ Adding an appropriate level of Gaussian noise could also be used to increase the effective number of training examples (not implemented here). 

# In[ ]:


from sklearn.model_selection import train_test_split

X = normalize_rows(train_df.iloc[:, 1:].values)
y = train_df['LABEL'].values

X_t, X_CV, y_t, y_CV = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def flip_positives(X, y):
    positives = X[y == 1]
    X_flip = np.fliplr(positives)
    X_new = np.vstack([X_flip, X])
    y_new = np.hstack([np.ones(len(X_flip)), y])
    return X_new, y_new

X_train, y_train = flip_positives(X_t, y_t)


# I will sort the training data by label to enable easy limiting of the training set size (without removing positive examples). 
# 
# The feature arrays also need to be reshaped for input into the neural network. 

# In[ ]:


X_train_pos = X_train[y_train==1]
X_train_neg = X_train[y_train==0]

# Add time-reversed positive examples
X_train = np.vstack([X_train_pos, X_train_neg])
y_train = np.hstack([np.ones(len(X_train_pos)), np.zeros(len(X_train_neg))])

# reshape feature arrays for Keras
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_CV = X_CV.reshape(X_CV.shape[0], X_CV.shape[1], 1)

X_test = normalize_rows(test_df.iloc[:, 1:].values)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
y_test = test_df['LABEL'].values

# Check positive rate in CV set
CV_pr = sum(y_CV == 1) / float(len(y_CV))
print('CV set positive rate', CV_pr)
print(sum(y_CV == 1), 'examples with confirmed exoplanets')


# **Model building and tuning**
# 
# As far as possible, I will try to implement a simplified version of the best performing model in the paper, without 'magic numbers' - all hyperparameter choices should be justified either by reference to the paper (or general literature on neural networks), or should be selected via cross validation. 
# 
# The best performing network in the paper above uses over 10 layers, from two separate input arrays (reflecting different methods of preprocessing. I want to keep things simple - I will start with a single convolutional layer. As in the paper, hidden layers use a ReLU activation function, whereas the output layer uses a sigmoid function. Dropout should be applied to each hidden layer to prevent overfitting (0.5 is generally agreed to be a reasonable threshold for hidden layers). Training uses the Adam method, with a binary crossentropy loss function. These choices follow the reported method closely. 
# 
# The convolutional layer needs two hyperparameters - the kernel size ('window'), and the dimensionality of the output space ('filters') - see https://keras.io/layers/convolutional/. There is one significant difficulty - the time series data here are not directly comparable to the (processed) Kepler data (it is difficult to find data online to fully understand this dataset). As a result, I will explore a range of possible kernel sizes and number of outputs (somewhat naively, in the ballpark of the values used in the reported model). I have not attempted to replicate the preprocessing methods. 
# 
# As mentioned above, I will limit the number of negative training examples (to 1000). In previous kernels, I have attempted to use all the negative training examples - this resulted in higher precision but lower recall (the result was robust). In this case, a model would presumably be used to narrow down the candidate pool, and thus achieving high recall is more important that high precision (although clearly both are important).
# 
# I have chosen a batch size of 100 - this will affect computational speed but should not affect the results. Similarly, the number of epochs should be sufficient for the network to reach an optimum (as can be seen by looking at training metrics) and experimentally, the network seems to be close to optimally trained after 100 epochs. These 
# 
# (The original paper uses a batch size of 64 for 50 epochs for training the best model).

# In[ ]:


from keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten, Dropout
from keras.models import Sequential
from sklearn.utils import shuffle
from keras.optimizers import SGD
from sklearn.metrics import *


# In[ ]:


filters_list = [16, 32, 64]
windows = [5, 10]

for filters in filters_list:
    for window in windows:
        model = Sequential((
             Convolution1D(filters, window, input_shape = X_train.shape[1:3], activation = 'relu'),
             MaxPooling1D(),
             Dropout(0.5),
             Flatten(),
             Dense(1, activation='sigmoid'),
             ))
        model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['mae'])
        model.fit(X_train[:1000, :, :], y_train[:1000], batch_size = 100, epochs = 100, verbose = 0)
    
        y_prob = model.predict(X_CV)
        precision, recall, thresholds = precision_recall_curve(y_CV, y_prob, pos_label=1)
        auprc = auc(recall, precision)
        print(filters, 'features, kernel size =', window)
        print('AUPRC:', auprc)
    
        y_pred = y_prob
        for i in range(len(y_prob)):
            y_pred[i] = np.round(y_prob[i])

        cm = confusion_matrix(y_pred, y_CV)
        print('Confusion matrix:\n', cm, '\n')


# The results are not particularly sensitive to the choice of parameters. The model consistently achieves a recall of about half, and a high level of precision.

# In[ ]:


model = Sequential((
    Convolution1D(64, 5, input_shape = X_train.shape[1:3], activation = 'relu'),
    MaxPooling1D(),
    Dropout(0.5),
    Flatten(),
    Dense(1, activation='sigmoid'),
    ))

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['mae'])
model.fit(X_train[:1000, :, :], y_train[:1000], batch_size = 100, epochs = 100, verbose = 0)

y_prob = model.predict(X_test)
precision, recall, thresholds = precision_recall_curve(y_test, y_prob, pos_label=1)
auprc = auc(recall, precision)
print('64 features, kernel size = 5\nTest set performance')
print('AUPRC:', auprc)
    
y_pred = y_prob
for i in range(len(y_prob)):
    y_pred[i] = np.round(y_prob[i])

cm = confusion_matrix(y_pred, y_test)
print('Confusion matrix:\n', cm, '\n')


# Even a single layer performs surprisingly well, and this performance is fairly robust to the details of the convolutional layer. Even without preprocessing I would expected that performance can be improved to some extent by adding further layers - but the kernel timeout precludes the necessary hyperparameter tuning. 
