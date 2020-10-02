#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# This notebook documents my look into the Digit Recognizer competition. Several models (Logistic Regression, Multiple Layer Perceptrons and Convolutional Neural Networks) are used to predict labeling of handwriting data. 

# In[ ]:


#%% Data extraction
train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')
Y = train.label
X = train.drop('label', axis = 1)
X_test = test


# In[ ]:


#%% Data standardization
X = X/255
X_test = X_test/255


# In[ ]:


#%% Data exploration
import matplotlib.pyplot as plt
import seaborn as sns
fig, axs = plt.subplots()
g = sns.countplot(Y)


# From the histogram we can see that the data is fairly evenly distributed, and would not require special touches for rare classes. Let's look into the data a little more by printing some of the digit images provided:

# In[ ]:


#%% extract first 10 occurances of each digit for visualization
from collections import defaultdict
occurances = defaultdict(list)
for i in range(10):
    for digit in range(10):
        occurances[digit].append(Y[Y==digit].index[i])

fig, axes = plt.subplots(10,10, sharex = True, sharey = True, figsize = (10,12))
axes = axes.flatten()

for digit in occurances:
    for i in range(len(occurances[digit])):
        image = X.values[occurances[digit][i]].reshape(28,28)
        axes[digit*10 + i].imshow(image, cmap = 'gray')
        axes[digit*10 + i].axis('off')
        axes[digit*10 + i].set_title(digit)
plt.tight_layout()


# A quick scan through the sample images show that for some of the training data provided, it could prove difficult to identify the digit even as a human (for example, the 7th 6 could easily be identified as a 4).

# # 1) Logistic Regression Model
# Logistic Regression models are simple and often it is useful to get a good feel of the dataset by using a relatively interpreble statistical model. The maximum number of iterations was set to be 1000 because the convergence was above tolerance for 100 iterations. 
# 
# This simple Logistic Regression model without any cross validation was able to achieve a test accuracy of 92%. Our neural network models will need to do much better then!

# In[ ]:


#%% Simple Logistic Regression
from sklearn.linear_model import LogisticRegression
LR_model = LogisticRegression(max_iter = 1000, random_state = 0, solver='lbfgs', multi_class = 'multinomial').fit(X,Y)
prediction = LR_model.predict(X_test)

#%% Print result to CSV
prediction = pd.DataFrame(prediction, columns = ['Label'])
prediction.index += 1
prediction.to_csv(index_label = 'ImageId',path_or_buf = 'LR_model.csv')


# # 2) Multiple Layer Perceptron using scikit-learn
# scikit-learn provides a built-in MLP model with the MLPClassifier. Using a validation set approach, the model was trained to have 2 hidden layers and a test accuracy rate of 0.98128 - on par with human recognition capabilities. 
# 
# The downside however, is that the training is quite time-intensive, as the input feature space and dataset is quite large (784 pixels and over 40000 images)

# In[ ]:


#%% Simple Logistic Regression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
MLP_model = MLPClassifier(random_state = 0)
param_grid = {'hidden_layer_sizes':[(100,),(350,150),(250,150,50)]}
bestModel = GridSearchCV(MLP_model, param_grid, verbose = False, cv = 2).fit(X,Y)
prediction = bestModel.predict(X_test)
#%% Print result to CSV
prediction = pd.DataFrame(prediction, columns = ['Label'])
prediction.index += 1
prediction.to_csv(index_label = 'ImageId',path_or_buf = 'MLP_model_CV.csv')


# # 3) Convolutional Neural Network using Keras
# 
# Keras is high-level wrapper API for tensorflow, which utilizes more of the GPU's computing power to train deep neural networks. This is advantageous over scikit-learn, which is a higher-level library. The model uses 2 covolutional layers of 3x3 kernels, with relu activation for hidden layers and softmax for output layer.
# 
# The advantage of CNN vs MLP shines in images with medium to larger resolutions - for a mere 28x28 BW pixel image such as the dataset given in this competition, we must use 784 inputs as featurespace. If we were given 1200x800 RGB images to identify, using MLP would not be practical. Convolutional and pooling layers serve to reduce the input dimension while preserving important features in the image. 

# In[ ]:


#%% re-extract data
train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')
Y = train.label
X = train.drop('label', axis = 1)
X_test = test

#%% Data standardization
X = X/255
X_test = X_test/255


# In[ ]:


#%% Split data in to training and validation set
split = 35700
X_train = X[:split]
Y_train = Y[:split]

X_val = X[split:]
Y_val = Y[split:]

#%% categorize data
from keras.utils.np_utils import to_categorical
Y_train = to_categorical(Y_train,10)
Y_val = to_categorical(Y_val,10)


# To use Keras' Conv2D method, we must reshape the feature space to widthXheighXdepth, with depth being 1 for BW images.

# In[ ]:


#%% reshape input feature dimension from split X 784 to split X 28 X 28
img_rows, img_cols = 28, 28
input_shape = (28,28,1) # the input shape is 28x28x1 because the pixels are BW
X_train = X_train.values.reshape(split, img_rows, img_cols,1)
X_val = X_val.values.reshape(len(X)-split, img_rows, img_cols,1)


# Train model with 20 epochs.

# In[ ]:


#%% Create Keras Sequential model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

CNN_model = Sequential()
batchsize = int(split/20)

CNN_model.add(Conv2D(32, kernel_size = (3,3), activation = 'relu', input_shape = input_shape))
CNN_model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
CNN_model.add(MaxPooling2D(pool_size = (2,2)))
CNN_model.add(Dropout(0.25))
CNN_model.add(Flatten())
CNN_model.add(Dense(128, activation = 'relu'))
CNN_model.add(Dropout(0.33))
CNN_model.add(Dense(10, activation = 'softmax'))

CNN_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
history_CNN = CNN_model.fit(X_train, Y_train, validation_data = (X_val, Y_val), epochs = 20, batch_size = batchsize)


# A validation accuracy of 98.79% is quite impressive and an improvement over our MLP model using scikit-learn. A closer look at some of the misidentified labels show that they are actually quite difficult to tell even by human eyes, with the model labeling and the correct labeling in the bracket.

# In[ ]:


#%% Prediction of training set
X = X.values.reshape(len(X), img_rows, img_cols,1)
train_pred = CNN_model.predict_classes(X)
sum(train_pred != Y.values)

digits = []
i = 0
for i in range(len(Y.values)):
    if Y.values[i] != train_pred[i]:
        digits.append(i)

fig, axes = plt.subplots(1, 10, sharex = True, sharey = True, figsize = (10,2))
axes = axes.flatten()
for i in range(10):
    image = X[digits[i]].reshape(img_rows, img_cols)
    axes[i].imshow(image, cmap = 'gray')
    axes[i].axis('off')
    axes[i].set_title(str(train_pred[digits[i]]) + ' ('+str(Y[digits[i]]) +')')
plt.tight_layout()


# # 4 Conclusions
# 
# 1) For this ideal dataset, even logistic regression performed quite well (92% accuracy without any cross validation). The usual classifiers (trees, SVM etc.) are expected to perform similarly well and even better (95%+) with more model tuning. 
# 2) Multiple Layer Perceptron model using 2 hidden layers was able to achieve an accuracy of 98% using scikit-learn. A CNN model with two convolutional layers and one dense layer performed slightly better (close to 99%). Training both models prooved time-intensive. 
# 
# Further improvement could always be expected from parameter tuning and deepening, since deeper layers generally yield better results. A closer look into the accuracy - time tradeoff could also be interesting, since for certain cases high training and prediction costs may not justify a more complex model over a simpler one. 
