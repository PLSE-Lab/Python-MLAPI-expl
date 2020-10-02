#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import sklearn.datasets
import numpy as np

# Loading dataset

breast_cancer = sklearn.datasets.load_breast_cancer()

X = breast_cancer.data
Y = breast_cancer.target

print(X)
print(Y)

print(X.shape, Y.shape)

import pandas as pd

data = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)

data['class'] = breast_cancer.target

data.head()

data.describe()

print(data['class'].value_counts())

print(breast_cancer.target_names)

data.groupby('class').mean()

# Train test split

from sklearn.model_selection import train_test_split

X = data.drop('class', axis=1)
Y = data['class']

type(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

print(Y.shape, Y_train.shape, Y_test.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

print(Y.mean(), Y_train.mean(), Y_test.mean())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify = Y)

print(X_train.mean(), X_test.mean(), X.mean())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify = Y, random_state=1)

print(X_train.mean(), X_test.mean(), X.mean())

# Binarisation of input

import matplotlib.pyplot as plt

plt.plot(X_test.T, '*')
plt.xticks(rotation='vertical')
plt.show()

X_binarised_3_train = X_train['mean area'].map(lambda x: 0 if x < 1000 else 1)

plt.plot(X_binarised_3_train, '*')

X_binarised_train = X_train.apply(pd.cut, bins=2, labels=[1,0])

plt.plot(X_binarised_train.T, '*')
plt.xticks(rotation='vertical')
plt.show()

X_binarised_test = X_test.apply(pd.cut, bins=2, labels=[1,0])

type(X_binarised_test)

X_binarised_test = X_binarised_test.values
X_binarised_train = X_binarised_train.values

type(X_binarised_test)

# MP neuron model

from random import randint

b = 3

i = randint(0, X_binarised_train.shape[0])

print('For row', i)

if (np.sum(X_binarised_train[100, :]) >= b):
  print('MP Neuron inference is malignant')
else:
  print('MP Neuron inference is benign')
  
if (Y_train[i] == 1):
  print('Ground truth is malignant')
else:
  print('Ground truth is benign')

b = 3

Y_pred_train = []
accurate_rows = 0

for x, y in zip(X_binarised_train, Y_train):
  y_pred = (np.sum(x) >= b)
  Y_pred_train.append(y_pred)
  accurate_rows += (y == y_pred)
  
print(accurate_rows, accurate_rows/X_binarised_train.shape[0])
  

for b in range(X_binarised_train.shape[1] + 1):
  Y_pred_train = []
  accurate_rows = 0

  for x, y in zip(X_binarised_train, Y_train):
    y_pred = (np.sum(x) >= b)
    Y_pred_train.append(y_pred)
    accurate_rows += (y == y_pred)

  print(b, accurate_rows/X_binarised_train.shape[0])  

from sklearn.metrics import accuracy_score

b = 28

Y_pred_test = []

for x in X_binarised_test:
  y_pred = (np.sum(x) >= b)
  Y_pred_test.append(y_pred)

accuracy = accuracy_score(Y_pred_test, Y_test)

print(b, accuracy)  

# MP Neuron Class

class MPNeuron:
  
  def __init__(self):
    self.b = None
    
  def model(self, x):
    return(sum(x) >= self.b)
  
  def predict(self, X):
    Y = []
    for x in X:
      result = self.model(x)
      Y.append(result)
    return np.array(Y)
  
  def fit(self, X, Y):
    accuracy = {}
    
    for b in range(X.shape[1] + 1):
      self.b = b
      Y_pred = self.predict(X)
      accuracy[b] = accuracy_score(Y_pred, Y)
      
    best_b = max(accuracy, key = accuracy.get)
    self.b = best_b
    
    print('Optimal value of b is', best_b)
    print('Highest accuracy is', accuracy[best_b])

mp_neuron = MPNeuron()
mp_neuron.fit(X_binarised_train, Y_train)

Y_test_pred = mp_neuron.predict(X_binarised_test)
accuracy_test = accuracy_score(Y_test_pred, Y_test)

print(accuracy_test)

# Perceptron Class

X_train = X_train.values
X_test = X_test.values

$y = 1, \mbox{if} \sum_i w_i x_i >= b$

$y =  0, \mbox{otherwise}$

class Perceptron:
  
  def __init__ (self):
    self.w = None
    self.b = None
    
  def model(self, x):
    return 1 if (np.dot(self.w, x) >= self.b) else 0
    
  def predict(self, X):
    Y = []
    for x in X:
      result = self.model(x)
      Y.append(result)
    return np.array(Y)
    
  def fit(self, X, Y, epochs = 1, lr = 1):
    
    self.w = np.ones(X.shape[1])
    self.b = 0
    
    accuracy = {}
    max_accuracy = 0
    
    wt_matrix = []
    
    for i in range(epochs):
      for x, y in zip(X, Y):
        y_pred = self.model(x)
        if y == 1 and y_pred == 0:
          self.w = self.w + lr * x
          self.b = self.b - lr * 1
        elif y == 0 and y_pred == 1:
          self.w = self.w - lr * x
          self.b = self.b + lr * 1
          
      wt_matrix.append(self.w)    
          
      accuracy[i] = accuracy_score(self.predict(X), Y)
      if (accuracy[i] > max_accuracy):
        max_accuracy = accuracy[i]
        chkptw = self.w
        chkptb = self.b
        
    self.w = chkptw
    self.b = chkptb
        
    print(max_accuracy)
    
    plt.plot(accuracy.values())
    plt.ylim([0, 1])
    plt.show()
    
    return np.array(wt_matrix)

perceptron = Perceptron()

wt_matrix = perceptron.fit(X_train, Y_train, 10000, 0.5)

Y_pred_test = perceptron.predict(X_test)
print(accuracy_score(Y_pred_test, Y_test))

plt.plot(wt_matrix[-1,:])
plt.show()

get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import animation, rc
from IPython.display import HTML

# First set up the figure, the axis, and the plot element we want to animate
fig, ax = plt.subplots()

ax.set_xlim(( 0, wt_matrix.shape[1]))
ax.set_ylim((-15000, 25000))

line, = ax.plot([], [], lw=2)

# animation function. This is called sequentially
def animate(i):
    x = list(range(wt_matrix.shape[1]))
    y = wt_matrix[i, :]
    line.set_data(x, y)
    return (line,)

# call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, frames=100, interval=200, blit=True)

HTML(anim.to_html5_video())

