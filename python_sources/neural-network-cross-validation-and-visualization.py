#!/usr/bin/env python
# coding: utf-8

# # Neural network train, cross-validation and visualization
# Example from [MachineLearningMastery](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/) website.
# Dataset from [patient medical record data for Pima Indians](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv) with information whether they had an onset of diabetes within five years. The dataset is described [here](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names).

# In[ ]:


# first neural network with keras make predictions
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
# load the dataset
dataset = loadtxt('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]
# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10, verbose=0)
# evaluate the keras model
_, accuracy = model.evaluate(X, y, verbose=0)
# make class predictions with the model
predictions = model.predict_classes(X)
# show input/output sets
print("Original dataset size: {}\nInput dataset size:    {}\nOutput dataset size:   {}\n".format(dataset.shape, X.shape, y.shape))
# show accuracy
print('Accuracy: %.2f\n' % (accuracy*100))
# summarize the first 5 cases
print('[Patient variables] => Predicted class (Real class)')
for i in range(5):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))


# Cross-validation example from [MachineLearningMastery](https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/) website.
# Dataset from [patient medical record data for Pima Indians](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv) with information whether they had an onset of diabetes within five years. The dataset is described [here](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names).

# In[ ]:


# MLP for Pima Indians Dataset with 10-fold cross validation
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv', delimiter=',')
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X, Y):
  # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(X[train], Y[train], epochs=150, batch_size=10, verbose=0)
    # evaluate the model
    scores = model.evaluate(X[test], Y[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))


# Viz example from [TowardsDataScience](https://towardsdatascience.com/visualizing-artificial-neural-networks-anns-with-just-one-line-of-code-b4233607209e) website.
# Dataset from [patient medical record data for Pima Indians](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv) with information whether they had an onset of diabetes within five years. The dataset is described [here](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names).

# In[ ]:


get_ipython().system('pip3 install ann_visualizer')
get_ipython().system('pip install graphviz')


# In[ ]:


from ann_visualizer.visualize import ann_viz
from graphviz import Source

ann_viz(model, view=True, title="Neural network plot")
Source.from_file('network.gv')

