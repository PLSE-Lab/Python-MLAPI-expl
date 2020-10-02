#!/usr/bin/env python
# coding: utf-8

# # ** 1. Introduction**
# * In this kernel, I'll show you my multi-label classification of digits from 0 to 9 by using MLP.
# * Precision-recall curve and F1 score as well as ROC curve and AUC score will be demonstrated as metrics which are used in evaluation of classification models
# 
# Let's start by importing libraries & data.

# 

# In[ ]:


# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# import warnings
import warnings
# filter warnings
warnings.filterwarnings('ignore')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.
import os
print(os.listdir("../input"))


# In[ ]:


data = pd.read_csv('../input/digit-recognizer/train.csv')  # importing data in csv format with pandas library
test = pd.read_csv('../input/boratest/test.csv')
data.head()                           #  


# In[ ]:


data.info()


# # **2. Data Overview**
# * The data I used which is imported as 'train.csv' has 785 columns and 42,000 rows. 
# * Rows are the number of images of the digits we have in our data.
# * As for the columns, the first column is 'label' column, which contains the given answers, here are some label-encoded digits between 0 - 9 corresponding every images, like in any supervised learning.
# * The rest of the columns are the pixels of our 2D images, whose size is 28x28. Since we use a 2D matrix, it's already converted to 2D by multiplying the pixels with each other.
# 
# Let me show you some of the images below:

# In[ ]:


x = data.drop(['label'], axis=1).values/255 # make input values numpy array, then normalize by dividing with 255.
for i in range(9):   
    
    plt.subplot(3,3,i+1)
    plt.imshow(x[i].reshape(28,28), cmap='gray')
    plt.axis('off') 


# # **3. Preprocessing**

# In[ ]:


### seperating label (y) values and One-Hot encoding for multi-label classification ###

# It doesen't matter in what format you'll form y, but after one-hot encoding, it must be converted to array by .toarray()
#y = pd.DataFrame(data.label)
y = data.label.values.reshape(-1,1)
x_test = test.values/255
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
y = ohe.fit_transform(y).toarray()

# Now every column in y corresponds to a class.

y.shape


# In[ ]:


### train test split ###

from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.2, random_state = 23)


# # **4. Creating Model**

# In[ ]:


### building ANN function
# importing libraries
from keras.models import Sequential # initializing neural network library
from keras.layers import Dense, Dropout # building layers

# feed-forward neural network classifier is assigned as "model".
model = Sequential()  
# we use dropout in the ratio of 0.25 to prevent overfitting.
model.add(Dropout(0.25)) 
# 8 units for the first layer, also the input shape must be given in this line. 
# ReLU activation function is more useful than tanh function due to vanishing gradient problem.
# weights are initialized as "random uniform".
model.add(Dense(8, activation='relu', kernel_initializer='random_uniform', input_dim = x_train.shape[1])) 
# 16 nodes for the second layer
model.add(Dense(16, activation='relu', kernel_initializer='random_uniform'))
# since we have 10 outputs, in the last layer we need to enter 10 nodes. The output of the softmax function can be used to represent a categorical distribution. 
model.add(Dense(10, activation='softmax', kernel_initializer='random_uniform'))

# we compile our model by using "adadelta" optimizer. 
# since we have categorical outputs, loss function must be the cross entropy. if you use grid search, you need to use "sparse_categoricalentropy".
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

# fit the model with below batch size and number of epochs.
# verbose integers 0,1,2 sets the appearance of progress bar. "2" shows just a line.
history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs = 10, batch_size = 155, verbose = 2)


# # **5. Model Evaluation**

# In[ ]:


# a look on test data

# since we don't have any labels on test data that helps to find accuracy, we take a look at our first 9 predictions.

predicted_classes = pd.DataFrame(model.predict(test)) # make a dataframe from prediction values because their index will be needed.
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[i].reshape(28,28), cmap='gray')
    plt.title(predicted_classes.iloc[i].idxmax(axis=1)) # idmax gives us the column name(which are our outputs) of the maximum value in a row
    plt.axis('off') # don't show the axis


# # **Test Loss & Accuracy Visualization**
# * The fit() method of Keras model returns a "**history**" object. The history.history attribute is a dictionary recording training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values.
# * In below figure, the visualization of change in validation loss and validation accuracy is shown. We can learn from this graph how many epochs are enough for our model where it started not to show a significant decreasing in loss after a certain epoch. Since the model parameters are assigned as random, it will change in every run of code.
# * You can also see the change in the loss and the accuracy interactively by holding mouse on scatter points

# In[ ]:


### Test Loss Visualization ###
loss = go.Scatter(y= history.history['val_loss'], x=np.arange(0,10), mode = "lines+markers", name='Test Loss') 
accuracy = go.Scatter(y= history.history['val_acc'], x=np.arange(0,10), mode = "lines+markers", name='Test Accuracy') 
layout = dict(title = 'Test Loss & Accuracy Visualization',
              xaxis= dict(title= 'Epochs',ticklen= 5,zeroline= True),
              yaxis= dict(title= 'Loss & Accuracy',ticklen= 5,zeroline= True))
data = [loss, accuracy]
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# # **Confusion Matrix**
# 
# * A confusion matrix gives us the number of correct and incorrect predictions of a classification model compared to the actual outcomes. Size of a confusion matrix is NxN, where N is the number of classes. Performance of such models is commonly evaluated using the data in the matrix.

# In[ ]:


# Importing confusion matrix
from sklearn.metrics import confusion_matrix

# Since we don't have the labels for "test" data like in real life, we will only create a confusion matrix of validation values.

# Predict test values.
y_predicted = model.predict(x_valid)

# Find the column indices of maximum values which corresponds to predicted digits.
# An alternative method to do this, as it's done in subplots above, to convert the matrix to dataframe first, then find maximum column indices with "idxmax".
y_predicted = np.argmax(y_predicted, axis = 1) 
y_true = np.argmax(y_valid, axis = 1) 

# Create the confusion matrix.
confusion__matrix = confusion_matrix(y_true, y_predicted) 

# Plot it!
plt.figure(figsize=(10,10))
sns.heatmap(confusion__matrix, annot=True, linewidths=0.2, cmap="Blues",linecolor="black",  fmt= '.1f')
plt.xlabel("Predicted Labels", fontsize=15)
plt.ylabel("True Labels", fontsize=15)
plt.title("Confusion Matrix", color = 'red', fontsize = 20)
plt.show()


# # **Precision - Recall Curve & F1 Score**

# * Now it's time to evaluate our model. We use "Precision-Recall" metric for evaluating classifier models. High scores for both show that the classifier is returning accurate results (high precision), as well as returning a majority of all positive results (high recall).
#     * Precision is defined as the number of true positives(TP) over the number of true positives(TP) plus the number of false positives(FP). = TP / (TP + FP)
#     * Recall is defined as the number of true positives(TP) over the number of true positives(TP) plus the number of false negatives(FN).          = TP / (TP + FN)
#     * The relationship between recall and precision can be observed in the stairstep area of the plot which will be done below.
#     * You can think of positive(P) and negatives(N) as an evaluation between just 2 categories in "confusion matrix" we've plotted in heatmap above. But since we have 10 different labels in our model, we will use one-vs-all method by plotting "precision-recall" graph.

# In[ ]:


from sklearn.metrics import precision_recall_curve
classes = y.shape[1]

precision = dict()
recall = dict()
y_predict = model.predict(x_valid)
for i in range(classes):
    precision[i], recall[i], _ = precision_recall_curve(y_valid[:, i], y_predict[:, i])


# In[ ]:


colors = ['red', 'blue', 'green', 'black', 'cyan', 'purple', 'pink', 'navy', 'yellow', 'brown']
lines = []
labels = []
plt.figure(figsize=(10,10))

for i in range(classes):
    plt.plot(recall[i], precision[i], color=colors[i])
    labels.append('Precision-recall for class {0}'.format(i+1))
    
plt.ylim([0.0, 1.03])
plt.xlim([0.0, 1.03])
plt.xlabel('Recall',fontsize=15)
plt.ylabel('Precision', fontsize = 15)
plt.title('Recall vs Precision',fontsize = 20)
plt.legend(labels, loc=(.3, .3), prop={'size':12})
plt.show()


# * To observe every plot clearly, we can try to see all of them in subplots.

# In[ ]:


colors = ['red', 'blue', 'green', 'black', 'cyan', 'purple', 'pink', 'navy', 'yellow', 'brown']
lines = []
labels = []

plt.figure(figsize=(10,30))

for i in range(classes):
    plt.subplot(5,2,i+1)
    labels.append('Precision-recall for class {}'.format(i+1))
    plt.plot(recall[i], precision[i], color=colors[i], label=labels[i])
    plt.legend(loc=(.1, .3), prop={'size':12})
    plt.title('Class {}'.format(i+1),fontsize = 15)
    plt.xlabel('Recall',fontsize=10)
    plt.ylabel('Precision', fontsize = 10)

    
plt.show()


# * Eventually, to find a specific score of precision-recall for all labels, we need to calculate F1 score. 
# * The F1 score is the harmonic average of the precision and recall, where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.
# * Since we have multiple labels, we need to determine a method in "average" parameter in which I prefered using "macro" which calculate metrics for each label, and find their unweighted mean.
# * Notice that, both arrays including "true" and "predicted" labels must consist of only label values to be able to evaluate F1 score, NOT in one-hot encoded forms. But if you use 'samples' in average parameter, then the classes must be in one-hot encoded form.

# In[ ]:


from sklearn.metrics import f1_score
print('F1 Score: {}'.format(f1_score(y_true, y_predicted, average='macro')))


# # **ROC Curve & AUC Score**

# * ROC curves typically feature true positive rate(**TPR**) on the y axis, and false positive rate(**FPR**) on the x axis
# * Ideal point of a ROC curve is on top left where TP rate is 1 and FP rate is 0.
# * AUC score equals the area under the ROC curve. When it equals 1, then the classification is done without any errors.

# - To find ROC curve in multiclass problems, we need binarized labels.

# In[ ]:


y_predicted = y_predicted.T
y_true = y_true.T

from sklearn.preprocessing import label_binarize
y_true_roc = label_binarize(y_true,classes=[0,1,2,3,4,5,6,7,8,9])
y_pred_roc= label_binarize(y_predicted, classes=[0,1,2,3,4,5,6,7,8,9])

fpr = {} # false positive rate
tpr = {} #  true positive rate
roc_auc = {}
from sklearn.metrics import roc_curve, auc
for i in range(y_true_roc.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_pred_roc[:, i], y_true_roc[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


# In[ ]:


colors = ['red', 'blue', 'green', 'black', 'cyan', 'purple', 'pink', 'navy', 'yellow', 'brown']
lines = []
labels = []

plt.figure(figsize=(10,10))
for i in range(y_true_roc.shape[1]):
    labels.append('ROC curve for class {} & Area = {:f}'.format(i+1, roc_auc[i])) 
    plt.plot(fpr[i], tpr[i], color = colors[i],label=labels[i])
    plt.legend(loc=(.2, .3), prop={'size':15})
    plt.ylim([0.0, 1.03])
    plt.xlim([0.0, 1.03])
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize =15)
    plt.title('ROC curves & AUC scores'.format(i+1), fontsize=15)
plt.show()


# In[ ]:


colors = ['red', 'blue', 'green', 'black', 'cyan', 'purple', 'pink', 'navy', 'yellow', 'brown']
lines = []
labels = []

plt.figure(figsize=(10,30))

for i in range(classes):
    plt.subplot(5,2,i+1)
    labels.append('Precision-recall for class {}'.format(i+1))
    plt.plot(recall[i], precision[i], color=colors[i], label=labels[i])
    plt.legend(loc=(.1, .3), prop={'size':12})
    plt.title('Class {}'.format(i+1),fontsize = 15)
    plt.xlabel('Recall',fontsize=10)
    plt.ylabel('Precision', fontsize = 10)

    
plt.show()


# I'll be thankful if you upvote this kernel, thanks in advance. 
# 
# **END**
