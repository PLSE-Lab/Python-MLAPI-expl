#!/usr/bin/env python
# coding: utf-8

# # **INTRODUCTION**
# * In this kernel, I'll show you my  classification of 10 digits from 0 to 9 with Convolutional Neural Networks.
# * After modeling and results obtained, there will be some plots in which we can observe changes in loss and accuracies of the model which are returned by "history" of Keras library.
# * Also there will be some bar plots with "pandas" and "plotly" comparing the "Feed-forward Neural Network" model which is my preceeding kernel with the same dataset and CNN model.
# * Finally, our classifier will be evaluated with **Precision-Recall metric and F1 score** and plotted as well as **ROC curve and AUC score**. Usually, one of those metrics is enough but I want to demonstrate both in my kernel. 
# 
# Let's start by importing libraries & data.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
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


test = pd.read_csv('../input/test.csv')  # importing data in csv format with pandas library
train = pd.read_csv('../input/train.csv')
train.head()       


# In[ ]:


train.info()


#  # **1. DATA OVERVIEW**
# * The data I used which is imported as 'train' has 785 columns and 42,000 rows and 'test.csv' file which will be used for testing has 28000 samples. 
# * Rows are the number of images of the digits we have in our data.
# * As for the columns, the first column is 'label' column, which contains the given answers, here are some label-encoded digits between 0 - 9 corresponding every images, like in any supervised learning.
# * The rest of the columns are the pixels of our 2D images, whose size is 28x28. Since we use a 2D matrix, it's already converted to 2D by multiplying the pixels with each other.
# 
# Let me show you some of the images below:

# In[ ]:


### Splitting Data & Normalization ###

x = train.drop(['label'], axis=1).values/255 # make input values numpy array, then normalize by dividing with 255.
y = train.label
x_test = test.values/255


# In[ ]:


### Let's check the first 9 handwritten images ###
for i in range(9):   
    
    plt.subplot(3,3,i+1)
    plt.imshow(x[i].reshape(28,28), cmap='gray')
    plt.axis('off') 


# In[ ]:


### Value Counts of Digits & Countplot ###
plt.figure(figsize=(15,5))
sns.countplot(y, palette='icefire')
plt.title('Counts of Each Class')
plt.xlabel('Classes')
plt.ylabel('Counts')
y.value_counts()


#  # **2. PREPROCESSING**
# * Unlike feed-forward neural networks, CNN needs images in 3D format. So it's required to reshape the images.
# *  -1 : sample size;
# *  28, 28 : pixels of the images
# *  1 indicates that there is one color channel which is "grayscale". If it'd be 3, then we would have 3 channels as "Red", "Green" and "Blue".

# In[ ]:


### Reshape ###
x = x.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)


# * Since we have 10 different labels, our labels must be encoded one-hot.
# * If you decide to use some model selection methods like "Grid Search", "Cross Validation" etc. then it's ok for you to apply only label encoding due to the difference of return types of predict method.

# In[ ]:


from keras.utils.np_utils import to_categorical
y = to_categorical(y, num_classes=10)
y.shape


# In[ ]:


### Train Test Split ###
from sklearn.model_selection import train_test_split
X_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size = 0.1, random_state = 42)


#  # **3. CREATING MODEL**

# In[ ]:


### Importing Neural Network Libraries ###
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D


# In[ ]:


### Convolution Part ###

# Assign the framework as "model"
model = Sequential()

# I choose the kernel(filter) as (5,5) matrix with 32 feature maps.
# ReLU activation function is a better choice than tanh to avoid vanishing gradient problem unless if you use "batch normalization".
# Like in feed-forward neural networks, we need to indicate the input shape at the beginning.
model.add(Conv2D(32, (5,5), padding ='Same', activation ='relu', input_shape =(28,28,1)))

# We use "max pooling" method as (2,2) matrix to keep more information after having a smaller matrix in convolution operation
model.add(MaxPool2D(pool_size=(2,2)))

# dropout helps avoding overfitting problem. Generally 0.2 - 0.3 are the best choices
model.add(Dropout(0.25))

# Second convolution layer has 32 feature maps again, but our kernel is an (3,3) matrix now, since we have reduced parameters.
model.add(Conv2D(32, (3,3), padding ='Same', activation ='relu'))

# Max pooling again.
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

# Dropout before fully connected layer
model.add(Dropout(0.25))


# In[ ]:


### Fully Connected Layer ###

# Fully connected layer is where the classification will be done, so we flatten it firstly since we have multiple dimensions.
model.add(Flatten())
# 256 nodes in first layer
model.add(Dense(256, activation = "relu"))
# Dropout to avoid overfitting
model.add(Dropout(0.25))
# Output layer with 10 different labels.
model.add(Dense(10, activation = "softmax"))

# Compile & Fit Model

# Since we have multiple classes, we need to use categorical cross entropy. I used adam as an optimizer.
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fit the model with 392 batch size which is almost half of the pixels, with verbose = 2 showing only a line while in progress and 5 epochs.
# Model parameters are recorded in "history" for further uses.
history = model.fit(X_train, y_train, batch_size=392, verbose = 2, validation_data=(x_validation, y_validation), epochs = 5)


#  # **4. PLOTTING & MODEL EVALUATION**
# * Since we don't have any labels on our real life test data, we can't find the accuracy. Then we take a look at our first 16 predictions.

# In[ ]:


# Demonstrating Predicted Classes

predicted_classes = pd.DataFrame(model.predict(test))
plt.figure(figsize=(10,10)) # Setting the figure size
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(x_test[i].reshape(28,28), cmap='gray')
    plt.title(predicted_classes.iloc[i].idxmax(axis=1)) # "idmax" of pandas library gives us the column name(which are our outputs) of the maximum value in a row
    plt.axis('off') # don't show the axis
    plt.axis('off')


# # Test Loss & Accuracy Visualization
# * The fit() method of Keras model returns a "**history**" object. The history.history attribute is a dictionary recording training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values.
# * In below figure, the visualization of change in validation loss and validation accuracy is shown. We can learn from this graph how many epochs are enough for our model where it started not to show a significant decreasing in loss after a certain epoch. Since the model parameters are assigned as random, it will change in every run of code.
# * You can also see the change in the loss and the accuracy interactively by holding mouse on scatter points

# In[ ]:


loss = go.Scatter(y= history.history['val_loss'], x=np.arange(0,5), mode = "lines+markers", name='Test Loss') 
accuracy = go.Scatter(y= history.history['val_acc'], x=np.arange(0,5), mode = "lines+markers", name='Test Accuracy') 
layout = dict(title = 'Test Loss & Accuracy Visualization',
              xaxis= dict(title= 'Epochs',ticklen= 5,zeroline= True),
              yaxis= dict(title= 'Loss & Accuracy',ticklen= 5,zeroline= True))
data = [loss, accuracy]
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# # Confusion Matrix
# 
# * A confusion matrix gives us the number of correct and incorrect predictions of a classification model compared to the actual outcomes. Size of a confusion matrix is NxN, where N is the number of classes. Performance of such models is commonly evaluated using the data in the matrix.

# In[ ]:


# Importing confusion matrix
from sklearn.metrics import confusion_matrix

# Since we don't have the labels for "test" data like in real life, we will only create a confusion matrix of validation values.

# Predict test values.
y_predicted = model.predict(x_validation)

# Find the column indices of maximum values which corresponds to predicted digits.
# An alternative method to do this, as it's done in subplots above, to convert the matrix to dataframe first, then find maximum column indices with "idxmax".
y_predicted = np.argmax(y_predicted, axis = 1) 
y_true = np.argmax(y_validation, axis = 1) 

# Create the confusion matrix.
confusion__matrix = confusion_matrix(y_true, y_predicted) 

# Plot it!
plt.figure(figsize=(10,10))
sns.heatmap(confusion__matrix, annot=True, linewidths=0.2, cmap="Paired",linecolor="black",  fmt= '.1f')
plt.xlabel("Predicted Labels", fontsize=15)
plt.ylabel("True Labels", fontsize=15)
plt.title("Confusion Matrix", color = 'red', fontsize = 20)
plt.show()


# * Below, CNN accuracies and ANN accuracies of my previous kernel will be concatenated in "data" dataframe in order that we can bar charts grouped side by side in a plot.

# In[ ]:


cnn= pd.DataFrame(history.history).iloc[4]
cnn_accuracies = pd.DataFrame(cnn).T

cnn_accuracies.drop(['loss', 'val_loss'],inplace=True,axis=1)

ann_accuracies = {'acc':[0.8913], 'val_acc': [0.9075]}
ann_accuracies = pd.DataFrame(ann_accuracies)

data = pd.concat([ann_accuracies,cnn_accuracies], ignore_index=True)
data['methods']= ['ANN','CNN']
data


# * You can see the comparison of ANN and CNN accuracies below through pandas plot method.
# * If you prefer to see them in each one bars, you can delete the comment sign #, and see clearly by setting the appropriate colors and opacity(alpha).

# In[ ]:


### Comparing Accuracies via pandas.DataFrame.barplot ###
data.plot(x='methods', y=['val_acc','acc'], kind='bar', figsize=(10,7))
plt.xticks(rotation = 0)
plt.show()

#sns.barplot(data=dataa, x='methods', y='acc', color='yellow' alpha=1)
#sns.barplot(data=dataa, x='methods', y='val_acc', color='red', alpha = 05)


# * Plotly is my all time favoite visualization library.
# * Thanks to its interactive feature, you can observe even the smallest decimals corresponding to the points in graphs just by holding your cursor on it.

# In[ ]:


### Comparing Accuracies via Bar Charts of Plotly ###

bar1 = go.Bar(
                x = data.methods,
                y = data.acc,
                name = 'Train'
                )

bar2 = go.Bar(
                x = data.methods,
                y = data.val_acc,
                name = 'Validation'
                )

data2 = [bar1, bar2]
layout = go.Layout(barmode = 'group')
fig = go.Figure(data = data2, layout = layout)
iplot(fig)


# # Precision - Recall Curve & F1 Score
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
y_predict = model.predict(x_validation)
for i in range(classes):
    precision[i], recall[i], _ = precision_recall_curve(y_validation[:, i], y_predict[:, i])


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


# * Since the preceeding plots cannot be observed clearly, let's see all of them in subplots.

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


# # ROC Curve & AUC Score
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

plt.figure(figsize=(17,40))
for i in range(y_true_roc.shape[1]):
    plt.subplot(5,2,i+1)
    labels.append('ROC curve for class {} & Area = {:f}'.format(i+1, roc_auc[i])) 
    plt.plot(fpr[i], tpr[i], color = colors[i],label=labels[i])
    plt.legend(loc=(.1, .3), prop={'size':15})
    plt.ylim([0.0, 1.03])
    plt.xlim([0.0, 1.03])
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize =15)
    plt.title('ROC curves & AUC scores'.format(i+1), fontsize=15)
plt.show()


# **I'd be glad if this kernel could help you, thanks in advance.**
# # **END**
# 
# 
# 
# 
# 
# 
# 
# 
