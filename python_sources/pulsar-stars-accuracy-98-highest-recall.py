#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import the necessary packages
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD
from keras import regularizers

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# In[ ]:


# Loading the dataset
dataset = pd.read_csv('../input/pulsar_stars.csv')


# 1. Once the dataset is loaded, let's see some relevant information about the dataset such as column's titles and types, number of records and shape.

# In[ ]:


# EDA
print(dataset.head())


# 2. Now, let's divide our dataset into two subsets: data and target. In this case, I am doing a Standardization procedure in order obtain zero mean and a standard deviation equals to 1. This preserves Gaussian and Gaussian-like distributions whilst normalizing the central tendencies for each attribute.

# In[ ]:


# Droping the target and assigning the rest to the data variable
data = dataset.drop(['target_class'], axis=1)

# Standardization procedure
scaler = StandardScaler()
data = scaler.fit_transform(data)

target = dataset[['target_class']]


# In[ ]:


# Construct the training and testing splits 
trainX, testX, trainY, testY = train_test_split(data, target, test_size=0.25)


# Machine Learning algorithms works better when integer labels are transformed into vector labels. In order to accomplish this transformation I will instantiate a LabelBinarizer object and apply the transformation methods into our trainY and testY sets.

# In[ ]:


lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)


# With keras, it is possible to define models to our neural network (nn). In this case, we are going to work with a Sequential nn, which is just the nn as we already know, i.e., each layer has as its input the output of the former layer. It is worth mention that our neural network is 8-4-2-1.

# In[ ]:


# Defining the model
model = Sequential()
model.add(Dense(4, input_shape=(8,), activation='sigmoid'))
model.add(Dense(2, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))


# We are going to use the Stochastic Gradient Descent technique as an optimizer, initially with a learning rate = 0.12 and a momentum = 0.4. Furthermore, as this is a binary classification problem, in this case a common loss function to use is the binary cross-entropy. However, as we have imbalanced classes, we need to tell the network that during the training, the positive class is more important than the negative. This has been done through the class weight dictionary, in this case, I am telling the network that during the training data positive points are 2 times more important than the negative ones.

# In[ ]:


sgd = SGD(0.12, momentum=0.4)

model.compile(loss='binary_crossentropy', optimizer=sgd,
    metrics=["accuracy"])

class_weight = {0 : 1., 1 : 2.}

H = model.fit(trainX, trainY, validation_data=(testX, testY), 
              batch_size=128, epochs=200, class_weight=class_weight, verbose=0)

scores = model.evaluate(testX, testY, verbose = 0)


# At this point we are ready to analyze the results from our neural network. Calling the .predict method on our model will give us the predictions from our testing set. In addition, as the output from our network is given by the sigmoid activation function, the outputs values are real number in the range [0,1], so, we need to apply a step function to threshold the outputs to binary class labels. Lastly, we print a report showing us the performance of the model.

# In[ ]:


predictions = model.predict(testX, batch_size=128)

# apply a step function to threshold the outputs to binary
# class labels
predictions[predictions < 0.5] = 0
predictions[predictions >= 0.5] = 1

report = classification_report(testY, predictions, 
                               target_names=['Non-pulsar Star', 'Pulsar Star'])

print('Accuracy = {:.7f}'.format(scores[1]))
print(report)


# **ACCURACY = 0.98**
# 
# It is also important to visualize the confusion matrix of our predictions, this can lead us to a more precise visualization and comprehension about where the numbers in our report came from. 

# In[ ]:


conf_matrix = confusion_matrix(testY, predictions)

# Plot the confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt     

plt.figure(figsize=(10,8))
ax = plt.subplot()
sns.heatmap(conf_matrix, annot=True, ax = ax, fmt='d') #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['Non-Pulsar', 'Pulsar'])
ax.yaxis.set_ticklabels(['Non-Pulsar', 'Pulsar'])
#plt.savefig('confusion_matrix_wcw.png')


# As we want to classify the class labeled 1 (Pulsar stars), we are especially interessed in obtain a low rate of false negatives. What I mean is, our classes are extremely unbalanced, if we only care about accuracy and obtain high rate of false negatives, it means that our network is not performing well on what it was created for. 
# 
# What I did here was tuning the parameters of the network to obtain the least rate of false negatives while increasing accuracy. The drawback of this approach was that as the false negative occurrencies decreased so increased the occurencies of false positives. In a real worl context, this means that we are better at classifying Pulsar stars, meanwhile our network fails more in classify a Non-pulsar Star as a Pulsar Star (There's no free lunch, you know). Particularly, this behavior is worth because we will be predicting better at what the classifier was meant for. For those who are not acquainted with statistics, the false negative occurrencies are in the bottom -left cell of the confusion matrix and the false positive one are at the top-right cell. 

# In[ ]:


# Plotting the curve Epoch vs. Loss/Accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 200), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 200), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 200), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 200), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()


# Note that the training loss is much higher than the validation loss, why is it?
# 
# A Keras model has two modes: training and testing. Regularization mechanisms, such as Dropout and L1/L2 weight regularization, are turned off at testing time.
# 
# Besides, the training loss is the average of the losses over each batch of training data. Because your model is changing over time, the loss over the first batches of an epoch is generally higher than over the last batches. On the other hand, the testing loss for an epoch is computed using the model as it is at the end of the epoch, resulting in a lower loss.
# 
# For more informations, see: http://blog.datumbox.com/the-batch-normalization-layer-of-keras-is-broken/ https://forums.fast.ai/t/validation-loss-lower-than-training-loss/4581/2

# Another Important metric for this problem is the Area Under Roc Curve, which can be seen following. Note that the model achieved a very good result, yieling a value greater than 0.93.

# In[ ]:


fpr_keras, tpr_keras, thresholds_keras = roc_curve(testY, predictions)

auc_keras = auc(fpr_keras, tpr_keras)


# In[ ]:


plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


# In[ ]:




