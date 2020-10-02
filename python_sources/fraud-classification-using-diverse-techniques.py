#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Import for model plotting
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from matplotlib.ticker import PercentFormatter
import hdbscan

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/creditcard.csv')
data.head(10)


# # Dataset evaluation
# 
# For the purpose of this analysis, I decided to go for a simple Kmeans clustering along with an HDBSCAN clustering to define the data that are classified as fraud and those thare aren't. To do so, we must first understand the general idea behind the dataset and identify the particularities of the dataset.
# 
# ## Negative and positive values
# 
# We first classify the values by positive and negative to see what percentage of each category represent:
# 

# In[ ]:


classification = data['Class'].value_counts()
print(classification)
N = 2
y_values = classification.values
x_values = [0,1]
ind = np.arange(N)
plt.bar(x_values,y_values)
plt.xticks(ind, ('0','1'))
plt.show()


# ## Data points distribution
# 
# We want to know what is the general distribution of the different features. To do so, we will simply associate the different values present in the features table, flatten the whole set and compute a frequency plot. 

# In[ ]:


value_plot = []
cols_names = ['V'+str(i) for i in range(1,29)]
n_bins = 1000

values_set = data.iloc[:, 1:29].values
values = np.array(values_set).flatten()


plt.hist(values,bins=n_bins)
plt.xlim(-5, 5)
plt.show()



# ## Model configuration
# 
# Now that our distribution is proposed and displayed, we need to define both our Kmeans and HDBScan models. The advantage that we have with kmeans is the fact that the number of classes is already defined. One disadvantage of HDBScan is the fact that the number of classes will be automatically defined through density and linkage analysis, meaning that it may result in more classes that expected.
# 
# ### KMEANS model

# In[ ]:


number_cluster = 2    # We have a fixed number of clusters
random_state = 23     # We define a seed for our random state for result reproduction

def kmeans_evaluation(values_set,number_clusters=2,random_state=23,n_init=100,number_iterations = 10000,tol=0.00001):
    y_pred = KMeans(n_clusters=number_clusters, verbose=2, random_state=random_state,max_iter=number_iterations,tol = tol).fit_predict(values_set)
    return y_pred


# In[ ]:


def acc(y_def,y_pred):
    classification_result = classification_report(y_def, y_pred)
    return classification_result


# In[ ]:


result = acc(data.iloc[:,-1].values,kmeans_evaluation(values_set))
print('\nThe accuracy provided by kmeans =====> \n'+str(result))


# We discover that our kmeans model provides very poor results with a classification of 47.37%. This is really low accuracy, meaning that the model cannot be used for classification purposes. Now we try another unsupervised classification algorithm that is more complex than kmeans and uses linkage and density to increase the quality of the classification.
# 
# ### HDBSCAN model

# In[ ]:


def hdbscan_evaluation(values_set,number_clusters=2,random_state=23,n_init=100,number_iterations = 10000,tol=0.00001):
    y_pred = hdbscan.HDBSCAN().fit(values_set)
    return y_pred.labels_
    


# In[ ]:


#result = acc(data.iloc[:,-1].values,hdbscan_evaluation(values_set))
#print('\nThe accuracy provided by kmeans =====> '+str(result))


# Our accuracy provided by HDBSCAN is not better than the accuracy provided by kmeans. In effect, for this type of exercise, having a predefined number of labels helps for the clustering. Not all elements can be classified using a similarity of patterns such as distance. In this case, we need to see what is the mathematical or even non linear relationship between the different elements analyzed to see what is their influence in the determination of fraud. This is why we go for the next set of models that will be linear regression and logistic regression. Before doing so, we need to define a kfold cross validation that will be used throughout the system for results consistency and statistical validity.
# 
# ### Kfold cross validation
# 
# 

# In[ ]:


from sklearn.model_selection import KFold

# We decide to define 5 folds as we have a small size dataset

n_folds = 5
kf = KFold(n_splits=n_folds)
folds = kf.split(values_set)

# Hence we defined our separation for training and testing. This will be used for training and testing
# It will be applied for the training and testing for supervised training


# ### Linear classification model

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def linear_reg_model(values_set,labels,test_set,test_labels):
    regr = LinearRegression()
    # Train the model using the training sets
    regr.fit(values_set, labels)

    # Make predictions using the testing set
    y_pred = regr.predict(test_set)
    y_pred[y_pred<=0.5] = 0
    y_pred[y_pred>0.5] = 1
    
    return y_pred


# In[ ]:


labels_set = data.iloc[:,-1].values
accuracies = []
for train_index, test_index in folds:
    X_train, X_test = values_set[train_index], values_set[test_index]
    y_train, y_test = labels_set[train_index], labels_set[test_index]
    result = acc(y_test,linear_reg_model(X_train,y_train,X_test,y_test))
    accuracies.append(result)
for accuracy in accuracies:   
    print(accuracy)



# In[ ]:


values_set[:5]


# Setting our threshold to 0.5, we have a training accuracy of 99.89%. For the testing, we have the highest accuracy recorded at 99.91%, hence setting a record in term of classification performance. Now that our linear classification has been set, we can go ahead and work with another classification method called logistic regression. This is another regression problem that define the probabilistic capacity of each entity using a sigmoid function. For this one, we will also implement an ROC curve to determine what threshold provide the best separation for the true positive and true negative values.
# 
# ### Logistic regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

def logistic_reg_model(x_train,x_test,y_train,y_test,iterations,tolerance):
    rt_lm = LogisticRegression(solver='lbfgs',verbose=2, max_iter=iterations,tol = tolerance)
    rt_lm.fit(X_train, y_train)
    y_pred = rt_lm.predict_proba(X_test)[:,1]
    fpr_rt_lm, tpr_rt_lm, threshold = roc_curve(y_test, y_pred)
    
    i = np.arange(len(tpr_rt_lm)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr_rt_lm-(1-fpr_rt_lm), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.ix[(roc.tf-0).abs().argsort()[:1]]

    optimal_threshold = list(roc_t['threshold'])[0]
    
    y_pred[y_pred<=optimal_threshold] = 0
    y_pred[y_pred>optimal_threshold] = 1
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rt_lm, tpr_rt_lm, label='Logistic Regression')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    
    return optimal_threshold, y_pred


# In[ ]:


n_folds = 5
kf = KFold(n_splits=n_folds)
lr_folds = kf.split(values_set)
iterations = 1000
tolerance = 0.01

accuracies = []
thresholds = []
for train_index, test_index in lr_folds:
    X_train, X_test = values_set[train_index], values_set[test_index]
    y_train, y_test = labels_set[train_index], labels_set[test_index]
    threshold, model_pred = logistic_reg_model(X_train,X_test,y_train,y_test,iterations,tolerance)
    result = acc(y_test,model_pred)
    accuracies.append(result)
    thresholds.append(threshold)
    
for accuracy in accuracies:
    print(accuracy)
print(thresholds)


# For our logistic regression, we concluded with a threshold close from 0.001 with a median accuracy of 91.5%. This result proves that we can use logistic regression and end up with satisfying results. Now that we are done with out linear models, we want to define classifiers using Neural Networks, CNN and Reinforcement Networks. They will be for most of them pretty simple and used for simple binary classification. The framework used is Keras. It offers the possibility to work on the GPU but we will focuss on working strictly with the CPU.
# 
# ### Simple Neural Network

# In[ ]:


# Important Keras Imports

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.utils import class_weight

# We define our hyperparameters that will be used in the different models

learning_rate = 0.000025
batch_size = 32
epochs = 100


# In[ ]:


# In this part we define our simple neural network model
# We decided to go for a really simple model for a start

def defineNNModel(training_data,testing_data,validation_data,training_label,testing_label,validation_label,num_classes = 2,
               training_sample=0):
    
    # input image dimensions
    img_x, img_y = 7, 4
    
    class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(training_label),
                                                 training_label)
    
    x_train = training_data.reshape(training_data.shape[0], img_x, img_y, 1)
    x_test = testing_data.reshape(testing_data.shape[0], img_x, img_y, 1)
    x_validation = validation_data.reshape(validation_data.shape[0], img_x, img_y, 1)
    
    y_train = keras.utils.to_categorical(training_label, num_classes)
    y_test = keras.utils.to_categorical(testing_label, num_classes)
    y_validation = keras.utils.to_categorical(validation_label, num_classes)
    
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(500,input_dim=28,activation="sigmoid"))
    model.add(Dense(1000,activation='relu'))
    model.add(Dense(100,activation='sigmoid'))
    model.add(Dense(2,activation='softmax'))
    
    model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.SGD(lr=learning_rate),
              metrics=['acc'])
    
    # Our model will use mean squared error for the estimation of loss, along with a Stochastic Gradient Descent for optimization
    # The metric we want to gather is the accuracy, reason why we define it.
    
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_validation, y_validation),
              class_weight=class_weights)
    
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig("NN_model_accuracy"+str(training_sample)+'.png')
    plt.show()
    
    
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig("NN_model_loss"+str(training_sample)+'.png')
    plt.show()
    
    print(model.summary())
    
    score = model.evaluate(x_test, y_test, verbose=1)
    y_pred = model.predict_classes(x_test)
    report = acc(testing_label,y_pred)
    
    return score[1], score[0], report


# In[ ]:


training_cycle = 0
accuracies_NN = []
losses_NN = []

for train_index, test_index in kf.split(values_set):
    
    print('\n==========  TRAINING CYCLE #'+str(training_cycle)+' ==========\n')
    
    data_train, data_test = values_set[train_index], values_set[test_index]
    label_train, label_test = labels_set[train_index], labels_set[test_index]
    
    # We need to define the validation data and validation labels of our data train and label train
    # We decide to take the last 100 elements of the training data as our validation data
    
    data_validation = data_train[-50000:]
    label_validation = label_train[-50000:]
    
    # We go for our NN model
    accuracy_nn, loss_nn, report = defineNNModel(data_train,data_test,data_validation,label_train,label_test,label_validation,training_sample=training_cycle)
    accuracies_NN.append(accuracy_nn)
    losses_NN.append(loss_nn)
    print(report)
    
    training_cycle+=1   # We update the training iteration


# ### Convolutional Neural Network
# 
# Any array of size N can be converted into a matrix of size NxP representing an image format. This is what we will use for the training of our convolutional neural network. The result provided should be either close or better than the one provided by the simple neural network. We will use a kernel size of 2x2 and a stride of 1

# In[ ]:


def defineCNNModel(training_data,testing_data,validation_data,training_label,testing_label,validation_label,num_classes = 2,
                training_sample=0):
    
    # input image dimensions
    img_x, img_y = 7, 4
    
    # reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
    # because the MNIST is greyscale, we only have a single channel - RGB colour images would have 3
    x_train = training_data.reshape(training_data.shape[0], img_x, img_y, 1)
    x_test = testing_data.reshape(testing_data.shape[0], img_x, img_y, 1)
    x_validation = validation_data.reshape(validation_data.shape[0], img_x, img_y, 1)
    input_shape = (img_x, img_y, 1)
    
    y_train = keras.utils.to_categorical(training_label, num_classes)
    y_test = keras.utils.to_categorical(testing_label, num_classes)
    y_validation = keras.utils.to_categorical(validation_label, num_classes)
    
    class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(training_label),
                                                 training_label)
    
    model = Sequential()
    
    model.add(Conv2D(32, kernel_size=(2, 2),input_shape=input_shape))
    # We first define a convolutional layer with 32 output vectors. The kernel size is set to be 2 by 2
    # The input shape is the same as out image data.
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # pooling is used in convolutional neural networks to make the detection of certain features 
    # in the input invariant to scale and orientation changes
    
    model.add(Conv2D(32, kernel_size=(3, 3),input_shape=input_shape))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    # We flatten our output to enter the fully connected layer
    model.add(Dense(100,activation='sigmoid'))
    model.add(Dense(20,activation='tanh'))
    
    model.add(Dense(num_classes, activation='softmax'))
    # Softmax classification, or output layer, which is the size of the number of our classes 
    
    model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.SGD(lr=learning_rate),
              metrics=['acc'])
    
    # Our model will use mean squared error for the estimation of loss, along with a Stochastic Gradient Descent for optimization
    # The metric we want to gather is the accuracy, reason why we define it.
    
    history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_validation, y_validation),
          class_weight=class_weights)
    
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig("CNN_model_accuracy"+str(training_sample)+'.png')
    plt.show()
    
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig("CNN_model_loss"+str(training_sample)+'.png')
    plt.show()
    
    print(model.summary())
    
    score = model.evaluate(x_test, y_test, verbose=1)
    y_pred = model.predict_classes(x_test)
    report = acc(testing_label,y_pred)
    
    return score[1], score[0], report


# In[ ]:


training_cycle = 0
accuracies_CNN = []
losses_CNN = []

for train_index, test_index in kf.split(values_set):
    
    print('\n==========  TRAINING CYCLE #'+str(training_cycle)+' ==========\n')
    
    data_train, data_test = values_set[train_index], values_set[test_index]
    label_train, label_test = labels_set[train_index], labels_set[test_index]
    
    # We need to define the validation data and validation labels of our data train and label train
    # We decide to take the last 100 elements of the training data as our validation data
    
    data_validation = data_train[-50000:]
    label_validation = label_train[-50000:]
    
    # Now we simply apply our different models and gather their information
    
    # We start with the CNN model
    accuracy, loss, report = defineCNNModel(data_train,data_test,data_validation,label_train,label_test,label_validation,training_sample=training_cycle)
    accuracies_CNN.append(accuracy)
    losses_CNN.append(loss)
    print(report)
    
    training_cycle+=1   # We update the training iteration


# In[ ]:


accuracies_CNN_mean = np.mean(accuracies_CNN)
accuracies_NN_mean = np.mean(accuracies_NN)

print("CNN Accuracy : "+str(accuracies_CNN_mean)+"\nNN Accuracy : "+str(accuracies_NN_mean))

