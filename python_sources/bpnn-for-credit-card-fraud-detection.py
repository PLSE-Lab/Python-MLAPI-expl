#!/usr/bin/env python
# coding: utf-8

# # Import useful APIs for Machine Learning #

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from imblearn.over_sampling import ADASYN
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

get_ipython().run_line_magic('matplotlib', 'inline')

import tensorflow as tf
from tensorflow import keras

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# 
# Before starting the experiment, a testable hypothesis is defined. Since the aim of this project is to implement a ANN algorithmn for credit card fraud detection. To demonstrate whether the number of hidden layers can influence the classification performance of ANN, a hypothesis here is defined:
# 
# **Hypothesis**: The classification performance of a neural network classifier on the Credit Card Fraud Detection dataset is affected by the number of hidden layer neurons.

# **H0**: the null hypothesis: insufficient evidence to support hypothesis. 
# 
# **H1**: the alternate hypothesis: evidence suggests the hypothesis is likely true.

# # Data Preprocessing #

# In[ ]:


credit_card=pd.read_csv("../input/creditcard.csv")
credit_card.head()
#Take a look of our dataset and variables to get brief understanding of the data we'll be working with


# In[ ]:


#Check the size of our dataset
credit_card.shape


# Here, we can see that there are **284807 rows** and **31 columns** in the dataset.

# In[ ]:


credit_card["Class"].value_counts()


# According to the values of "Class", we can see the dataset is highly imbalanced. Class "1" consist of 284315 samples while Class "0" consist of 492 samples. Therefore later we need to deal with the problem.

# In[ ]:


#Check if there is missing value
null_data=pd.isnull(credit_card).sum()
print(null_data)


# It seems there is **no missing value** within our dataset. Great!
# 
# Next, we need to select valuable variables that would be used for later classification. 

# # Data Visualization

# Firstly, we can check the distribution of the amount of transactions using different transaction time.

# In[ ]:


f, (fraud, normal) = plt.subplots(2, 1, sharex=True, figsize=(12,4))

bins = 50

fraud.hist(credit_card.Time[credit_card.Class == 1], bins = bins)
fraud.set_title('Fraud')

normal.hist(credit_card.Time[credit_card.Class == 0], bins = bins)
normal.set_title('Normal')

plt.xlabel('Time (in Seconds)')
plt.ylabel('Number of Transactions')
plt.show()


# It can hardly see a uniform distribution in fraudulent transactions, while a cyclical distribution is found in normal transactions.

# In[ ]:


f, (fraud, normal) = plt.subplots(2, 1, sharex=False, figsize=(12,7))

bins = 30

fraud.hist(credit_card.Amount[credit_card.Class == 1], bins = bins)
fraud.set_title('Fraud')

normal.hist(credit_card.Amount[credit_card.Class == 0], bins = bins)
normal.set_title('Normal')

plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.yscale('log')
plt.show()


# Fraudulent transactions are mostly small amount. Only one fraud is over $1000, most of them are less than $100. Normal transactions have maximum amount at approximately $25000.

# In[ ]:


f, (fraud, normal) = plt.subplots(2, 1, sharex=False, figsize=(12,8))

fraud.scatter(credit_card.Time[credit_card.Class == 1], credit_card.Amount[credit_card.Class == 1])
fraud.set_title('Fraud')

normal.scatter(credit_card.Time[credit_card.Class == 0], credit_card.Amount[credit_card.Class == 0])
normal.set_title('Normal')

plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()


# No clear ralationship between Amount and Time in both fraud and normal classes.

# In[ ]:


v_variable=credit_card.iloc[:,1:29].columns


# In[ ]:


plt.figure(figsize=(12,28*4))
gs = gridspec.GridSpec(28, 1)
for i, cn in enumerate(credit_card[v_variable]):
    ax = plt.subplot(gs[i])
    sns.distplot(credit_card[cn][credit_card.Class == 1], bins=50)
    sns.distplot(credit_card[cn][credit_card.Class == 0], bins=50)
    ax.set_xlabel('')
    plt.legend(credit_card["Class"])
    ax.set_title('histogram of feature: ' + str(cn))
    
plt.show()


# # Stratified Cross-Validation - Split training/testing datasets

# The samples of "Time" are integers rather than float, which are in a different form from other variable samples. Here I will drop it to avoid noise.

# In[ ]:


#select the values of "V1" to "Amount"
creditcard_v=credit_card.iloc[:,1:30].values


# In[ ]:


print(creditcard_v)


# In[ ]:


creditcard_v.shape


# Next we need to split the samples to training and testing data. Due to highly imbalanced nature of our dataset, the model-validation technique we will use is **Stratified cross-validation**. 
# 
# > Stratified cross-validation is a good technique in the case of highly imbalanced classes. For binary classification with a training/test split rather than cross-validation, this involves the training set having the same proportion of positive-labeled points as the test set (and hence the same as the overall training set). Such a split is easily accomplished by splitting your points by label resulting in two sets, shuffling each of these sets, and then placing the first x% of each set into the training set and the last x% of each set into the test set.
# 
#    - StackExchange (https://stats.stackexchange.com/questions/91922/splitting-an-imbalanced-dataset-for-training-and-testing)

# Here we are using **Stratified ShuffleSplit cross_validator**, which is a merge of StratifiedKFold and ShuffleSplit. We split data into **5 folds**, each fold has **30% test** sets and **70% train** sets. 
# 
# *Note: Each fold is made by the preserving percentage of samples for each class, in this case is Class 0: Class 1 = 284315 : 492*
# 
# > Stratified ShuffleSplit cross-validator
# Provides train/test indices to split data in train/test sets.
# This cross-validation object is a merge of StratifiedKFold and ShuffleSplit, which returns stratified randomized folds. The folds are made by preserving the percentage of samples for each class.
# Note: like the ShuffleSplit strategy, stratified random splits do not guarantee that all folds will be different, although this is still very likely for sizeable datasets.
# 
# - Scikit-learn organization (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html#sklearn.model_selection.StratifiedShuffleSplit)

# In[ ]:


#Set 30 variables as "X" and "Class" as y
X=creditcard_v
y=credit_card["Class"].values

sss=StratifiedShuffleSplit(n_splits=5, test_size=0.3,random_state=0)
sss.get_n_splits(X,y)


# In[ ]:


#So this is the cross-validator that we are using
print(sss)


# In[ ]:


#Split train/test sets of X and y
for train_index, test_index in sss.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train,X_test=X[train_index], X[test_index]
    y_train,y_test=y[train_index], y[test_index]


# In[ ]:


#Let's see the number of sets in each class of training and testing datasets
print(pd.Series(y_train).value_counts())
print(pd.Series(y_test).value_counts())


# Next we need to verify sample size sufficiency to make sure the sample size of training and testing datasets are sufficient to demonstrate the hypothese test.

# # 2-D visualisation of imbalanced training data

# Before resampling, let's reduce the dimensionality of our input variables and visualise the imbalanced training dataset first. Here we reduce variables to 2 dimensions.

# In[ ]:


pca=PCA(n_components=2)
data_2d=pd.DataFrame(pca.fit_transform(X_train))


# In[ ]:


data_2d


# In[ ]:


data_2d=pd.concat([data_2d, pd.DataFrame(y_train)], axis=1)
data_2d.columns=["x","y","fraud"]


# In[ ]:


data_2d


# In[ ]:


#visualise the 2D training data
sns.set_style("darkgrid")
sns.lmplot(x="x", y="y", data=data_2d, fit_reg=False, hue="fraud",height=5, aspect=2)
plt.title("Scatter Plot of imbalanced Training Data")


# Now we got a plot distribution of imbalanced training data. The figure shows some complex non-linear relationship between x and y. 
# 
# As our training data is still highly imbalanced, which we can see the majority points are in blue and minority are in yellow. Thus we need to resample the training sets to create balanced training data for later machine learning.

# # Oversampling Training Data

# In order to maximize machine learning outcomes, we have to oversample the data, which is to create new synthetic samples that simulate the minority class to balance the dataset. Now we can use SMOTE( ) to oversample the training data.
# 
# > ADASYN:  ADAptive SYNthetic (ADASYN) is based on the idea of adaptively generating minority data samples according to their distributions using K nearest neighbor. The algorithm adaptively updates the distribution and there are no assumptions made for the underlying distribution of the data.  The algorithm uses Euclidean distance for KNN Algorithm. The key difference between ADASYN and SMOTE is that the former uses a density distribution, as a criterion to automatically decide the number of synthetic samples that must be generated for each minority sample by adaptively changing the weights of the different minority samples to compensate for the skewed distributions. The latter generates the same number of synthetic samples for each original minority sample.
# 
# -- Rohit Walimbe, Data Science Central (https://www.datasciencecentral.com/profiles/blogs/handling-imbalanced-data-sets-in-supervised-learning-using-family)

# In[ ]:


ada= ADASYN()
x_resample,y_resample=ada.fit_sample(X_train,y_train)


# In[ ]:


#concat oversampled "x" and "y" into one DataFrame
data_oversampled=pd.concat([pd.DataFrame(x_resample),pd.DataFrame(y_resample)],axis=1)
#replace column labels using the labels of original datasets
data_oversampled.columns=credit_card.columns[1:31]


# In[ ]:


#while the label of column 30 is "Class",we can rename it to "fraud"
data_oversampled.rename(columns={"Class":"fraud"},inplace=True)


# Let's see how many samples in each class after oversampling.

# In[ ]:


data_oversampled["fraud"].value_counts()


# Each class has nearly the same amount of samples, the training dataset now is balanced.
# 
# Now we can use the same approach as earlier to reduce the dimensionality of our balanced training samples and visualise it using scatterplot.

# In[ ]:


#reduce dimensionality to 2 dimensions
oversampled_train2d=pd.DataFrame(pca.fit_transform(data_oversampled.iloc[:,0:29]))
oversampled_train2d=pd.concat([oversampled_train2d,data_oversampled["fraud"]],axis=1)
oversampled_train2d.columns= ["x","y","fraud"]


# In[ ]:


#visualise data
sns.set_style("darkgrid")
sns.lmplot(x="x", y="y", data=oversampled_train2d, fit_reg=False, hue="fraud",height=5, aspect=2)
plt.title("Scatter Plot of balanced Training Data")


# # One-Hot Encoding for Class labels

# In[ ]:


y_resample


# Since we have two classes which are already integers of y_resample. 1 represents fraud, 0 represents normal.
# 
# Now we need to apply one-hot encoding to reformat y_resample to be a 2-dimensional vector in terms of [1. 0.] or [0. 1.].

# In[ ]:


#use one-hot encoding to reformat
Y_resample=keras.utils.to_categorical(y_resample,num_classes=None)


# In[ ]:


print(Y_resample)


# # Designing and Configuring BPNN algorithms

# Now we design a ANN model using Kera from TensorFlow to train the training dataset as a **control arm**. The model consists of one hidden layer with four neurons on it. 

# In[ ]:


ANN=keras.Sequential()

ANN.add(keras.layers.Dense(4, input_shape=(29,),activation="relu"))
ANN.add(keras.layers.Dense(2, activation="softmax"))


# In[ ]:


ANN.compile(keras.optimizers.Adam(lr=0.04), "categorical_crossentropy",metrics=['accuracy'])


# In[ ]:


ANN.summary()


# Next, train the model using training set.

# In[ ]:


ANN.fit(x_resample,Y_resample, epochs=30, verbose=0)


# Next, test the model and provide evaluations

# In[ ]:


#Evaluate the model based on accuracy
Y_test=keras.utils.to_categorical(y_test,num_classes=None)
control_accuracy=ANN.evaluate(X_test,Y_test)[1]
print("Accuracy: {}".format(control_accuracy))
print("\n")

#Use sklearn to calculate precision and recall of the model
#Create a classification report
prediction_control=ANN.predict(X_test, verbose=1)
labels=["Normal","Fraud"]
y_pred=np.argmax(prediction_control,axis=1)
precision=precision_score(y_test,y_pred,labels=labels)
recall=recall_score(y_test,y_pred,labels=labels)
print("Fraud Precision:{}".format(precision))
print("Fraud Recall:{}".format(recall))

print("\n")
print(classification_report(y_test,y_pred, target_names=labels,digits=8))


# Then we design experimental arm.

# In[ ]:


ANN_exp=keras.Sequential()
ANN_exp.add(keras.layers.Dense(4, input_shape=(29,),activation="relu"))
ANN_exp.add(keras.layers.Dense(4, activation="relu"))
ANN_exp.add(keras.layers.Dense(2, activation="softmax"))
    
ANN_exp.compile(keras.optimizers.Adam(lr=0.04), "categorical_crossentropy",metrics=["accuracy"])
    
ANN_exp.fit(x_resample,Y_resample, epochs=30, verbose=0)


# In[ ]:


#Evaluate the model based on accuracy
Y_test=keras.utils.to_categorical(y_test,num_classes=None)
exp_accuracy=ANN_exp.evaluate(X_test,Y_test)[1]
print("Accuracy: {}".format(exp_accuracy))
print("\n")

#Use sklearn to calculate precision and recall of the model
#Create a classification report
prediction_exp=ANN_exp.predict(X_test, verbose=1)
labels=["Normal","Fraud"]
y_pred_exp=np.argmax(prediction_exp,axis=1)
precision=precision_score(y_test,y_pred_exp,labels=labels)
recall=recall_score(y_test,y_pred_exp,labels=labels)
print("Fraud Precision:{}".format(precision))
print("Fraud Recall:{}".format(recall))

print("\n")
print(classification_report(y_test,y_pred_exp, target_names=labels,digits=8))


# In[ ]:


import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:


cnf_matrix_control = confusion_matrix(y_test,y_pred)

class_names = ["Normal","Fraud"]
plt.figure(figsize=(6,6))
plot_confusion_matrix(cnf_matrix_control
                      , classes=class_names
                      , title='Confusion matrix of control arm')
plt.show()

cnf_matrix_exp = confusion_matrix(y_test,y_pred_exp)

class_names = ["Normal","Fraud"]
plt.figure(figsize=(6,6))
plot_confusion_matrix(cnf_matrix_exp
                      , classes=class_names
                      , title='Confusion matrix of experimental arm')
plt.show()


# # Generating 30 predictions of control arm

# Based on the model we designed above, we can receive evaluation results thirty times using the loop below.

# In[ ]:


results_control_evaluation= []
for i in range(0,30):
    model=keras.Sequential()
    model.add(keras.layers.Dense(4, input_shape=(29,),activation="relu"))
    model.add(keras.layers.Dense(2, activation="softmax"))
    model.compile(keras.optimizers.Adam(lr=0.04), "categorical_crossentropy",metrics=["accuracy"])
    
    model.fit(x_resample,Y_resample, epochs=30, verbose=0)
    
    Y_test=keras.utils.to_categorical(y_test,num_classes=None)
    accuracy=model.evaluate(X_test,Y_test)[1]
    
    prediction_control=model.predict(X_test, verbose=0)
    labels=["Normal","Fraud"]
    y_pred=np.argmax(prediction_control,axis=1)
    precision=precision_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred)
    evaluation=pd.DataFrame([accuracy,precision,recall])
    results_control_evaluation.append(evaluation)
result_control=pd.concat(results_control_evaluation,axis=1)
print(result_control)


# # Generating 30 predictions of experimental arm

# The same approach is used for experimental arm.

# In[ ]:


results_experimental_evaluation= []
for i in range(0,30):
    model_exp=keras.Sequential()
    model_exp.add(keras.layers.Dense(4, input_shape=(29,),activation="relu"))
    model_exp.add(keras.layers.Dense(4, activation="relu"))
    model_exp.add(keras.layers.Dense(2, activation="softmax"))
    
    model_exp.compile(keras.optimizers.Adam(lr=0.04), "categorical_crossentropy",metrics=["accuracy"])
    
    model_exp.fit(x_resample,Y_resample, epochs=30, verbose=0)
    
    Y_test=keras.utils.to_categorical(y_test,num_classes=None)
    accuracy=model_exp.evaluate(X_test,Y_test)[1]

    prediction_experimental=model_exp.predict(X_test, verbose=0)
    labels=["Normal","Fraud"]
    y_exp_pred=np.argmax(prediction_control,axis=1)
    precision=precision_score(y_test,y_exp_pred)
    recall=recall_score(y_test,y_exp_pred)
    evaluation=pd.DataFrame([accuracy,precision,recall])
    results_experimental_evaluation.append(evaluation)
result_experimental=pd.concat(results_experimental_evaluation,axis=1)
print(result_experimental)


# # Statistic Normal Test

# According to definition:
# > scipy.stats.normaltest(a, axis=0, nan_policy='propagate')
# Test whether a sample differs from a normal distribution. This function tests the null hypothesis that a sample comes from a normal distribution.
# - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html
# 
# 
# For this test,
# 
# Null hypothesis: H0: Samples come from a normal distribution
# 
# H1: Samples do not come from a normal distribution
# 

# In[ ]:


from scipy import stats

alpha = 0.05;

s, p = stats.normaltest(result_control.iloc[0,:])
if p < alpha:
  print('Control data is not normal')
else:
  print('Control data is normal')
print("p-value:{}".format(p))
print("\n")
s, p = stats.normaltest(result_experimental.iloc[0,:])
if p < alpha:
  print('Experimental data is not normal')
else:
  print('Experimental data is normal')

print ("p-value:{}".format(p))


# # Non-parametric test and significance test

# Since the control and experimental arm are both not normal distribution, non-parametric test in terms of median and Wilcoxon signed-rank test are selected to test the hypothesis.

# In[ ]:


median_control_acc=result_control.iloc[0,:].median()
median_control_preci=result_control.iloc[1,:].median()
median_control_recall=result_control.iloc[2,:].median()

median_exp_acc=result_experimental.iloc[0,:].median()
median_exp_preci=result_experimental.iloc[1,:].median()
median_exp_recall=result_experimental.iloc[2,:].median()


# In[ ]:


print("Median control accuracy:{}".format(median_control_acc))
print("Median experim accuracy:{}".format(median_exp_acc))
print("\n")

#Significance test of accuracy
s, p = stats.wilcoxon(result_control.iloc[0,:], result_experimental.iloc[0,:])
print("p-value:{}".format(p))
if p < 0.05:
  print('null hypothesis rejected, significant difference between the data-sets')
else:
  print('null hypothesis accepted, no significant difference between the data-sets')


# In[ ]:


print("Median control precision:{}".format(median_control_preci))
print("Median experim precision:{}".format(median_exp_preci))
print("\n")

#Significance test of Precision
s, p = stats.wilcoxon(result_control.iloc[1,:], result_experimental.iloc[1,:])
print("p-value:{}".format(p))
if p < 0.05:
  print('null hypothesis rejected, significant difference between the data-sets')
else:
  print('null hypothesis accepted, no significant difference between the data-sets')


# In[ ]:


print("Median control recall:{}".format(median_control_recall))
print("Median experim recall:{}".format(median_exp_recall))
print("\n")

#Significance test of Precision
s, p = stats.wilcoxon(result_control.iloc[2,:], result_experimental.iloc[2,:])
print("p-value:{}".format(p))
if p < 0.05:
  print('null hypothesis rejected, significant difference between the data-sets')
else:
  print('null hypothesis accepted, no significant difference between the data-sets')


# In[ ]:


result_accuracy=pd.concat([result_control.iloc[0,:],result_experimental.iloc[0,:]],axis=1)
result_precision=pd.concat([result_control.iloc[1,:],result_experimental.iloc[1,:]],axis=1)
result_recall=pd.concat([result_control.iloc[2,:],result_experimental.iloc[2,:]],axis=1)

result_accuracy.columns=["control","experimental"]
result_precision.columns=["control","experimental"]
result_recall.columns=["control","experimental"]


# In[ ]:


result_accuracy.boxplot()
plt.title("Boxplot of Accuracy")


# In[ ]:


result_precision.boxplot()
plt.title("Boxplot of Precision")


# In[ ]:


result_recall.boxplot()
plt.title("Boxplot of Recall")

