#!/usr/bin/env python
# coding: utf-8

# ### Digital Image Recognizer Classification using Support Vector Machine Algorithm

# ### The code was built on local desktop and executed on kaggle as the image dataset was on kaggle and execution time was an issue on the local machine. 

# In[1]:


# Importing python libraries
#kaggle/python docker image: https://github.com/kaggle/docker-python
import os
print(os.listdir("../input")) # Any results you write to the current directory are saved as output.
# Input data files are available in the "../input/" directory in kaggle.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')


# In[2]:


# importing the image dataset from kaggle. First dataset was called on kaggle page and then uploaded to this 
#file fetch command of '../input/train.csv' Please run the same on kaggle.
#Importing train dataset in kaggle
MNISTtrain = pd.read_csv('../input/train.csv')
#Importing test dataset in kaggle
MNISTtest = pd.read_csv("../input/test.csv")
#Checking the dataframe
MNISTtrain.head()


# In[3]:


# Inspecting train dataframe
print(MNISTtrain.columns)
print(MNISTtrain.shape)
MNISTtrain.info()


# In[4]:


# Checking dataset for missing values - both rows and columns
MNISTtrain.isnull().values.any()


# In[5]:


#inspecting test dataframe
print(MNISTtest.columns)
print(MNISTtest.shape)
MNISTtest.info()


# In[6]:


# Checking for the missing values
MNISTtest.isnull().values.any()


# In[7]:


# increasing the display limit as there are 785 columns
pd.set_option('display.max_columns', 785)

# lets visualize the basic statistics of the variables
MNISTtrain.describe()


# In[8]:


#Finding is any duplication image is there
order = list(np.sort(MNISTtrain['label'].unique()))
print(order)


# In[9]:


#Finding the mean of count of digits
digit_means = MNISTtrain.groupby('label').mean()
digit_means.head()


# ### Data Visualization - EDA

# In[10]:


# Heatmap to find if pixels are correlated
plt.figure(figsize=(30, 20))
sns.heatmap(digit_means)


# In[11]:


#See the distribution of the digits
sns.countplot(MNISTtrain['label'])
plt.show()


# In[12]:


# lets see the distribution in numbers
MNISTtrain['label'].astype('category').value_counts()


# In[13]:


#Taking only 20% of the dataset in training
subset_train = MNISTtrain[0:8000]

y = subset_train.iloc[:,0]

X = subset_train.iloc[:,1:]

print(y.shape)
print(X.shape)


# In[14]:


# Converting 1D array to 2D 28x28 array using reshape , to plot and view grayscale images

#Lets see digit "3" images in the data.

plt.figure(figsize=(28,28))

digit_3 = subset_train.loc[subset_train.label==3,:]
image = digit_3.iloc[:,1:]
subplots_loc = 191

for i in range(1,9):
    plt.subplot(subplots_loc)
    four = image.iloc[i].values.reshape(28, 28)
    plt.imshow(four, cmap='gray')
    subplots_loc = subplots_loc +1


# In[15]:


# Converting 1D array to 2D 28x28 array using reshape , to plot and view grayscale images

#Lets see digit "4" images in the data.

plt.figure(figsize=(28,28))

digit_4 = subset_train.loc[subset_train.label==4,:]
image = digit_4.iloc[:,1:]
subplots_loc = 191

for i in range(1,9):
    plt.subplot(subplots_loc)
    four = image.iloc[i].values.reshape(28, 28)
    plt.imshow(four, cmap='gray')
    subplots_loc = subplots_loc +1


# In[16]:


# Converting 1D array to 2D 28x28 array using reshape , to plot and view grayscale images

#Lets see digit "1" images in the data.


plt.figure(figsize=(28,28))

digit_1 = subset_train.loc[subset_train.label==1,:]
image = digit_1.iloc[:,1:]
subplots_loc = 191

for i in range(1,9):
    plt.subplot(subplots_loc)
    one = image.iloc[i].values.reshape(28, 28)
    plt.imshow(one, cmap='gray')
    subplots_loc = subplots_loc +1


# In[17]:


# Converting 1D array to 2D 28x28 array using reshape , to plot and view grayscale images

#Lets see digit "0" images in the data.

plt.figure(figsize=(28,28))

digit_0 = subset_train.loc[subset_train.label==0,:]
image = digit_0.iloc[:,1:]
subplots_loc = 191

for i in range(1,9):
    plt.subplot(subplots_loc)
    zero = image.iloc[i].values.reshape(28, 28)
    plt.imshow(zero, cmap='gray')
    subplots_loc = subplots_loc +1


# In[18]:


# Converting 1D array to 2D 28x28 array using reshape , to plot and view grayscale images

#Lets see digit "7" images in the data.

plt.figure(figsize=(28,28))

digit_7 = subset_train.loc[subset_train.label==7,:]
image = digit_7.iloc[:,1:]
subplots_loc = 191

for i in range(1,9):
    plt.subplot(subplots_loc)
    seven = image.iloc[i].values.reshape(28, 28)
    plt.imshow(seven, cmap='gray')
    subplots_loc = subplots_loc +1


# In[ ]:





# In[19]:


#See the distribution of the labels in sliced data
sns.countplot(subset_train.label)


# In[20]:


# average feature values
round(MNISTtrain.drop('label', axis=1).mean(), 2)


# ### Feature Scaling and Model Building

# In[21]:


# Data splitting in train and test data
X_train, X_test,y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)


# In[22]:


# splitting into X and y
X = MNISTtrain.drop("label", axis = 1)
y = MNISTtrain['label']


# In[23]:


#Scaling the data
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
X_train_s = scale.fit_transform(X_train)
X_test_s = scale.transform(X_test)


# ### Building the Linear SVM Model

# In[24]:


#Importing ML algorithm libraries
from sklearn import svm
from sklearn import metrics

# An initial SVM model with linear kernel is built
svm_linear = svm.SVC(kernel='linear')

# fitting the data in the model
svm_linear.fit(X_train_s, y_train)


# In[25]:


# predicting from the built model
predictions = svm_linear.predict(X_test_s)
predictions[:10]


# In[26]:


# confusion matrix and accuracy of linear SVM model

# accuracy
print("accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=predictions), "\n")

# cm
print(metrics.confusion_matrix(y_true=y_test, y_pred=predictions))


# In[27]:


# measure accuracy of linear SVM model
metrics.accuracy_score(y_true=y_test, y_pred=predictions)


# In[28]:


# class-wise accuracy
class_wise = metrics.classification_report(y_true=y_test, y_pred=predictions)
print(class_wise)


# ### Overall accuracy is 91% in the linear SVM model :
# - The digit 6 shows highest precision, showing it is correctly identified in 97% cases.
# - The model is worst for digit 3, it has only  85% of precision.
# - F1 score is >80% for all digits indicating model is good

# ### Building the SVM Model using RDF Kernel

# In[29]:


# rbf kernel with other hyperparameters kept to default 
svm_rbf = svm.SVC(kernel='rbf')
svm_rbf.fit(X_train_s, y_train)


# In[30]:


# predict
predictions = svm_rbf.predict(X_test_s)

# accuracy 
print(metrics.accuracy_score(y_true=y_test, y_pred=predictions))


# In[31]:


# class-wise accuracy
class_wise = metrics.classification_report(y_true=y_test, y_pred=predictions)
print(class_wise)


# #### Overall rbf model accuracy is 90% :
# - The digit 6 and 0 show highest precision in prediction (94%).
# - The precision is lowest for digit 3 ->  83%.
# - F1 score for all digit predictions is showing greater than 0.80 which indicates model is good in classification prediction

# ### Grid Search: Hyperparameter Tuning

# In[ ]:


from sklearn.model_selection import KFold
# creating a KFold object with 3 splits 
folds = KFold(n_splits = 5, shuffle = True, random_state = 4)

# specify range of parameters (C)  and (gamma) as a list
hyper_params = [ {'gamma': [1e-2, 1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}]

# specify model
model = svm.SVC(kernel='rbf')

# set up GridSearchCV()
model_cv = GridSearchCV(estimator = model, 
                        param_grid = hyper_params, 
                        scoring= 'accuracy', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score=True)   

# fit
model_cv.fit(X_train_s, y_train)


# In[ ]:


# results
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results


# In[ ]:


# converting C to numeric type for plotting on x-axis
cv_results['param_C'] = cv_results['param_C'].astype('int')

# # plotting
plt.figure(figsize=(16,6))

# subplot 1/3
plt.subplot(131)
gamma_01 = cv_results[cv_results['param_gamma']==0.01]

plt.plot(gamma_01["param_C"], gamma_01["mean_test_score"])
plt.plot(gamma_01["param_C"], gamma_01["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.01")
plt.ylim([0.70, 1.05])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')

# subplot 2/3
plt.subplot(132)
gamma_001 = cv_results[cv_results['param_gamma']==0.001]

plt.plot(gamma_001["param_C"], gamma_001["mean_test_score"])
plt.plot(gamma_001["param_C"], gamma_001["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.001")
plt.ylim([0.70, 1.05])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')


# subplot 3/3
plt.subplot(133)
gamma_0001 = cv_results[cv_results['param_gamma']==0.0001]

plt.plot(gamma_0001["param_C"], gamma_0001["mean_test_score"])
plt.plot(gamma_0001["param_C"], gamma_0001["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.0001")
plt.ylim([0.70, 1.05])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')


# ##### The plots above show some useful insights:
# 
# Non-linear models (high gamma) perform much better than the linear ones
# At any value of gamma, a high value of C leads to better performance
# None of the models tend to overfit (even the complex ones), since the training and test accuracies closely follow each other
# This suggests that the problem and the data is inherently non-linear in nature, and a complex model will outperform simple, linear models in this case.

# ##### Let's now choose the best hyperparameters.

# In[ ]:


# printing the optimal accuracy score and hyperparameters
best_score = model_cv.best_score_
best_hyperparams = model_cv.best_params_

print("The best test score is {0} corresponding to hyperparameters {1}".format(best_score, best_hyperparams))


# ##### Building and Evaluating the Final Model

# In[ ]:


# model with optimal hyperparameters

# model
model = svm.SVC(C=10, gamma=0.01, kernel="rbf")

model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)

# metrics
print("accuracy", metrics.accuracy_score(y_test, y_pred), "\n")
print(metrics.confusion_matrix(y_test, y_pred), "\n")


# In[ ]:


#Prediction of test data
#scaling of test.csv data
X_test_df = scale.transform(MNISTtest)


# In[ ]:


# prediction of test data
predicted_digit = model.predict(X_test_df)


# In[ ]:


# shape of the predicted digits
predicted_digit.shape


# In[ ]:


# Creating dataframe
data = pd.DataFrame({'Label': predicted_digit})
data.head()


# In[ ]:


#Exporting test output
data.to_csv('digi_predictions.csv', sep=",")


# #### Conclusion: The accuracy achieved using a non-linear kernel (~0.95) is much higher than that of a linear kernel (~0.85). We can conclude that the problem is highly non-linear.
