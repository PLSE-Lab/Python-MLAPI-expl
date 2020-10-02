#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing different packages

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')


# ### Reading data
# 
# As there is no labels in the test data, we do not need for the our assignment.

# In[ ]:


# get data to dataframe variables
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv("../input/test.csv")
# data visualization
train_df


# In[ ]:


#inspecting train dataframe
print(train_df.shape)
train_df.info()


# In[ ]:


# Checking for the missing values
train_df.isnull().values.any()


# In[ ]:


#inspecting test dataframe
print(test_df.shape)
test_df.info()


# In[ ]:


# Checking for the missing values
test_df.isnull().values.any()


# ### Summary
#     There are no missing values.
#     All the variables are of int datatype.
#     Test data has no labels, and hence we have to send the predicted labels to the kaggle.

# In[ ]:


# some columns are hidden, so increasing the display limit
pd.set_option('display.max_columns', 785)

# lets visualize the basic statistics of the variables
train_df.describe()


# There are some columns which are having all zero value. Those can be dropped but I choose to keep it as it should not affect the performance of the alogorithms as it is all zero.

# ### Distribution of the labels

# In[ ]:


#See the distribution of the labels
sns.countplot(train_df.label)


# In[ ]:


# lets see the distribution in numbers
train_df.label.astype('category').value_counts()


# The data labels are almost equally distributed. So no problem there.

# ### Since training dataset is quite large, we will consider the 20% of the data to train and validate the model and also to find the optimal hyperparameter.

# In[ ]:


digits = train_df[0:8000]

y = digits.iloc[:,0]

X = digits.iloc[:,1:]

print(y.shape)
print(X.shape)


# In[ ]:


#See the distribution of the labels in sliced data
sns.countplot(digits.label)


# In[ ]:


# Converting 1D array to 2D 28x28 array using reshape , to plot and view grayscale images

#Lets see digit "3" images in the data.

plt.figure(figsize=(28,28))

digit_3 = digits.loc[digits.label==3,:]
digit_image = digit_3.iloc[:,1:]
subplots_loc = 191

for i in range(1,9):
    plt.subplot(subplots_loc)
    four = digit_image.iloc[i].values.reshape(28, 28)
    plt.imshow(four, cmap='gray')
    subplots_loc = subplots_loc +1


# In[ ]:


# Converting 1D array to 2D 28x28 array using reshape , to plot and view grayscale images

#Lets see digit "4" images in the data.

plt.figure(figsize=(28,28))

digit_3 = digits.loc[digits.label==4,:]
digit_image = digit_3.iloc[:,1:]
subplots_loc = 191

for i in range(1,9):
    plt.subplot(subplots_loc)
    four = digit_image.iloc[i].values.reshape(28, 28)
    plt.imshow(four, cmap='gray')
    subplots_loc = subplots_loc +1


# In[ ]:


# Converting 1D array to 2D 28x28 array using reshape , to plot and view grayscale images

#Lets see digit "1" images in the data.


plt.figure(figsize=(28,28))

digit_3 = digits.loc[digits.label==1,:]
digit_image = digit_3.iloc[:,1:]
subplots_loc = 191

for i in range(1,9):
    plt.subplot(subplots_loc)
    four = digit_image.iloc[i].values.reshape(28, 28)
    plt.imshow(four, cmap='gray')
    subplots_loc = subplots_loc +1


# ### Splitting data into Train and Test Data and Labels
# <br>
# 
#     Dividing Data randomly into train and test. 
#     Out of 8000 data "train.csv" Data, 80% is kept as train for training the model
#     and 20% is kept as test to validate the model. 

# In[ ]:


# Data splitting in train and test data
X_train, X_test,y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)


# In[ ]:


# Shapes of the data
print("Train Data Shape: ",X_train.shape)
print("Train Label Shape: ",y_train.shape)
print("Test Data Shape: ",X_test.shape)
print("Test Label Shape: ",y_test.shape)


# In[ ]:


# lets see the distribution of the data in the train and test data
plt.figure(figsize=(12,6))
plt.subplot(121)
sns.countplot(y_train.to_frame(name='label').label)
plt.title('train data')
plt.subplot(122)
sns.countplot(y_test.to_frame(name='label').label)
plt.title('test data')


# Data distribution seems ok in both the train and test dataset.

# ### Scaling of the data
# Necessary because it take less iteration to converse to the solution

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

#Scaling data
scale = MinMaxScaler()
X_train_s = scale.fit_transform(X_train)
X_test_s = scale.transform(X_test)


# ## Model building
# Let's fist build two basic models - linear and non-linear with default hyperparameters, and compare the accuracies.
# 
# ### Linear SVM
# 
# Let's first try building a linear SVM model (i.e. a linear kernel).

# In[ ]:


from sklearn import svm
from sklearn import metrics

# an initial SVM model with linear kernel   
svm_linear = svm.SVC(kernel='linear')

# fit
svm_linear.fit(X_train_s, y_train)


# In[ ]:


# predict
predictions = svm_linear.predict(X_test_s)
predictions[:10]


# In[ ]:


# measure accuracy
metrics.accuracy_score(y_true=y_test, y_pred=predictions)


# In[ ]:


# class-wise accuracy
class_wise = metrics.classification_report(y_true=y_test, y_pred=predictions)
print(class_wise)


# ### Summar of result
# 
#     Overall accuracy is 91% of the model.
#     The digit 6 is correctly identified in 97% cases.
#     The model is worst for digit 3, it has only accuracy of 85%.

# ### Non-Linear SVM: RBF kernal
# 
# Let's now try a non-linear model with the RBF kernel.

# In[ ]:


# rbf kernel with other hyperparameters kept to default 
svm_rbf = svm.SVC(kernel='rbf')
svm_rbf.fit(X_train_s, y_train)


# In[ ]:


# predict
predictions = svm_rbf.predict(X_test_s)

# accuracy 
print(metrics.accuracy_score(y_true=y_test, y_pred=predictions))


# In[ ]:


# class-wise accuracy
class_wise = metrics.classification_report(y_true=y_test, y_pred=predictions)
print(class_wise)


# ### Summar of result
# 
#     Overall accuracy is 90% of the model.
#     The digit 6 and 0 is correctly identified in 94% cases.
#     The model is worst for digit 3, it has only accuracy of 83%.

# ## Grid Search to find Optimal Hyperparameter C and gamma
# 
# The accuracy achieved with a linear kernel is almost same as non-linear(RBF) one.
# 
# Lets use the non-linear classifier model and optimize the hyperparameters C and gamma through grid search CV.

# In[ ]:


from sklearn.model_selection import KFold
# creating a KFold object with 5 splits 
folds = KFold(n_splits = 5, shuffle = True, random_state = 4)

# specify range of parameters (C)  and (gamma) as a list
hyper_params = [ {'gamma': [1e-2, 1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}]

# specify model
model = svm.SVC(kernel="rbf")

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


# Summary of the plots above:
# - At very high gamma (0.01), the model is achieving almost 100% accuracy on the training data, when C > 10 and the test score is decreasing or flatening. 
# Thus, the model is overfitting when C>10.
# 
# - At gamma=0.001, the training and test scores are widening significantly, when C>30, i.e. the model starts to overfit at higher values of C.
# 
# - At gamma=0.0001, the model does not overfit till C=10 but accuracy is about 90%. When C >10, accuracy increasing on train data but not on test data.
# 
# Thus, it seems that the best combination is gamma=0.01 and C=10, which gives the highest test accuracy (~95%) while avoiding overfitting.
# 
# Let's now build the final model and see the performance on test data.
# 
# 

# ### Building and Evaluating the Final Model
# 
# Let's now build and evaluate the final model, i.e. the model with highest test accuracy.

# In[ ]:


# printing the optimal accuracy score and hyperparameters
best_score = model_cv.best_score_
best_hyperparams = model_cv.best_params_

print("The best test score is {0} corresponding to hyperparameters {1}".format(best_score, best_hyperparams))


# In[ ]:


# model with optimal hyperparameters

# model
model = svm.SVC(C=10, gamma=0.01, kernel="rbf")

model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)

# metrics
print("accuracy", metrics.accuracy_score(y_test, y_pred), "\n")
print(metrics.confusion_matrix(y_test, y_pred), "\n")


# ## Summary of the result
# 
#     The accuracy achieved using a non-linear kernel (~0.96) is mush higher than that of a linear one (~0.91). 
#     
#     We can conclude that the problem is highly non-linear in nature.

# ## Prediction on the test.csv 

# In[ ]:


#scaling of test.csv data
X_test_df = scale.transform(test_df)


# In[ ]:


# predict
predicted_digit = model.predict(X_test_df)


# In[ ]:


# shape of the predicted digits
predicted_digit.shape


# In[ ]:


# Creating dataframe
data = pd.DataFrame({'Label': predicted_digit})
data.head()


# In[ ]:


data.to_csv('digi_predictions.csv', sep=",")

