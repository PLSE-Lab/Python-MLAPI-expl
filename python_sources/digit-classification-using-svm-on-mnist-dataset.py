#!/usr/bin/env python
# coding: utf-8

# # MNIST Digits - Classification Using SVM
# 
# In this notebook, we'll explore the popular MNIST dataset and build an SVM model to classify handwritten digits. <a href='http://yann.lecun.com/exdb/mnist/'>Here is a detailed description of the dataset.</a>
# 
# We'll divide the analysis into the following parts:
# - Data understanding and cleaning
# - Data preparation for model building
# - Building an SVM model - hyperparameter tuning, model evaluation etc.
# 

# ## Data Understanding and Cleaning
#  
#  Let's understand the dataset and see if it needs some cleaning etc.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import gc
import cv2


# In[ ]:


# read the dataset
digits = pd.read_csv("../input/mnist-svm-m4/train.csv")
digits.info()


# In[ ]:


# head
digits.head()


# In[ ]:


four = digits.iloc[3, 1:]
four.shape


# In[ ]:


four = four.values.reshape(28, 28)
plt.imshow(four, cmap='gray')


# #### Side note: Indexing Recall ####
# `list =    [0, 4, 2, 10, 22, 101, 10]` <br>
# `indices = [0, 1, 2, 3, ...,        ]` <br>
# `reverse = [-n           -3  -2   -1]` <br>

# In[ ]:


# visualise the array
print(four[5:-5, 5:-5])


# In[ ]:


# Summarise the counts of 'label' to see how many labels of each digit are present
digits.label.astype('category').value_counts()


# In[ ]:


# Summarise count in terms of percentage 
100*(round(digits.label.astype('category').value_counts()/len(digits.index), 4))


# Thus, each digit/label has an approximately 9%-11% fraction in the dataset and the **dataset is balanced**. This is an important factor in considering the choices of models to be used, especially SVM, since **SVMs rarely perform well on imbalanced data** (think about why that might be the case).
# 
# Let's quickly look at missing values, if any.

# In[ ]:


# missing values - there are none
digits.isnull().sum()


# Also, let's look at the average values of each column, since we'll need to do some rescaling in case the ranges vary too much.

# In[ ]:


# average values/distributions of features
description = digits.describe()
description


# You can see that the max value of the mean and maximum values of some features (pixels) is 139, 255 etc., whereas most features lie in much lower ranges  (look at description of pixel 0, pixel 1 etc. above).
# 
# Thus, it seems like a good idea to rescale the features.

# ## Data Preparation for Model Building
# 
# Let's now prepare the dataset for building the model. We'll only use a fraction of the data else training will take a long time.
# 

# In[ ]:


# Creating training and test sets
# Splitting the data into train and test
X = digits.iloc[:, 1:]
Y = digits.iloc[:, 0]

# Rescaling the features
from sklearn.preprocessing import scale
X = scale(X)

# train test split with train_size=10% and test size=90%
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.10, random_state=101)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


# delete test set from memory, to avoid a memory error
# we'll anyway use CV to evaluate the model, and can use the separate test.csv file as well
# to evaluate the model finally

# del x_test
# del y_test


# ## Model Building
# 
# Let's now build the model and tune the hyperparameters. Let's start with a **linear model** first.
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
svm_linear.fit(x_train, y_train)


# In[ ]:


# predict
predictions = svm_linear.predict(x_test)
predictions[:10]


# In[ ]:


# evaluation: accuracy
# C(i, j) represents the number of points known to be in class i 
# but predicted to be in class j
confusion = metrics.confusion_matrix(y_true = y_test, y_pred = predictions)
confusion


# In[ ]:


# measure accuracy
metrics.accuracy_score(y_true=y_test, y_pred=predictions)


# In[ ]:


# class-wise accuracy
class_wise = metrics.classification_report(y_true=y_test, y_pred=predictions)
print(class_wise)


# In[ ]:


# run gc.collect() (garbage collect) to free up memory
# else, since the dataset is large and SVM is computationally heavy,
# it'll throw a memory error while training
gc.collect()


# ### Non-Linear SVM
# 
# Let's now try a non-linear model with the RBF kernel.

# In[ ]:


# rbf kernel with other hyperparameters kept to default 
svm_rbf = svm.SVC(kernel='rbf')
svm_rbf.fit(x_train, y_train)


# In[ ]:


# predict
predictions = svm_rbf.predict(x_test)

# accuracy 
print(metrics.accuracy_score(y_true=y_test, y_pred=predictions))


# The accuracy achieved with a non-linear kernel is slightly higher than a linear one. Let's now do a grid search CV to tune the hyperparameters C and gamma.
# 
# ### Grid Search Cross-Validation

# In[ ]:


# conduct (grid search) cross-validation to find the optimal values 
# of cost C and the choice of kernel

from sklearn.model_selection import GridSearchCV

parameters = {'C':[1, 10, 100], 
             'gamma': [1e-2, 1e-3, 1e-4]}

# instantiate a model 
svc_grid_search = svm.SVC(kernel="rbf")

# create a classifier to perform grid search
clf = GridSearchCV(svc_grid_search, param_grid=parameters, scoring='accuracy',return_train_score=True)

# fit
clf.fit(x_train, y_train)


# In[ ]:


# results
cv_results = pd.DataFrame(clf.cv_results_)
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
plt.ylim([0.60, 1])
plt.legend(['test accuracy', 'train accuracy'], loc='lower right')
plt.xscale('log')

# subplot 2/3
plt.subplot(132)
gamma_001 = cv_results[cv_results['param_gamma']==0.001]

plt.plot(gamma_001["param_C"], gamma_001["mean_test_score"])
plt.plot(gamma_001["param_C"], gamma_001["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.001")
plt.ylim([0.60, 1])
plt.legend(['test accuracy', 'train accuracy'], loc='lower right')
plt.xscale('log')


# subplot 3/3
plt.subplot(133)
gamma_0001 = cv_results[cv_results['param_gamma']==0.0001]

plt.plot(gamma_0001["param_C"], gamma_0001["mean_test_score"])
plt.plot(gamma_0001["param_C"], gamma_0001["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.0001")
plt.ylim([0.60, 1])
plt.legend(['test accuracy', 'train accuracy'], loc='lower right')
plt.xscale('log')

plt.show()


# From the plot above, we can observe that (from higher to lower gamma / left to right):
# - At very high gamma (0.01), the model is achieving 100% accuracy on the training data, though the test score is quite low (<75%). Thus, the model is overfitting.
# 
# - At gamma=0.001, the training and test scores are comparable at around C=1, though the model starts to overfit at higher values of C
# 
# - At gamma=0.0001, the model does not overfit till C=10 but starts showing signs at C=100. Also, the training and test scores are slightly lower than at gamma=0.001.
# 
# Thus, it seems that the best combination is gamma=0.001 and C=1 (the plot in the middle), which gives the highest test accuracy (~92%) while avoiding overfitting.
# 
# Let's now build the final model and see the performance on test data.
# 
# ### Final Model
# 
# Let's now build the final model with chosen hyperparameters.

# In[ ]:


# optimal hyperparameters
best_C = 1
best_gamma = 0.001

# model
svm_final = svm.SVC(kernel='rbf', C=best_C, gamma=best_gamma)

# fit
svm_final.fit(x_train, y_train)


# In[ ]:


# predict
predictions = svm_final.predict(x_test)


# In[ ]:


# evaluation: CM 
confusion = metrics.confusion_matrix(y_true = y_test, y_pred = predictions)

# measure accuracy
test_accuracy = metrics.accuracy_score(y_true=y_test, y_pred=predictions)

print(test_accuracy, "\n")
print(confusion)


# ### Conclusion
# 
# The final accuracy on test data is approx. 92%. Note that this can be significantly increased by using the entire training data of 42,000 images (we have used just 10% of that!). 
# 
# 
