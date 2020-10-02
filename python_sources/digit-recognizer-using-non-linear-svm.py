#!/usr/bin/env python
# coding: utf-8

# # Assignment - Support Vector Machines
# 
# Let's now tackle a slightly more complex problem - letter recognition. We'll first explore the dataset a bit, prepare it (scale etc.) and then experiment with linear and non-linear SVMs with various hyperparameters.
# 
# 
# ## Data Understanding 
# 
# Let's first understand the shape, attributes etc. of the dataset.

# In[3]:


# libraries
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale

import warnings
warnings.filterwarnings('ignore')


# In[19]:


# dataset
train = pd.read_csv("../input/train.csv")
train.head()


# In[20]:


mnist.shape


# In[21]:


A, mnist=train_test_split(train, test_size = 0.40, random_state = 42)


# In[22]:


mnist.shape


# In[23]:


mnist.info()


# In[24]:


mnist.describe


# In[25]:


mnist.isnull().sum()


# In[26]:


mnist.drop_duplicates(subset=None, keep='first', inplace=True)


# In[27]:


mnist.shape


# In[28]:


# lets see the distribution in numbers
mnist.label.astype('category').value_counts()


# ## Data Preparation
# 
# Let's conduct some data preparation steps before modeling. Firstly, let's see if it is important to **rescale** the features, since they may have varying ranges. For example, here are the average values:

# In this case, the average values do not vary a lot (e.g. having a diff of an order of magnitude). Nevertheless, it is better to rescale them.

# In[29]:


# splitting into X and y
X = mnist.drop("label", axis = 1)
y = mnist['label']


# In[30]:


# scaling the features
X_scaled = scale(X)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.3, random_state = 101)


# ## Model Building
# 
# Let's fist build two basic models - linear and non-linear with default hyperparameters, and compare the accuracies.

# In[31]:


# linear model

model_linear = SVC(kernel='linear')
model_linear.fit(X_train, y_train)

# predict
y_pred = model_linear.predict(X_test)


# In[32]:


# confusion matrix

print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))


# In[33]:


# print other metrics

# accuracy
print("accuracy", metrics.accuracy_score(y_true=y_test, y_pred=y_pred),"\n")

# precision
print("precision", metrics.precision_score(y_true=y_test, y_pred=y_pred, average='macro'),"\n")

# recall/sensitivity
print("recall", metrics.recall_score(y_true=y_test, y_pred=y_pred, average='macro'),"\n")


# The linear model gives approx. 91% accuracy. Let's look at a sufficiently non-linear model with randomly chosen hyperparameters.

# In[34]:


# non-linear model
# using rbf kernel, C=1, default value of gamma

# model
non_linear_model = SVC(kernel='rbf')

# fit
non_linear_model.fit(X_train, y_train)

# predict
y_pred = non_linear_model.predict(X_test)


# In[35]:


# confusion matrix

print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))


# In[36]:


# print other metrics

# accuracy
print("accuracy", metrics.accuracy_score(y_true=y_test, y_pred=y_pred),"\n")

# precision
print("precision", metrics.precision_score(y_true=y_test, y_pred=y_pred, average='macro'),"\n")

# recall/sensitivity
print("recall", metrics.recall_score(y_true=y_test, y_pred=y_pred, average='macro'),"\n")


# The non-linear model gives approx. 93% accuracy. Thus, going forward, let's choose hyperparameters corresponding to non-linear models.

# ## Grid Search: Hyperparameter Tuning
# 
# Let's now tune the model to find the optimal values of C and gamma corresponding to an RBF kernel. We'll use 5-fold cross validation.

# In[37]:


# creating a KFold object with 5 splits 
folds = KFold(n_splits = 5, shuffle = True, random_state = 100)

# specify range of hyperparameters
# Set the parameters by cross-validation
hyper_params = [ {'gamma': [1e-2, 1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}]


# specify model
model = SVC(kernel="rbf")

# set up GridSearchCV()
model_cv = GridSearchCV(estimator = model, 
                        param_grid = hyper_params, 
                        scoring= 'accuracy', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score=True)      

# fit the model
model_cv.fit(X_train, y_train)                  


# In[38]:


# cv results
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results.head()


# In[39]:


# print 5 accuracies obtained from the 5 folds
print(cv_results)
print("mean accuracy = {}".format(cv_results.mean()))


# In[40]:


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


# The plots above show some useful insights:
# - Non-linear models (high gamma) perform *much better* than the linear ones
# - At any value of gamma, a high value of C leads to better performance
# - None of the models tend to overfit (even the complex ones), since the training and test accuracies closely follow each other
# 
# This suggests that the problem and the data is **inherently non-linear** in nature, and a complex model will outperform simple, linear models in this case.

# Let's now choose the best hyperparameters. 

# In[41]:


# printing the optimal accuracy score and hyperparameters
best_score = model_cv.best_score_
best_hyperparams = model_cv.best_params_

print("The best test score is {0} corresponding to hyperparameters {1}".format(best_score, best_hyperparams))


# ### Building and Evaluating the Final Model
# 
# Let's now build and evaluate the final model, i.e. the model with highest test accuracy.

# In[42]:


# model with optimal hyperparameters

# model
model = SVC(C=10, gamma=0.001, kernel="rbf")

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# metrics
print("accuracy", metrics.accuracy_score(y_test, y_pred), "\n")
print(metrics.confusion_matrix(y_test, y_pred), "\n")


# In[43]:


test = pd.read_csv("../input/test.csv")


# In[44]:


test1 = scale(test)


# In[45]:


predicted_digit = model.predict(test1)


# In[46]:


submission = pd.DataFrame({'ImageId': range(1,len(test)+1) ,'Label': predicted_digit })


# In[47]:


submission.to_csv("submission.csv",index=False)

