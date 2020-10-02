#!/usr/bin/env python
# coding: utf-8

# # Step 1: Reading and Understanding the Data

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


pd.options.display.max_columns = None
pd.options.display.max_rows = None
sns.set(style="whitegrid", color_codes=True)


# In[ ]:


train = pd.read_csv('../input/train.csv')


# In[ ]:


holdout = pd.read_csv('../input/test.csv')


# In[ ]:


# Read sample submission file
ss = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


df=train.sample(frac=0.1,random_state=200)


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


X = df.drop('label', axis = 1)


# In[ ]:


y = df['label']


# In[ ]:


y.shape


# In[ ]:


y.value_counts().plot(kind = 'bar')


# In[ ]:


df.describe()


# # Step 2: Data Cleaning

# In[ ]:


# Missing values & Duplicates


# In[ ]:


train.isna().sum().sum()


# In[ ]:


test.isna().sum().sum()


# In[ ]:


train.duplicated().sum()


# In[ ]:


images = X.values.reshape(-1,28,28,1)


# In[ ]:


g = plt.imshow(images[0][:,:,0])


# In[ ]:


y.iloc[0]


# # Step 3: Data Preparation

# In[ ]:


# Scaling the features


# In[ ]:


X = X/255


# In[ ]:


X.describe()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)


# # Step 4: SVM Modelling

# In[ ]:


from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# ## Base modeling (Non-linear kernel)

# In[ ]:


base_model = SVC(kernel = 'rbf')


# In[ ]:


base_model.fit(X_train, y_train)


# In[ ]:


y_pred = base_model.predict(X_test)


# In[ ]:


print('Accuracy = {}%'.format(round(metrics.accuracy_score(y_test, y_pred),3)*100))


# In[ ]:


plt.figure(figsize = (8,5))
sns.heatmap(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred), annot = True, fmt = '0.3g')
plt.xlabel('Predicted label')
plt.ylabel('True label')


# ## Hyperparameter Tuning

# In[ ]:


folds = KFold(n_splits = 5, shuffle = True, random_state = 101)


# In[ ]:


params = [{'gamma': [0.0001, 0.001, 0.01], 'C': [1, 10, 100, 1000]}]


# In[ ]:


model = SVC(kernel = 'rbf')


# In[ ]:


model_cv = GridSearchCV(estimator=model, param_grid = params, scoring = 'accuracy', 
                        cv = folds, verbose = 1, 
                       return_train_score=True)


# In[ ]:


model_cv.fit(X_train, y_train)


# In[ ]:


cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results[['param_C', 'param_gamma','mean_train_score', 'mean_test_score']].sort_values(by = 'mean_test_score', 
                                                                                        ascending = False)


# In[ ]:


plt.figure(figsize = (16,6))


plt.subplot(1,3,1)
gamma_01 = cv_results[cv_results['param_gamma'] == 0.01]
sns.lineplot(x = 'param_C', y = 'mean_test_score', data = gamma_01)
sns.lineplot(x = 'param_C', y = 'mean_train_score', data = gamma_01)
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.ylim([0.6,1])
plt.title('Gamma = 0.01')
plt.legend(['test_accuracy', 'train_accuracy'], loc = 'upper_left')
plt.xscale('log')             

                      
plt.subplot(1,3,2)
gamma_001 = cv_results[cv_results['param_gamma'] == 0.001]
sns.lineplot(x = 'param_C', y = 'mean_test_score', data = gamma_001)
sns.lineplot(x = 'param_C', y = 'mean_train_score', data = gamma_001)
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.ylim([0.6,1])
plt.title('Gamma = 0.001')
plt.legend(['test_accuracy', 'train_accuracy'], loc = 'upper_left')
plt.xscale('log')
                      

plt.subplot(1,3,3)
gamma_0001 = cv_results[cv_results['param_gamma'] == 0.0001]
sns.lineplot(x = 'param_C', y = 'mean_test_score', data = gamma_0001)
sns.lineplot(x = 'param_C', y = 'mean_train_score', data = gamma_0001)
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.ylim([0.6,1])
plt.title('Gamma = 0.0001')
plt.legend(['test_accuracy', 'train_accuracy'], loc = 'upper_left')
plt.xscale('log')


# In[ ]:


# printing the optimal accuracy score and hyperparameters
best_score = model_cv.best_score_
best_hyperparams = model_cv.best_params_

print("The best test score is {0} corresponding to hyperparameters {1}".format(best_score, best_hyperparams))


# ### Further hyperparameter tuning
# Trying more values for gamma & C

# In[ ]:


params_2 = [{'gamma': [0.001, 0.01,0.05], 'C': [0.1,1, 10, 100]}]


# In[ ]:


model_cv_2 = GridSearchCV(estimator=model, param_grid = params_2, scoring = 'accuracy', 
                        cv = folds, verbose = 1, 
                       return_train_score=True)


# In[ ]:


model_cv_2.fit(X_train, y_train)


# In[ ]:


cv_results_2 = pd.DataFrame(model_cv_2.cv_results_)
cv_results_2[['param_C', 'param_gamma','mean_train_score', 'mean_test_score']].sort_values(by = 'mean_test_score', 
                                                                                        ascending = False)


# In[ ]:


plt.figure(figsize = (16,6))


plt.subplot(1,3,1)
gamma_05 = cv_results_2[cv_results_2['param_gamma'] == 0.05]
sns.lineplot(x = 'param_C', y = 'mean_test_score', data = gamma_05)
sns.lineplot(x = 'param_C', y = 'mean_train_score', data = gamma_05)
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.ylim([0.6,1])
plt.title('Gamma = 0.05')
plt.legend(['test_accuracy', 'train_accuracy'], loc = 'upper_left')
plt.xscale('log')             

                      
plt.subplot(1,3,2)
gamma_01 = cv_results_2[cv_results_2['param_gamma'] == 0.01]
sns.lineplot(x = 'param_C', y = 'mean_test_score', data = gamma_01)
sns.lineplot(x = 'param_C', y = 'mean_train_score', data = gamma_01)
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.ylim([0.6,1])
plt.title('Gamma = 0.01')
plt.legend(['test_accuracy', 'train_accuracy'], loc = 'upper_left')
plt.xscale('log')
                      

plt.subplot(1,3,3)
gamma_001 = cv_results_2[cv_results_2['param_gamma'] == 0.001]
sns.lineplot(x = 'param_C', y = 'mean_test_score', data = gamma_001)
sns.lineplot(x = 'param_C', y = 'mean_train_score', data = gamma_001)
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.ylim([0.6,1])
plt.title('Gamma = 0.001')
plt.legend(['test_accuracy', 'train_accuracy'], loc = 'upper_left')
plt.xscale('log')


# **With higher gamma (0.05 from 0.01) - we don't see any significant improvement in test accuracy - so we'll keep the optimal hyperparamaters as identified in original hyperparameter tuning, i.e. C = 10, gamma = 0.01**

# # Step 5: Prediction for Test & Holdout data

# In[ ]:


C_final = model_cv.best_params_['C']
gamma_final = model_cv.best_params_['gamma']


# In[ ]:


model_f = SVC(C = C_final, gamma = gamma_final, kernel = 'rbf')


# In[ ]:


model_f.fit(X_train, y_train)


# ## Prediction for test data

# In[ ]:


y_test_pred = model_f.predict(X_test)


# In[ ]:


print("Accuracy on test data = {}%".format(round(metrics.accuracy_score(y_test, y_test_pred),2)*100))


# In[ ]:


plt.figure(figsize = (8,5))
sns.heatmap(metrics.confusion_matrix(y_true=y_test, y_pred=y_test_pred), annot = True, fmt = '0.3g')
plt.xlabel('Predicted label')
plt.ylabel('True label')


# ## Prediction for Holdout data

# In[ ]:


holdout.head()


# In[ ]:


holdout.shape


# In[ ]:


holdout_scaled = holdout/255


# In[ ]:


holdout_pred = model_f.predict(holdout_scaled)


# In[ ]:


holdout_pred


# In[ ]:


# Checking sample submission file
ss.head()


# In[ ]:


submission = pd.DataFrame(list(zip(holdout.index, holdout_pred)), columns = ['ImageId', 'Label'])


# In[ ]:


submission['ImageId'] = submission['ImageId'].apply(lambda x: x+1)


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv("Nischay_svm_mnist.csv",index=False)


# In[ ]:


submission.shape

