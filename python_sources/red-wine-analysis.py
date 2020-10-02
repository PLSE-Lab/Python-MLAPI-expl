#!/usr/bin/env python
# coding: utf-8

# # Red wine analysis
# ### by Rhodium Beng
# Started on 10 October 2017
# 
# This kernel is based on a tutorial in EliteDataScience. In that tutorial, a Random Forest Regressor was used and involved standardizing the data and hyperparameter tuning. Here, I am using a Random Forest Classifier instead, to do a binary classification of good wine and not-so-good wine.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Load data

# In[2]:


# For this kernel, I amm only using the red wine dataset
data = pd.read_csv('../input/winequality-red.csv')
data.head()


# ### Summary statistics

# In[3]:


data.describe()


# #### All columns has the same number of data points, so it looks like there are no missing data. 
# #### Are there duplicated rows in the data?

# In[4]:


extra = data[data.duplicated()]
extra.shape


# #### There are 240 duplicates. But I think it is wise to keep these "extra" as I am assuming that the quality ratings for the same/similar wine were given by different wine tasters.

# In[5]:


# Let's proceed to separate 'quality' as the target variable and the rest as features.
y = data.quality                  # set 'quality' as target
X = data.drop('quality', axis=1)  # rest are features
print(y.shape, X.shape)           # check correctness


# ## Visualize data through plots

# In[10]:


# data.hist(figsize=(10,10))
sns.set()
data.hist(figsize=(10,10), color='red')
plt.show()


# #### Quality are in discrete numbers, and not a continous variable. Most of the wine are rated '5' & '6', with much fewer in the other numbers. Let's look at the correlation among the variables using Correlation chart.

# In[12]:


colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Correlation of Features', y=1.05, size=15)
sns.heatmap(data.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, 
            linecolor='white', annot=True)


# #### Observations:
# - Alcohol has the highest correlation with wine quality, followed by the various acidity, sulphates, density & chlorides.
# - Let's use all the features in the classifiers.

# ## Group the wine into two groups; with 'quality' > 5 as "good wine"

# In[13]:


# Create a new y1
y1 = (y > 5).astype(int)
y1.head()


# In[18]:


# plot histogram
ax = y1.plot.hist(color='green')
ax.set_title('Wine quality distribution', fontsize=14)
ax.set_xlabel('aggregated target value')


# ## Use Random Forest Classifier to train a prediction model

# In[19]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import confusion_matrix


# ### Split data into training and test datasets

# In[20]:


seed = 8 # set seed for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.2,
                                                    random_state=seed)


# In[21]:


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# ### Train and evaluate the Random Forest Classifier with Cross Validation

# In[22]:


# Instantiate the Random Forest Classifier
RF_clf = RandomForestClassifier(random_state=seed)
RF_clf


# In[23]:


# Compute k-fold cross validation on training dataset and see mean accuracy score
cv_scores = cross_val_score(RF_clf,X_train, y_train, cv=10, scoring='accuracy')
print('The accuracy scores for the iterations are {}'.format(cv_scores))
print('The mean accuracy score is {}'.format(cv_scores.mean()))


# ## Perform predictions

# In[24]:


RF_clf.fit(X_train, y_train)
pred_RF = RF_clf.predict(X_test)


# In[25]:


# Print 5 results to see
for i in range(0,5):
    print('Actual wine quality is ', y_test.iloc[i], ' and predicted is ', pred_RF[i])


# #### The first five predictions look ok. Out of first five, there are one mistake. Let's look at the metrics.

# ## Accuracy, log loss and confusion matrix

# In[26]:


print(accuracy_score(y_test, pred_RF))
print(log_loss(y_test, pred_RF))


# In[27]:


print(confusion_matrix(y_test, pred_RF))


# #### There is a total of 66 classification errors.

# ## Let's fit a Logistic Regression model as a classifier

# In[28]:


# Import and istantiate the Logistic Regression model
from sklearn.linear_model import LogisticRegression
LR_clf = LogisticRegression(random_state=seed)
LR_clf


# ### Train and evaluate the Logistic Regression Classifier with Cross Validation

# In[29]:


# Compute cross validation scores on training dataset and see mean score
cv_scores = cross_val_score(LR_clf, X_train, y_train, cv=10, scoring='accuracy')
print('The cv scores from the iterations are {}'.format(cv_scores))
print('The mean cv score is {}'.format(cv_scores.mean()))


# ## Perform predictions

# In[30]:


LR_clf.fit(X_train, y_train)
pred_LR = LR_clf.predict(X_test)


# In[31]:


# Print 5 results to see
for i in range(0,5):
    print('Actual wine quality is ', y_test.iloc[i], ' and predicted is ', pred_LR[i])


# #### Out of the first five predictions, there are already two classification errors.

# ## Accuracy, log loss and confusion matrix

# In[32]:


print(accuracy_score(y_test, pred_LR))
print(log_loss(y_test, pred_LR))


# In[33]:


print(confusion_matrix(y_test, pred_LR))


# #### There is a total of 81 classification errors.
# #### Compared to the Logistic Regression classifier, the Random Forest classifier is a shade better.

# ## Let's tune hyperparameters of the Random Forest classifier

# In[ ]:


from sklearn.model_selection import GridSearchCV
grid_values = {'n_estimators':[50,100,200],'max_depth':[None,30,15,5],
               'max_features':['auto','sqrt','log2'],'min_samples_leaf':[1,20,50,100]}
grid_RF = GridSearchCV(RF_clf,param_grid=grid_values,scoring='accuracy')
grid_RF.fit(X_train, y_train)


# In[ ]:


grid_RF.best_params_


# #### Other than number of estimators, the other recommended values are the defaults.

# In[ ]:


RF_clf = RandomForestClassifier(n_estimators=100,random_state=seed)
RF_clf.fit(X_train,y_train)
pred_RF = RF_clf.predict(X_test)


# In[ ]:


print(accuracy_score(y_test,pred_RF))
print(log_loss(y_test,pred_RF))


# In[ ]:


print(confusion_matrix(y_test,pred_RF))


# #### With hyperparameter tuning, the accuracy of the RF classifier has improved to 82.5% with a corresponding reduction in the log loss value. The number of classification errors are also reduced to 56.

# # End Remarks
# ### A prediction accuracy of 82.5% looks reasonable to use the Random Forest Classifier as a basic recommender to classify a red wine as "recommended" (6 & above quality rating) or "not recommended" (5 & below quality rating).
# ### Please upvote if you find this basic analysis useful. 
# ### Tips, comments are welcomed.

# In[ ]:




