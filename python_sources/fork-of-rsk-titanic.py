#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
train_df.head()


# In[ ]:


train_df.info()


# In[ ]:


# Check for missing values

train_df.isnull().sum()


# In[ ]:


# Percentage of missing values

round(train_df.isnull().sum() * 100 / len(train_df.index), 2)


# This `Cabin` column has lots of missing values. So, lets' drop it from the dataframe.

# In[ ]:


train_df = train_df.drop('Cabin', axis=1)


# In[ ]:


train_df.info()


# Additionally, `Name` and `Ticket` columns are not going to help us in making prediction. So, let's drop them as well.

# In[ ]:


train_df = train_df.drop(columns=['Name', 'Ticket'], axis=1)
train_df.info()


# In[ ]:


embarked_counts = train_df['Embarked'].astype('category').value_counts()
embarked_counts


# In[ ]:


count_sex = train_df['Sex'].astype('category').value_counts()
count_sex


# In[ ]:


count_parch = train_df['Parch'].astype('category').value_counts()
count_parch


# In[ ]:


train_df['Age'].describe()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.boxplot(train_df['Age'])
plt.show()


# Let's impute the missing values in the `Age` column with the median values.

# In[ ]:


train_df.loc[train_df['Age'].isnull(), ['Age']] = train_df['Age'].median()
train_df['Age'].isnull().sum()


# In[ ]:


sns.boxplot(train_df['Age'])
plt.show()


# In[ ]:


train_df.head()


# In[ ]:


def binary_func(x):
    if x == 'male':
        x = 1
    elif x == 'female':
        x = 0
    return x

y = list(map(binary_func, train_df['Sex']))
y = np.array(y)
y = pd.Series(y)
train_df['Sex'] = y
train_df.head()


# In[ ]:


train_df.info()


# In[ ]:


embarked = pd.get_dummies(train_df['Embarked'])
embarked.head()


# In[ ]:


train_df = pd.concat([train_df, embarked], axis=1)
train_df.head()


# In[ ]:


train_df = train_df.drop(columns=['Embarked'])
train_df.head()


# In[ ]:


corr = train_df.corr()
corr


# In[ ]:


plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='YlGnBu')
plt.show()


# In[ ]:


train_df.isnull().sum()


# In[ ]:


# Check for outliers
train_df.describe(percentiles=[0.25, 0.50, 0.75, 0.95, 0.99])


# In[ ]:


sns.countplot(train_df['Survived'])
plt.show()


# In[ ]:


sns.countplot(train_df['Survived'], hue=train_df['Sex'])
plt.show()


# In[ ]:


sns.countplot(train_df['Survived'], hue=train_df['Pclass'])
plt.show()


# In[ ]:


sns.countplot(train_df['S'], hue=train_df['Survived'])
plt.show()


# In[ ]:


sns.countplot(train_df['C'], hue=train_df['Survived'])
plt.show()


# In[ ]:


sns.countplot(train_df['Q'], hue=train_df['Survived'])
plt.show()


# In[ ]:


sns.countplot(train_df['Survived'], hue=train_df['Parch'])
plt.show()


# In[ ]:


sns.countplot(train_df['Survived'], hue=train_df['SibSp'])
plt.show()


# In[ ]:


plt.hist(train_df['Pclass'])
plt.show()


# In[ ]:


plt.hist(train_df['Sex'])
plt.show()


# In[ ]:


plt.hist(train_df['SibSp'])
plt.show()


# In[ ]:


plt.hist(train_df['Age'])
plt.show()


# In[ ]:


plt.hist(train_df['Fare'])
plt.show()


# In[ ]:


X_train = train_df.drop(columns=['PassengerId', 'Survived'], axis=1)
X_train.head()


# In[ ]:


y_train = train_df['Survived']
y_train.head()


# ### Feature Scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[ ]:


X_train[['Age', 'Fare']] = scaler.fit_transform(X_train[['Age', 'Fare']])
X_train.head()


# In[ ]:


plt.figure(figsize=(12, 6))
sns.heatmap(X_train.corr(), annot=True, cmap='YlGnBu')
plt.show()


# ### Model Building

# In[ ]:


import statsmodels.api as sm


# In[ ]:


# Logistic regression model
log_reg_m1 = sm.GLM(y_train, (sm.add_constant(X_train)), family=sm.families.Binomial())
log_reg_m1_summary = log_reg_m1.fit().summary()
log_reg_m1_summary


# ### Feature Select Using RFE

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE


# In[ ]:


rfe = RFE(LogisticRegression(), 5) # running RFE with 5 variables as output
rfe = rfe.fit(X_train, y_train)
rfe.support_


# In[ ]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# Columns supported by RFE

# In[ ]:


col = X_train.columns[rfe.support_]
col


# Columns not supported by RFE.

# In[ ]:


X_train.columns[~rfe.support_]


# #### Assessing the model with StatsModels

# In[ ]:


X_train_sm = sm.add_constant(X_train[col])
log_reg_m2 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = log_reg_m2.fit()
res.summary()


# ### Checking VIFs

# In[ ]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# We need `Pclass` because according to p-Value, it is statistically significant. So, let's drop `S` column and rebuild the model.

# In[ ]:


col = col.drop('S', 1)
col


# In[ ]:


# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train[col])
log_reg_m3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = log_reg_m3.fit()
res.summary()


# The `Q` column is still statistically insignificant. Hence, dropping it and rebuilding the model.

# In[ ]:


col = col.drop('Q', 1)
col


# In[ ]:


# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train[col])
log_reg_m4 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = log_reg_m4.fit()
res.summary()


# All the variables are statistically significant.

# In[ ]:


# Recalculating the VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# All the variables have a VIF value less than 5 which is great.

# ### Making Predictions on the Train Data

# In[ ]:


y_train_pred = res.predict(X_train_sm).values.reshape(-1)
y_train_pred[:10]


# In[ ]:


y_train_pred_final = pd.DataFrame({'Survived': y_train.values, 'Survival_Prob':y_train_pred})
y_train_pred_final['PassengerId'] = y_train.index
y_train_pred_final.head()


# #### Creating new column 'Predicted' with 1 if Churn_Prob > 0.5 else 0

# In[ ]:


y_train_pred_final['Predicted'] = y_train_pred_final.Survival_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# In[ ]:


from sklearn import metrics


# In[ ]:


# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final.Predicted )
print(confusion)


# In[ ]:


# Predicted     not_survived    survived
# Actual
# not_survived        443      106
# survived            92       250  


# In[ ]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Survived, y_train_pred_final.Predicted))


# So, the accuracy is almost 78%. This is quite good. Let's check for other metrics as well.

# ## Metrics beyond simply accuracy

# In[ ]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[ ]:


# Checking the sensitivity of the logistic regression model
TP / float(TP+FN)


# In[ ]:


# Checking specificity
TN / float(TN+FP)


# In[ ]:


# Calculating false postive rate - predicting survival when a passenger has died.
print(FP/ float(TN+FP))


# In[ ]:


# positive predictive value 
print (TP / float(TP+FP))


# In[ ]:


# Negative predictive value
print (TN / float(TN+ FN))


# ### ROC Curve

# In[ ]:


def draw_roc(actual, probs):
    fpr, tpr, thresholds = metrics.roc_curve(actual, probs, drop_intermediate = False)
    auc_score = metrics.roc_auc_score(actual, probs)
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC Curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[ ]:


fpr, tpr, thresholds = metrics.roc_curve(y_train_pred_final.Survived, y_train_pred_final.Survival_Prob, drop_intermediate = False)


# In[ ]:


draw_roc(y_train_pred_final.Survived, y_train_pred_final.Survival_Prob)


# **### Compute Optimal Survival Probablity Cutoff Value

# In[ ]:


# Create columns with different probability cutoff values
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Survival_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[ ]:


# Calculate accuracy, sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame(columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    conf_mat_1 = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final[i] )
    total1=sum(sum(conf_mat_1))
    accuracy = (conf_mat_1[0,0]+conf_mat_1[1,1])/total1
    
    speci = conf_mat_1[0,0]/(conf_mat_1[0,0]+conf_mat_1[0,1])
    sensi = conf_mat_1[1,1]/(conf_mat_1[1,0]+conf_mat_1[1,1])
    cutoff_df.loc[i] =[i, accuracy,sensi,speci]
print(cutoff_df)


# In[ ]:


# Plotting accuracy, sensitivity and specificity for various probabilities.
plt.figure(figsize=(12, 8))
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# #### From the curve above, 0.38 is the optimum point to consider it as a cutoff probability.

# In[ ]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Survival_Prob.map(lambda x: 1 if x > 0.38 else 0)
y_train_pred_final.head()


# In[ ]:


# Check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.Survived, y_train_pred_final.final_predicted)


# So, the accuracy value gets reduced to 75% from ~78%.

# In[ ]:


confusion2 = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final.final_predicted )
confusion2


# In[ ]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[ ]:


# Recalculating the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[ ]:


# Specificity
TN / float(TN+FP)


# In[ ]:


# False postive rate - predicting survival when a passenge is alive
print(FP/ float(TN+FP))


# In[ ]:


# Positive predictive value 
print(TP / float(TP+FP))


# In[ ]:


# Negative predictive value
print(TN / float(TN+ FN))


# #### Precision and Recall
# 
# Let's go over the confusion matrix again.****

# In[ ]:


confusion = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final.final_predicted)
confusion


# ##### Precision
# TP / TP + FP

# In[ ]:


confusion[1,1]/(confusion[0,1]+confusion[1,1])


# ##### Recall
# TP / TP + FN

# In[ ]:


confusion[1,1]/(confusion[1,0]+confusion[1,1])


# #### Using sklearn utilities for the same metrics

# In[ ]:


from sklearn.metrics import precision_score, recall_score


# In[ ]:


precision_score(y_train_pred_final.Survived, y_train_pred_final.final_predicted)


# In[ ]:


recall_score(y_train_pred_final.Survived, y_train_pred_final.final_predicted)


# ### Precision and recall tradeoff

# In[ ]:


from sklearn.metrics import precision_recall_curve


# In[ ]:


y_train_pred_final.Survived, y_train_pred_final.final_predicted


# In[ ]:


p, r, thresholds = precision_recall_curve(y_train_pred_final.Survived, y_train_pred_final.Survival_Prob)


# In[ ]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# ### Making predictions on the test set

# In[ ]:


test_df = pd.read_csv('../input/test.csv')
test_df.head()


# In[ ]:


test_df.info()


# In[ ]:


test_df.describe()


# Based on the quartiles, mean and standard deviation, there seem to be no outliers.

# In[ ]:


test_df.shape


# In[ ]:


# Checking for missing value
test_df.isnull().sum()


# Based on our Logistic Regression model, we are concnered with the ```Pclass```, ```Sex``` and ```C```. These features are not missing from the test database, so we don't have to treat the missing values. However, we have to binary encode the ```Sex``` values and extract the ```C``` values from the test dataframe.

# In[ ]:


z = list(map(binary_func, test_df['Sex']))
z = np.array(z)
z = pd.Series(z)
test_df['Sex'] = z
test_df.head()


# In[ ]:


test_embarked = pd.get_dummies(test_df['Embarked'])
test_embarked.head()


# In[ ]:


test_df = pd.concat([test_df, test_embarked], axis=1)
test_df.head()


# In[ ]:


test_df = test_df.drop(columns=['Embarked'])
test_df.head()


# In[ ]:


test_df = test_df[['PassengerId', 'Pclass', 'Sex', 'C']]
test_df.head()


# In[ ]:


X_test = test_df[['Pclass', 'Sex', 'C']]
X_test.head()


# No need to scale the values as they are comparable.

# In[ ]:


X_test_sm = sm.add_constant(X_test)


# Making predictions on the test set

# In[ ]:


y_test_pred = res.predict(X_test_sm)
y_test_pred[:10]


# In[ ]:


y_test_pred_final = pd.DataFrame(y_test_pred)
y_test_pred_final['Survival_Prob'] = pd.DataFrame(y_test_pred)


# In[ ]:


y_test_pred_final['Survived'] = y_test_pred_final.Survival_Prob.map(lambda x: 1 if x > 0.38 else 0)
y_test_pred_final = y_test_pred_final[['Survival_Prob', 'Survived']]
y_test_pred_final['PassengerId'] = test_df.PassengerId
y_test_pred_final.head()


# In[ ]:


df_for_submission = y_test_pred_final[['PassengerId', 'Survived']]
df_for_submission.head()


# In[ ]:


df_for_submission.shape


# In[ ]:


df_for_submission.to_csv('titanic-second-submission.csv')

