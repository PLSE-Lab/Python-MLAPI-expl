#!/usr/bin/env python
# coding: utf-8

# **My new to data science. Please my notebook solution and let me know the inputs to improve the solution**

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


import numpy as np
import pandas as pd
import seaborn as sns
# Suppressing Warnings
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df_heart=pd.read_csv('../input/heart.csv')


# In[ ]:


df_heart.dtypes


# In[ ]:


df_heart.head()


# In[ ]:


df_heart.describe()


# In[ ]:


df_heart.isnull().sum()


# ## Univariate Analysis:

# ###  Categorical Features

# In[ ]:


sns.countplot(x='age',data=df_heart)


# In[ ]:


df_heart['sex'].value_counts().plot.pie()


# In[ ]:


df_heart['cp'].value_counts().plot.pie()


# In[ ]:


df_heart['fbs'].value_counts().plot.bar()


# In[ ]:


df_heart['exang'].value_counts().plot.bar()


# In[ ]:


df_heart['restecg'].value_counts().plot.bar()


# In[ ]:


df_heart['ca'].value_counts().plot.bar()


# In[ ]:


df_heart['thal'].value_counts().plot.bar()


# In[ ]:


df_heart['slope'].value_counts().plot.bar()


# In[ ]:


df_heart['target'].value_counts().plot.bar()


# ### Numerical Features:

# In[ ]:


sns.distplot(df_heart['chol'])


# In[ ]:


sns.distplot(df_heart['thalach'])


# In[ ]:


sns.distplot(df_heart['oldpeak'])


# In[ ]:


sns.distplot(df_heart['trestbps'])


# ## Bi-Variate Analysis

# In[ ]:


sns.countplot(x="age", hue="target", data=df_heart)


# In[ ]:


sns.countplot(x="sex", hue="target", data=df_heart)


# In[ ]:


sns.countplot(x="cp", hue="target", data=df_heart)


# In[ ]:


sns.countplot(x="fbs", hue="target", data=df_heart)


# In[ ]:


sns.countplot(x="restecg", hue="target", data=df_heart)


# In[ ]:


sns.countplot(x="exang", hue="target", data=df_heart)


# In[ ]:


sns.countplot(x="ca", hue="target", data=df_heart)


# In[ ]:


sns.countplot(x="thal", hue="target", data=df_heart)


# In[ ]:


a = pd.get_dummies(df_heart['cp'], prefix = "cp")
b = pd.get_dummies(df_heart['thal'], prefix = "thal")
c = pd.get_dummies(df_heart['slope'], prefix = "slope")
d = pd.get_dummies(df_heart['ca'], prefix = "ca") 
e = pd.get_dummies(df_heart['restecg'], prefix = "restecg") 


# In[ ]:


frames = [df_heart, a, b, c,d,e]
df_heart = pd.concat(frames, axis = 1)
df_heart.head()


# In[ ]:


df_heart = df_heart.drop(columns = ['cp', 'thal', 'slope','ca','restecg'])
df_heart.head()


# ### Modeling : Classification

# ### 1. Train Test Split

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


# Putting feature variable to X
X = df_heart.drop(['target'], axis=1)

X.head()


# In[ ]:


# Putting response variable to y
y = df_heart['target']

y.head()


# In[ ]:


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# ### 2. Feature Scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler = StandardScaler()

X_train[['age','trestbps','chol','thalach','oldpeak']] = scaler.fit_transform(X_train[['age','trestbps','chol','thalach','oldpeak']])

X_train.head()


# In[ ]:


### Checking the disease Rate
target = (sum(df_heart['target'])/len(df_heart['target'].index))*100
target


# ### 3. Looking at cormelations

# In[ ]:


# Let's see the correlation matrix 
import matplotlib.pyplot as plt
plt.figure(figsize = (20,10))        # Size of the figure
sns.heatmap(df_heart.corr(),annot = True)
plt.show()


# ### 4. Model Building

# In[ ]:


import statsmodels.api as sm


# In[ ]:


# Logistic regression model
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# ### 5. Feature Selection Using RFE

# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[ ]:


from sklearn.feature_selection import RFE
rfe = RFE(logreg, 15)             # running RFE with 15 variables as output
rfe = rfe.fit(X_train, y_train)


# In[ ]:


rfe.support_


# In[ ]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[ ]:


col = X_train.columns[rfe.support_]


# In[ ]:


X_train.columns[~rfe.support_]


# In[ ]:


X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[ ]:


# Getting the predicted values on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[ ]:


y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# In[ ]:


y_train_pred_final = pd.DataFrame({'target':y_train.values, 'target_Prob':y_train_pred})
y_train_pred_final.head()


# In[ ]:


y_train_pred_final['predicted'] = y_train_pred_final.target_Prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()


# In[ ]:


from sklearn import metrics


# In[ ]:


# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.target, y_train_pred_final.predicted )
print(confusion)


# In[ ]:


# Predicted     not_have    have_disease
# Actual
# no_disease        78      14
# have_disease     10     110 


# In[ ]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.target, y_train_pred_final.predicted))


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


# In[ ]:


col = col.drop('ca_0', 1)
col


# In[ ]:


# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train[col])
logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm3.fit()
res.summary()


# In[ ]:


y_train_pred = res.predict(X_train_sm).values.reshape(-1)


# In[ ]:


y_train_pred[:10]


# In[ ]:


y_train_pred_final['target_Prob'] = y_train_pred


# In[ ]:


# Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0
y_train_pred_final['predicted'] = y_train_pred_final.target_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# In[ ]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.target, y_train_pred_final.predicted))


# In[ ]:


vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


# Let's take a look at the confusion matrix again 
confusion = metrics.confusion_matrix(y_train_pred_final.target, y_train_pred_final.predicted )
confusion


# In[ ]:


# Predicted     not_have    have
# Actual
# not_have        78      14
# have            10     110 


# In[ ]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[ ]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[ ]:


# Let us calculate specificity
TN / float(TN+FP)


# In[ ]:


# Calculate false postive rate 
print(FP/ float(TN+FP))


# In[ ]:


# positive predictive value 
print (TP / float(TP+FP))


# In[ ]:


# Negative predictive value
print (TN / float(TN+ FN))


# In[ ]:


col = col.drop('slope_2', 1)
col


# In[ ]:


# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train[col])
logm4 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm4.fit()
res.summary()


# In[ ]:


y_train_pred = res.predict(X_train_sm).values.reshape(-1)


# In[ ]:


y_train_pred[:10]


# In[ ]:


y_train_pred_final['target_Prob'] = y_train_pred


# In[ ]:


# Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0
y_train_pred_final['predicted'] = y_train_pred_final.target_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# In[ ]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.target, y_train_pred_final.predicted))


# In[ ]:


vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


col = col.drop('ca_4', 1)
col


# In[ ]:


# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train[col])
logm5 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm5.fit()
res.summary()


# In[ ]:


y_train_pred = res.predict(X_train_sm).values.reshape(-1)


# In[ ]:


y_train_pred[:10]


# In[ ]:


y_train_pred_final['target_Prob'] = y_train_pred


# In[ ]:


# Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0
y_train_pred_final['predicted'] = y_train_pred_final.target_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# In[ ]:


vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


col = col.drop('thal_2', 1)
col


# In[ ]:


# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train[col])
logm6 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm6.fit()
res.summary()


# In[ ]:


y_train_pred = res.predict(X_train_sm).values.reshape(-1)


# In[ ]:


y_train_pred[:10]


# In[ ]:


y_train_pred_final['target_Prob'] = y_train_pred


# In[ ]:


# Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0
y_train_pred_final['predicted'] = y_train_pred_final.target_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# In[ ]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.target, y_train_pred_final.predicted))


# In[ ]:


vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


col = col.drop('restecg_1', 1)
col


# In[ ]:


# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train[col])
logm7 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm7.fit()
res.summary()


# In[ ]:


y_train_pred = res.predict(X_train_sm).values.reshape(-1)


# In[ ]:


y_train_pred[:10]


# In[ ]:


y_train_pred_final['target_Prob'] = y_train_pred


# In[ ]:


# Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0
y_train_pred_final['predicted'] = y_train_pred_final.target_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# In[ ]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.target, y_train_pred_final.predicted))


# In[ ]:


vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


col = col.drop('cp_3', 1)
col


# In[ ]:


# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train[col])
logm8 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm8.fit()
res.summary()


# In[ ]:


y_train_pred = res.predict(X_train_sm).values.reshape(-1)


# In[ ]:


y_train_pred[:10]


# In[ ]:


y_train_pred_final['target_Prob'] = y_train_pred


# In[ ]:


# Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0
y_train_pred_final['predicted'] = y_train_pred_final.target_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# In[ ]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.target, y_train_pred_final.predicted))


# In[ ]:


vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[ ]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.target, y_train_pred_final.target_Prob, drop_intermediate = False )


# In[ ]:


draw_roc(y_train_pred_final.target, y_train_pred_final.target_Prob)


# In[ ]:


# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.target_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[ ]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

#TP = confusion[1,1] # true positive 
#TN = confusion[0,0] # true negatives
#FP = confusion[0,1] # false positives
#FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.target, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[ ]:


# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# In[ ]:


y_train_pred_final['final_predicted'] = y_train_pred_final.target_Prob.map( lambda x: 1 if x > 0.57 else 0)

y_train_pred_final.head()


# In[ ]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.target, y_train_pred_final.final_predicted)


# In[ ]:


confusion2 = metrics.confusion_matrix(y_train_pred_final.target, y_train_pred_final.final_predicted )
confusion2


# In[ ]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[ ]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[ ]:


# Let us calculate specificity
TN / float(TN+FP)


# In[ ]:


# Calculate false postive rate - predicting churn when customer does not have churned
print(FP/ float(TN+FP))


# In[ ]:


# Positive predictive value 
print (TP / float(TP+FP))


# In[ ]:


# Negative predictive value
print (TN / float(TN+ FN))


# ## Precision and Recall

# In[ ]:


confusion = metrics.confusion_matrix(y_train_pred_final.target, y_train_pred_final.predicted )
confusion


# ##### Precision
# TP / TP + FP

# In[ ]:


confusion[1,1]/(confusion[0,1]+confusion[1,1])


# ##### Recall
# TP / TP + FN

# In[ ]:


confusion[1,1]/(confusion[1,0]+confusion[1,1])


# In[ ]:


from sklearn.metrics import precision_score, recall_score


# In[ ]:


precision_score(y_train_pred_final.target, y_train_pred_final.predicted)


# In[ ]:


recall_score(y_train_pred_final.target, y_train_pred_final.predicted)


# ### Precision and recall tradeoff

# In[ ]:


from sklearn.metrics import precision_recall_curve


# In[ ]:


y_train_pred_final.target, y_train_pred_final.predicted


# In[ ]:


p, r, thresholds = precision_recall_curve(y_train_pred_final.target, y_train_pred_final.target_Prob)


# In[ ]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# In[ ]:


X_test[['age','trestbps','chol','thalach','oldpeak']] = scaler.transform(X_test[['age','trestbps','chol','thalach','oldpeak']])


# In[ ]:


X_test = X_test[col]
X_test.head()


# In[ ]:


X_test_sm = sm.add_constant(X_test)


# In[ ]:


y_test_pred = res.predict(X_test_sm)


# In[ ]:


y_test_pred[:10]


# In[ ]:


# Converting y_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)


# In[ ]:


# Let's see the head
y_pred_1.head()


# In[ ]:


# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)


# In[ ]:


# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[ ]:


# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)


# In[ ]:


# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'target_Prob'})


# In[ ]:


# Rearranging the columns
y_pred_final = y_pred_final.reindex_axis(['target','target_Prob'], axis=1)


# In[ ]:


# Let's see the head of y_pred_final
y_pred_final.head()


# In[ ]:


y_pred_final['final_predicted'] = y_pred_final.target_Prob.map(lambda x: 1 if x > 0.60 else 0)


# In[ ]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_pred_final.target, y_pred_final.final_predicted)


# In[ ]:


confusion2 = metrics.confusion_matrix(y_pred_final.target, y_pred_final.final_predicted )
confusion2


# In[ ]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[ ]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[ ]:


# Let us calculate specificity
TN / float(TN+FP)


# In[ ]:


# Calculate false postive rate - predicting churn when customer does not have churned
print(FP/ float(TN+FP))


# In[ ]:


# Positive predictive value 
print (TP / float(TP+FP))


# In[ ]:


# Negative predictive value
print (TN / float(TN+ FN))


# In[ ]:


fpr, tpr, thresholds = metrics.roc_curve( y_pred_final.target, y_pred_final.target_Prob, drop_intermediate = False )


# In[ ]:


draw_roc(y_pred_final.target, y_pred_final.target_Prob)

