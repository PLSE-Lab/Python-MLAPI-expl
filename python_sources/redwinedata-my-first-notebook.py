#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## Hi,

## This is my first notebook. Please let me know the changes required for this.


# "https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009"
# 

# In[ ]:


# importing required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from  matplotlib.pyplot import subplot


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Read the dataset

df = pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
df.head()


# In[ ]:


df.shape


# In[ ]:


df.quality.value_counts()


# In[ ]:


df.isnull().sum()


# Insights:
# - All are numerical fields
# - Quaity is the dependent variable(discrete). All other fields are continuous.
# - Data is free from missing values

# In[ ]:





# ## Data Analysis

# In[ ]:


# check the number of records for all quality wine

sns.countplot(df['quality'])
#plt.xticks(rotation=90)


# Insight: records with wine quality 5 & 6 are more

# In[ ]:


# check the relationship between each variables
plt.figure(figsize=(20,10))
sns.heatmap(df.corr(), annot=True,cmap='Blues')
plt.show()


# In[ ]:


df.corr()


# In[ ]:



sns.barplot(y='fixed acidity', x='quality', data=df)


# In[ ]:



sns.barplot(y='volatile acidity', x='quality', data=df)


# In[ ]:



sns.barplot(y='citric acid', x='quality', data=df)


# In[ ]:



sns.barplot(y='residual sugar', x='quality', data=df)


# In[ ]:



sns.barplot(y='chlorides', x='quality', data=df)


# In[ ]:



sns.barplot(y='free sulfur dioxide', x='quality', data=df)


# In[ ]:



sns.barplot(y='total sulfur dioxide', x='quality', data=df)


# In[ ]:



sns.barplot(y='density', x='quality', data=df)


# In[ ]:



sns.barplot(y='pH', x='quality', data=df)


# In[ ]:



sns.barplot(y='sulphates', x='quality', data=df)


# In[ ]:


sns.barplot(y='alcohol', x='quality', data=df)


# In[ ]:





# ## Data pre processing

# #### From the Details of "https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009"
# 
# What might be an interesting thing to do, is aside from using regression modelling, is to set an arbitrary cutoff for your dependent variable (wine quality) at e.g. 7 or higher getting classified as 'good/1' and the remainder as 'not good/0'. This allows you to practice with hyper parameter tuning on e.g. decision tree algorithms looking at the ROC curve and the AUC value. Without doing any kind of feature engineering or overfitting you should be able to get an AUC of .88 (without even using random forest algorithm)
# 
# KNIME is a great tool (GUI) that can be used for this.
# 1. - File Reader (for csv) to linear correlation node and to interactive histogram for basic EDA.
# 2. - File Reader to 'Rule Engine Node' to turn the 10 point scale to dichtome variable (good wine and rest), the code to put in the rule engine is something like this:
# 
# $quality$ > 6.5 => "good"
# TRUE => "bad"
# 3. - Rule Engine Node output to input of Column Filter node to filter out your original 10point feature (this prevent leaking)
# 4. - Column Filter Node output to input of Partitioning Node (your standard train/tes split, e.g. 75%/25%, choose 'random' or 'stratified')
# 5. - Partitioning Node train data split output to input of Train data split to input Decision Tree Learner node and
# 6. - Partitioning Node test data split output to input Decision Tree predictor Node
# 7. - Decision Tree learner Node output to input Decision Tree Node input
# 8. - Decision Tree output to input ROC Node.. (here you can evaluate your model base on AUC value)

# In[ ]:


# making binary classification for the dependent variable Quality

# 0 -> bad
# 1 -> good

bins = (2, 6.5, 8)
group_names = [0, 1]
df['quality'] = pd.cut(df['quality'], bins = bins, labels = group_names)
df['quality'].value_counts()

# or

#df['quality']= df['quality'].replace(df[df['quality']<=6.5]['quality'],'bad')
#df['quality']= df['quality'].replace(df[df['quality']>6.5]['quality'],'good')


# In[ ]:





# In[ ]:


d=df.groupby('quality').mean().reset_index().T
d=d.reset_index().iloc[1:,:]
d.columns=['independent_variables','bad','good']
d


# # Model building

# In[ ]:


# check the number of records with bad and good quality wine

sns.countplot(df['quality'])


# insight: Data is imbalanced

# In[ ]:


x=df.drop('quality', axis = 1)
y= df['quality']


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


# Normalize using MinMaxScaler to constrain values to between 0 and 1.

scaler = MinMaxScaler(feature_range = (0,1))

scaler.fit_transform(x)

x.head()


# In[ ]:





# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)


# ### Logistic regression

# In[ ]:


#fit logistic regression model

from sklearn.linear_model import LogisticRegression
classifier_log = LogisticRegression()
model = classifier_log.fit(x_train,y_train)

y_pred_log = classifier_log.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred_log, y_test)*100)


# In[ ]:


probs = classifier_log.predict_proba(x_test)

from sklearn import metrics
prob_positive = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, prob_positive)

roc_auc = metrics.auc(fpr, tpr)
print('Area under the curve:', roc_auc)

plt.title('Receiver Operating characteristics')

plt.plot(fpr,tpr, 'orange', label= 'auc %0.2f'  )
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# ### Decision tree classifier

# In[ ]:


# fit decision tree

from sklearn.tree import DecisionTreeClassifier

# doing pruning to avoid overfitting
classifier_tree=DecisionTreeClassifier(criterion ='gini', splitter = 'random',
                         max_leaf_nodes = 10, min_samples_leaf = 5, 
                         max_depth = 6)
model = classifier_tree.fit(x_train, y_train)

y_pred_tree = classifier_tree.predict(x_test)

print(accuracy_score(y_pred_tree, y_test)*100)


# In[ ]:


probs = classifier_tree.predict_proba(x_test)

from sklearn import metrics
prob_positive = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, prob_positive)

roc_auc = metrics.auc(fpr, tpr)
print('Area under the curve:', roc_auc)

plt.title('Receiver Operating characteristics')

#plt.title('receiver operating characteristics')
plt.plot(fpr, tpr, 'orange', label='AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')

plt.plot([0,1],[0,1],color='darkblue', linestyle='--')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.show()


# ### Random forest classifier

# In[ ]:


# fit random forest

from sklearn.ensemble import RandomForestClassifier
classifier_forest = RandomForestClassifier(n_estimators=100,
                                           criterion = 'entropy',
                                           random_state = 0,max_leaf_nodes = 10, min_samples_leaf = 5, 
                         max_depth = 6)

model = classifier_forest.fit(x_train,y_train)

y_pred_tree = classifier_forest.predict(x_test)

print(accuracy_score(y_pred_tree, y_test)*100)


# In[ ]:


probs = classifier_forest.predict_proba(x_test)

from sklearn import metrics
prob_positive = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, prob_positive)

roc_auc = metrics.auc(fpr, tpr)
print('Area under the curve:', roc_auc)

plt.title('Receiver Operating characteristics')

#plt.title('receiver operating characteristics')
plt.plot(fpr, tpr, 'orange', label='AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')

plt.plot([0,1],[0,1],color='darkblue', linestyle='--')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.show()


# ## Upsampling

# In[ ]:





# In[ ]:


print(df['quality'].value_counts())
df_majority= df[df['quality'] == 0]
df_minority = df[df['quality']== 1]


# In[ ]:


import sklearn.utils as ut

df_minority_upsampled = ut.resample(df_minority, 
                                   replace = True, # sample with replacement
                                   n_samples = 1382, # to match majority class
                                   random_state = 1)  # reproducible results


# In[ ]:





# In[ ]:


df_upsampled = pd.concat([df_majority, df_minority_upsampled])
df_upsampled.head()


# In[ ]:


df_upsampled.quality.value_counts()


# In[ ]:


x_upsampled = df_upsampled.drop('quality', axis = 1)
y_upsampled = df_upsampled['quality']


# Normalize using MinMaxScaler to constrain values to between 0 and 1.

scaler = MinMaxScaler(feature_range = (0,1))

scaler.fit_transform(x_upsampled)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(
                                                    x_upsampled, y_upsampled,
                                                    test_size=0.2, random_state=0)


# ### random forest

# In[ ]:


# Random forest

classifier_forest = RandomForestClassifier(n_estimators=100,
                                           criterion = 'entropy',
                                           random_state = 0)

model = classifier_forest.fit(x_train,y_train)

y_pred_tree = classifier_forest.predict(x_test)

print(accuracy_score(y_pred_tree, y_test)*100)


# In[ ]:


probs = classifier_forest.predict_proba(x_test)

from sklearn import metrics
prob_positive = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, prob_positive)

roc_auc = metrics.auc(fpr, tpr)
print('Area under the curve:', roc_auc)

plt.title('Receiver Operating characteristics')

#plt.title('receiver operating characteristics')
plt.plot(fpr, tpr, 'orange', label='AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')

plt.plot([0,1],[0,1],color='darkblue', linestyle='--')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.show()


# In[ ]:




