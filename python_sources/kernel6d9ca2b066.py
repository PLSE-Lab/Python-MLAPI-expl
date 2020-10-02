#!/usr/bin/env python
# coding: utf-8

# In[152]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from sklearn.utils.extmath import weighted_mode


# In[153]:


train = pd.read_csv("train.csv")


# In[154]:


def count_days(row, begin, end, powers):
    idx = 0
    days = [0] * 7
    
    for day in row:
        if day < begin or day > end:
            continue
        days[get_weekday(day) - 1] += powers[day - 1]
        idx += 1
    
    days = pd.Series(days)
    days /= days.abs().sum()
    return days


# In[155]:


def get_weekday(day):
    return (day - 1) % 7 + 1


# In[156]:


def parse_row(row):
    return list(map(int, row.split(' ')[1:]))


# In[157]:


train['visits'] = train['visits'].apply(parse_row)


# In[158]:


train['answer'] = train['visits'].apply(lambda x: get_weekday(x[-1]))


# In[159]:


powers = [1.001 ** (i - i % 7) if i > 100 else 0 for i in range(1099)]

def get_weighted_mode(row):
    days = [get_weekday(i) for i in row]
    weigths = [powers[i - 1] for i in row]
    return int(weighted_mode(days, weigths)[0][0])


# In[160]:


train['answer'] = train['visits'].apply(get_weighted_mode)


# In[161]:


train.sample(5)


# In[162]:


res = pd.DataFrame(columns=['id', 'nextvisit'])
res['id'] = train['id']
res['nextvisit'] = train['answer'].apply(lambda x: ' ' + str(x))
res.to_csv('solution.csv', index=False, sep=',')


# I've wanted a decent model with features, but deadline always is an unexpected surprise =(

# In[163]:


exit(0)


# In[ ]:





# In[ ]:





# In[ ]:





# In[149]:


count_days(train.loc[190684].visits, 1, 1100, powers).append(count_days(train.loc[190684].visits, 1000, 1100, [1] * 1100))


# In[150]:


count_days([1, 2, 3, 9, 10, 16], 10, 17, [1] * 100) + count_days()


# In[ ]:





# In[3]:


train.info(), test.info()


# In[4]:


# set "PassengerId" variable as index
train.set_index("PassengerId", inplace=True)
test.set_index("PassengerId", inplace=True)


# In[5]:


# generate training target set (y_train)
y_train = train["Survived"]


# In[6]:


# delete column "Survived" from train set
train.drop(labels="Survived", axis=1, inplace=True)


# In[7]:


# shapes of train and test sets
train.shape, test.shape


# In[8]:


# join train and test sets to form a new train_test set
train_test =  train.append(test)


# In[9]:


# delete columns that are not used as features for training and prediction
columns_to_drop = ["Name", "Age", "SibSp", "Ticket", "Cabin", "Parch", "Embarked"]
train_test.drop(labels=columns_to_drop, axis=1, inplace=True)


# In[10]:


# convert objects to numbers by pandas.get_dummies
train_test_dummies = pd.get_dummies(train_test, columns=["Sex"])


# In[11]:


# check the dimension
train_test_dummies.shape


# In[12]:


# replace nulls with 0.0
train_test_dummies.fillna(value=0.0, inplace=True)


# In[13]:


# generate feature sets (X)
X_train = train_test_dummies.values[0:891]
X_test = train_test_dummies.values[891:]


# In[14]:


X_train.shape, X_test.shape


# In[15]:


# transform data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)


# In[16]:


# split training feature and target sets into training and validation subsets
from sklearn.model_selection import train_test_split

X_train_sub, X_validation_sub, y_train_sub, y_validation_sub = train_test_split(X_train_scale, y_train, random_state=0)


# In[17]:


# import machine learning algorithms
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc


# In[18]:


# train with Gradient Boosting algorithm
# compute the accuracy scores on train and validation sets when training with different learning rates

learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
for learning_rate in learning_rates:
    gb = GradientBoostingClassifier(n_estimators=20, learning_rate = learning_rate, max_features=2, max_depth = 2, random_state = 0)
    gb.fit(X_train_sub, y_train_sub)
    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb.score(X_train_sub, y_train_sub)))
    print("Accuracy score (validation): {0:.3f}".format(gb.score(X_validation_sub, y_validation_sub)))
    print()


# In[19]:


# Output confusion matrix and classification report of Gradient Boosting algorithm on validation set

gb = GradientBoostingClassifier(n_estimators=20, learning_rate = 0.5, max_features=2, max_depth = 2, random_state = 0)
gb.fit(X_train_sub, y_train_sub)
predictions = gb.predict(X_validation_sub)

print("Confusion Matrix:")
print(confusion_matrix(y_validation_sub, predictions))
print()
print("Classification Report")
print(classification_report(y_validation_sub, predictions))


# In[20]:


# ROC curve and Area-Under-Curve (AUC)

y_scores_gb = gb.decision_function(X_validation_sub)
fpr_gb, tpr_gb, _ = roc_curve(y_validation_sub, y_scores_gb)
roc_auc_gb = auc(fpr_gb, tpr_gb)

print("Area under ROC curve = {:0.2f}".format(roc_auc_gb))


# In[21]:




