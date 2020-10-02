#!/usr/bin/env python
# coding: utf-8

# # Ghouls, Goblins, and Ghosts
# ## Multiclass Classification Task
# 
# Falconi Nicasio
# 
# April 23 2019

# In[6]:


# load data
import pandas as pd
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")


# In[7]:


from sklearn.preprocessing import LabelEncoder

# encode color features
gle = LabelEncoder()
train_color_labels = gle.fit_transform(train_data['color'])
test_color_labels = gle.fit_transform(test_data['color'])
train_data['color_labels'] = train_color_labels
test_data['color_labels'] = test_color_labels


# In[8]:


# take target out of training set
Y = train_data['type']
train_data = train_data.drop(['type', 'id', 'color'], axis=1)
test_data = test_data.drop(['id', 'color'], axis=1)


# In[9]:


# standardize values
train_data = (train_data - train_data.mean()) / train_data.std()
test_data = (test_data - test_data.mean()) / test_data.std()


# In[10]:


train_data.describe()


# In[11]:


#check correlation for possible feature engineering
train_data.corr()


# In[12]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# initiate model
model_to_set = OneVsRestClassifier(LogisticRegression(penalty = 'l2', max_iter = 1000))

# create parameter options
parameters = {
    "estimator__C": [0.1, 0.5, 0.7, 1.0, 1.2, 2, 5, 10, 20, 100],
    "estimator__solver": ['newton-cg', 'lbfgs', 'sag'],
    "estimator__multi_class" : ['multinomial', 'ovr']
}

# Cross validation
model_tunning = GridSearchCV(model_to_set, param_grid=parameters,
                             cv = 2)

model_tunning.fit(train_data, Y)

print(model_tunning.best_score_)
print(model_tunning.best_params_)


# In[13]:


# create model
ovr = OneVsRestClassifier(LogisticRegression(penalty = "l2", C = 0.1, multi_class = 'ovr', solver = 'newton-cg'))
# train model
ovr_fitted = ovr.fit(train_data, Y)
# predict test_data
res = ovr_fitted.predict(test_data)


# In[14]:


# save predictions
sample_data = pd.read_csv("../input/sample_submission.csv")
sample_data['type'] = res
sample_data.to_csv('prediction.csv', index = False)

