#!/usr/bin/env python
# coding: utf-8

# ## This kernel is dedicated to a solution including data engineering, ordinal encoding, standardization, and random forest classifier use

# In[ ]:


import numpy as np
import pandas as pd

sample_submission = pd.read_csv('../input/sample_submission.csv')
test_f = pd.read_csv('../input/test_features.csv')
train_f = pd.read_csv('../input/train_features.csv')
train_l = pd.read_csv('../input/train_labels.csv')


# In[ ]:


# Grabbing the mode of a column after eliminating null values
train_f_s = train_f
train_f_s.head()
train_f_s = train_f_s[train_f_s['construction_year'] !=0]
train_f_s['construction_year'].mode()[0]


# In[ ]:


# Replacing null values with the mode of the column
train_f['construction_year'].replace({0:2010}, inplace=True)
test_f['construction_year'].replace({0:2010}, inplace=True)


# In[ ]:


import category_encoders as ce
import numpy as np


# Combining train and test sets
train_objs_num = len(train_f)
dataset = pd.concat(objs=[train_f, test_f], axis=0)

# Binning longitude and latitiude
step = 20
to_bin = lambda x: np.floor(x / step) * step
dataset["latbin"] = dataset.latitude.map(to_bin)
dataset["lonbin"] = dataset.longitude.map(to_bin)

# Dropping the highest cardinality features
dataset.drop(['longitude', 'latitude', 'wpt_name'],axis=1, inplace=True)

# Ordinal encoding the combined dataset
ordinal = ce.OrdinalEncoder()
ordinal.fit(dataset)
dataset = ordinal.transform(dataset)

# The train and test sets are separated again
train_f = dataset[:train_objs_num]
test_f = dataset[train_objs_num:]


# In[ ]:


from sklearn.model_selection import train_test_split
# Making the train and test datasets

X = train_f
y = train_l['status_group']

X_train, X_test, y_train, y_test = train_test_split(X, y)


# ## The model below takes a while to run due to the 'n_estimators' paramater being so high. It is a large number because it improves the performance of the model.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# Model being used for predictions

pipeline = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(n_estimators=3000,criterion = 'entropy')
)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)


# ## The last step is to fit the model to the test data:

# In[ ]:


y_pred = pipeline.predict(test_f)


# In[ ]:




