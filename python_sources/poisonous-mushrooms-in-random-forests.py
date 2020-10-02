#!/usr/bin/env python
# coding: utf-8

# Just testing random forests on mushrooms dataset.  When I used 70% of the data for 
# training and 30% for testing the random forest algorithm performed too well. Zero false positives 
# and zero true negatives. So I decided to use smaller training sets just to see how well it performs.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


# In[ ]:


data = pd.read_csv('../input/mushrooms.csv')


# In[ ]:


features = data.columns[1:-1]


# In[ ]:



cat_cols = data.columns[:-1]
for col in cat_cols:
    data[col] = data[col].astype('category')
cat_cols = data.select_dtypes(['category']).columns
data[cat_cols] = data[cat_cols].apply(lambda x : x.cat.codes)

xs01 = []
xs10 = []
for p in np.arange(0.01, 1.0, 0.01):
    data['is_train'] = np.random.uniform(0, 1, len(data)) <= p # percentage used for testing 
    train, test = data[data['is_train']== True], data[data['is_train'] == False]
    
    rfclf = RandomForestClassifier()
    rfclf.fit(train[features], train['class'])
    preds = rfclf.predict(test[features])
    ct = pd.crosstab(test['class'], preds, rownames=['actual'], colnames=['pred'])
    # print(ct)
    nTest = len(data[data['is_train'] == False])
    xs01.append(float(ct[0][1])/nTest)
    xs10.append(float(ct[1][0])/nTest)


# In[ ]:


plt.plot( np.arange(0.01,1.0,0.01), xs01)
plt.plot( np.arange(0.01,1.0,0.01), xs10)


# As the fraction of trainng samples increase the false positive and false negative rates drop rapidly. 
# Why is this dataset so easy? 

# TODO(pratik): gather and average xs01/10s plot 

# In[ ]:




