#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


train = pd.read_csv("../input/train.tsv", sep = '\t')
train.columns


# In[ ]:


train


# In[ ]:


train.describe()


# In[ ]:


category_price = train[['category_name', 'price']]
mean_regressor = category_price.groupby(['category_name']).mean().to_dict()['price']

def predict(category):
    if category in mean_regressor :
        return mean_regressor[category]
    else :
        return 0.0

print(predict('Beauty/Bath & Body/Scrubs & Body Treatments'))
print(predict('Beauty/Fragrance/Men'))


# In[ ]:


test = pd.read_csv("../input/test.tsv", sep = '\t', header = 0)
print(test.columns)
test.head(5)


# In[ ]:


test['price'] = test['category_name'].map(predict)
test.describe()


# In[ ]:


test[['test_id', 'price']].to_csv("mean_submission.csv", index = False)


# In[ ]:




