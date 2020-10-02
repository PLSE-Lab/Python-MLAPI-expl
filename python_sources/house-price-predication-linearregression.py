#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
# import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


theredata=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
theredata.head()


# In[ ]:


theredata.columns


# In[ ]:


Dataset=theredata[['LotArea','YrSold','OverallQual','OverallCond','YearBuilt','TotalBsmtSF','1stFlrSF','GrLivArea','GarageArea','YrSold','SalePrice']]


# In[ ]:


Dataset.head()
Dataset.dropna(inplace=True)


# In[ ]:


from sklearn.model_selection import train_test_split
Dataset.head()
Dataset.dropna()
Dataset.shape


# In[ ]:


y=Dataset['SalePrice']
X=Dataset.drop('SalePrice',axis=1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


Dataset.describe()


# In[ ]:


import seaborn as sns
sns.distplot(Dataset['SalePrice'])


# In[ ]:


sns.heatmap(X.corr())


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lm = LinearRegression()


# In[ ]:


lm.fit(X_train,y_train)


# In[ ]:


print(lm.intercept_)


# In[ ]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[ ]:


predictions = lm.predict(X_test)


# In[ ]:


plt.scatter(y_test,predictions)


# In[ ]:


sns.distplot((y_test-predictions),bins=50);


# In[ ]:


test_data_set=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")


# In[ ]:


feature_data_kaggle=test_data_set[['LotArea','YrSold','OverallQual','OverallCond','YearBuilt','TotalBsmtSF','1stFlrSF','GrLivArea','GarageArea','YrSold']]
feature_data_kaggle=feature_data_kaggle.apply(lambda row: row.fillna(row.mean()), axis=1)


# In[ ]:


predictions=lm.predict(feature_data_kaggle)
data={'Id':test_data_set['Id'],'SalePrice':predictions}
new_prediction=pd.DataFrame(data,columns=['Id','SalePrice']).astype('int')


# In[ ]:


new_prediction.to_csv("../working/submission.csv",index=False)


# In[ ]:


new_prediction.shape


# In[ ]:




