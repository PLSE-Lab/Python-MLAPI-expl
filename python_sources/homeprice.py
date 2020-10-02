#!/usr/bin/env python
# coding: utf-8

# In[246]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# In[247]:


import os
print(os.listdir("../input"))


# In[248]:


print(os.listdir())


# In[249]:


test = pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")


# In[250]:


len(train.columns)


# In[251]:


len(test.columns)


# In[252]:


train.head()


# In[253]:


test.head()


# In[254]:


train.describe()


# In[255]:


missing = train.isnull().sum()
missing = missing[missing > 0]
missing.plot.bar()


# In[256]:


missing.index


# In[257]:


len(train.columns)


# In[258]:


dropArr = ['LotFrontage', 'Alley', 'MasVnrType', 'MasVnrArea', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Electrical', 'FireplaceQu', 'GarageType', 'GarageYrBlt',
       'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence',
       'MiscFeature']
train = train.drop(dropArr, axis=1)
test = test.drop(dropArr, axis=1)
test = pd.get_dummies(test)
train = pd.get_dummies(train)


# In[259]:


len(train.columns)
test = test.fillna(method='ffill')
testID = test["Id"]
train = train.drop(["Id"], axis=1)
test = test.drop(["Id"], axis=1)


# In[260]:


trainSalePrice = train['SalePrice']
train.columns


# In[261]:


def plot_corr(df,size=62):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);


# In[262]:


#plot_corr(train)


# In[263]:


corr = train.corr()
corr.columns


# In[264]:


corr = corr.sort_values(by=['SalePrice'] , ascending=False).iloc[:,-1].abs()


# In[265]:


corr


# In[266]:


corr = corr.nlargest(30)
corr


# In[267]:


selectedColumns = corr.index.values


# In[268]:


selectedColumns = np.delete(selectedColumns,4)
selectedColumns = np.delete(selectedColumns,0)
selectedColumns = np.delete(selectedColumns,2)
selectedColumns = np.delete(selectedColumns,3)
selectedColumns = np.delete(selectedColumns,1)


# In[269]:


corr


# In[270]:


train = train[selectedColumns]
test = test[selectedColumns]
len(test)


# In[271]:


from sklearn import linear_model
reg = linear_model.LinearRegression()


# In[272]:


reg.fit(train,trainSalePrice)


# In[273]:


reg.coef_


# In[274]:


pred = reg.predict(test)


# In[275]:


submit = pd.DataFrame()
submit["Id"] = testID
submit["SalePrice"] = pred


# In[276]:


submit.set_index('Id')
submit.head()


# In[277]:


submit.info()


# In[278]:


len(submit)


# In[279]:


# import the modules we'll need
from IPython.display import HTML
import base64
# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)


# In[280]:


create_download_link(submit)

