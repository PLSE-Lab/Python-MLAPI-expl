#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


target=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")
target.head()


# In[ ]:


df=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
df.head()


# In[ ]:


df=df.select_dtypes(include="int64").head()


# In[ ]:


df.head()


# In[ ]:


df.isnull().sum()


# In[ ]:


from sklearn import linear_model
y=df["SalePrice"]
#xdf=df.drop(["SalePrice"],axis=1)
X=df[[ 'MSSubClass', 'LotArea', 'OverallQual', 'OverallCond',
       'YearBuilt', 'YearRemodAdd', 'BsmtFinSF1', 'BsmtUnfSF',
       'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
       'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
       'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars',
       'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 
       'ScreenPorch','MiscVal', 'MoSold', 'YrSold']]
model=linear_model.LinearRegression()
model.fit(X,y)


# In[ ]:


tstdata=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
tstdata.head()


# In[ ]:


xtest=tstdata[[ 'MSSubClass', 'LotArea', 'OverallQual', 'OverallCond',
       'YearBuilt', 'YearRemodAdd', 'BsmtFinSF1', 'BsmtUnfSF',
       'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
       'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
       'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars',
       'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 
       'ScreenPorch','MiscVal', 'MoSold', 'YrSold']]


# In[ ]:


xtest=xtest.fillna(method="bfill")
xtest.isnull().sum()


# In[ ]:


sncdata=pd.DataFrame(tstdata["Id"])
sncdata.head()
sncdata["SalePrice"]=model.predict(xtest)
sncdata.head(126)


# In[ ]:


from IPython.display import HTML
import pandas as pd
import numpy as np
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

# create a random sample dataframe
df = pd.DataFrame(sncdata)

# create a link to download the dataframe
create_download_link(df)

