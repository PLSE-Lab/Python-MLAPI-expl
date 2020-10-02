#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from mlxtend.frequent_patterns import apriori, association_rules 
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


pd.options.display.max_columns
pd.options.display.max_rows = 100


# In[ ]:


store_data = pd.read_csv('/kaggle/input/satislar.csv',sep=';',low_memory=False, header=None)


# In[ ]:


store_data.head()


# In[ ]:


store_data.columns = ['BranchId', 'PosId', 'InvoiceDate', 'InvoiceNo','StockCode','Line','Quantity','CategoryCode','CategoryName']
store_data.head()


# In[ ]:


for i in range(len(store_data.CategoryName)):
    if store_data.CategoryName[i][-1]==",":
        store_data.CategoryName[i]=store_data.CategoryName[i][:-1]
    if store_data.CategoryName[i][-1]==",":
        store_data.CategoryName[i]=store_data.CategoryName[i][:-1]


# In[ ]:


store_data.head()


# In[ ]:


store_data.dropna(subset= ["CategoryCode"],inplace= True)
store_data.shape


# In[ ]:


store_data["CategoryName"].nunique()


# In[ ]:


store_data["BranchId"].value_counts()


# In[ ]:


store_data.info()


# In[ ]:


store_data["InvoiceNo"].nunique()


# In[ ]:


store_data.describe().T


# In[ ]:


# Stripping extra spaces in the description 
store_data['CategoryName'] = store_data['CategoryName'].str.strip() 


# In[ ]:


# Dropping the rows without any invoice number 
store_data.dropna(subset =['InvoiceNo'], inplace = True) 


# In[ ]:


store_data


# In[ ]:


maskBrachId = store_data["BranchId"] == 4010
df= store_data[maskBrachId]


# In[ ]:


df


# In[ ]:


df.dropna(subset= ["CategoryName"],inplace= True)


# In[ ]:


df


# In[ ]:


pd.DataFrame(df["CategoryName"].value_counts(normalize=True)).head(10)


# In[ ]:


df["BranchId"].value_counts()


# In[ ]:


df["PosId"].value_counts()


# In[ ]:


df.info()


# In[ ]:


df['Quantity'] = [x.replace(',', '.') for x in df['Quantity']]


# In[ ]:


df["Quantity"] = df["Quantity"].astype("float")


# In[ ]:


df["Quantity"] = df["Quantity"].astype("int")


# In[ ]:


df.info()


# In[ ]:


branch_order = (df.groupby(['InvoiceNo', 'CategoryName'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNo')) 


# In[ ]:


branch_order.head(20)


# In[ ]:


# Encoding the datasets 
branch_encoded = branch_order.applymap(lambda x: 0 if x<=0 else 1) 
basket_branch = branch_encoded 


# In[ ]:


frq_items = apriori(basket_branch, min_support = 0.0005, use_colnames = True)


# In[ ]:


frq_items


# In[ ]:


from mlxtend.frequent_patterns import association_rules
ass_rules = association_rules(frq_items, metric ="confidence", min_threshold = 0.20) 
ass_rules.head() 


# In[ ]:


rules = association_rules(frq_items, metric="lift", min_threshold=1.2)
rules


# In[ ]:


rules2 = rules.sort_values(['lift','confidence'], ascending =[False, False]) 
rules2.head(100)


# In[ ]:




