#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


test = pd.read_excel("../input/Data_Test.xlsx")
train = pd.read_excel("../input/Data_Train.xlsx")
submission = pd.read_excel("../input/Sample_submission.xlsx")


# In[ ]:


submission.head()


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.columns.difference(test.columns)


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train["Source"] = "train"
test["Source"] = "test"


# In[ ]:


df = pd.concat([train,test])


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.duplicated().sum()


# In[ ]:


df.isna().sum()


# In[ ]:


comma_data = df["TITLE"].str.split(",", expand = True)


# In[ ]:


df["TITLE1"] = comma_data[0]
df["TITLE2"] = comma_data[1]


# In[ ]:


df["CUISINES"].str.split(",", expand = True)


# In[ ]:


comma_cuisines = df["CUISINES"].str.split(",", expand = True)
df["CUISINES1"] = comma_cuisines[0]
df["CUISINES2"] = comma_cuisines[1]
df["CUISINES3"] = comma_cuisines[2]
df["CUISINES4"] = comma_cuisines[3]
df["CUISINES5"] = comma_cuisines[4]
df["CUISINES6"] = comma_cuisines[5]
df["CUISINES7"] = comma_cuisines[6]
df["CUISINES8"] = comma_cuisines[7]
df.drop(columns =["TITLE"], inplace = True) 
df.drop(columns =["CUISINES"], inplace = True)


# In[ ]:


df


# In[ ]:


reshaped = (df.set_index(df.columns.drop('CITY',1).tolist())
   .CITY.str.split(',', expand=True)
   .stack()
   .reset_index()
   .rename(columns={0:'CITY'})
   .loc[:, df.columns]
)


# In[ ]:


reshaped


# In[ ]:


reshaped1 = (reshaped.set_index(reshaped.columns.drop('LOCALITY',1).tolist())
   .LOCALITY.str.split(',', expand=True)
   .stack()
   .reset_index()
   .rename(columns={0:'LOCALITY'})
   .loc[:, reshaped.columns]
)


# In[ ]:


reshaped1


# In[ ]:


reshaped2 = (df.set_index(df.columns.drop('TIME',1).tolist())
   .TIME.str.split('(', expand=True)
   .stack()
   .reset_index()
   .rename(columns={0:'TIME'})
   .loc[:, df.columns]
)


# In[ ]:


reshaped2


# In[ ]:


reshaped3 = (reshaped2.set_index(reshaped2.columns.drop('TIME',1).tolist())
   .TIME.str.split(')', expand=True)
   .stack()
   .reset_index()
   .rename(columns={0:'TIME'})
   .loc[:, reshaped2.columns]
)


# In[ ]:


reshaped3


# In[ ]:


df1 = (reshaped3.set_index(reshaped3.columns.drop('TIME',1).tolist())
   .TIME.str.split(',', expand=True)
   .stack()
   .reset_index()
   .rename(columns={0:'TIME'})
   .loc[:, reshaped3.columns]
)


# In[ ]:


df1


# In[ ]:


df2 = (df1.set_index(reshaped3.columns.drop('VOTES',1).tolist())
   .VOTES.str.split('votes', expand=True)
   .stack()
   .reset_index()
   .rename(columns={0:'VOTES'})
   .loc[:, df1.columns]
)


# In[ ]:


df2


# In[ ]:


df2.isna().sum()


# In[ ]:


df2.duplicated().sum()


# In[ ]:


df2 = df2.drop_duplicates()


# In[ ]:


df2.duplicated().sum()


# In[ ]:


df2["LOCALITY"]= df2['LOCALITY'].fillna("Not_specified")
print("Not_specified")


# In[ ]:


df2["CITY"]= df2['CITY'].fillna("Not_specified")
print("Not_specified")


# In[ ]:


df2["CUISINES2"]= df2['CUISINES2'].fillna("Not_specified")
df2["CUISINES3"]= df2['CUISINES3'].fillna("Not_specified")
df2["CUISINES4"]= df2['CUISINES4'].fillna("Not_specified")
df2["CUISINES5"]= df2['CUISINES5'].fillna("Not_specified")
df2["CUISINES6"]= df2['CUISINES6'].fillna("Not_specified")
df2["CUISINES7"]= df2['CUISINES7'].fillna("Not_specified")
df2["CUISINES8"]= df2['CUISINES8'].fillna("Not_specified")
print("Not_specified")


# In[ ]:


df2.isna().sum()


# In[ ]:


df2_column_numeric = df2.select_dtypes(include=np.number).columns


# In[ ]:


df2_column_category = df2.select_dtypes(exclude=np.number).columns


# In[ ]:


df2_column_category


# In[ ]:


df2.select_dtypes(exclude=np.number)


# In[ ]:


#One hot encoding
encoded_cat_col = pd.get_dummies(df2_column_category)


# In[ ]:


encoded_cat_col


# In[ ]:


df_final = pd.concat([df2[df2_column_numeric],encoded_cat_col], axis = 1)
df_final
df_final.fillna(df2.mode(),inplace=True)
df_final


# In[ ]:


train_final = df_final[df_final.Source=="train"]
test_final = df_final[df_final.Source=="test"]


# In[ ]:


train_final.drop(columns="Source",inplace=True)


# In[ ]:


train_final.drop(columns="RESTAURANT_ID",inplace=True)


# In[ ]:


df2_column_numeric


# In[ ]:


test_final.isna().sum()


# In[ ]:


train_X = train_final.drop(columns=["COST"])


# In[ ]:


train_Y = train_final["COST"]


# In[ ]:


test_X = test_final.drop(columns=["VOTES"])


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[ ]:


model = LinearRegression()


# In[ ]:


model.fit(train_X, train_Y)
        
#Predict training set:
dtrain_predictions = model.predict(train_X)


# In[ ]:


dtrain_predictions


# In[ ]:


#Print model report:
print("\nModel Report")
print("RMSE : %.4g" % np.sqrt(mean_squared_error(train_Y.values, dtrain_predictions)))
    
#Predict on testing data:
test_final["res_linear"] =  model.predict(test_X)

