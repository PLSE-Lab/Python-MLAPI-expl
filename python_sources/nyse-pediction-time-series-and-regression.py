#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Let's have a glimpse of all the files we have.
# 

# In[ ]:


df_fun=pd.read_csv("../input/fundamentals.csv")
df_fun.head()


# In[ ]:


df_price=pd.read_csv("../input/prices.csv")
df_price.head()


# In[ ]:


df_price.symbol.unique()


# In[ ]:


df_sec=pd.read_csv("../input/securities.csv")
df_sec.head()


# In[ ]:


df_psa=pd.read_csv("../input/prices-split-adjusted.csv",parse_dates=["date"])
df_psa.head()


# In[ ]:


type(df_psa.date[0])  #date time index check


# In[ ]:


df_psa.set_index("date",inplace=True)


# Grouping based on companies

# In[ ]:


df_psa_grp=df_psa.groupby(df_psa.symbol)


# 3 different companies taken for analysis

# In[ ]:


df_wltw=df_psa_grp.get_group("WLTW") 
df_acn=df_psa_grp.get_group("ACN") #accenture
df_abt=df_psa_grp.get_group("ABT") #abbot lab


# In[ ]:


df_wltw.head()


# In[ ]:


df_wltw.close.resample("M").mean()# monthly resampled data for closing points
df_acn.close.resample("M").mean()
df_abt.close.resample("M").mean()


# In[ ]:


#monthly closing points of 3 different securities.
df_wltw.close.resample("M").mean().plot(label="wltw")
df_acn.close.resample("M").mean().plot(label="accenture")
df_abt.close.resample("M").mean().plot(label="abbot lab")
plt.legend()


# #Lets see how accenture did in year 2010

# In[ ]:


df_acn_2010=df_acn["2010-01-01":"2010-12-31"]#accenture data of year 2010
df_acn_2010.tail()


# In[ ]:


#accenture quartely results shown in plot
df_acn_2010.close.resample("Q").mean().plot(kind="bar",label="accenture",color="green")


# #lets focus on accenture data for predictions on closing prices

# In[ ]:


df_acn.head()
df1=df_acn[["open","low","high","volume"]]

df1.corr()


# In[ ]:


X=df1
y=df_acn.close


# DimensionalityReduction: choosing the best features
# As we can see that the features are highly correlated, we can use SelectKBest to find best high scoring features
# 

# In[ ]:


from sklearn.feature_selection import SelectKBest
sk=SelectKBest(k=4)
X=sk.fit_transform(X,y)


# In[ ]:


sk.pvalues_


# In[ ]:


sk.scores_


# High scoring features are "low". Lets use them as features for our model.[](http://)

# In[ ]:


#X=df_acn["high"].values.reshape(-1,1) #since we are using single variable
X=df1[["low"]]


# In[ ]:


#label
y=df_acn["close"]


# In[ ]:


#train test and split data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=1)


# Feature scaling

# In[ ]:


#lets try to scale the features
from sklearn.preprocessing import MinMaxScaler
mm=MinMaxScaler()
X_train_mm=mm.fit_transform(X_train)


# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
gbr=GradientBoostingRegressor()


# In[ ]:


#training
#my_fit=gbr.fit(X_train_mm,y_train)
#my_fit

my_fit=gbr.fit(X_train,y_train)
my_fit


# In[ ]:


y_pred=gbr.predict(X_test)
y_pred


# In[ ]:


gbr.score(X_test,y_test)


# In[ ]:




