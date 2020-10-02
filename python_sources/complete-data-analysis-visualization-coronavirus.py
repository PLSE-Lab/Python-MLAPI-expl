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


import pandas as pd
df = pd.read_csv("../input/2019-coronavirus-dataset-01212020-01262020/2019_nC0v_20200121_20200126_cleaned.csv")


# In[ ]:


df.head()


# In[ ]:


df.pop('Unnamed: 0')


# In[ ]:


df.head(2)


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df.dtypes


# In[ ]:


df.shape


# In[ ]:


df.isnull().sum()


# In[ ]:


df.info()


# In[ ]:


df.columns


# In[ ]:


df.loc[df.Deaths == max(df.Deaths )]


# In[ ]:


df.describe(include=[np.number])


# In[ ]:


df['Province/State'].unique()


# In[ ]:


df.Country.unique()


# In[ ]:





# **  find countries with confirmed cases**

# In[ ]:


confirmed_con = df.loc[df.Confirmed >=1]
confirmed_con.head()


# In[ ]:


country_details  = pd.pivot_table(df,index=["Country"] ,aggfunc=np.sum).sort_values(by='Confirmed', ascending=False)


# In[ ]:


country_details 


# In[ ]:


# country_details.sort_values(by='Confirmed',ascending=False).head()


# In[ ]:


# country_details.sort_values(by='Deaths',ascending=False)


# In[ ]:


list(zip(country_details.index , country_details.Confirmed))


# In[ ]:


df.loc[df.Country == 'Australia'].agg('sum')[3:]


# In[ ]:


df.loc[df.Country == 'Nepal'].agg('sum')[3:]


# In[ ]:


df['Province/State'].unique()


# > Note we have same values of Province/States for some data of different countries

# In[ ]:


pd.pivot_table(df,index=["Country","Province/State"] ,aggfunc=np.sum)


# In[ ]:





# **In Avove pivot table we have found that value of Province/State 	= 0 **
# 
# ***
# Now distinguish them :
# ***

# In[ ]:


ind =[]
for index in range(df.shape[0]):
    
    if df.iloc[index][0]=='0':
        #print('Column Number : ', index)
        ind.append(index)


# In[ ]:


df.head()


# In[ ]:


df.loc[ind,'Province/State'] = df.loc[ind].Country


# In[ ]:


df[df.Country=='Nepal']


# In[ ]:


pd.pivot_table(df,index=["Country","Province/State"] ,aggfunc=np.sum)


# In[ ]:





# In[ ]:


#  country_details  - country


# In[ ]:


province_state_country = pd.pivot_table(df,index=["Province/State"] ,aggfunc=np.sum).sort_values(by='Confirmed', ascending=False)


# In[ ]:


province_state_country[:10]


# In[ ]:


# province_state_country[:5].plot(kind='pie', subplots=True, figsize=(100, 100))


# In[ ]:


province_state_country[:10].plot(kind='bar' ,figsize=(10, 4), width=1,zorder=12)


# In[ ]:


province_state_country[1:10].plot(kind='bar' ,figsize=(10, 4), width=2)


# In[ ]:


country_details[:5]


# In[ ]:


country_details[0:5].plot(kind='bar' ,figsize=(10, 4), width=1,zorder=12,rot=2)


# In[ ]:


country_details[1:6].plot(kind='bar' ,figsize=(10, 4), width=1,zorder=12,rot=1)


# In[ ]:





# **Trend Analysis**

# In[ ]:


cpd = pd.pivot_table(df,index=["Country","Province/State",'Date last updated'] )


# In[ ]:


cpd


# In[ ]:


cpd.loc[('Australia', ), :]


# In[ ]:


aus = cpd.loc[('Australia', ), :]


# In[ ]:





# In[ ]:


aus.loc['Australia'].index


# In[ ]:


aus.loc['Australia'].sort_index()


# In[ ]:


aus.loc['Australia'].sort_index().plot.line()


# In[ ]:


# for china


# In[ ]:


china = df[df['Country'] == 'Mainland China']
china_d = pd.DataFrame(china.groupby(['Province/State'])['Confirmed','Suspected','Recovered','Deaths'].agg('sum')).reset_index()
china_d.head(35)


# In[ ]:


import seaborn as sns
china_d.sort_values(by=['Confirmed'], inplace=True,ascending=False)
plt.figure(figsize=(30,15))
plt.title("Patients Confirmed Infected by Corona Virus by States")
sns.barplot(x=china_d['Province/State'],y=china_d['Confirmed'],orient='v')
plt.ylabel("Confirmed Patients")


# In[ ]:




