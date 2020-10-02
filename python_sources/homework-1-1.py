#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import  matplotlib.pyplot as plt
import seaborn as sns #Visualition tool
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_csv('../input/master.csv')


# In[ ]:


data.info()


# In[ ]:


data.corr


# In[ ]:


#correlation map
f,ax=plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(),annot=True,linewidths=.5,fmt='.2f',ax=ax)


# In[ ]:


data.head


# In[ ]:


data.columns


# In[ ]:


data.year.plot(kind='line',color='g',label='year',linewidth=1,alpha=.9,grid=True,linestyle=':')
data.suicides_no.plot(color='r',label='suicides_no',linewidth=1,alpha=.5,grid=True,linestyle='-.')
plt.legend(loc='upper right')
plt.xlabel('suicides_no')
plt.ylabel('year')
plt.title('line plot')
plt.show()


# In[ ]:


data.plot(kind='scatter',y='suicides_no',x='year',color='r',alpha=.5)


# In[ ]:


data.suicides_no.plot(kind='hist',bins=10,figsize=(18,18))
plt.show()


# In[ ]:


data.tail()


# In[ ]:


data=pd.read_csv('../input/master.csv')
data.head(8)


# In[ ]:


data.columns


# In[ ]:


data.shape


# In[ ]:


data.info


# In[ ]:


data.columns


# In[ ]:


v=data['age'].value_counts(dropna=False)
print(v)


# In[ ]:


print(data['generation'].value_counts(dropna=False))


# In[ ]:


data.describe()


# In[ ]:


data.boxplot(column='age',by='suicides_no')


# In[ ]:


data_1=data.head()
data_1


# In[ ]:


melted=pd.melt(frame=data_1,id_vars='sex',value_vars=['suicides_no','gdp_per_capita ($)'])
melted


# In[ ]:


data.columns


# In[ ]:


data1=data['sex'].head()
data2=data['suicides_no'].head()
con_con=pd.concat([data1,data2],axis=1)
con_con


# In[ ]:


data1=data.head()
data2=data.tail()
concat1=pd.concat([data1,data2],axis=0,ignore_index=True)
concat1


# In[ ]:


data.dtypes


# In[ ]:


data['population']=data['population'].astype('float32')
data['country']=data['country'].astype('category')
data.dtypes


# In[ ]:


data['HDI for year'].value_counts(dropna='False')


# In[ ]:


assert data['HDI for year'].notnull().all()


# In[ ]:


assert data.age.dtypes==np.int


# ## Building data frames from scracth

# In[ ]:


country=['turkey','france']
population=['10','5']
list_label=['country','population']
list_col=[country,population]
zipped=zip(list_label,list_col)
zipped=list(zipped)
zipped
data_dict=dict(zipped)
data_dict
df=pd.DataFrame(data_dict)
df


# In[ ]:


df['capital']=['ankara','paris']
df


# In[ ]:


data.columns


# In[ ]:


data1=data.loc[:,['suicides/100k pop','gdp_per_capita ($)','population']]
data1.plot()


# In[ ]:


data1.plot(subplots='True')
plt.show()


# In[ ]:


data1.plot(kind='scatter',y='suicides/100k pop',x='gdp_per_capita ($)')
plt.show()


# In[ ]:


data1.plot(kind='hist',y='suicides/100k pop',bins=50,range=(0,250),normed=True)
plt.show()


# ## Indexing pandas time series

# In[ ]:


time_list=["1994-1-10","1994-2-11","1994-3-12"]
print(type(time_list))
datetime_object=pd.to_datetime(time_list)
print(type(datetime_object))


# In[ ]:


data2=data.head()
time_list=["1994-1-10","1994-1-9","1994-3-10","1994-4-10","1994-5-10"]
time_obj=pd.to_datetime(time_list)
data2["date"]=time_obj
data2=data2.set_index("date")
data2


# In[ ]:


data2.resample("M").mean()


# In[ ]:


data2.resample("M").first().interpolate("linear")


# ## Indexing data frames

# In[ ]:


data=pd.read_csv('../input/master.csv')
#data=data.set_index('country')
data.head()


# In[ ]:


data[["sex","age","population"]]


# ## Slicing data frame 

# In[ ]:


print(type(data["age"]))
print(type(data[["age"]]))


# In[ ]:


data.loc[1:10,"year":"suicides_no"]


# In[ ]:


boolean=data.suicides_no>100
data[boolean]


# In[ ]:


first_filter=data.population>100000
sec_filter=data.year>2000
third_filter=data.suicides_no<10
data[first_filter & sec_filter & third_filter]


# In[ ]:


print(data.index.name)


# In[ ]:


data.index.name="index name"
data.head()


# In[ ]:


data3=data.copy()
data3.index=range(100,27920,1)
data3.head()


# In[ ]:


data=pd.read_csv('../input/master.csv')
data.head()


# In[ ]:


data1=data.set_index(["country","sex"])
data1.head(1000)


# In[ ]:


dict1={"treatment":["A","A","B","B"],"response":[1,2,9,10],"gender":["M","F","M","M"],"age":[10,20,15,30]}
df=pd.DataFrame(dict1)
df


# In[ ]:


# pivoting
df.pivot(index="treatment",columns = "gender",values="age")


# In[ ]:


df1=df.set_index(["treatment","gender"])
df1


# In[ ]:


df1.unstack(level=0)


# In[ ]:


df1.unstack(level=1)


# In[ ]:





# ## Melting data frames

# In[ ]:


df


# In[ ]:


pd.melt(df,id_vars="treatment",value_vars=["age","response"])


# In[ ]:


pd.melt(data.head(10),id_vars="country",value_vars=["age","sex"])


# In[ ]:


df.groupby("response").mean()


# In[ ]:


df.groupby("treatment").age.max()


# In[ ]:


df.groupby("treatment")[["age","response"]].min() 


# In[ ]:


df.groupby("gender")[["age","response"]].min() 


# In[ ]:


data.head()


# In[ ]:


data.groupby("year")[["population","suicides_no","suicides/100k pop"]].mean()


# In[ ]:




