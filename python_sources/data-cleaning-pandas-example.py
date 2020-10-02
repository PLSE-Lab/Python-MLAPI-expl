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


data = pd.read_csv("../input/pokemon.csv")
data.head() #first 5 data.head()=data.head(5)


# In[ ]:


data.tail() #last 5 


# In[ ]:


data.columns


# In[ ]:


data.shape #800 rows*12 columns


# 386 pokemons don't have type 2 info. in this data.

# In[ ]:


data.info() 


# **EDA:Explotary Data Analysis**

# In[ ]:


print(data["Type 1"].value_counts(dropna = False)) #counting pokemon types with value_counts method


# Outlier are very high or low value from data. (O =Q1-1.5(Q3-Q1) )
# Q3 = 75%
# Q1 = 25%
# Median = 50%
# 

# In[ ]:


data.describe()       #only numeric values


# In[ ]:


data.boxplot(column ='Attack' ,by = 'Legendary' ) #boxplot shows outlier, median,Q3,Q1


# In[ ]:


data_new = data.head(5)
data_new


# In[ ]:


melted =pd.melt(frame=data_new ,id_vars ="Name",value_vars=["Attack","Defense"])  #frame=what data we want melt
melted


# In[ ]:


melted.pivot(index="Name",columns="variable",values="value")  #pivoting data= reverse of melting


# In[ ]:


data1 = data.head()
data2 = data.tail()
conc_data_row = pd.concat([data1,data2],axis = 0 ,ignore_index=True) #concatenating data=add data to data
conc_data_row


# In[ ]:


data1 = data['Attack'].head()
data2= data['Defense'].head()
conc_data_col = pd.concat([data1,data2],axis =1) 
conc_data_col


# 5 Basic data types :  object(str), boolean, int, float, categorical.

# In[ ]:


data.dtypes


# In[ ]:


data["Speed"] = data["Speed"].astype("float")
data.dtypes


# In[ ]:


data.info()


# In[ ]:


data["Type 2"].value_counts(dropna=False) #dropna false= show me nan


# In[ ]:


data1=data
data1["Type 2"].dropna(inplace=True)


# In[ ]:


assert data["Type 2"].notnull().all()  


# In[ ]:


country =["Spain","France"] #list1
population =["11","12"]     #list2
list_label=["country","population"]
list_col= [country,population]
zipped = list(zip(list_label,list_col)) #list1 add to list2 and create zipped
data_dict=dict(zipped) #convert to dictionary
df =pd.DataFrame(data_dict) 
df


# In[ ]:


df["capital"] =["madrid","paris"]  #we added new column
df


# In[ ]:


df["income"] = 0
df


# In[ ]:


data1 =data.loc[:,["Attack","Defense","Speed"] ]
data1.plot()


# In[ ]:


data1.plot(subplots =True)


# In[ ]:


data1.plot(kind= "scatter",x="Attack", y="Defense")


# In[ ]:


data1.plot(kind="hist", y="Defense", bins = 50, range=(0,250), normed=True)


# In[ ]:


data.describe() 


# In[ ]:


time_list= ["1992-03-08","1992-04-12"]
print(type(time_list[1]))
datetime_object = pd.to_datetime(time_list)
print(type(datetime_object))


# 

# In[ ]:


date2 =data.head()
date_list =["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16" ]
datetime_object = pd.to_datetime(date_list)
date2["date"]=datetime_object                                      #time series data
date2=date2.set_index("date")
date2


# In[ ]:


date2.loc[ "1992-02-10": "1993-03-15"]


# In[ ]:


#??
data2.resample["M"].mean()
data2.resample("M").first().interpolate("linear")


# In[ ]:


data = pd.read_csv('../input/pokemon.csv')
data = data.set_index("#") #index start with 1
data.head()


# In[ ]:


data["HP"][1]     


# In[ ]:


data.HP[1]


# In[ ]:


data.loc[1,["HP"]]


# In[ ]:


data.head()[["Attack","Defense"]]


# In[ ]:


print(type(data["HP"]))   #series
print(type(data[["HP"]])) #data frame


# In[ ]:


data.loc[1:10,"HP":"Defense"]  #slicing


# In[ ]:


data.loc[10:1:-1, "HP":"Defense"]   #reverse slicing


# In[ ]:


data.loc[1:10,"Speed":]   #from speed to end


# In[ ]:


boolean =data.HP>200
data[boolean]


# In[ ]:


first_filter = data.HP >150
second_filter = data.Speed > 50
data[first_filter & second_filter]


# In[ ]:


data.HP[data.Speed>150]


# In[ ]:


def div(n):
    return n/2
data.head().HP.apply(div)


# In[ ]:


data["total_power"] = data.Attack +data.Defense
data.head()


# In[ ]:


datax = data.set_index["Type 1","Type 2"]
datax.head()


# In[ ]:


dict ={ "treatment":["A","A","B","B"] ,"gender":["F","M","F","M"] ,"response":["10","45","5","9"],"age" :["15","4","72","25"]}
df =pd.DataFrame(dict)
df


# In[ ]:


df.pivot(index="treatment",columns="gender",values="response")     #pivoting data


# In[ ]:


df1 =df.set_index(["treatment","gender"])        #level0= treatment  ,level1= gender
df1


# In[ ]:


df1.unstack(level=0)


# In[ ]:


df1.unstack(level=1)


# In[ ]:


df2 = df1.swaplevel(0,1)
df2                        # change inner and outer level index position


# In[ ]:


df


# In[ ]:


pd.melt(df, id_vars="treatment",value_vars=["age","response"])


# In[ ]:


df.groupby("treatment").min()


# In[ ]:


df.groupby("treatment")[["age","response"]].min()


# In[ ]:




