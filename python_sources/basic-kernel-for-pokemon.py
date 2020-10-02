#!/usr/bin/env python
# coding: utf-8

# **Pokemon Basic Kernel**
# ![Pokemon_Image](https://zaytung.com/fotos/pokemon_burclar_zaytung_blog.jpg)
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # visualization tool 1
import seaborn as sns  # visualization tool 2

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.aa


# In[ ]:


df = pd.read_csv("/kaggle/input/pokemon-challenge/pokemon.csv")


# Reading file from kaggle location.

# In[ ]:


df.info()
df.shape


# What is inside and their types. (And shapes)

# In[ ]:


df.describe()


# Describe method has statistical information about the dataFrame. (Only numeric datas)

# In[ ]:


df.head(2)


# In[ ]:


df.tail(2)


# In[ ]:


df.dtypes


# We're getting basic properties basically.

# In[ ]:


df.Speed.plot(kind="line",label="Speed",grid=True,linestyle = ':',alpha=0.6,color="green")
df.Defense.plot(kind="line",label="Defense",grid=True,alpha=0.4,linestyle="-.",color="red")
plt.title("Speed vs Defence")
plt.legend()
plt.xlabel=('Speed')
plt.ylabel=('Defense')
plt.show()


# **Line** Plot:  Speed and Defense of Dataframe. (Defense is more distributed than Speed****.)

# In[ ]:


df.plot(kind="scatter",x="Speed",y="Defense",color="brown",alpha=0.5)
plt.title('Attack Defense Scatter Plot')
plt.show()


# Diffrent plot style (**scatter**) for Speed vs Defense. 

# In[ ]:


df.plot(kind="line",x="Generation",y="Attack")


# In[ ]:


df.Speed.plot(kind="hist",bins=50,figsize=(12,12))
plt.title("Speed Histograms")
plt.show()


# In[ ]:


df.Speed.plot(kind="hist",bins=50,figsize=(12,12))
plt.title("Speed Histograms")
plt.clf()
plt.show()


# In[ ]:


series = df["Defense"]
print(type(series))
dataFrame = df[["Defense"]]
print(type(dataFrame))


# In[ ]:


power_filter = df["Sp. Atk"]>150
print(df[power_filter])


# Which pokemons have more power than 150.

# In[ ]:


#Double conditions
df[np.logical_and(df["Defense"]>200,df["Attack"]>100)]


# In[ ]:


df[(df["Defense"]>150) & (df["Attack"]>50) & (df["Sp. Atk"]>50)]


# Different double condition query.

# Adding a new feature and properties on the .cvs file.

# In[ ]:


mean = np.sum(df.Speed)/len(df.Speed)

df["Speed_Status"] = ["High" if each > mean else "Normal" for each in df.Speed]
print("The mean speed value is: ",mean)
df.loc[20:25,["Speed_Status","Speed"]]
# print(repr(df.Speed_Status))


# Data Count for getting dataFrame infos.

# In[ ]:


print(df["Type 1"].value_counts(dropna=False))
print(df["Type 2"].value_counts(dropna=False))


# Visualize basic datas with box plots.

# In[ ]:


df.boxplot(column="Sp. Atk",by="Speed_Status")
plt.show()


# Melting

# In[ ]:


will_be_melt = df.head(30)
melted_data = pd.melt(frame=will_be_melt,id_vars = ["Name","Speed_Status","Speed"],value_vars = ["Attack","Defense"])
melted_data


# Pivoting

# In[ ]:


melted_data.pivot (index =  "Name",columns="variable",values="value")


# Concatenating

# In[ ]:


df1 = df.head(25)
df2 = df.tail(25)

cancat_data = pd.concat([df1,df2],axis=0,ignore_index=True)
cancat_data

Dropna, notnull and fillna functions.
# In[ ]:


df["Type 2"].dropna(inplace = True)


# In[ ]:


assert df["Type 2"].notnull().all() # we can check it is true / false. e.g. assert df[1] == "Name"


# In[ ]:


df["Type 2"].fillna("BlankArea",inplace = True)


# Creating a new DataFrame.

# In[ ]:


properties1 = df["Name"].head(10)
properties2 = df["Attack"].head(10)
labels = ["Name","Attack"]
zipped_data = list(zip(labels,[properties1,properties2]))
data_dict = dict(zipped_data)
data_frame = pd.DataFrame(data_dict)
data_frame


# Adding a new column.

# In[ ]:


data_frame["Defense"] = df["Defense"].head(10)
data_frame["Owner"]=0
data_frame


# Ploting the new data

# In[ ]:


data_frame.loc[:,["Attack","Defense","Owner"]]
data_frame.plot()
plt.show()


# In[ ]:


data_frame.plot(subplots=True)
plt.show()


# In[ ]:


data_frame.plot(kind="scatter",x="Attack",y="Defense")
plt.show()


# In[ ]:


data_frame.plot(kind="hist",y="Attack",bins=50,range = (0,150), normed=True)
plt.show()


# In[ ]:


data_frame.plot(kind="hist",y="Attack",bins=50,range = (0,150), normed=True,cumulative=True)
plt.show()


# #### Time Series

# In[ ]:


the_times = ["1998-08-25","2017-06-25"]
print(type(the_times))
the_timeSeries_object = pd.to_datetime(the_times)
print(type(the_timeSeries_object))


# #### Adding the datetime index on the data_frame // Time Series Data

# In[ ]:


the_times = ["1998-8-25","1998-8-26","1998-8-27","1998-8-28","1998-8-29","1998-8-30","1998-8-31","1998-8-24","1998-8-23","1998-8-22",]
the_times = pd.to_datetime(the_times)
data_frame["DATE"] = the_times
data_frame = data_frame.set_index("DATE")
data_frame


# Printing elements with datetime index.

# In[ ]:


print(data_frame.loc["1998-08-25":"1998-08-30"])


# Resampling
# 

# In[ ]:


data_frame.resample("A").mean() 


# Linear Interpolate

# In[ ]:


data_frame.resample("A").first().interpolate("linear")


# In[ ]:


Basic selections


# In[ ]:


copy_df = df.copy()
copy_df.HP[0]


# In[ ]:


copy_df["HP"][0]


# In[ ]:


copy_df.loc[0,"HP"]


# In[ ]:


copy_df[["Speed_Status","Attack"]]


# In[ ]:


copy_df.loc[0:25,"Attack":"Speed_Status"]


# In[ ]:


#Revers
copy_df.loc[25:0:-1,"Attack":"Speed_Status"]


# In[ ]:


speed_status_filter = copy_df["Speed_Status"] == "High"
copy_df[speed_status_filter]


# In[ ]:


speed_status_filter = copy_df["Speed_Status"] == "High"
speed_filter = copy_df["Speed"] > 150
copy_df[speed_status_filter & speed_filter]


# In[ ]:


copy_df.Speed[copy_df.HP > 130]


# Transforming Data

# In[ ]:


def midLevel(pokeAttack):
    return pokeAttack*2.5/1.8
copy_df.Attack.apply(midLevel)


# In[ ]:


copy_df.Defense.apply(lambda dfns: dfns * 2.5/0.8)


# Index objects and labeled data

# In[ ]:


copy_df.set_index('#')


# In[ ]:


print(copy_df.index.name)
copy_df.index.name = "INDEX"
copy_df.head()


# In[ ]:


copy_df.index = range(25,825,1)
copy_df


# In[ ]:


second_copy = df.copy()
second_copy = second_copy.set_index(["Type 1","Type 2"])
second_copy


# ## Pivoting

# In[ ]:


newDatas = {"Treatment":["A","B","A","B"],"Gender":["F","F","M","M"],"Response":[23,76,25,94]}
newDataFrame = pd.DataFrame(newDatas)
newDataFrame


# In[ ]:


newDataFrame.pivot(index="Treatment",columns="Gender",values="Response")


# In[ ]:


multiIndex = newDataFrame.set_index(["Treatment","Gender"])
multiIndex


# In[ ]:


multiIndex2 = multiIndex.copy()
multiIndex2.unstack(level=1)
multiIndex2


# ## Melting

# In[ ]:


pd.melt(newDataFrame,id_vars="Treatment",value_vars=["Gender","Response"])


# Grouping & Categorizied

# In[ ]:


newDataFrame.groupby("Treatment").mean()

