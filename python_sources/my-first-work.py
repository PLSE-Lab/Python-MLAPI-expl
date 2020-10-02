#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/2017.csv")


# In[ ]:


data.info()


# In[ ]:


data.corr()


# In[ ]:


sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f')
plt.show()


# In[ ]:


data.head(10)


# In[ ]:


data.columns


# In[ ]:


data.plot(kind = 'line', x="Happiness.Rank", y="Happiness.Score", color = 'g',label = 'Happiness.Score',linewidth=2,alpha = 1,grid = True,linestyle = '-')

plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()


# In[ ]:


data.plot(kind='scatter', x='Economy..GDP.per.Capita.', y='Happiness.Score',alpha = 0.5,color = 'red')
plt.xlabel('Economy..GDP.per.Capita.')              # label = name of label
plt.ylabel('Happiness.Score')
plt.title('Economy..GDP.per.Capita. vs Happiness.Score Scatter Plot')   


# In[ ]:


series = data['Country']        # data['Defense'] = series
print(type(series))
data_frame = data[['Country']]  # data[['Defense']] = data frame
print(type(data_frame))


# In[ ]:


x=data[data["Economy..GDP.per.Capita."]>1.5]


# In[ ]:


y=data[(data["Economy..GDP.per.Capita."]>1.5) & (data["Happiness.Score"]>5.0)]
y1=data[np.logical_and(data["Economy..GDP.per.Capita."]>1.5, data["Happiness.Score"]>5.0 )]


# In[ ]:


y


# In[ ]:


y1


# In[ ]:


for index,value in data[['Country']][0:6].iterrows():
    print(index," : ",value)


# In[ ]:


score = sum(data["Happiness.Score"])/len(data["Happiness.Score"])


# In[ ]:


score


# In[ ]:


data["Happiness_level"] = ["high" if i > score else "low" for i in data["Happiness.Score"]]


# In[ ]:


data.loc[0:100,["Country","Happiness.Score","Happiness_level"]]


# In[ ]:


data.head()


# In[ ]:


data.tail()


# In[ ]:


data.info()


# In[ ]:


print(data["Happiness_level"].value_counts(dropna =False))


# In[ ]:


data.describe()


# In[ ]:


data.boxplot(column="Happiness.Score", by = "Happiness_level")


# In[ ]:


data1=data.head(10)


# In[ ]:


data1


# In[ ]:


data3 = data1[['Happiness.Score','Family']]


# In[ ]:


data3


# In[ ]:


data2=pd.melt(frame=data1, id_vars = 'Country',value_vars= ['Happiness.Score','Family'] )


# In[ ]:


data2


# In[ ]:


data2.pivot(index = 'Country', columns = 'variable',values='value')


# In[ ]:


data1=data.head()


# In[ ]:


data2=data.tail()


# In[ ]:


data1


# In[ ]:


data2


# In[ ]:


conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =False)


# In[ ]:


conc_data_row


# In[ ]:


data1 = data['Happiness.Rank'].head()
data2 = data['Happiness.Score'].head()
conc_data_col = pd.concat([data1,data2],axis =1) # axis = 0 : adds dataframes in row
conc_data_col


# In[ ]:


data3=data[['Happiness.Rank','Happiness.Score']].head()


# In[ ]:


data3


# In[ ]:


data.dtypes


# In[ ]:


data.info()


# In[ ]:


assert  data['Country'].notnull().all()


# In[ ]:


data1=data[["Happiness.Score","Freedom","Family"]]
data1.plot()
plt.show()


# In[ ]:


data1.plot(subplots = True)
plt.show()


# In[ ]:


data1.plot(kind="scatter", x="Family", y="Happiness.Score")
plt.show()


# In[ ]:


data1.plot(kind="scatter", x="Freedom", y="Happiness.Score")
plt.show()


# In[ ]:


data1["Happy"]=[int(i) for i in data1["Happiness.Score"]]
data1


# In[ ]:


data1["Happy"].value_counts(dropna=False)


# In[ ]:


data1.plot(kind = "hist",y = "Happy",bins = 20,range= (0,10), normed=False)


# In[ ]:


data2=data.head()
data2


# In[ ]:


date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]
datetime_object = pd.to_datetime(date_list)
data2["date"]=datetime_object
data2


# In[ ]:


data2 = data2.set_index("date")
data2


# In[ ]:


print(data2.loc["1993-03-16"])


# In[ ]:


print(data2.loc["1992-03-10":"1993-03-16"])


# In[ ]:


data2.resample("A").mean()


# In[ ]:


data2.resample("M").mean()


# In[ ]:


data2.resample("M").first().interpolate("linear")


# In[ ]:


data2.resample("M").mean().interpolate("linear")


# In[ ]:


data.head()


# In[ ]:


data1=data.head(10)


# In[ ]:


data1


# In[ ]:


data.Family[1]


# In[ ]:


data["Family"][1]


# In[ ]:


data1.loc[1,["Family"]]


# In[ ]:


data1[["Country","Happiness.Rank"]]


# In[ ]:


data1.loc[1:5,"Country":"Family"]


# In[ ]:


data1.loc[5:1:-1,"Country":"Family"]


# In[ ]:


data1


# In[ ]:


x=data1.Family>1.5
data1[x]


# In[ ]:


data1[data1.Family>1.5]


# In[ ]:


data1.Family>1.5


# In[ ]:


data1[data1["Family"]>1.5]


# In[ ]:


x=data1["Family"]>1.5
y=data1["Happiness.Score"]>7.5
data1[x&y]


# In[ ]:


data1["Happiness.Score"][data1["Family"]>1.5]


# In[ ]:


def div(n):
    return n/2
data1.Family=data1.Family.apply(div)


# In[ ]:


data1


# In[ ]:


data1.Family=data1.Family.apply(lambda n : n/2)


# In[ ]:


data1


# In[ ]:


data1["berk"]=data1["Happiness.Score"]+data1.Family


# In[ ]:


data1


# In[ ]:


print(data1.index.name)
data1.index.name="index_name"
print(data1.index.name)


# In[ ]:


data1


# In[ ]:


data1.index=range(100,110,1)


# In[ ]:


data1


# In[ ]:


data=pd.read_csv("../input/2017.csv")


# In[ ]:


data.head()


# In[ ]:


data1=data.head(10)
data1


# In[ ]:


data2=data1.set_index(["Happiness.Rank"])


# In[ ]:


data2


# In[ ]:


data3.index=data1["Happiness.Rank"]


# In[ ]:


data3


# In[ ]:


data4=data1.set_index(["Happiness.Rank","Family"])


# In[ ]:


data4


# In[ ]:


dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}
df = pd.DataFrame(dic)
df


# In[ ]:


df.index=range(1,5)


# In[ ]:


df


# In[ ]:


df1=df.head()


# In[ ]:


df1


# In[ ]:


df.pivot(index="treatment",columns = "gender",values="response")


# In[ ]:


df1 = df.set_index(["treatment","gender"])
df1


# In[ ]:


df1.unstack(level=0)


# In[ ]:


df1.unstack(level=1)


# In[ ]:


df2 = df1.swaplevel(0,1)
df2


# In[ ]:


pd.melt(df,id_vars="treatment",value_vars=["age","response"])


# In[ ]:


df


# In[ ]:


df.groupby("treatment").mean() 


# In[ ]:


df.groupby("treatment").age.max() 


# In[ ]:


df.groupby("treatment")[["age","response"]].min() 


# In[ ]:




