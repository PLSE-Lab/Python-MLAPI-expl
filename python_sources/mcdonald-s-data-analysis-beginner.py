#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

import os
print(os.listdir("../input"))

data=pd.read_csv('../input/menu.csv')


# In[ ]:


data.columns


# Check for missing **calories**(data :D)

# In[ ]:


data.isnull().sum()


# In[ ]:


data.tail()


# In[ ]:


data.corr()


# In[ ]:


data.pivot_table('Protein','Category').plot(kind='bar',stacked=True,color='y')


# In[ ]:


data.pivot_table('Vitamin A (% Daily Value)','Category').plot(kind='bar',stacked=True,color='b')


# In[ ]:


f,ax=plt.subplots(figsize=(18,11))
sns.heatmap(data.corr(),annot=True,linewidths=5,fmt='.1f',ax=ax)


# In[ ]:


data.Calories.plot(kind='hist',bins=40,figsize=(18,11))
plt.xlabel('Calories')
plt.ylabel('Sugars')


# In[ ]:


#Histogram
data.info() #to remember what categories i have


# In[ ]:


#Scatter Plot
data.plot(kind='scatter',x='Cholesterol',y='Sugars',alpha=0.4,color='m')
plt.xlabel('Cholesterol')
plt.ylabel('Sugars')
plt.title('does it matter?')


# In[ ]:


data


# In[ ]:


data['Category']


# In[ ]:


dataFrame=data[['Calories']]
print(type(dataFrame))
print('')
series=data['Calories']
print(type(series))


# In[ ]:


data.shape #rows and columns


# In[ ]:


data.describe() #to remember what i ve


# In[ ]:


x=data['Sugars'] > 30
data[x]


# In[ ]:


data[(data['Calories']>300) & (data['Total Fat']>15)]


# In[ ]:


data[np.logical_and(data['Carbohydrates']>35, data['Calories']>400)]


# In[ ]:


data.loc[:5,"Carbohydrates"]


# In[ ]:


data.loc[:5, ["Category","Item"]]


# In[ ]:


data.loc[:3,"Item" :"Cholesterol"] #between Item and Cholesterol


# In[ ]:


threshold=sum(data['Calories'])/len(data['Calories'])
print('threshold is', threshold)

data["cal_level"]=["high" if i>threshold else "low" for i in data['Calories']]
data.loc[:15,["cal_level","Calories","Item"]] 


# In[ ]:


data['Calories'].value_counts().head(10).plot.bar()


# In[ ]:


data['Trans Fat'].value_counts().sort_index().plot.line()


# In[ ]:


data['Cholesterol'].value_counts().sort_index().plot.area()


# In[ ]:


data.plot.scatter(x='Carbohydrates (% Daily Value)',y='Carbohydrates')


# In[ ]:


data.info()


# In[ ]:


print(data['Item'].value_counts(dropna=False))


# In[ ]:


data.boxplot(column='Calories', by='Calcium (% Daily Value)')


# > Tidy Data

# In[ ]:


data_new=data.head()
data_new


# > lets try to melt

# In[ ]:


melted_data=pd.melt(frame=data_new,id_vars = 'Item',value_vars=['Calories','Protein'])
melted_data


# > 

# > Pivoting Data , reverse of melting.

# In[ ]:


melted_data.pivot(index='Item',columns='variable',values='value')


# > Concatenating Data , lets concatenating two dataframes.

# In[ ]:


data1=data.head()
data2=data.tail()
conc_data=pd.concat([data1,data2],axis=0,ignore_index=True)
conc_data


# In[ ]:


data3=data['Calories'].head()
data4=data['Calories from Fat'].head()
conc_data_col=pd.concat([data3,data4],axis=1)
conc_data_col


# >Data Types

# In[ ]:


data.dtypes


# In[ ]:


data['Cholesterol']=data['Cholesterol'].astype('float64')
data['Serving Size']=data['Serving Size'].astype('category')


# In[ ]:


data.dtypes


# 

# In[ ]:


data5=data.loc[:,["Calcium (% Daily Value)","Vitamin A (% Daily Value)","Vitamin C (% Daily Value)"]]
data5.plot()


# >subplots

# In[ ]:


data5.plot(subplots=True)
plt.show()


# >scatter plot

# In[ ]:


data.plot(kind="scatter",x="Calories",y="Vitamin C (% Daily Value)")
plt.show()


# >hist plot

# In[ ]:


data5.plot(kind='hist',y='Calcium (% Daily Value)',bins=50,range=(0,160),normed=True)


# In[ ]:


# histogram subplot with non cumulative and cumulative
fig, axes = plt.subplots(nrows=2,ncols=1)
data.plot(kind = "hist",y = "Calories",bins = 50,range= (0,250),normed = True,ax = axes[0])
data.plot(kind = "hist",y = "Total Fat (% Daily Value)",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




