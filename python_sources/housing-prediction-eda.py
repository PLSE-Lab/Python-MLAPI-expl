#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.read_csv('/kaggle/input/Housing.csv')


# In[ ]:


df.head()


# In[ ]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.corr()


# In[ ]:


sns.heatmap(df.corr(),annot=True,cmap='Blues')
plt.show()


# In[ ]:


def plot_feature(df,col):
    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    if df[col].dtype == 'int64':
        df[col].value_counts().sort_index().plot()
    else:
        mean = df.groupby(col)['price'].mean()
        df[col] = df[col].astype('category')
        levels = mean.sort_values().index.tolist()
        df[col].cat.reorder_categories(levels,inplace=True)
        df[col].value_counts().plot()
    plt.xticks(rotation=45)
    plt.xlabel(col)
    plt.ylabel('Counts')
    plt.subplot(1,2,2)
    
    if df[col].dtype == 'int64' or col == 'price':
        mean = df.groupby(col)['price'].mean()
        std = df.groupby(col)['price'].std()
        mean.plot()
        plt.fill_between(range(len(std.index)),mean.values-std.values,mean.values + std.values,                         alpha=0.1)
    else:
        sns.boxplot(x = col,y='price',data=df)
    plt.xticks(rotation=45)
    plt.ylabel('price')
    plt.show()    


# In[ ]:


plot_feature(df,'area')


# In[ ]:


plot_feature(df,'bedrooms')


# In[ ]:


plot_feature(df,'bathrooms')


# In[ ]:


plot_feature(df,'stories')


# In[ ]:


plot_feature(df,'mainroad')


# In[ ]:


plot_feature(df,'guestroom')


# In[ ]:


plot_feature(df,'basement')


# In[ ]:


plot_feature(df,'hotwaterheating')


# In[ ]:


plot_feature(df,'airconditioning')


# In[ ]:


plot_feature(df,'parking')


# In[ ]:


plot_feature(df,'prefarea')


# In[ ]:


plot_feature(df,'furnishingstatus')


# In[ ]:


plot_feature(df,'price')


# In[ ]:


sns.pairplot(df,diag_kind='kde')
plt.show()


# In[ ]:


plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
sns.boxplot(df.price)
plt.subplot(1,2,2)
sns.distplot(df.price,bins=20)
plt.show()


# In[ ]:


for col in df:
    plt.figure(figsize=(10,8))
    sns.barplot(x=df[col],y=df['price'])
    


# In[ ]:


for col in df:
    plt.figure(figsize=(10,8))
    sns.boxplot(x=df[col],y=df['price'])


# In[ ]:


for col in df:
    plt.figure(figsize=(10,8))
    sns.violinplot(x=df[col],y=df['price'])


# In[ ]:


df.columns


# In[ ]:


def plot_count(x,fig):
    plt.subplot(4,2,fig)
   
    sns.countplot(df[x],palette=("magma"))
    plt.subplot(4,2,(fig+1))
    
    sns.boxplot(x=df[x], y=df.price, palette=("magma"))
    
plt.figure(figsize=(15,20))

plot_count('area', 1)
plot_count('bedrooms', 3)
plot_count('stories', 5)
plot_count('mainroad', 7)



plt.tight_layout()
plt.show()


# In[ ]:


def plot_count(x,fig):
    plt.subplot(4,2,fig)
   
    sns.countplot(df[x],palette=("magma"))
    plt.subplot(4,2,(fig+1))
    
    sns.boxplot(x=df[x], y=df.price, palette=("magma"))
    
plt.figure(figsize=(15,20))

plot_count('guestroom', 1)
plot_count('basement', 3)
plot_count('hotwaterheating',5)
plot_count('airconditioning',7)




plt.tight_layout()
plt.show()


# In[ ]:


def plot_count(x,fig):
    plt.subplot(4,2,fig)
   
    sns.countplot(df[x],palette=("magma"))
    plt.subplot(4,2,(fig+1))
    
    sns.boxplot(x=df[x], y=df.price, palette=("magma"))
    
plt.figure(figsize=(15,20))

plot_count('parking', 1)
plot_count('prefarea', 3)
plot_count('furnishingstatus', 5)
plot_count('price', 7)



plt.tight_layout()
plt.show()


# In[ ]:


df.columns


# In[ ]:


df.head()


# In[ ]:


num = ['area', 'bedrooms', 'bathrooms', 'stories','parking']


# In[ ]:


num


# In[ ]:


num = ['area', 'bedrooms', 'bathrooms', 'stories','parking']

fig, axis = plt.subplots(2, 3,  figsize=(25, 20))

counter = 0
for items in num:
    value_counts = df[items].value_counts()
    
    trace_x = counter // 3
    trace_y = counter % 3
    x_pos = np.arange(0, len(value_counts))
    my_colors = 'rgbkymc'
    
    axis[trace_x, trace_y].bar(x_pos, value_counts.values, tick_label = value_counts.index,color=my_colors)
    
    axis[trace_x, trace_y].set_title(items)
    
    for tick in axis[trace_x, trace_y].get_xticklabels():
        tick.set_rotation(90)
    
    counter += 1

plt.tight_layout()
plt.show()


# In[ ]:


fig, axis = plt.subplots(2, 3, sharex=False, sharey=False, figsize=(20, 15))

counter = 0
for items in num:
    
    trace_x = counter // 3
    trace_y = counter % 3
    
    
    axis[trace_x, trace_y].hist(df[items])
    
    axis[trace_x, trace_y].set_title(items)
    
    counter += 1

plt.tight_layout()
plt.show()


# In[ ]:


for col in num:
    plt.figure(figsize=(10,8))
    sns.jointplot(x=df[col],y=df['price'],kind='reg')


# In[ ]:





# In[ ]:


plt.figure(figsize=(14,6))
df['area'].plot()
plt.show()


# In[ ]:


plt.figure(figsize=(14,6))
df['bedrooms'].plot()
plt.show()


# In[ ]:


plt.figure(figsize=(14,6))
df['bathrooms'].plot()
plt.show()


# In[ ]:


plt.figure(figsize=(14,6))
df['stories'].plot()
plt.show()


# In[ ]:


plt.figure(figsize=(14,6))
df['parking'].plot()
plt.show()


# In[ ]:


plt.figure(figsize=(14,6))
df['price'].plot()
plt.show()


# In[ ]:


plt.figure(figsize=(20, 12))
plt.plot(df['area'],df['price'], color='orange')
plt.xlabel('area', size=30)
plt.ylabel('price', size=30)
plt.xticks(rotation=50, size=15)
plt.show()


# In[ ]:


plt.figure(figsize=(20, 12))
plt.plot(df['bedrooms'],df['price'], color='orange')
plt.xlabel('bedrooms', size=30)
plt.ylabel('price', size=30)
plt.xticks(rotation=50, size=15)
plt.show()


# In[ ]:


plt.figure(figsize=(20, 12))
plt.plot(df['bathrooms'],df['price'], color='orange')
plt.xlabel('bathrooms', size=30)
plt.ylabel('price', size=30)
plt.xticks(rotation=50, size=15)
plt.show()


# In[ ]:


plt.figure(figsize=(20, 12))
plt.plot(df['stories'],df['price'], color='orange')
plt.xlabel('stories', size=30)
plt.ylabel('price', size=30)
plt.xticks(rotation=50, size=15)
plt.show()


# In[ ]:


plt.figure(figsize=(20, 12))
plt.plot(df['parking'],df['price'], color='orange')
plt.xlabel('parking', size=30)
plt.ylabel('price', size=30)
plt.xticks(rotation=50, size=15)
plt.show()


# In[ ]:


plt.figure(figsize=(20, 12))
plt.plot(df['area'], color='red')
plt.xlabel('area', size=30)
plt.ylabel('price', size=30)
plt.xticks(rotation=50, size=15)
plt.show()


# In[ ]:


plt.figure(figsize=(20, 12))
plt.plot(df['bedrooms'], color='orange')
plt.xlabel('area', size=30)
plt.ylabel('price', size=30)
plt.xticks(rotation=50, size=15)
plt.show()


# In[ ]:


plt.figure(figsize=(20, 12))
plt.plot(df['bathrooms'], color='green')
plt.xlabel('area', size=30)
plt.ylabel('price', size=30)
plt.xticks(rotation=50, size=15)
plt.show()


# In[ ]:


plt.figure(figsize=(20, 12))
plt.plot(df['stories'], color='black')
plt.xlabel('area', size=30)
plt.ylabel('price', size=30)
plt.xticks(rotation=50, size=15)
plt.show()


# In[ ]:


plt.figure(figsize=(20, 12))
plt.plot(df['parking'], color='blue')
plt.xlabel('area', size=30)
plt.ylabel('price', size=30)
plt.xticks(rotation=50, size=15)
plt.show()


# In[ ]:


for col in num:
    plt.figure(figsize=(10,8))
    sns.scatterplot(x=df[col],y=df['price'])
    


# In[ ]:


plt.figure(figsize=(9,6))
sns.boxplot(data=df,x='area')
plt.show()


# In[ ]:


plt.figure(figsize=(9,6))
sns.boxplot(data=df,x='bedrooms')
plt.show()


# In[ ]:


plt.figure(figsize=(9,6))
sns.boxplot(data=df,x='bathrooms')
plt.show()


# In[ ]:


plt.figure(figsize=(9,6))
sns.boxplot(data=df,x='stories')
plt.show()


# In[ ]:


plt.figure(figsize=(9,6))
sns.boxplot(data=df,x='parking')
plt.show()


# In[ ]:


for col in num:
    plt.figure(figsize=(10,9))
    sns.distplot(df[col],color='red')
    plt.show()
    


# In[ ]:




