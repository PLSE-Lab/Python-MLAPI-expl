#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv('../input/usa-cers-dataset/USA_cars_datasets.csv')
df.head()


# In[ ]:


df.drop(['Unnamed: 0','vin'],axis=1,inplace=True)
df.head()


# In[ ]:


df.hist()


# # #Plotting a correlation, between No. of cars, and Prices.**

# In[ ]:


plt.hist(df['price'], color = 'blue', edgecolor = 'black',bins=10)

# Add labels
plt.title('Range of prices')
plt.xlabel('Prices')
plt.ylabel('No. of cars')


# (We see that most of the cars, are having the price range of $10k to $20k)

# # Now considering, how many cars originate from which country,considering our dataset.

# In[ ]:


plt.hist(df['country'], color = 'peru', edgecolor = 'black',bins=6)


# (96-97% of cars are from Usa, while only 3-4% are from canada)

# # Now,Lets consider the relation between the prices and the mileage given by the cars,using scatterplot.

# In[ ]:


plt.figure(figsize=(12,6))
sns.scatterplot(x=df['price'], y=df['mileage']);


# # The same relation,with just different representation,using Jointplots.

# In[ ]:


sns.jointplot(x=df['price'], y=df['mileage']);


# # Now, Lets see, Price range of various Models of cars,from the given dataset, using Seaborn Boxplot.

# In[ ]:


plt.figure(figsize=(12,6))
sorted_nb = df.groupby(['brand'])['price'].median().sort_values()
sns.boxplot(x=df['brand'], y=df['price'], order=list(sorted_nb.index))
plt.xticks(rotation=70)


# # Considering, How many cars,come from which state of USA, as well as Canada, we'll just take Top-30 states,for better presentation.

# In[ ]:


df['state'].value_counts().head(30).plot(kind='barh', figsize=(6,10))


# In[ ]:


plt.figure(figsize=(15,8))
sns.countplot(df['color']);
plt.xticks(rotation=90)


# # And Lastly, we'll plot the correalation Heatmap of each and every features,using Seaborn.

# In[ ]:


#correlation map
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# # If you, like the work, do upvote,also comment your views,
# # There can be many relations,which can be shown/plotted within the features using    Matplotlib and seaborn(which I have used), but I have just sticked to basic one's.
# # # Thank-You!

# In[ ]:




