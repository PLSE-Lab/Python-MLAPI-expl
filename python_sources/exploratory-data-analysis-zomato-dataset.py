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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = "ticks")  #style = 'dark','darkgrid','whitegrid' are some other styles


# In[ ]:


df = pd.read_csv("../input/zomato.csv") #reading the document


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


#dropping a few columns 


# In[ ]:


df = df.drop(columns = ["url", "address","phone","location", "reviews_list", "menu_item", "listed_in(type)"], axis =1)


# In[ ]:


df.columns


# In[ ]:


df.columns = ['name','online_orders','prebooking','rate','votes','type',"best_dishes","cuisines",'cost','area'] #renaming the columns we will work with


# In[ ]:


df.dtypes #here cost dtype is object


# In[ ]:


df.dropna(subset = ['cost'], axis = 0, how = 'all').tail() #dropping na/nan values


# In[ ]:


df.shape #shape of the df 


# In[ ]:


v1 = lambda x:float(x[1:-1]) #conversion to float


# In[ ]:


df["cost"] = df["cost"].str.replace(",","").astype(float)


# In[ ]:


df.dropna(subset = ['cost'], how = 'all', axis = 0).tail() 


# In[ ]:


df = df.dropna(how='any')           # assign back
df.dropna(how='any', inplace=True)  # set inplace parameter


# In[ ]:


df.best_dishes.tail()


# In[ ]:


df.tail()


# In[ ]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud
##### Download using conda install -c conda-forge wordcloud

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(background_color='black',).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    plt.imshow(wordcloud)
    plt.show()


# In[ ]:


show_wordcloud(df['cuisines'])


# In[ ]:


#plotting histograms


# In[ ]:


df ['cost'].plot(kind = 'hist', bins = 100)
plt.xlabel('cost')


# In[ ]:


df.plot.hist(subplots=True, layout=(2,2), figsize=(10, 10), bins=5)


# In[ ]:


#plotting bar graphs


# In[ ]:


df['online_orders'].value_counts().sort_index().plot.barh() #online orders


# In[ ]:


df['prebooking'].value_counts().sort_index().plot.barh() #prebooking


# In[ ]:


#bar graphs


# In[ ]:


df.groupby('area').cost.mean().sort_values(ascending = False).plot.bar() #cost


# In[ ]:


df.groupby('area').votes.mean().sort_values(ascending = True).plot.bar() #votes


# In[ ]:


#plotting with seaborn


# In[ ]:


sns.countplot(df['online_orders'])


# In[ ]:


#plotting with plotly
#Download using pip install plotly + pip install cufflinks


# In[ ]:


# Standard plotly imports
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode 
# Using plotly + cufflinks in offline mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)


# In[ ]:


df['type'].iplot(kind = 'hist', xTitle='type', yTitle='count', title='Rest_type distribution')


# In[ ]:


df['name'].iplot(kind = 'hist', xTitle='name', yTitle='count', title='Name distribution')


# In[ ]:


df[['cost','votes']].iplot( kind = 'spread') #spread


# In[ ]:


df.iplot(kind='bubble',x='cost',y='votes',size='votes') #bubble


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




