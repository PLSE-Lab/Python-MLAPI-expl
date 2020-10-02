#!/usr/bin/env python
# coding: utf-8

#  First, lets import all useful lib that we will use

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


# In[ ]:


df = pd.read_csv('../input/alcohol-consumption-in-russia/russia_alcohol.csv')


# In[ ]:


df.head()


# In[ ]:


df.describe()


# After looking at df head and df descibe it might be useful to see how our data is distributed, we can do it by using hist command 

# In[ ]:


g=df.hist(figsize=(20,15))


# On this plots we can see that most of them have normal distribution, excepting brandy and champagne. df['year'] have some lack of data in between 2005 and 2007 years  

# Maybe some correlation exists between some type of alco, so to check this, we need to compute pairwise correlation for our type of alco

# In[ ]:


mask=np.zeros_like(df.corr(), dtype=bool)
mask[np.triu_indices_from(mask)]=True
sns.heatmap(df.corr(), mask=mask)


# By looking at plot, we can suggest that set of people who buy champagne most probably have big part of intersection with set of people who buy brandy

# It will be easier to build our plots, if our data will be represented like following: first 3 column will be the same and instead of separated columns for each type of alco we will create one column, that will display it

# In[ ]:


df_transformed=pd.DataFrame(columns=['year','region','alco','sale'])
for index,row in df.iterrows():
    s = pd.Series([row['year'], row['region'], 'wine', row['wine']],df_transformed.columns)
    s1 = pd.Series([row['year'], row['region'], 'beer', row['beer']],df_transformed.columns)
    s2 = pd.Series([row['year'], row['region'], 'vodka', row['vodka']],df_transformed.columns)
    s3 = pd.Series([row['year'], row['region'], 'champagne', row['champagne']],df_transformed.columns)
    s4 = pd.Series([row['year'], row['region'], 'brandy', row['brandy']],df_transformed.columns)
    df_transformed= df_transformed.append([s,s1,s2,s3,s4],ignore_index=True) 


# In[ ]:


df_transformed


# Now we can build our plots, and see the difference between type of alco by using hue parameter 

# In[ ]:


g=sns.relplot(x="year", y="sale",
            hue="alco",
            kind="line", data=df_transformed,size=10, aspect=2);


# The most dynamic changes type of alco - beer 

# # Top 10 regions by mean sales beer in liter per capita for period from 1998 to 2016

# In[ ]:


top_beer_region=df_transformed[df_transformed['alco']=='beer'].groupby('region').mean()
top_beer_region=top_beer_region.reset_index().sort_values(by='sale',ascending=False)
plt.figure(figsize=(15,7))
g=sns.barplot(data=top_beer_region[:10],y='sale',x='region')
g.set_xticklabels(g.get_xticklabels(), rotation=90)
g.tick_params(labelsize=10)


# # Top 10 regions by mean sales wine in liter per capita for period from 1998 to 2016

# In[ ]:


top_wine_region=df_transformed[df_transformed['alco']=='wine'].groupby('region').mean()
top_wine_region=top_wine_region.reset_index().sort_values(by='sale',ascending=False)
plt.figure(figsize=(15,7))
g=sns.barplot(data=top_wine_region[:10],y='sale',x='region')
g.set_xticklabels(g.get_xticklabels(), rotation=90)
g.tick_params(labelsize=10)


# # Top 10 regions by mean sales vodka in liter per capita for period from 1998 to 2016

# In[ ]:


top_vodka_region=df_transformed[df_transformed['alco']=='vodka'].groupby('region').mean()
top_vodka_region=top_vodka_region.reset_index().sort_values(by='sale',ascending=False)
plt.figure(figsize=(15,7))
g=sns.barplot(data=top_vodka_region[:10],y='sale',x='region')
g.set_xticklabels(g.get_xticklabels(), rotation=90)
g.tick_params(labelsize=10)

