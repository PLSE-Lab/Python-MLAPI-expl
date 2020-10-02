#!/usr/bin/env python
# coding: utf-8

# In[244]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
plt.style.use('fivethirtyeight')
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ![](https://i.ytimg.com/vi/GD5ozrtGCyk/maxresdefault.jpg)

# In[245]:


Hp15=pd.read_csv('../input/2015.csv')
Hp16=pd.read_csv('../input/2016.csv')
Hp17=pd.read_csv('../input/2017.csv')


# *Adding the Year column in the dataframe*

# In[246]:


Hp15['Year']='2015'
Hp16['Year']='2016'
Hp17['Year']='2017'


# **Check for the column names in all the dataframes and make it common accordingly**

# In[247]:


Hp15.columns


# In[248]:


Hp16.columns


# In[249]:


Hp17.columns


# In[250]:


Hp17.columns=['Country','Happiness Rank','Happiness Score','Whisker high','Whisker low','Economy (GDP per Capita)','Family','Health (Life Expectancy)','Freedom','Generosity','Trust (Government Corruption)','Dystopia Residual','Year']


# In[253]:


twentyfiften = dict(zip(list(Hp15['Country']), list(Hp15['Region'])))
twentysixten = dict(zip(list(Hp16['Country']), list(Hp16['Region'])))
regions = dict(twentyfiften, **twentysixten)

def find_region(row):
    return regions.get(row['Country'])


Hp17['Region'] = Hp17.apply(lambda row: find_region(row), axis=1)


# *Checking for null values in the Region and filling it*

# In[255]:


Hp17[Hp17['Region'].isnull()]['Country']


# In[256]:


Hp17 = Hp17.fillna(value = {'Region': regions['China']})


# **Merging all the dataframes together to get a complete dataframe**

# In[257]:


hreport=pd.concat([Hp15,Hp16,Hp17])


# In[258]:


hreport.head()


# In[259]:


hreport.fillna(0,inplace=True)


# **Visualizing the average happiness score for the top 10 countries in the year 2015-2017**

# In[262]:


avghappscore=hreport.groupby(['Year','Country'])['Happiness Score'].mean().reset_index().sort_values(by='Happiness Score',ascending=False)
avghappscore=avghappscore.pivot('Country','Year','Happiness Score').fillna(0)


# In[263]:


hscore=avghappscore.sort_values(by='2017',ascending=False)[:11]
hscore.plot.barh(width=0.8,figsize=(10,10))


# **Top Countries leading the happiness ranking for the year 2015-2017**

# In[265]:


groupcountryandyear=hreport.groupby(['Year','Country']).sum()
a=groupcountryandyear['Happiness Rank'].groupby(level=0, group_keys=False)
top10=a.nsmallest(10).reset_index()
yearwisetop10=pd.pivot_table(index='Country',columns='Year',data=top10,values='Happiness Rank')
ax=plt.figure(figsize=(10,10))
fig=ax.add_axes([1,1,1,1])
fig.set_xlabel('Country')
fig.set_ylabel('Rank')
yearwisetop10.plot.bar(ax=fig,cmap='YlOrRd')


# *Checking the heatmap for the correlation*

# In[266]:


sns.heatmap(hreport.drop(['Whisker high','Whisker low','Upper Confidence Interval','Lower Confidence Interval'],axis=1).corr(),annot=True,cmap='RdYlGn')


# > Global happiness ranking for the year 2017

# In[267]:


twentyseventen=hreport[hreport['Year']=='2017']


# In[268]:


data = dict(type = 'choropleth', 
           locations = twentyseventen['Country'],
           locationmode = 'country names',
           z = twentyseventen['Happiness Rank'], 
           text = twentyseventen['Country'],
           colorbar = {'title':'Happiness'})
layout = dict(title = 'Global Happiness', 
             geo = dict(showframe = False, 
                       projection = {'type': 'Mercator'}))
choromap3 = go.Figure(data = [data], layout=layout)
iplot(choromap3)


# **Lets dig into the prediction part, We will use linear regression for the rank prediction of a country for the upcoming years**

# In[271]:


from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score,confusion_matrix


# In[270]:


hreport.head()


# > Choosing the X features based on the correlation factors we saw above

# In[272]:


x=hreport[['Economy (GDP per Capita)','Trust (Government Corruption)','Freedom','Health (Life Expectancy)','Family','Dystopia Residual']]
y=hreport['Happiness Rank']


# In[273]:


lm=LinearRegression()


# In[274]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=101)
lm.fit(X_train,y_train)


# In[275]:


ypred=lm.predict(X_test)


# In[276]:


plt.scatter(y_test,ypred)


# > As you can see the prediction is almost linear and its fitting the model well

# *lets check for the coefficients values*

# In[277]:


coef = zip(x.columns, lm.coef_)
coef_df = pd.DataFrame(list(zip(x.columns, lm.coef_)), columns=['features', 'coefficients'])
coef_df


# **How accurate our prediction is! Lets check our R2 score**

# In[279]:


r2_score(y_test,ypred)


# > Well predicted and thus we can see, our x features are well chosen and the model is designed well accordingly

# In[ ]:




