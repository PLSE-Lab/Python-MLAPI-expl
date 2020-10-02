#!/usr/bin/env python
# coding: utf-8

# **Imports I'll nedd**

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True) 
get_ipython().run_line_magic('matplotlib', 'inline')


# **Loading data from CSV**

# In[ ]:


df = pd.read_csv('../input/athlete-events/athlete_events.csv')
df.head()


# **In this one I'll check out wich Country won more Gold Medals**

# In[ ]:


top_5_gold = df[df['Medal'] == 'Gold']['Team'].value_counts().head(5)


# In[ ]:


top_fg = pd.DataFrame(top_5_gold)


# In[ ]:


top_fg.reset_index(inplace=True)


# In[ ]:


top_fg


# In[ ]:


top_fg.rename(columns={'index':'Country'}, inplace=True)
plt.figure(figsize=(12,8))
sns.barplot(x='Country', y='Team', data=top_fg)


# **No surprise here, first USA and second SU**

# **Now, lets look for a few statistical data**

# In[ ]:


df['Age'].mean()


# In[ ]:


df[df['Medal'] == 'Gold']['Age'].mean()


# In[ ]:





# In[ ]:



df[df['Medal'] == 'Silver']['Age'].mean()


# In[ ]:


df[df['Medal'] == 'Bronze']['Age'].mean()


# In[ ]:


df['Age'].min()


# In[ ]:


df['Age'].max()


# **Or all togheter with 'describe'**

# In[ ]:


df.describe()


# *It's nice to notice that the mean age of the medals winners is near 25 years old.*

# **Now I'll find out how many categories we had in Olympics trought the years.**

# In[ ]:


cat_by_year = df.groupby('Year')['Sport'].nunique()
cat_by_year = pd.DataFrame(cat_by_year)


# In[ ]:


cat_by_year.reset_index(inplace=True)


# In[ ]:


cat_by_year.head()


# In[ ]:


plt.figure(figsize=(20,8))
sns.barplot(x='Year', y='Sport', data=cat_by_year, palette='inferno')


# As explained in the link bellow, from 1994 we have Summer and Winter games splited, happening every two years, that's why in 94/98/2002/2006/2010/2014 we have low sports.
# Link: https://pt.wikipedia.org/wiki/Jogos_Ol%C3%ADmpicos_de_Inverno_de_1994 (in Portuguese)

# **Now let's look at the amount of medals by countries.**

# In[ ]:


df.groupby('NOC')['Medal'].count().head()


# In[ ]:


med_by_country = df.groupby('NOC')['Medal'].count()


# In[ ]:


med_by_country = pd.DataFrame(med_by_country)


# In[ ]:


med_by_country.reset_index(inplace=True)


# In[ ]:


med_by_country.head()


# In[ ]:


med_by_country = med_by_country.sort_values(by='Medal',ascending=False).head()


# In[ ]:


med_by_country


# No surprise again.

# **Looking for more interesting data.** 

# In[ ]:


df.groupby('Season')['City'].nunique()


# In[ ]:


df['Event'].nunique()


# **Here I create a DF with most played sports in the Olympics.**

# In[ ]:


most_played = df['Event'].value_counts().head()


# In[ ]:


most_played = pd.DataFrame(most_played)


# In[ ]:


most_played.reset_index(inplace=True)


# In[ ]:


most_played.rename(columns={'index':'Sport'}, inplace=True)


# In[ ]:


most_played


# In[ ]:


plt.figure(figsize=(16,8))
sns.barplot(x='Sport',y='Event', data=most_played)


# As a brazilian, I'm happy to look that it was Soccer.

# **Looking for a correlation, the only one I've founded was between height and weight but thats nothing to be surprised with as soon as athletes must be in shape.**

# In[ ]:


df.corr()


# **Here I create a DF with the countries that had played more in all categories.**

# In[ ]:


most_act = df['NOC'].value_counts()


# In[ ]:


most_act = pd.DataFrame(most_act)


# In[ ]:


most_act.reset_index(inplace=True)


# In[ ]:


most_act.rename(columns={'index':'Country'}, inplace=True)


# In[ ]:


most_act.head()


# In[ ]:


data = dict(
        type = 'choropleth',
        locations = most_act['Country'],
        z = most_act['NOC'],
        text = most_act['Country'],
        colorbar = {'title' : 'Most active countries in Olympic Games'},
      ) 


# In[ ]:


layout = dict(
    title = 'Most active countries in Olympic Games',
    geo = dict(
        showframe = True
    )
)


# In[ ]:


choromap = go.Figure(data = [data],layout = layout)
iplot(choromap)


# USA was the most active country in history of the Olympic Games!

# **Now I'm just checking who won more medals.**

# In[ ]:


biggest_winners = df.groupby('Name')['Medal'].value_counts()


# In[ ]:


biggest_winners = pd.DataFrame(biggest_winners)


# In[ ]:


biggest_winners.sort_values(by='Medal', ascending=False).head()


# The myth Michael Fred Phelps!

# **Now I'll try to predict who'd win a gold medal using the historical data.**

# *First we need to convert categorical data into numbers.*

# In[ ]:


df['Gold'] = pd.get_dummies(df['Medal']=='Gold', drop_first=True)


# In[ ]:


from sklearn import preprocessing


# In[ ]:


le = preprocessing.LabelEncoder()


# In[ ]:


le.fit(df['NOC'])


# Medal Ok

# In[ ]:


df['country_cod'] = le.transform(df['NOC'])


# In[ ]:


le.fit(df['Event'])


# In[ ]:


le.transform(df['Event'])


# In[ ]:


df['event_cod'] = le.transform(df['Event'])


# In[ ]:


df.head()


# **Now let me import model_selection module train_split_test**

# In[ ]:


from sklearn.model_selection import train_test_split


# **Filling null values with mean age.**

# In[ ]:


df['Age'].fillna(df['Age'].mean(), inplace=True)


# In[ ]:


x = df[['Age', 'Year', 'country_cod', 'event_cod']]
y = df['Gold']


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=101, test_size=0.30)


# **Importing the ML model I've selected, I've tried others but this one performed really good with no adjustments, and without overfitting.**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rcf = RandomForestClassifier()


# In[ ]:


rcf.fit(x_train, y_train)


# **Predicting..**

# In[ ]:


pred = rcf.predict(x_test)


# **Importing Metrics to show my results.**

# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test, pred))


# Well, I've reached 96% of accuracy in this model, seams good to me!
# See you arround folks!

# In[ ]:




