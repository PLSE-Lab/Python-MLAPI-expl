#!/usr/bin/env python
# coding: utf-8

# # Forecasting the number of infected people in Brazil

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime


# In[ ]:


df = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv', parse_dates=['Date'], index_col='Id')
df.rename(columns={'Date': 'date',
                     'Province/State':'state',
                     'Country/Region':'country',
                     'Lat':'lat',
                     'Long': 'long',
                     'ConfirmedCases': 'confirmed',
                     'Fatalities':'deaths',
                    }, inplace=True)
df.head()


# In[ ]:


df = df.drop(['state'], axis=1).groupby(['country','date']).sum().reset_index()
confirmed = df.pivot_table(index='date', columns='country', values='confirmed')
confirmed.tail()


# In[ ]:


drop_countries = confirmed.sum()==.0
confirmed.drop(drop_countries.index[drop_countries==True].tolist(), axis=1, inplace=True)
confirmed.tail()


# In[ ]:


confirmed_corr = confirmed_norm.corr()
confirmed_corr['Brazil'].sort_values(ascending=False).head(10)


# Okay, so we can see that Brazil's evolution is highly correlated with other countries such as the US and Portugual, which have more confirmed cases, but also Cuba and Burkina Faso, which have considerably fewer cases. If we want to predict what will happen in Brazil based on what **has happened** in other countries, we need to change the index from the calendar date to the date since the first case was discovered.

# In[ ]:


first_BR = (confirmed['Brazil']>10).idxmax()
days_since_first = confirmed.index[-1] - first_BR
print(first_BR)


# In[ ]:


drop_countries_1 = confirmed.loc[first_BR] < 1
confirmed_BR = confirmed.drop(drop_countries_1.index[drop_countries_1==True].tolist(), axis=1)

drop_countries_2 = confirmed_BR.iloc[-1] <= confirmed_BR.iloc[-1].loc['Brazil']
confirmed_BR = confirmed_BR.drop(drop_countries_2.index[drop_countries_2==True].tolist(), axis=1)

confirmed_BR_countries = confirmed_BR.columns.tolist()
confirmed_BR.tail()


# In[ ]:


aligned = pd.DataFrame()
aligned['Brazil'] = confirmed.loc[first_BR:,'Brazil'].values
aligned.head()


# In[ ]:


for country in confirmed_BR_countries:
    first = (confirmed[country]>10).idxmax()
    aligned[country] = confirmed.loc[first:first+days_since_first,country].values
    print(first, country)
    
aligned.head()


# Since the 10th Brazilian case was confirmed on March 6, we'll exclude countries who detected their 10th infected case during March. This way, we will get a total of 6 days of forecast into the future using the remaining countries.

# In[ ]:


aligned.drop(['Austria', 'Belgium', 'Netherlands', 'Portugal'], axis=1, inplace=True)
fig,ax = plt.subplots(figsize=(10,10))
aligned.plot(ax=ax, cmap='tab20')


# In[ ]:


aligned.corr()['Brazil'].sort_values(ascending=False)


# In[ ]:


aligned.drop(['Germany','Korea, South'], axis=1, inplace=True)


# According to the correlation matrix, Brazil could follow Spain or Italy's footsteps. We dropped Germany and South Korea due to the low correlations; they were likely to contribute poorly to the model anyway. We're ready to perform our Multiple Linear Regression model: we'll train our model to predict Brazil's outcome knowing what happened to other countries.

# In[ ]:


from sklearn.linear_model import LinearRegression

X = aligned.drop('Brazil', axis=1)
y = aligned['Brazil']

lm = LinearRegression()
lm.fit(X,y)
coefs = lm.coef_
print(lm.score(X,y))


# In[ ]:


sns.barplot(y=X.columns.tolist(),x=coefs)


# In[ ]:


aligned_new = pd.DataFrame()
days_rest = days_since_first + datetime.timedelta(days=6)

for country in X.columns.tolist():
    first = (confirmed[country]>10).idxmax()
    aligned_new[country] = confirmed.loc[first:first+days_rest,country].values
    
aligned_new.tail()


# In[ ]:


aligned_new['Brazil_pred'] = lm.predict(aligned_new)


# In[ ]:


fig, ax = plt.subplots(figsize=(12,12))
aligned_new.plot(cmap='tab20', ax=ax)


# Our analysis indicates that the number of Brazilians infected won't grow as fast as other countries such as Italy, Spain or the US. According to our model, we can expect the number of cases to continue growing in a controlled manner, similar to what is happening in Iran and Switzerland where the number of infections is growing but not exponentially as it is in the US.
# 
# As days go by and the pandemic continues to develop, new data will allow us to update our model and verify our predictions.
# 
# There are 2 relevant ideas to keep in mind. First of all, our model is limited by the distance between the predictor countries (the 6 days between the 10th case in Brazil and the 10th case in other countries), which means we can only forecast a few days into the future. Second, there is a lot of noice in the dataset, since the number of tests may vary from day to day and not all confirmed cases may be reported in time. A longer pandemic would certainly mean a better dataset, but **let's hope this won't last much longer**.
