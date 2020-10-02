#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv("../input/NBA.csv")


# In[ ]:


df.columns


# In[ ]:


df.dtypes


# In[ ]:


df.drop(['FINAL_MARGIN'], axis = 1, inplace = True)


# In[ ]:


df['MATCHUP'] = df['MATCHUP'].str[:12]


# In[ ]:


df['MATCHUP'] = pd.to_datetime(df.MATCHUP)


# In[ ]:


df['Day'] = df.MATCHUP.dt.day
df['Month'] = df.MATCHUP.dt.month
df['Year'] = df.MATCHUP.dt.year


# In[ ]:


df['GAME_CLOCK'] = pd.to_datetime(df.GAME_CLOCK, format = '%M:%S')


# In[ ]:


df['GAME_CLOCK'] = df.GAME_CLOCK.dt.time


# In[ ]:


df.dropna(subset = ['SHOT_CLOCK'], how = 'any', inplace = True)


# In[ ]:


df['SHOT_CLOCK'] = pd.to_datetime(df.SHOT_CLOCK, format = '%S')


# In[ ]:


df['SHOT_CLOCK'] = df.SHOT_CLOCK.dt.time


# In[ ]:


df['LOCATION'] = df['LOCATION'].str.replace('H','0').str.replace('A','1')


# In[ ]:


df['LOCATION'] = df['LOCATION'].astype(int)


# In[ ]:


df['SHOT_RESULT'] = df['SHOT_RESULT'].str.replace('missed','0').str.replace('made','1').astype(int)


# In[ ]:


df.drop('FGM',axis = 1, inplace = True)


# In[ ]:


df.head()


# In[ ]:


df.groupby(['player_name']).SHOT_RESULT.sum().sort_values(ascending = False).head(5)


# In[ ]:


df.groupby(['player_name']).PTS.sum().sort_values(ascending = False).head(5)


# In[ ]:


df.groupby(['CLOSEST_DEFENDER']).CLOSE_DEF_DIST.mean().sort_values().head(50)


# In[ ]:





# In[ ]:


df.MATCHUP.min()


# In[ ]:


df.MATCHUP.max()


# In[ ]:


df['MY'] = df.Year.map(str) + '-' + df.Month.map(str)


# In[ ]:


df.groupby(['Month','Year','player_name']).PTS.sum().sort_values(ascending = False).head(5)


# In[ ]:


df.groupby(['player_name']).SHOT_DIST.mean().sort_values(ascending = False).head(5)


# In[ ]:


df.groupby(['Month']).PTS.sum()


# In[ ]:


df.dtypes


# In[ ]:


df.groupby(['player_name']).PTS.sum().sort_values(ascending = False).head(10).plot.bar()
plt.title('Most Points during October 2014 to March 2015')
plt.xlabel('Player')
plt.xticks(color = 'B')
plt.ylabel('Points Scored')


# In[ ]:


df.groupby(['MY']).PTS.sum().plot(c = 'r')
plt.title('Point Total by Month')
plt.xlabel('Year-Month')
plt.ylabel('Points')
plt.yticks([5000,10000,15000,20000,25000,30000,35000],['5,000',
                                                      '10,000','15,000','20,000','25,000','30,000','35,000'])


# In[ ]:


df.corr()


# In[ ]:


plt.scatter(x=df.CLOSE_DEF_DIST,y=df.SHOT_DIST, c = 'b')
plt.title('Defender Distance Vs Shot Distance')
plt.xlabel('Defender Distance')
plt.ylabel('Shot Distance')


# In[ ]:


plt.scatter(x=df.TOUCH_TIME,y=df.SHOT_DIST, c = 'b')
plt.title('Touch Time Vs Shot Distance')
plt.xlabel('Touch Time')
plt.ylabel('Shot Distance')


# In[ ]:


df[df.TOUCH_TIME <-80]


# In[ ]:


df.drop([5574], axis = 0, inplace = True)


# In[ ]:


plt.scatter(x=df.TOUCH_TIME,y=df.SHOT_DIST, c = 'b')
plt.title('Touch Time Vs Shot Distance')
plt.xlabel('Touch Time')
plt.ylabel('Shot Distance')


# In[ ]:


df[df.TOUCH_TIME <-0].head()


# In[ ]:


df.drop(df[df.TOUCH_TIME <-0].index, inplace = True)


# In[ ]:


plt.scatter(x=df.TOUCH_TIME,y=df.SHOT_DIST, c = 'b')
plt.title('Touch Time Vs Shot Distance')
plt.xlabel('Touch Time')
plt.ylabel('Shot Distance')


# In[ ]:


X = df.ix[:,[9,15]]
Y = df.SHOT_DIST
x = sm.add_constant(X)


# In[ ]:


Reg = sm.OLS(Y,x)
FitReg = Reg.fit()
FitReg.summary()


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(df[['SHOT_DIST','CLOSE_DEF_DIST','TOUCH_TIME','DRIBBLES']],df.SHOT_RESULT,test_size = 0.3)


# In[ ]:


model = LogisticRegression()
model.fit(X_train,y_train)


# In[ ]:


model.predict(X_test)


# In[ ]:


model.score(X_test,y_test)


# In[ ]:


model.predict_proba(X_test)


# In[ ]:


df.columns


# In[ ]:


df.head(1)


# In[ ]:


df.SHOT_RESULT


# In[ ]:




