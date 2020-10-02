#!/usr/bin/env python
# coding: utf-8

# # Data Exploration - Disney Movies 

# #### Source for the dataset - Kaggle

# ### Setting up

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Read the dataset
df=pd.read_csv('/kaggle/input/disney-movies/disney_movies.csv', parse_dates=['release_date'], index_col=['release_date'])
df.head()


# In[ ]:


#Info
display(df.shape)
display(df.info())


# In[ ]:


df.describe()


# ### Analysing the Data

# #### How movie genre impacted the earnings

# In[ ]:


earnings_genre = df[['total_gross', 'inflation_adjusted_gross', 'genre']].groupby(['genre']).mean().sort_values(by='inflation_adjusted_gross',ascending=False)
display(earnings_genre)
display(earnings_genre.plot(kind='bar'))


# In[ ]:


plot1 = sns.swarmplot(x = "genre", y = "total_gross", data = df, size = 5)


# In[ ]:


plot2 = sns.swarmplot(x = 'genre', y = "inflation_adjusted_gross", data = df, size = 5)


# #### How mpaa rating impacted the earnings

# In[ ]:


#Effect of mpaa_rating on earnings
earnings_rating = df[['mpaa_rating', 'total_gross', 'inflation_adjusted_gross']].groupby(['mpaa_rating']).mean().sort_values(by='inflation_adjusted_gross',ascending=False)
display(earnings_rating)
display(earnings_rating.plot(kind='bar'))


# In[ ]:


plot3 = sns.swarmplot(x = "mpaa_rating", y = "total_gross", data = df, size = 5)
plot3


# In[ ]:


plot4 = sns.swarmplot(x='mpaa_rating', y='inflation_adjusted_gross', data=df, size=5)
plot4


# #### How earnings changed with time

# In[ ]:


df.index


# In[ ]:


dates = df.index.values
inflation_adjusted = df['inflation_adjusted_gross']
gross = df['total_gross']


# In[ ]:


plt.rcParams['figure.figsize']=[16,8]
print(plt.scatter(x=dates, y=inflation_adjusted, color='green', alpha=0.9))
print(plt.scatter(x=dates, y=gross, color='lightblue', alpha = 0.5))


# In[ ]:


sns.lineplot(x=dates, y=inflation_adjusted)
sns.lineplot(x=dates, y=gross, alpha = 0.6)


# ## Predicting Earnings 

# In[ ]:


x = df.reset_index()


# In[ ]:


y = x.inflation_adjusted_gross


# In[ ]:


x = x.drop(['total_gross','movie_title','inflation_adjusted_gross'], axis = 1)
x.head()


# In[ ]:


g = pd.get_dummies(x.genre)
g.head()


# In[ ]:


r = pd.get_dummies(x.mpaa_rating)
r.head()


# In[ ]:


x = pd.concat([x,g,r], axis = 1)
x.head()


# In[ ]:


x = x.drop(['genre','Western','R','mpaa_rating','release_date'], axis=1)
x.head()


# In[ ]:


from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression


# In[ ]:


xTrain,xTest,yTrain,yTest=tts(x,y,test_size=0.3)


# In[ ]:


Linreg=LinearRegression()
Linreg.fit(xTrain,yTrain)
y_pred=Linreg.predict(xTest)


# In[ ]:


p = pd.DataFrame(y_pred, columns=['Actual'])
p1 = np.asarray(yTest)
p2 = pd.DataFrame(p1, columns=['Pred'])
cm = pd.concat([p,p2], axis=1)
cm


# In[ ]:


from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
#Linear Regression Training MSE
print((mean_squared_error(np.square(Linreg.predict(xTrain)),np.square(yTrain))))
#Linear Regression Testing MSE 
print((mean_squared_error(np.square(Linreg.predict(xTest)),np.square(yTest))))
#Linear Regression Mean Absolute Error Training
print((mean_absolute_error(np.square(Linreg.predict(xTrain)),np.square(yTrain))))
#Linear Regression Mean Absolute Error Testing
print((mean_absolute_error(np.square(Linreg.predict(xTest)),np.square(yTest))))


# In[ ]:


sns.lineplot(x=cm.index.values, y=cm.Actual, color='purple')
sns.lineplot(x=cm.index.values, y=cm.Pred, alpha = 0.6)

