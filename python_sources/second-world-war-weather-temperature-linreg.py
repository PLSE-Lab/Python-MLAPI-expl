#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../input/weatherww2/Summary of Weather.csv')
df.head()


# In[ ]:


df = df[['Date', 'Precip', 'MaxTemp', 'MinTemp', 'MeanTemp', 'Snowfall', 'YR', 'MO']]
df.head()


# In[ ]:


corr = df.corr()
sns.heatmap(corr, cmap='YlGnBu', square=True)


# In[ ]:


df.info()


# In[ ]:


sns.regplot(x='MinTemp', 
            y = 'MaxTemp', 
            data = df, 
            scatter_kws = {'color': 'purple', 'alpha': 0.3}, 
            line_kws = {'color': 'teal', 'alpha': 0.3, 'lw':6})
sns.set_style('whitegrid')
plt.ylim(0,)


# In[ ]:


sns.residplot(df['MinTemp'], 
              df['MaxTemp'], 
              scatter_kws = {'color': 'Maroon', 'alpha': 0.3})


# In[ ]:


sns.jointplot(x ='MinTemp', y='MaxTemp', data = df,
              color= 'Peru', 
              alpha= 0.3)


# In[ ]:


from scipy import stats
pearson_coef, p_value = stats.pearsonr(df['MinTemp'], df['MaxTemp'])
print(pearson_coef, p_value)


# In[ ]:


df.dropna(inplace=True)


# In[ ]:


y_data = df['MaxTemp'].values.reshape(-1, 1)
x_data = df['MinTemp'].values.reshape(-1, 1)


# In[ ]:


from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=1)


print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


# In[ ]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train, y_train)


# In[ ]:


lm.score(x_train, y_train)


# In[ ]:


lm.score(x_test,y_test)


# In[ ]:


from sklearn.model_selection import cross_val_score
rcross = cross_val_score(lm, x_data, y_data, cv=10)
rcross


# In[ ]:


print("The mean of the folds are", rcross.mean(), "and the standard deviation is" , rcross.std())


# In[ ]:


from sklearn.model_selection import cross_val_predict
yhat = cross_val_predict(lm, x_data, y_data, cv=10)
yhat[0:5]


# In[ ]:


ax1 = sns.distplot(y_data, hist=False, color="LightCoral", label="Max Temperature")
sns.distplot(yhat, hist=False, color="LightSkyBlue", label="Fitted Max Temperature" , ax=ax1)

plt.show()


# In[ ]:


inpMaxTemp = [[21.87]]
yhat2 = lm.predict(inpMaxTemp)
print('The predicted maximum tempreature based on the given minimum temperature is:', yhat2)


# In[ ]:




