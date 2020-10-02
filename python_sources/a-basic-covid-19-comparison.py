#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# Across the country

# In[ ]:


import pandas as pd
import numpy as np
df = pd.read_csv('../input/us-counties-covid-19-dataset/us-counties.csv')

data=df.drop(['fips'], axis = 1) 
data


# In[ ]:


data.sort_values(by=['cases'], ascending=False)


# In[ ]:


data.sort_values(by=['deaths'], ascending=False)


# In[ ]:


plt.figure(figsize=(12,8)) # Figure size
data.groupby("state")['cases'].max().plot(kind='bar', color='olivedrab')


# In[ ]:


data.plot.line()


# In[ ]:


plt.figure() # for defining figure sizes
data.plot(x='state', y='deaths', figsize=(12,8), color='goldenrod')


# In[ ]:


WA=data.loc[data['state']== 'Washington']
WA


# In[ ]:


WA.groupby('county').plot(x='date', y='deaths')


# In[ ]:


NY=data.loc[data['state']== 'New York']
NY


# In[ ]:


DF1 = pd.concat([WA,NY])
DF1


# In[ ]:


DFGroup = DF1.groupby(['cases'])

DFGPlot = DFGroup.sum().unstack().plot(kind='bar', figsize=(15,10))


# In[ ]:


plt.style.use('ggplot')
WA.plot(kind='bar', figsize=(16,9))
plt.ylabel('total')


# In[ ]:


DF1.plot('state',['deaths','cases'],kind = 'line', figsize=(16,9))


# In[ ]:


plt.figure(figsize=(16,9)) # Figure size
sns.lineplot(x='date', y='cases', data=WA, marker='o', color='lightseagreen') 
plt.title('Cases in Washington') # Title
plt.xticks(DF1.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees
plt.show()


# In[ ]:


plt.figure(figsize=(16,9)) # Figure size
sns.lineplot(x='date', y='cases', data=NY, marker='o', color='darkmagenta') 
plt.title('Cases in New York state') # Title
plt.xticks(DF1.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees
plt.show()


# In[ ]:


plt.figure(figsize=(16,9)) # Figure size
sns.lineplot(x='date', y='cases', data=DF1, marker='o', color='royalblue') 
plt.title('Cases in the states of WA and NY') # Title
plt.xticks(DF1.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees
plt.show()


# Correlation:

# In[ ]:


DF1.corr().style.background_gradient(cmap='magma')


# In[ ]:


ax = WA.plot()
NY.plot(ax=ax)


# In[ ]:


plt.figure(figsize=(16,11))
sns.lineplot(x="cases", y="deaths", hue="county",data=WA)


# In[ ]:


plt.figure(figsize=(16,11))
sns.lineplot(x="cases", y="deaths", hue="state",data=DF1)


# Regressions

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split 


# In[ ]:


X = WA['cases'].values.reshape(-1,1)
y = WA['deaths'].values.reshape(-1,1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm


# In[ ]:


print(regressor.intercept_)
print(regressor.coef_)


# In[ ]:


y_pred = regressor.predict(X_test)


# In[ ]:


preds1 = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
preds1


# In[ ]:


plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:


LogReg = LogisticRegression()  
LogReg.fit(X_train, y_train) #training the algorithm


# In[ ]:


y_pred = LogReg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(LogReg.score(X_test, y_test)))


# In[ ]:


accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy_percentage = 100 * accuracy
accuracy_percentage

