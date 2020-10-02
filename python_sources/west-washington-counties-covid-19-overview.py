#!/usr/bin/env python
# coding: utf-8

# # **West Washington counties: King, Snohomish, Pierce**    
# 
# By. MM Figueroa Lopez   
# Written on. April 1, 2020   
# 

# **The overall US picture** 

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import pandas as pd
import numpy as np
df = pd.read_csv('../input/us-counties-covid-19-dataset/us-counties.csv')

data=df.drop(['fips'], axis = 1) 
data


# As of **March 31, 2020**, the death toll is highest in New York, and followed by Washington State.   

# In[ ]:


plt.figure(figsize=(12,8)) # Figure size
data.groupby("state")['deaths'].max().plot(kind='bar', color='darkred')


# **Washington state** overall

# In[ ]:


WA=data.loc[data['state']== 'Washington']
WA


# In[ ]:


WA.sort_values(by=['cases'], ascending=False)


# In[ ]:


WA.sort_values(by=['deaths'], ascending=False)


# King county continues to lead in case and death totals.

# ** Washinton state counties**

# In[ ]:


plt.figure(figsize=(12,8)) # Figure size
WA.groupby("county")['cases'].max().plot(kind='bar', color='olive')


# The graph above shows that King and Snohomish have the highest rates of cases.

# In[ ]:


plt.figure(figsize=(12,8)) # Figure size
WA.groupby("county")['deaths'].max().plot(kind='bar', color='darkgray')


# The above greaph shows how King county surpasses other counties in WA in total deaths related to COVID19.

# In[ ]:


plt.figure(figsize=(16,11))
sns.lineplot(x="date", y="deaths", hue="county",data=WA)
plt.xticks(WA.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees
plt.show()


# The above data shows that King County began reporting COVID-19 deaths before most of Washington.

# In[ ]:


plt.figure(figsize=(16,11))
sns.lineplot(x="date", y="cases", hue="county",data=WA)
plt.xticks(WA.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees
plt.show()


# The above shows a pattern on case rises: First, King county, then Snohomish county, and then Pierce county. These counties are neighbors.  
# 

# * **King, Pirece, and Snohomish Counties**

#  **King County** 

# In[ ]:


King=WA.loc[WA['county']== 'King']
King


# In[ ]:


plt.figure(figsize=(16,9)) # Figure size
sns.lineplot(x='date', y='cases', data=King, marker='o', color='peru') 
plt.title('Cases in King county') # Title
plt.xticks(King.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees
plt.show()


# In[ ]:


plt.figure(figsize=(16,9)) # Figure size
sns.lineplot(x='date', y='deaths', data=King, marker='o', color='steelblue') 
plt.title('Reported deaths in King county') # Title
plt.xticks(King.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees
plt.show()


# In cases and deaths, there seems to be a curving for King county. Perhaps, King county reached its peak. But this may not be truw for the rest of the state.   

#  **Pierce county**

# In[ ]:


Prc=WA.loc[WA['county']== 'Pierce']
Prc


# In[ ]:


plt.figure(figsize=(16,9)) # Figure size
sns.lineplot(x='date', y='cases', data=Prc, marker='o', color='tomato') 
plt.title('Cases in Pierce county') # Title
plt.xticks(Prc.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees
plt.show()


# In[ ]:


plt.figure(figsize=(16,9)) # Figure size
sns.lineplot(x='date', y='deaths', data=Prc, marker='o', color='darkkhaki') 
plt.title('Deaths in Pierce county') # Title
plt.xticks(Prc.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees
plt.show()


# It does not seem clear if Pierce county cases of COVID-19 are peaking, but deaths seem supressed in this county. 

#  **Snohomish county**

# In[ ]:


Sno=WA.loc[WA['county']== 'Snohomish']
Sno


# In[ ]:


plt.figure(figsize=(16,9)) # Figure size
sns.lineplot(x='date', y='cases', data=Sno, marker='o', color='indigo') 
plt.title('Cases in Snohomish county') # Title
plt.xticks(Sno.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees
plt.show()


# There could be a curve at the top of the graph, but it may still be too early to determine if Snohomish county is peaking in COVID-19 cases.

# In[ ]:


plt.figure(figsize=(16,9)) # Figure size
sns.lineplot(x='date', y='deaths', data=Sno, marker='o', color='dimgrey') 
plt.title('Deaths in Snohomish county') # Title
plt.xticks(Sno.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees
plt.show()


# Comparing King, Snohomish, and Piecr counties.

# In[ ]:


#concat dfs

DF1 = pd.concat([King,Prc, Sno])
DF1


# In[ ]:


plt.figure(figsize=(16,11))
sns.lineplot(x="cases", y="deaths", hue="county",data=DF1)
plt.title('Death across counties') # Title


# In[ ]:


plt.figure(figsize=(16,11))
sns.lineplot(x="date", y="cases", hue="county",data=DF1)
plt.title('Cases across counties') # Title
plt.xticks(DF1.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees
plt.show()


# Correlation:

# In[ ]:


DF1.corr().style.background_gradient(cmap='magma')


# There may be a correlation in cases-deaths across the three neighboring counties.

# Regressions/prediction

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split 


# King county

# In[ ]:


X = King['cases'].values.reshape(-1,1)
y = King['deaths'].values.reshape(-1,1)


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


plt.scatter(X_test, y_test,  color='palevioletred')
plt.plot(X_test, y_pred, color='black', linewidth=1)
plt.show()


# There seems to be a linear regression fits in the case of King County, WA

# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# The model and predictions above are not ideal for conclusive results or decision-making, but still informative.   

# In[ ]:


X1 = Sno['cases'].values.reshape(-1,1)
y1 = Sno['deaths'].values.reshape(-1,1)


# In[ ]:


X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=1)


# Different than in King county's linear regression, here I used 1 random state

# In[ ]:


regressor1 = LinearRegression()  
regressor1.fit(X_train1, y_train1) #training the algorithm


# In[ ]:


y_pred2 = regressor1.predict(X_test1)


# In[ ]:


preds2 = pd.DataFrame({'Actual': y_test1.flatten(), 'Predicted': y_pred2.flatten()})
preds2


# In[ ]:


plt.scatter(X_test1, y_test1,  color='rosybrown')
plt.plot(X_test1, y_pred2, color='black', linewidth=1)
plt.show()


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test1, y_pred2))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test1, y_pred2))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test1, y_pred2)))


# The above linear regression and predictions for Snohomish county data seems to fit very well.

# **Pierce County**

# In[ ]:


X2 = Prc['cases'].values.reshape(-1,1)
y2 = Prc['deaths'].values.reshape(-1,1)


# In[ ]:


X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.1, random_state=0)


# Like in the case of King county, I used random state 0 because 1 did not fit, like in Snohomish county data. I also decreased the test size. Pierce county has less data.

# In[ ]:


regressor2 = LinearRegression()  
regressor2.fit(X_train2, y_train2) #training the algorithm


# In[ ]:


y_pred3 = regressor2.predict(X_test2)


# In[ ]:


preds3 = pd.DataFrame({'Actual': y_test2.flatten(), 'Predicted': y_pred3.flatten()})
preds3


# In[ ]:


plt.scatter(X_test2, y_test2,  color='goldenrod')
plt.plot(X_test2, y_pred3, color='black', linewidth=1)
plt.show()

