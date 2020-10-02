#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df = pd.read_csv('/kaggle/input/brasilian-houses-to-rent/houses_to_rent.csv',index_col=['Unnamed: 0'])
df.head()


# In[ ]:


df = pd.read_csv('/kaggle/input/brasilian-houses-to-rent/houses_to_rent.csv',index_col=['Unnamed: 0'])
df.drop_duplicates(inplace=True)
df['floor'] = df['floor'].replace('-',np.nan)
df['floor']= df['floor'].fillna(df['floor'].median()).astype('int')
df.head()


# In[ ]:


df['animal'] = df['animal'].map({
    'acept':'Yes',
    'not acept':'No'
})
df['furniture']= df['furniture'].map({
    'not furnished':'unfurnished',
    'furnished':'furnished'
})


# In[ ]:


import re
df['total'] = df['total'].map(lambda x: re.sub(r'\D+', '', x))
df['hoa'] = df['hoa'].map(lambda x: re.sub(r'\D+', '', x))
df['rent amount'] = df['rent amount'].map(lambda x: re.sub(r'\D+', '', x))
df['fire insurance'] = df['fire insurance'].map(lambda x: re.sub(r'\D+', '', x))
df['property tax'] = df['property tax'].map(lambda x: re.sub(r'\D+', '', x))


# In[ ]:


df


# In[ ]:


df.isnull().sum()
#df['hoa'] = df['hoa'].fillna(df['hoa'].median())
#df['property tax'] = df['property tax'].fillna(df['property tax'].median())


# In[ ]:


la = LabelEncoder()
df['animal'] = la.fit_transform(df['animal'])
df['furniture'] =la.fit_transform(df['furniture'])


# In[ ]:


new_df =df.copy()


# In[ ]:


new_df['animal'].value_counts()


# In[ ]:


size = [4535,1347]
labels = ['yes', 'no']
colors = ['cyan', 'lightblue']
explode = [0, 0.1]

plt.pie(size, colors = colors, labels = labels, shadow = True, explode = explode, autopct = '%.2f%%')
plt.title('A Pie Chart representing the animal pet', fontsize = 20)
plt.legend()
plt.show()


# In[ ]:


df.describe()


# In[ ]:


corrmat = df.corr()
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corrmat, vmax=.8, annot=True);


# In[ ]:


df['total'] = df['total'].astype('float')


# In[ ]:


from scipy import stats
from scipy.stats import norm, skew #for some statistics

sns.distplot(df['total'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(df['total'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

fig = plt.figure()
res = stats.probplot(df['total'], plot=plt)
plt.show()


# In[ ]:


df.total = np.log1p(df.total)
y = df.total


# In[ ]:


df


# In[ ]:


plt.scatter(y =df.total,x = df.area,c = 'black')
plt.show()


# In[ ]:


df.columns


# In[ ]:


f, axes = plt.subplots(1, 2,figsize=(15,5))
sns.boxplot(x=df['bathroom'],y=df['total'], ax=axes[0])
sns.boxplot(x=df['floor'],y=df['total'], ax=axes[1])
sns.despine(left=True, bottom=True)
axes[0].set(xlabel='bathroom', ylabel='Price')
axes[0].yaxis.tick_left()
axes[1].yaxis.set_label_position("right")
axes[1].yaxis.tick_right()
axes[1].set(xlabel='Floor', ylabel='rent amount')

f, axe = plt.subplots(1, 1,figsize=(12.18,5))
sns.despine(left=True, bottom=True)
sns.boxplot(x=df['bathroom'],y=df['total'], ax=axe)
axe.yaxis.tick_left()
axe.set(xlabel='bathroom / bathroom', ylabel='Price')


# In[ ]:


new_df['hoa'] = pd.to_numeric(new_df['hoa'])
new_df['hoa'] = new_df['hoa'].astype('float')
new_df['rent amount'] = new_df['rent amount'].astype('float')
new_df['property tax'] = pd.to_numeric(new_df['property tax'])
new_df['property tax'] = new_df['property tax'].astype('float')
new_df['fire insurance'] = new_df['fire insurance'].astype('float')
new_df['total'] = new_df['total'].astype('float')


# In[ ]:


model = linear_model.LinearRegression()


# In[ ]:


new_df['hoa'] = new_df['hoa'].fillna(0)
new_df['property tax'] = new_df['property tax'].fillna(0)


# In[ ]:


x = new_df.drop('total',axis='columns')
y = new_df.total


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=0)


# In[ ]:


model.fit(X_train,y_train)


# In[ ]:


model.score(X_train,y_train)


# In[ ]:


y_predicted = model.predict(X_test)


# In[ ]:


# finding the mean_squared error
mse = mean_squared_error(y_test, y_predicted)
print("RMSE Error:", np.sqrt(mse))

# finding the r2 score or the variance
r2 = r2_score(y_test, y_predicted)
print("R2 Score:", r2)


# In[ ]:


new_df['animal'].value_counts()


# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state =1 )
model = linear_model.LinearRegression()
parameters = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
grid = GridSearchCV(model,parameters, cv=None)
grid.fit(X_train, y_train)

print("Residual sum of squares: %.2f"
              % np.mean((grid.predict(X_test) - y_test) ** 2))


# In[ ]:


print ("r2 / variance : ",grid.best_score_)


# In[ ]:




