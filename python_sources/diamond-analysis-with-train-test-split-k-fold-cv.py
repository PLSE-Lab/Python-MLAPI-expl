#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
df = pd.read_csv("../input/diamonds.csv")
df.drop(['Unnamed: 0'],axis=1,inplace=True)
print (df.head())


# In[ ]:


df.describe()


# #### No null values in enitire Data Frame, but columns x,y,z are having zero(0) values as minimum, which makes no sence. So lets drop those records

# In[ ]:


df.shape


# In[ ]:


df = df[(df['x']!=0) & (df['y']!=0) & (df['z']!=0)]


# In[ ]:


df.shape


# In[ ]:


df['clarity'].value_counts()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks", color_codes=True)
sns.pairplot(df, vars=['carat','depth', 'table','price'])
plt.show()


# #### price has linear relationship with carat(weight of the diamond)

# In[ ]:


sns.distplot(df.carat)
plt.show()


# In[ ]:


sns.countplot(x=df.cut)
plt.show()


# In[ ]:


sns.countplot(x=df.color)
plt.show()


# In[ ]:


sns.countplot(x=df.clarity)
plt.show()


# In[ ]:


#sns.boxplot(x=df.drop(['carat'],axis=1),orient='v')
sns.boxplot(x=df['carat'],orient='v')
plt.show()


# In[ ]:


diamond_cut = {'Fair':0,
               'Good':1,
               'Very Good':2, 
               'Premium':3,
               'Ideal':4}

diamond_color = {'J':0,
                 'I':1, 
                 'H':2,
                 'G':3,
                 'F':4,
                 'E':5,
                 'D':6}

diamond_clarity = {'I1':0,
                   'SI2':1,
                   'SI1':2,
                   'VS2':3,
                   'VS1':4,
                   'VVS2':5,
                   'VVS1':6,
                   'IF':7}


# In[ ]:


df['cut'] = df['cut'].map(diamond_cut)
df['color'] = df['color'].map(diamond_color)
df['clarity'] = df['clarity'].map(diamond_clarity)


# In[ ]:


df.head(20)


# In[ ]:


df.describe()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
X = df.drop(['price'],axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[ ]:


from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

models = []
models.append(('LR', LinearRegression()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('DT', DecisionTreeRegressor()))
models.append(('SVM', SVR()))
models.append(('RF',RandomForestRegressor()))
# evaluate each model in turn
MSE = []
r2 = []
names = []
score = []
for name, model in models:
    Algo = model.fit(X_train,y_train)
    y_pred = Algo.predict(X_test)
    MSE.append(mean_squared_error(y_test,y_pred))
    r2.append(r2_score(y_test,y_pred))
    names.append(name)


# In[ ]:


df_TT = pd.DataFrame({'Name':names,'r2_score':r2,'MSE':MSE})
ax = sns.barplot(x="Name", y="r2_score", data=df_TT)
plt.show()


# In[ ]:


from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

models = []
models.append(('LR', LinearRegression()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('DT', DecisionTreeRegressor()))
models.append(('SVM', SVR()))
models.append(('RF',RandomForestRegressor()))
# evaluate each model in turn
MSE = []
names = []
r2_score = []
scoring = 'r2'
for name, model in models:
    Algo = model.fit(X_train,y_train)
    r2 =cross_val_score(Algo, X_train, y_train, cv=3, scoring=scoring)
    y_pred = Algo.predict(X_test)
    r2_score.append(np.mean(r2))
    MSE.append(mean_squared_error(y_test,y_pred))
    names.append(name)


# In[ ]:


df_cv = pd.DataFrame({'Name':names,'r2_score':r2_score,'MSE':MSE})
print (df_cv)


# In[ ]:


ax = sns.barplot(x="Name", y="r2_score", data=df_cv)
plt.show()


# In[ ]:


df_TT


# In[ ]:


df_cv


# Looking at above dataframes of r2_score under Train Test Spli & Cross Validation. **Random Forest under Cross validation** has better score to move forward with.
# 
# This is my first Kernel in Kaggle. So please suggest points for improvement if any so. I will be happy hear.

# In[ ]:




