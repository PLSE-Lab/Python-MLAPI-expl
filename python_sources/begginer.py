#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import ShuffleSplit,train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from scipy import stats
from scipy.stats import norm, skew
import folium
from mpl_toolkits.basemap import Basemap


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
df=pd.read_csv('../input/revised-notice-of-property-value-rnopv.csv')
df.head()


# In[ ]:


df_na= (df.isnull().sum() / len(df)) * 100
df_na = df_na.drop(df_na[df_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing %' :df_na})
print(df.columns.values)
missing_data.head(30)


# In[ ]:


df1=df.drop(['Country ','EASE','RC4','RC3','RC5'],axis=1)


# In[ ]:


df_na= (df1.isnull().sum() / len(df)) * 100
df_na = df_na.drop(df_na[df_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing %' :df_na})
missing_data.head(20)


# In[ ]:


df2= df1[np.isfinite(df['BBL'])]


# In[ ]:


df_na= (df2.isnull().sum() / len(df)) * 100
df_na = df_na.drop(df_na[df_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing %' :df_na})
missing_data.head(30)


# In[ ]:



df2=df2.dropna(axis=1,how='any')


# In[ ]:


print(df2.isnull().values.any())


# In[ ]:


from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

x = df2['MAILED DATE']
data = [go.Histogram(x=x)]

iplot(data, filename='basic histogram')


# Looks like 'MAILED DATE" is in two months-it should not have any influence on prices

# In[ ]:


df2=df2.drop(['MAILED DATE','Address 1','Address 2 ','Address 3 ','City, State, Zip ','NAME  ','NTA','RC2'],axis=1)
df2.head()
#data like Adress, zip code are impossible to calculate into pcize


# In[ ]:


x = df2['BLD Class']
data = [go.Histogram(x=x)]

iplot(data, filename='basic histogram')#lots of signs


# In[ ]:


x = df2['RC 1']
data = [go.Histogram(x=x)]

iplot(data, filename='basic histogram')


# In[ ]:


x = df2['Borough']
data = [go.Histogram(x=x)]

iplot(data, filename='basic histogram')#few parameters


# In[ ]:


df2.head()


# In[ ]:


df2 = pd.get_dummies(df2, prefix='BLD Class_', columns=['BLD Class'])
df2 = pd.get_dummies(df2, prefix='RC 1', columns=['RC 1'])
df2 = pd.get_dummies(df2, prefix='Borough_', columns=['Borough'])


# In[ ]:


df2.head()


# In[ ]:


import matplotlib.pyplot as plt

plt.matshow(df2.corr())
plt.colorbar()
plt.show()
print(df.columns.values)


# In[ ]:


#take transaction columns to other dataset
transactions=df2[['ORIGINAL MARKET VALUE','ORIGINAL ASSESSED VALUE','ORIGINAL EXEMPTION',
                           'ORIGINAL TRANSITIONAL  ASSESSED VALUE ','ORIGINAL TRANSITIONAL EXEMPTION',
                           'ORIGINAL TAXABLE VALUE','REVISED MARKET VALUE','REVISED ASSESSED VALUE',
                           'REVISED  EXEMPTION','REVISED TRANSITIONAL ASSESSED VALUE','REVISED TRANSITIONAL EXEMPTION',
                            'REVISED TAXABLE VALUE']].copy()

plt.matshow(transactions.corr())
plt.colorbar()
plt.show()


# In[ ]:


df3=df2.drop(['ORIGINAL ASSESSED VALUE','ORIGINAL EXEMPTION',
                'ORIGINAL TRANSITIONAL  ASSESSED VALUE ','ORIGINAL TRANSITIONAL EXEMPTION',
                'ORIGINAL TAXABLE VALUE','REVISED MARKET VALUE','REVISED ASSESSED VALUE',
                'REVISED  EXEMPTION','REVISED TRANSITIONAL ASSESSED VALUE',
              'REVISED TRANSITIONAL EXEMPTION','REVISED TAXABLE VALUE'],axis=1)
df3.head()


# Let find out if we can predict Original Market value only using "building parameters", what means without othect parameters corelated with taxt, cost etc
# df5 and X1 are corelated withou other cost parameters
# df4 and X are aditionaly corelated with cost parametes

# In[ ]:


labels1=df3.drop(['ORIGINAL MARKET VALUE'],axis=1)
X1=labels1.loc[:,:].values
y=df2.loc[:,'ORIGINAL MARKET VALUE'].values

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_classification
print(labels.columns.values)
print(labels.columns.values.shape)
print(X.shape)


# Building corelation WITHOUT other cost parameters-based on X1

# In[ ]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X1, y, train_size=0.8 , random_state=100)
scaler = StandardScaler().fit(X_train)
X_train_scaled = pd.DataFrame(scaler.transform(X_train))
X_test_scaled = pd.DataFrame(scaler.transform(X_test))


# In[ ]:


from sklearn.metrics import r2_score
reg = RandomForestRegressor(n_estimators=45,random_state=0, n_jobs=-1)
model = reg.fit(X_train_scaled, y_train)
model.score(X_train_scaled, y_train)
y_pred= model.predict(X_test_scaled)
print("R2 for test",r2_score(y_test,y_pred))


# Very poor result. 
# Now the same calculations with other costs.
# We have to remamber there are lots of 0 values, which we have to remove

# In[ ]:


df2.astype(bool).sum(axis=0)
df2 = df2.rename(columns={'ORIGINAL MARKET VALUE': 'VALUE'})
df2 = df2.rename(columns={'ORIGINAL TRANSITIONAL EXEMPTION': 'OTE'})
df2 = df2.rename(columns={'ORIGINAL TAXABLE VALUE': 'OTV'})

df2 = df2[df2.VALUE != 0]
df2 = df2[df2.OTE != 0]
df2 = df2[df2.OTV != 0]
df2.astype(bool).sum(axis=0)


# In[ ]:


labels=df2.drop(['VALUE'],axis=1)
X=labels.loc[:,:].values
y=df2.loc[:,'VALUE'].values


# In[ ]:



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8 , random_state=100)
scaler = StandardScaler().fit(X_train)
X_train_scaled = pd.DataFrame(scaler.transform(X_train))
X_test_scaled = pd.DataFrame(scaler.transform(X_test))


# In[ ]:


from sklearn.metrics import r2_score
reg = RandomForestRegressor(n_estimators=45,random_state=0, n_jobs=-1)
model = reg.fit(X_train_scaled, y_train)
model.score(X_train_scaled, y_train)
y_pred= model.predict(X_test_scaled)
print("R2 for test",r2_score(y_test,y_pred))


# Much better result. It means the real valu can not predict based only on presented building parameters, there are other important hiden parameters

# In[ ]:


feature_importances = pd.DataFrame(reg.feature_importances_,
                                   index = labels.columns.values,
                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances


# In[ ]:


imp=feature_importances[feature_importances['importance']>0.001]
f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=imp.index, y=imp.importance,palette="deep")
plt.xlabel('Features', fontsize=15)
plt.ylabel('Feature importance', fontsize=15)
plt.title('Feature importance using Random Forest model', fontsize=15)


# The highest influence has original assessed value and next revised assessed value, revised market value. The other values are not so much important
