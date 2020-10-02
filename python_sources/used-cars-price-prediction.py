#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[ ]:


df = pd.read_csv("../input/craigslist-carstrucks-data/vehicles.csv")


# In[ ]:


df.shape


# In[ ]:


df.columns


# In[ ]:


df.head()


# In[ ]:


df.nunique(axis=0)


# In[ ]:


df.describe().apply(lambda s: s.apply(lambda x: format(x, 'f')))


# In[ ]:


NA_val = df.isna().sum()
def na_filter(na, threshold = .4):
    col_pass = []
    for i in na.keys():
        if na[i]/df.shape[0]<threshold:
            col_pass.append(i)
    return col_pass


# In[ ]:


df_cleaned = df[na_filter(NA_val)]
df_cleaned.shape


# In[ ]:


NA_val = df.isna().sum()


# In[ ]:


df_cleaned = df_cleaned[df_cleaned['price'].between(999.99, 250000)] # Computing IQR
Q1 = df_cleaned['price'].quantile(0.25)
Q3 = df_cleaned['price'].quantile(0.75)
IQR = Q3 - Q1
# Filtering Values between Q1-1.5IQR and Q3+1.5IQR
df_filtered = df_cleaned.query('(@Q1 - 1.5 * @IQR) <= price <= (@Q3 + 1.5 * @IQR)')
df_filtered.boxplot('price')


# In[ ]:


df_filtered.describe().apply(lambda s: s.apply(lambda x: format(x, 'f')))


# In[ ]:


df_filtered = df_filtered[df_filtered['year'].between(1900, 2020)]# = 140000 + 1.5 * (140000-52379)
df_filtered = df_filtered[df_filtered['odometer'].between(0, 271431.5)]


# In[ ]:


df_final = df_filtered.copy().drop(['id','url','region_url','image_url','region','description','model','state','paint_color'], axis=1)
df_final.shape


# In[ ]:


import matplotlib.pylab as plt
import seaborn as sns
# calculate correlation matrix
corr = df_final.corr()# plot the heatmap
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))


# In[ ]:


df_final['manufacturer'].value_counts().plot(kind='bar')


# In[ ]:


df_cleaned['type'].value_counts().plot(kind='bar')


# In[ ]:


df_final = pd.get_dummies(df_final, drop_first=True)
df_final.shape
df_final


# In[ ]:


from sklearn.preprocessing import StandardScaler
X_head = df_final.iloc[:, df_final.columns != 'price']
X = df_final.loc[:, df_final.columns != 'price']
y = df_final['price']
X
X.isnull().sum();X.drop(['lat','long'],axis=1,inplace = True)
X = StandardScaler().fit_transform(X)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)
model = RandomForestRegressor(random_state=1)


# In[ ]:


X_train


# In[ ]:


y_train


# In[ ]:


model.fit(X_train, y_train)


# In[ ]:


pred = model.predict(X_test)


# In[ ]:


print(mae(y_test, pred))


# In[ ]:


print(model.score(X_test,y_test))

