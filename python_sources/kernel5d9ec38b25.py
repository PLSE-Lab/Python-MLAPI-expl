#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data = pd.read_csv('/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv')


# In[ ]:


data.head(10)


# In[ ]:


data.columns


# In[ ]:


data.describe()


# In[ ]:


data = data.drop(['Date'], axis=1)


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.RainToday


# In[ ]:


data['RainToday']


# In[ ]:


numerical_features = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation','Sunshine','WindGustSpeed', 
        'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am','Temp3pm']


# In[ ]:


len(numerical_features)


# In[ ]:


numeric_data = data[numerical_features]
numeric_data.info()


# In[ ]:


numeric_data


# In[ ]:


numeric_data['MinTemp'] = numeric_data['MinTemp'].fillna(12.186)


# In[ ]:


sum(numeric_data['MinTemp'])/len(numeric_data['MinTemp'])


# In[ ]:


np.mean(numeric_data['MinTemp'])


# In[ ]:


import seaborn as sns
from matplotlib.pyplot import plot as plt


# In[ ]:


numeric_data['MinTemp'].plot(figsize=(20,12)).line()


# In[ ]:


numeric_data['MinTemp'].values.sort()


# In[ ]:


sns.distplot(numeric_data['MinTemp'])


# In[ ]:


for i in numerical_features:
    numeric_data[i] = numeric_data[i].fillna(np.nanmean(numeric_data[i]))


# In[ ]:


list(enumerate(numerical_features))


# In[ ]:


import matplotlib

fig, axes = matplotlib.pyplot.subplots(16, figsize=(12,64))

for ix,el in enumerate(numerical_features):
    sns.distplot(numeric_data[el], ax=axes[ix])


# In[ ]:


numeric_data['Cloud3pm'].value_counts().plot.bar()


# In[ ]:


data.head()


# In[ ]:


y = data['RainTomorrow']


# In[ ]:


y.head()


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logres = LogisticRegression()


# In[ ]:


logres.fit(X=numeric_data, y=y)


# In[ ]:


logres.coef_


# In[ ]:


for ix, el in enumerate(numerical_features):
    print(el, logres.coef_[0][ix])


# In[ ]:


y_pred = logres.predict(numeric_data)


# In[ ]:


y.values


# In[ ]:


y_pred


# In[ ]:


assert len(y.values) == len(y_pred)


# In[ ]:


errors = 0
for i in range(len(y_pred)):
    if y.values[i] != y_pred[i]:
        errors += 1

print((1-errors/len(y_pred))*100)


# In[ ]:


logres.score(numeric_data, y)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


tree = DecisionTreeClassifier()


# In[ ]:


tree.fit(numeric_data, y)


# In[ ]:


for ix, el in enumerate(numerical_features):
    print(el, tree.feature_importances_[ix])


# In[ ]:


tree.score(numeric_data, y)


# In[ ]:


tree = DecisionTreeClassifier(max_depth=8)
tree.fit(numeric_data, y)
tree.score(numeric_data, y)


# In[ ]:


tree.predict(numeric_data)


# In[ ]:


tree_proba = tree.predict_proba(numeric_data)


# In[ ]:


logres_proba = logres.predict_proba(numeric_data)


# In[ ]:


logres_proba


# In[ ]:


mixed_proba = (tree_proba + logres_proba) / 2


# In[ ]:


y_pred_mixed = []
for i in mixed_proba:
    if i[0] > i[1]:
        y_pred_mixed.append('No')
    else:
        y_pred_mixed.append('Yes')


# In[ ]:


y_pred_mixed = ['No' if i[0] > i[1] else 'Yes' for i in mixed_proba]


# In[ ]:


y_pred_mixed = np.array(y_pred_mixed)


# In[ ]:


errors = 0
for i in range(len(y_pred_mixed)):
    if y.values[i] != y_pred_mixed[i]:
        errors += 1

print((1-errors/len(y_pred_mixed))*100)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rf = RandomForestClassifier(n_estimators=100)


# In[ ]:


rf.fit(numeric_data, y)


# In[ ]:


rf.score(numeric_data, y)


# In[ ]:


cat_features = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']


# In[ ]:


cat_data = data[cat_features]


# In[ ]:


cat_data['Location'].value_counts()


# In[ ]:


sns.distplot(cat_data['Location'].value_counts())


# In[ ]:


locations = set(cat_data['Location'].values)


# In[ ]:


locations


# In[ ]:


cat_data['Adelaide'] = 0


# In[ ]:


cat_data['Adelaide'] = cat_data['Location'].apply(lambda x: int(x == 'Adelaide'))


# In[ ]:


cat_data[cat_data['Adelaide'] == 1]


# In[ ]:


for i in locations:
    cat_data[i] = cat_data['Location'].apply(lambda x: int(x == i))


# In[ ]:


cat_data


# In[ ]:


cat_data.drop(['Location'], axis=1, inplace=True)


# In[ ]:


WindGustDir_dummies = pd.get_dummies(cat_data['WindGustDir'], prefix='WindGustDir')


# In[ ]:


WindDir9am_dummies = pd.get_dummies(cat_data['WindDir9am'], prefix='WindDir9am')
WindDir3am_dummies = pd.get_dummies(cat_data['WindDir3pm'], prefix='WindDir3pm')


# In[ ]:


cat_data.drop(['WindGustDir','WindDir9am','WindDir3pm'], axis=1, inplace=True)
cat_data.head()


# In[ ]:


cat_data = pd.concat([cat_data,WindGustDir_dummies, WindDir9am_dummies, WindDir3am_dummies],axis=1)


# In[ ]:


cat_data.head()


# In[ ]:


X = pd.concat([numeric_data, cat_data], axis=1)
X.head()


# In[ ]:


X.info()


# In[ ]:


logres_full = LogisticRegression()
logres_full.fit(X, y)
logres_full.score(X, y)

