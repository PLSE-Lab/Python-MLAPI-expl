#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from glob import glob


# In[ ]:


def metrics(y_true, y_preds):

    mse = mean_squared_error(y_true, y_preds)
    r2  = r2_score(y_true, y_preds)
    
    print("R2: ", r2)
    print("MSE: ", mse)


# In[ ]:


data = pd.read_csv('../input/DATASET.csv')
data.fillna("0", inplace = True)

data = data.drop(columns = ['id', 'nome', 'data', 'time'])


# In[ ]:


data.head()


# In[ ]:


# vectorizer = CountVectorizer()
# bow = vectorizer.fit_transform(list(data.text.values)).toarray()


# In[ ]:


vectorizer = TfidfVectorizer()
bow = vectorizer.fit_transform(list(data.text.values)).toarray()


# In[ ]:


lat_long = data.drop(columns = ['text'])


# In[ ]:


data = pd.concat([ 
    pd.DataFrame(bow),
    lat_long], 
    axis = 1
)


# In[ ]:


data.head()


# In[ ]:


data[[6150,6152,'lat','long']].describe()


# In[ ]:


X = data.drop(columns = ['lat', 'long'])
y = np.asarray([list(x) for x in zip(data.lat, data.long)])


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
    X.values, 
    y, 
    shuffle=True,
    test_size=0.33, 
    random_state=42
)


# In[ ]:


pca = PCA(n_components=500)
pca_X_train = pca.fit_transform(X_train)
pca_X_test  = pca.transform(X_test)


# In[ ]:


y_preds = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators = 250,
        max_depth = 80,
        max_features = 'auto'
    )
).fit(pca_X_train, y_train).predict(pca_X_test)


# In[ ]:


y_preds


# In[ ]:


metrics(y_test, y_preds)


# **No PCA**

# In[ ]:


y_preds2 = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators = 250,
        max_depth = 80,
        max_features = 'auto'
    )
).fit(X_train, y_train).predict(X_test)


# In[ ]:


metrics(y_test, y_preds2)


# In[ ]:




