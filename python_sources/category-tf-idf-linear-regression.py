#!/usr/bin/env python
# coding: utf-8

# 11/29/2017: TF-IDF on name + item_description fed into a Linear Regression. Log transformed price to improve performance.
# 
# I'm a huge proponent of the fact that simple regressors (linear or logistic) run the world. I wanted to put together a linear model that uses the category_name field to demonstrate that a linear regression is a nice baseline.

# In[1]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline

import pandas as pd
import numpy as np
import time
import re

seed = 101


# In[2]:


def tokenizer(text):
    if text:
        result = re.findall('[a-z]{2,}', text.lower())
    else:
        result = []
    return result


# In[3]:


df = pd.read_csv('../input/train.tsv', sep='\t')
df.head()


# Now let's train a simple tfidf vectorizer on the combination of the item name and description.

# In[4]:


df['item_description'].fillna(value='Missing', inplace=True)
X = (df['name'] + ' ' + df['item_description']).values
y = np.log1p(df['price'].values)

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=seed)


# In[5]:


vect = TfidfVectorizer(tokenizer=tokenizer, stop_words='english')
start = time.time()
X_train_vect = vect.fit_transform(X_train)
end = time.time()
print('Time to train vectorizer and transform training text: %0.2fs' % (end - start))


# Now let's train a linear regression model.

# In[6]:


# I was using a LinearRegression previously, but with the wider vocab it's too slow. 
# Let's use the SGDRegressor with ordinary least squares.
# Also, using mean squared error as the eval metric, since negative values crash mean squared log error.

model = SGDRegressor(loss='squared_loss', penalty='l2', random_state=seed, max_iter=5)
params = {'penalty':['none','l2','l1'],
          'alpha':[1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1]}
gs = GridSearchCV(estimator=model,
                  param_grid=params,
                  scoring='neg_mean_squared_error',
                  n_jobs=1,
                  cv=5,
                  verbose=3)
start = time.time()
gs.fit(X_train_vect, y_train)
end = time.time()
print('Time to train model: %0.2fs' % (end -start))


# In[7]:


model = gs.best_estimator_
print(gs.best_params_)
print(gs.best_score_)


# Now let's package everything up nicely in a pipeline and run it over the test set to check the performance.

# In[8]:


pipe = Pipeline([('vect',vect),('model',model)])
start = time.time()
y_pred = pipe.predict(X_test)
end = time.time()
print('Time to generate predictions on test set: %0.2fs' % (end - start))


# In[12]:


# Replace negative values with zero for the time being.
print(np.sqrt(mean_squared_log_error(np.exp(y_test)-1, np.exp(y_pred)-1)))


# Okay, now let's read in the test data and generate our submission file.

# In[13]:


df_test = pd.read_csv('../input/test.tsv', sep='\t')
df_test.head()


# In[14]:


df_test['item_description'].fillna('Missing', inplace=True)
df_test['price'] = np.exp(pipe.predict((df_test['name'] + ' ' + df_test['item_description']).values))-1
df_test.head()


# In[ ]:


df_test[['test_id','price']].to_csv('output.csv', index=False)


# In[ ]:




