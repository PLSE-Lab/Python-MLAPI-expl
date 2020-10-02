#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# ## Extra info about data:

# This is how the data was first scraped. This was entirely from the events and fights page of ufcstats. The data has simply been scraped, and thrown into a csv format. 

# In[ ]:


raw1 = pd.read_csv('../input/raw_total_fight_data.csv', sep=';')
raw1.head()


# This is the fighter details scraped from the fighters page of the ufcstats. This contains information about fighter date of birth, height, reach etc.

# In[ ]:


raw2 = pd.read_csv('../input/raw_fighter_details.csv')
raw2.head()


# This is the semi processed data. Most of the work went into this step. A lot of feature engineering and overall combining of both raw1 and raw2 data in a correct fashion.

# In[ ]:


data = pd.read_csv('../input/data.csv')
data.head()


# The final step was to do one hot encoding, treating missing values and removing unwanted columns and the result was this dataset

# In[ ]:


df = pd.read_csv('../input/preprocessed_data.csv')
df.head()


# ### Normalizing data

# In[ ]:


df_num = df.select_dtypes(include=[np.float, np.int])


# In[ ]:


scaler = StandardScaler()

df[list(df_num.columns)] = scaler.fit_transform(df[list(df_num.columns)])


# ### Splitting into train, test

# In[ ]:


y = df['Winner']
X = df.drop(columns = 'Winner')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=43)


# ### Fitting baseline Random Forest

# In[ ]:


model = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=43)


# In[ ]:


model.fit(X_train, y_train)


# In[ ]:


model.oob_score_


# In[ ]:


y_preds = model.predict(X_test)
accuracy_score(y_test, y_preds)


# In[ ]:


model.feature_importances_


# In[ ]:




