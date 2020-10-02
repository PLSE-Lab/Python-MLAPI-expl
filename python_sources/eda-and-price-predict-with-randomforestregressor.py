#!/usr/bin/env python
# coding: utf-8

# 1. **Import Packages**

# In[ ]:


import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv('/kaggle/input/craigslist-carstrucks-data/vehicles.csv')
data.head()


# 2. **Remove Useless Features**

# In[ ]:


#Visualize all columns of the dataset
data.columns


# In[ ]:


#Remove useless data
remove = ['id', 'url', 'region_url', 'model', 'vin', 'image_url', 'description', 'county', 'lat', 'long']
data.drop(remove, axis=1, inplace=True)


# 3. **Verify Missing Data and Remove Features With Missing Values Are More Than 20%**

# In[ ]:


#See total missing data by feature
data.isnull().sum()


# In[ ]:


#If the missing data correspond more than 20% of the feature, remove column
for col in data.columns:
    if(data[col].isnull().sum() > data.shape[0]*0.2):
        data.drop(col, axis=1, inplace=True)


# In[ ]:


data.isnull().sum()


# In[ ]:


data.head()


# 4. **Handle Missing Data, Impute Method**

# In[ ]:


numerical_col = ['year','odometer']
categorical_col = ['region', 'manufacturer','fuel', 'title_status', 'transmission', 'state']


# In[ ]:


#Impute with median method in the numerical features 
imputer = SimpleImputer(strategy="median")
data[numerical_col] = imputer.fit_transform(data[numerical_col])


# In[ ]:


#Impute with constant method in the categorical features 
imputer = SimpleImputer(strategy="constant")
data[categorical_col] = imputer.fit_transform(data[categorical_col])


# In[ ]:


data.isnull().sum()


# 5. **Visualize The Shape of the Distribution**

# In[ ]:


#See the distribution 
data.plot(kind='box', subplots=True, layout=(1,3), sharex=False, sharey=False, figsize=(10,6))
plt.show()


# As we can see above, there are many outiliers that negatively influence the creation of the model, eg: "price = 0", "year = 0" or "odometer very high".

# In[ ]:


data = data[(data['price'] > 1000) & (data['price'] < 35000)]


# So I've Used The Interquartile Range (IQR) Method to Remove outliers data

# In[ ]:


Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[ ]:


data_out = data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]


# In[ ]:


data_out.plot(kind='box', subplots=True, layout=(1,3), sharex=False, sharey=False, figsize=(10,6))
plt.show()


# In[ ]:


scaler = StandardScaler()
scaled = data_out[numerical_col].copy()
scaled = scaler.fit_transform(scaled)
data_out.loc[:][numerical_col] = scaled


# In[ ]:


data_out.plot(kind='box', subplots=True, layout=(1,3), sharex=False, sharey=False, figsize=(10,6))
plt.show()


# 6. **One-Hot Encoding The Categorical Data**

# In[ ]:


#Visualize the unique values of the categorical features, to ensure organization
for col in data_out[categorical_col].columns:
    print("Coluna:", col, " ", len(data[col].unique()))


# In[ ]:


one_hot = pd.get_dummies(data_out[categorical_col])
data_out.drop(categorical_col, axis=1, inplace=True)
df = pd.concat([data_out, one_hot], axis=1)


# In[ ]:


df.head()


# 7. **Build The Model**

# In[ ]:


#Predict the Price
y = df['price']
X = df.drop('price', axis=1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)


# In[ ]:


model = RandomForestRegressor(n_estimators=50)
model.fit(X_train, y_train)

preds = model.predict(X_test)


# In[ ]:


print(mean_absolute_error(y_test, preds))
print(r2_score(y_test, preds))

