#!/usr/bin/env python
# coding: utf-8

# # work on qsar fish dataset

# In[ ]:


#Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

#preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures


# In[ ]:


# column headers 
_headers = ['CIC0', 'SM1', 'GATS1i', 'NdsCH', 'Ndssc', 'MLOGP', 'response'] 
# read in data 
df = pd.read_csv('https://raw.githubusercontent.com/PacktWorkshops/The-Data-Science-Workshop/master/Chapter06/Dataset/qsar_fish_toxicity.csv', names=_headers, sep=';') 


# In[ ]:


df.head()


# # splitting data

# In[ ]:


# Let's split our data 

features = df.drop('response', axis=1).values 
labels = df[['response']].values 
X_train, X_eval, y_train, y_eval = train_test_split(features, labels, test_size=0.2, random_state=0) 
X_val, X_test, y_val, y_test = train_test_split(X_eval, y_eval, random_state=0) 


# In[ ]:


model = LinearRegression() 


# In[ ]:


model.fit(X_train, y_train) 


# In[ ]:


y_pred = model.predict(X_val) 


# In[ ]:


r2 = model.score(X_val, y_val) 
print('R^2 score: {}'.format(r2)) 


# In[ ]:


_ys = pd.DataFrame(dict(actuals=y_val.reshape(-1), predicted=y_pred.reshape(-1))) 
_ys.head() 


# In[ ]:


# Let's compute our MEAN ABSOLUTE ERROR
mae = mean_absolute_error(y_val, y_pred)
print('MAE: {}'.format(mae))


# In[ ]:


#Let's get the R2 score
r2 = model.score(X_val, y_val)
print('R^2 score: {}'.format(r2))


# In[ ]:


#create a pipeline and engineer quadratic features
steps = [
    ('scaler', MinMaxScaler()),
    ('poly', PolynomialFeatures(2)),
    ('model', LinearRegression())
]


# In[ ]:


#create a Linear Regression model
model = Pipeline(steps)


# In[ ]:


#train the model
model.fit(X_train, y_train)


# In[ ]:


#predict on validation dataset
y_pred = model.predict(X_val)


# # computing MAE for second model

# In[ ]:


#compute MAE
mae = mean_absolute_error(y_val, y_pred)
print('MAE: {}'.format(mae))


# In[ ]:


# let's get the R2 score
r2 = model.score(X_val, y_val)
print('R^2 score: {}'.format(r2))


# In[ ]:


from sklearn.externals import joblib


# In[ ]:


joblib.dump(model, './model.joblib')


# In[ ]:


m2 = joblib.load('./model.joblib')


# In[ ]:


m2_preds = m2.predict(X_val)


# In[ ]:


ys = pd.DataFrame(dict(predicted=y_pred.reshape(-1), m2=m2_preds.reshape(-1)))
ys.head()

