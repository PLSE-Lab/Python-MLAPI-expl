#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#I import some codes from here https://www.kaggle.com/olavomendes/rental-prices-in-brazil?fbclid=IwAR3LeW1eWl_dvfkOs2leVNA3l5Zc2HiDlu1BUqQdItVjdvc1y78YMkdoLG8
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# # Objective

# #In this Nootbook, 
# 1) We read the data then we view the first 10 coloumns and the last 10 coloumns.
# 2) we view numbers of coloumns(features) and numbers of rows(dataset samples).
# 3) we view some informations about data using data.describe order then data.info and comment the results.
# 4) we check null data and solve this problem if it found.
# 5) we do labelencoder for categorical features.
# 6) we choose half of features using one of feature tools.
# 7) we do datascaling using one of tools.
# 8) we divide data to testing and training.
# 9) we execute 5 algorithems  from sklearn library and view the score for each agorithm.
# 10) we use classification data ,so we view Percision and recall.
# 11) we use gridsearch tool with used algorithm to get best parameters.
# 12) we choose the best algorithm and use it to perdict X_test then we view first 10 rows results
# 13) in the case of law performances, we 'll comment this to do some steps to enhance the performance.
# 

# In[ ]:


raw_data = pd.read_csv('../input/brasilian-houses-to-rent/houses_to_rent_v2.csv')

raw_data.head(10)


# In[ ]:


#To view last 10 rows
raw_data.head(-10)


# The features are:
# * **city** - city where the property is located
# * **area** - property area
# * **rooms** - quantity of rooms
# * **bathroom** - quantity of bathrooms
# * **parking spaces** - quantity of parking spaces
# * **floor** - floor
# * **animal** - acept animals?
# * **furniture** - furniture?
# * **hoa** - Homeowners association tax
# * **property tax** - IPTU / property tax
# * **rent amount** - rental price
# * **fire insurance** - fire insurance
# * **total** - total value
# 

# # View number of coloumns and number of rows 

# In[ ]:


# Shape
print('ROWS: ', raw_data.shape[0])
print('COLUMNS: ', raw_data.shape[1])


# In[ ]:


# Basic info
raw_data.info()


# In[ ]:


# Basic description
#we notice in area values, the variance is big as min =11 and max=46335 so we have a small flats and a huge filla.
#we notice also in no. of rooms, the variance is big as min =1 and max=13 so we have a small flats and a huge filla
#so we conclude that we have outliers in our dataset.
raw_data.describe().T


# In[ ]:


# NULL values
#we conclude no null values.
raw_data.isnull().sum()


# ## Analysis of important features

# ### rooms

# In[ ]:


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.barplot(x=raw_data['rooms'], y=raw_data['rent amount (R$)'])

plt.subplot(1, 2, 2)
sns.boxplot(x=raw_data['rooms'])
plt.xticks(np.arange(raw_data['rooms'].min(), raw_data['rooms'].max(), step=1))


plt.show()


# The number of rooms usually varies between 1 and 4, and we noticed that the more rooms, the higher the rent, which is already expected.

# ### Bathroom
# 

# In[ ]:


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.barplot(x=raw_data['bathroom'], y=raw_data['rent amount (R$)'])

plt.subplot(1, 2, 2)
sns.boxplot(x=raw_data['bathroom'])
plt.xticks(np.arange(raw_data['bathroom'].min(), raw_data['bathroom'].max(), step=1))


plt.show()


# The number of bathrooms usually varies between 1 and 6, and we noticed that the more bathrooms, the higher the rent, which is already expected.

# ### Parking spaces

# In[ ]:


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.barplot(x=raw_data['parking spaces'], y=raw_data['rent amount (R$)'])

plt.subplot(1, 2, 2)
sns.boxplot(x=raw_data['parking spaces'])
plt.xticks(np.arange(raw_data['parking spaces'].min(), raw_data['parking spaces'].max(), step=1))


plt.show()


# The number of parking spaces usually varies between 0 and 5, and we noticed that the more parking spaces, the higher the rent, which is already expected.

# ### Fire insurance

# In[ ]:


plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
sns.regplot(x=raw_data['fire insurance (R$)'], y=raw_data['rent amount (R$)'], line_kws={'color': 'r'})

plt.show()


# ### Furniture

# In[ ]:


furniture = raw_data['furniture'].value_counts()
pd.DataFrame(furniture)


# There are about 3x more unfurnished houses than furnished

# In[ ]:


plt.figure(figsize=(11, 5))

plt.subplot(1, 2, 1)
plt.title('Furniture ratio')
plt.pie(furniture, labels = ['not furnished', 'furnished'], colors= ['r', 'g'], 
        explode = (0, 0.1), autopct='%1.1f%%')

plt.subplot(1, 2, 2)
plt.title('Furniture vs Rent amount')
sns.barplot(x=raw_data['furniture'], y=raw_data['rent amount (R$)'])

plt.tight_layout()
plt.show()


# The fact that the house is furnished increases the rent price

# # Testing ML models (data with outliers)

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ML models
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor


# #### Categorical columns handler

# In[ ]:


catTransformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


# ### Numerical columns handler

# In[ ]:


#Encoding
numTransformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])


# ### Select FEATURES (X)

# In[ ]:


#select half of features depend on previous analysis
cols = ['city', 'rooms', 'bathroom', 'parking spaces', 'fire insurance (R$)', 'furniture']
X = raw_data[cols]
X.head()


# In[ ]:


#scaling for fire insurence
for col in X:
    X = X.astype('object')
X['fire insurance (R$)'] = X['fire insurance (R$)'].astype('int64')
X.info()


# # Output seperation "rent amount"
# 

# In[ ]:


#we seperate our output rent amount
y = raw_data['rent amount (R$)']
y


# ### Select numerical features

# In[ ]:


#change "object" to "integer"
numFeatures = X.select_dtypes(include=['int64']).columns
numFeatures


# ### Select categorical features

# In[ ]:


catFeatures = X.select_dtypes(include=['object']).columns
catFeatures


# #### Handling numerical and categorical features

# In[ ]:


preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', numTransformer, numFeatures),
        ('categoric', catTransformer, catFeatures)])


# ### Select TRAIN and TEST data

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ### List of ML models

# In[ ]:


regressors = [
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    SVR(),
    LinearRegression(),
    XGBRegressor()
]


# ### Fit all ML models and select best

# In[ ]:


# Seed
np.random.seed(42)

for regressor in regressors:
    
    estimator = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])
    
    estimator.fit(X_train, y_train)
    preds = estimator.predict(X_test)
    
    print(regressor)

    print('MAE:', mean_absolute_error(y_test, preds))
    print('RMSE:', np.sqrt(mean_squared_error(y_test, preds)))
    print('R2:', r2_score(y_test, preds))
    print('-' * 40)


# **XGBoost** is the best model!

# ### GridSearchCV with the best model (XGBRegressor)

# In[ ]:


#search for the best parameters 
pipe = Pipeline(steps=[('preprocessor', preprocessor),
                       ('model', XGBRegressor(random_state=42))
                      ])


# In[ ]:


params = {
            'model__learning_rate': [0.01, 0.1],
            'model__n_estimators': [100, 150, 200],
            'model__max_depth': [2, 4, 6, 8],
            'model__subsample': [0.8, 1],
            'model__colsample_bytree': [0.8, 1],
            'model__gamma': [0, 1, 5]
            #'model__min_samples_split': [2, 4, 6, 8],
            #'model__min_samples_leaf': [2, 4, 6, 8],
         }


# In[ ]:


estimator = GridSearchCV(pipe, cv=10, param_grid=params)
estimator.fit(X_train,y_train)


# ### Best params

# In[ ]:


estimator.best_params_


# ### Predict with the best params

# In[ ]:


preds = estimator.predict(X_test)
preds[0:10]


# ### Evaluate

# In[ ]:


print('MAE:', mean_absolute_error(y_test, preds))
print('RMSE:', np.sqrt(mean_squared_error(y_test, preds)))
print('R2:', r2_score(y_test, preds))


# In[ ]:


plt.figure(figsize=(8, 6))

sns.boxplot(raw_data['city'], raw_data['rent amount (R$)'])

plt.show()


# In[ ]:


plt.figure(figsize=(8, 6))

sns.distplot(y_test, hist=False, color='b', label ='Actual')
sns.distplot(preds, hist=False, color='r', label = 'Predicted')

plt.show()


# # Save model

# In[ ]:


from joblib import dump, load
dump(estimator, 'model_2.joblib')
model = load('model_2.joblib')

