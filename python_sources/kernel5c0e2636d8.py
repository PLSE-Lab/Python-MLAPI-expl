#!/usr/bin/env python
# coding: utf-8

# ## Import libraries

# In[ ]:


get_ipython().system('pip install category_encoders')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import category_encoders as ce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier
pd.set_option('display.float_format', '{:.2f}'.format)


# ## Import and read CSV files

# In[ ]:


# Merge train_feartures.csv & train_labels.csv
train = pd.merge(pd.read_csv('https://drive.google.com/uc?export=download&id=14ULvX0uOgftTB2s97uS8lIx1nHGQIB0P'),
                pd.read_csv('https://drive.google.com/uc?export=download&id=1r441wLr7gKGHGLyPpKauvCuUOU556S2f'))

# Read test_features.csv & sample_admission.csv
test = pd.read_csv('https://drive.google.com/uc?export=download&id=1wvsYl9hbRbZuIuoaLWCsW_kbcxCdocHz')
sample_submission = pd.read_csv('https://drive.google.com/uc?export=download&id=1kfJewnmhowpUo381oSn3XqsQ6Eto23XV')

# Split train into train & val
train, val = train_test_split(train, train_size=.80, test_size=.20, 
                             stratify=train['status_group'], random_state=42)


# In[ ]:


# Identify the dimension of the 'train' data set
train.shape


# ## Exploring categorical features

# In[ ]:


# function that will give a vision on what columns double data:
def identify_double_columns_data(X):
  """Identify the columns that contain the same data in the data set,
  type of data categorical"""
  for column in X.select_dtypes(exclude='number').columns:
    if X[column].nunique() <= 30:
      print('--------' * 10)
      print('feature displayed: >>>', column, ' <<<')
      print('                      ', len(column) * '-')
      print('value     &    times present in column')
      print(X[column].value_counts())


# In[ ]:


identify_double_columns_data(train)


# #### Analysis of columns: source, source_type, source_class

# In[ ]:


train[['source', 'source_type', 'source_class']].isin(['unknown']).sum()


# In[ ]:


train[['source', 'source_type', 'source_class']].isin(['other']).sum()


# #### Analyze columns 'scheme_management', 'management', 'management_group'

# In[ ]:


train[['scheme_management', 'management', 'management_group']].isin(['unknown']).sum()


# In[ ]:


train[['scheme_management', 'management', 'management_group']].isin(['other']).sum()


# In[ ]:


train[['scheme_management', 'management', 'management_group']].isna().sum()


# In[ ]:


train[['scheme_management', 'management', 'management_group']]


# #### Analyze columns 'extraction_type', 'extraction_type_group', 'extraction_type_class'

# In[ ]:


train[['extraction_type', 'extraction_type_group', 'extraction_type_class']].isna().sum()


# In[ ]:


train[['extraction_type', 'extraction_type_group', 'extraction_type_class']].nunique()


# In[ ]:


train[['extraction_type', 'extraction_type_group', 'extraction_type_class']].isin(['unknown']).sum()


# In[ ]:


train[['extraction_type', 'extraction_type_group', 'extraction_type_class']].isin(['other']).sum()


# #### Analyzing the results above, I decided the next actions:
# 1. Delete column  'recorded_by'. It's contains the same value in every cell.
# 
# 2. Delete column 'quality_group'. Have same data as 'water_quality',  but the last have 2 values divided in 4 values 
# 
# 3. Doubled data in the columns 'quantity_group' and 'quantity'. Drop 'quantity_group' column
# 
# 4. Delete column 'waterpoint_type_group'. Have same data as 'waterpoint_type',  but the last have 1 category divided in 2 categories
# 
# 5. Columns 'payment' and 'payment_type' double the same values and I will drop the 'payment column'
# 
# 6. Columns  'source', 'source_type', 'source_class' contain the same values. Delete 'source_type', 'source_class'
# 
# 7. Delete column 'scheme_management' it's look like it double the 'management', and the last more or less is it doubled by 'management_group' column what will be deleted too
# 
# 8. Delete columns 'extraction_type_group', 'extraction_type_class' that double the column 'extraction_type' (the last have more type of value, and could be more useful)
# 
# #### Conclusion
# Delete columns:  recorded_by, quality_group, quantity_group, waterpoint_type_group, payment column, source_type, source_class, scheme_management, management_group, extraction_type_group, extraction_type_class

# In[ ]:


def wrangle(X):
  """Wrangle train, validate and test sets in the same way"""
  X = X.copy()
  
  # Convert data_recorded to datetime
  X['date_recorded'] = pd.to_datetime(X['date_recorded'], infer_datetime_format=True)
  
  # Extract components from date_recorded, then drop the original column
  X['year_recorded'] = X['date_recorded'].dt.year
  X['month_recorded'] = X['date_recorded'].dt.month
  X['day_recorded'] = X['date_recorded'].dt.day
  X = X.drop(columns='date_recorded')
  
  # Delete the columns that double the data
  X = X.drop(columns=['recorded_by', 'quality_group', 'quantity_group', 
              'waterpoint_type_group', 'payment', 'source_type', 
              'source_class', 'scheme_management', 'management_group', 
              'extraction_type_group', 'extraction_type_class', 'num_private'])
  
  # Delete 'id' column doesn't contain information as feature
  X.drop(columns='id', inplace=True)
  
  # About 3% of the time, latitude has small values near zero,
  # outside Tanzania, so we'll treat these like null values
  X['latitude'] = X['latitude'].replace(-2e-08, np.nan)

  # When columns have zeros and shouldn't, they are like null values
  for col in ['construction_year', 'longitude', 'latitude', 'gps_height', 'population']:
    X[col] = X[col].replace(0, np.nan)
    
  # Fill na values based on the feature's distribution

#   dist = X['construction_year'].value_counts(normalize=True)
#   X.loc[X['construction_year'].isna(),'construction_year'] = np.random.choice(
#       dist.index, size=X['construction_year'].isna().sum(),p=dist.values)
  
  # Calculate the age of the pump at the moment of recording
  X['age'] = X['year_recorded'] - X['construction_year']
  
  X.drop(columns='construction_year', inplace=True)
  

  
  return X


# #### Wrangle datasets

# In[ ]:


train = wrangle(train)
val = wrangle(val)
test = wrangle(test)


# In[ ]:


train.head()


# In[ ]:


train.n


# In[ ]:


train[['gps_height',	'longitude',	'latitude', 'basin']].isna().sum()


# In[ ]:


position = 


# In[ ]:


train.shape, val.shape, test.shape


# #### Define features and target

# In[ ]:


# The status_group column is the target
target = 'status_group'

# Get a dataframe with all train columns except the target
train_features = train.drop(columns=[target])

# Get a list of the numeric features
numeric_features = train_features.select_dtypes(include='number').columns.tolist()

# Get a series with the cardinality of the nonnumeric features
cardinality = train_features.select_dtypes(exclude='number').nunique()

# Get a list of all categorical features with cardinality <= 50
categorical_features = cardinality[cardinality <= 50].index.tolist()

# Combine the lists 
features = numeric_features + categorical_features


# In[ ]:


# Arrange data into X features matrix and y target vector 
X_train = train[features]
y_train = train[target]
X_val = val[features]
y_val = val[target]
X_test = test[features]


# ## Random Forest Classifier

# In[ ]:


get_ipython().run_cell_magic('time', '', "from sklearn.ensemble import RandomForestClassifier\n\npipeline = make_pipeline(\n    ce.OneHotEncoder(use_cat_names=True), \n    SimpleImputer(strategy='mean'), \n    RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)\n)\n\n# Fit on train, score on val\nmodel = pipeline.fit(X_train, y_train)\nprint('Validation Accuracy', pipeline.score(X_val, y_val))")


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# ## Ordinal encoding

# In[ ]:


# Arrange data into X features matrix and y target vector
X_train = train.drop(columns=target)
y_train = train[target]
X_val = val.drop(columns=target)
y_val = val[target]


# In[ ]:


X_train.shape, X_val.shape, X_test.shape


# In[ ]:


get_ipython().run_cell_magic('time', '', "\npipeline = make_pipeline(\n    ce.OrdinalEncoder(), \n    SimpleImputer(strategy='median'), \n    RandomForestClassifier(n_estimators=800, random_state=42, n_jobs=-1)\n)\n\n# Fit on train, score on val\npipeline.fit(X_train, y_train)\ny_pred = pipeline.predict(test)\nprint('Validation Accuracy', pipeline.score(X_val, y_val))")


# In[ ]:


submission = sample_submission.copy()
submission['status_group'] = y_pred
submission.to_csv('submission-10.csv', index=False)


# In[ ]:


from google.colab import files
files.download('submission-10.csv')


# ## XGBoost Classifier

# In[ ]:


get_ipython().system('pip install xgboost')


# In[ ]:


from xgboost import XGBClassifier


# In[ ]:


encoder = pipeline.named_steps['ordinalencoder']
imputer = pipeline.named_steps['simpleimputer']
X_train_encoded = encoder.transform(X_train)
X_train_imputed = imputer.transform(X_train_encoded)


# In[ ]:


model = XGBClassifier()
model.fit(X_train_encoded, y_train)


# In[ ]:


X_val_encoded = encoder.transform(X_val)
X_val_imputed = imputer.transform(X_val_encoded)


# In[ ]:


model.score(X_val_encoded, y_val)

