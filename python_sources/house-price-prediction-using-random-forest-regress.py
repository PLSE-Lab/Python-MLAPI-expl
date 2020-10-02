#!/usr/bin/env python
# coding: utf-8

# In[3]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ### Load Libraries

# In[4]:



# Data manipulation libraries
import pandas as pd
import numpy as np

##### Scikit Learn modules needed for Logistic Regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder,MinMaxScaler , StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Plotting libraries
from IPython.display import SVG
from graphviz import Source
from IPython.display import display
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes = True)
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Load Training Data

# In[15]:


df = pd.read_csv("../input/train.csv")
df.head()
print(df.describe())


# ### Explore Training Data

# In[6]:


# Select only numerical columns for data analysis
df_numeric = df._get_numeric_data()
print(df_numeric.columns)
exclude_dates = ['Id','YearBuilt','YearRemodAdd','MoSold', 'YrSold','SalePrice']
df_numeric = df_numeric.drop(exclude_dates,axis=1)
print(df_numeric.columns)


# In[7]:


# Explore data visually
# Build Correlation Matrix to study multi collinearity
correlation = df_numeric.corr()
#print(correlation)

fig , ax = plt.subplots()
fig.set_figwidth(18)
fig.set_figheight(18)
sns.heatmap(correlation,annot=True,cmap="YlGnBu")


# ### Visual Observation
# - Several numerical variables are strongly correlated viz Garage Cars with Garage Area or Ground Levl Area with Total rooms above
# - We could either remove of of the correlated values but can also engineer a metric as ratio of two quantities
# - In my current example I have kept all the correlated values and opted for reducing dimensions of numerical variables by using PCA

# ### Build Preprocessing Pipeline -
# - created separate strategy to handle numerical and categorical variables
# 
# #### Preprocessing of Numerical Features - 
# - Imputation using Mean (however added median as part of grid search in below code)
# - opted for standard scaling of numerical values
# - Dimentionality Reduction using PCA
# 
# #### Categorical Variables
# - Imputed missing values with word _'missing'_
# - Tranformation using One hot encoding

# In[8]:


# We create the preprocessing pipelines for both numeric and categorical data.

numeric_features = [x for x in df_numeric.columns]
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components= 2))])

all_numeric_columns = exclude_dates + numeric_features
categorical_features = [x for x in df.columns if x not in all_numeric_columns ]
# categorical_features = [x for x in df.columns if x not in df_numeric + exclude_dates]
#print(categorical_features)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestRegressor())])


# ### Split Data in Training & test sets (80/20 ratio)

# In[9]:


X_train, X_test, y_train, y_test = train_test_split(df[numeric_features + categorical_features], 
                                                    df["SalePrice"], test_size=0.2,random_state =42)


# In[10]:


param_grid = {
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'classifier__max_features': ["auto","sqrt", "log2"],
    #'classifier__max_iter' :[100,150,200],
    'classifier__n_estimators': [10,50,100,200],
    'classifier__max_depth':[2,4,8]
}

grid_search_rfr = GridSearchCV(clf, param_grid, cv=10, iid=False,verbose= 2 , n_jobs = -1)
grid_search_rfr.fit(X_train, y_train)

print(("best Linear Regression from grid search: %.3f"
       % grid_search_rfr.score(X_test, y_test)))
print("Best Parameter Setting is {}".format(grid_search_rfr.best_params_))


# In[11]:


test_df = pd.read_csv("../input/test.csv")
test_df_columns = [x for x in test_df if x not in exclude_dates]

# Load Submission File
sample_submission = pd.read_csv("../input/sample_submission.csv")


# In[12]:


y_prediction = grid_search_rfr.predict(test_df[test_df_columns])


# In[13]:


submission = pd.DataFrame({"Id":sample_submission["Id"].values, "SalePrice":y_prediction.tolist()})


# In[14]:


submission.to_csv("submission_randomfr_V1.csv",index=False)


# ### Thats all for the day folks !! Oh yes and I havent touched upon time based variables, kept them for next iterations to come :)

# In[ ]:




