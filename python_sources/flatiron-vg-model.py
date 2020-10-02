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

""" Create custom transformers"""

# Get sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Get keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping

# Load DataFrame
df = pd.read_csv('../input/train.csv', index_col=0)

# Get numerical and categorical data, resp.
num_cols = ['Critic_Count', 'Critic_Score', 'JP_Sales', 'User_Count', 'Year_of_Release']
cat_cols = [x for x in df.columns if x not in num_cols + ['NA_Sales']]


def data_clean(df_load, cat_cols=cat_cols):
    """ Cleans data as needed. """
    df = df_load.copy()
    
    # Cast tbd as nan
    df.loc[df.User_Score == 'tbd', ['User_Score']] = np.nan
    df.User_Score = df.User_Score.astype('float64')
    
    # Cast small ratings as nan
    for rating in ['EC', 'RP', 'K-A']:
        if any(df.Rating == rating):
            df.loc[df.Rating == rating, ['Rating']] = np.nan
    
    for col in cat_cols:
        #df[col].fillna("None", inplace=True)
        #df[col] = df[col].astype('str')
        df[col] = df[col].astype('category')
    
    return df

    
class DataFrameSelector(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer that provides column selection. """
    
    def __init__(self, columns=[]):
        """ Get selected columns. """
        self.columns = columns
        
    def transform(self, X):
        """ Returns df with selected columns. """
        return X[self.columns].copy()
    
    def fit(self, X, y=None):
        """ Do nothing operation. """
        return self
        

# Get Pipelines

# Fit numerical pipeline
num_pipeline = make_pipeline(
    DataFrameSelector(num_cols),
    SimpleImputer(strategy='median'),
    StandardScaler()
)

# Fit categorical pipeline
cat_pipeline = make_pipeline(
    DataFrameSelector(cat_cols),
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(handle_unknown='ignore', sparse=False)
)

# Union pipelines
full_pipeline = FeatureUnion(transformer_list=[
    ("cat_pipeline", cat_pipeline),
    ("num_pipeline", num_pipeline)
])


def get_X_y(df):
    """ Gets predictors X and target y. """
    x_cols = [x for x in df.columns if x != 'NA_Sales']
    X, y = df[x_cols], df.NA_Sales.values.reshape(-1, 1)
    X = full_pipeline.fit_transform(data_clean(X))
    return X, y
    

def get_model():
    """ Gets keras regression model. """
    n_cols = X.shape[1]
    model = Sequential()
    model.add(Dense(100, activation='relu', input_shape=(n_cols,)))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2)) 
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.2)) 
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
    return model


# Get data and model
X, y = get_X_y(df)
model = get_model()

# Compile and fit model
early_stopping_monitor = EarlyStopping(patience=5)
model.fit(X, y, 
          epochs=25, 
          callbacks=[early_stopping_monitor])

# Get test data
df_test = pd.read_csv('../input/test.csv', index_col=0)
X_test = full_pipeline.transform(data_clean(df_test))
y_pred = model.predict(X_test)
df_test['Prediction'] = y_pred
df_test[['Prediction']].to_csv('submssion.csv')