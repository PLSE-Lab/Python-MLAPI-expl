#!/usr/bin/env python
# coding: utf-8

# # Importing Data

# In[ ]:


# Load libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


train_df = pd.read_csv('../input/train_V2.csv')
test_df  = pd.read_csv('../input/test_V2.csv')

submission_df = pd.DataFrame()
submission_df['Id'] = test_df["Id"].copy()


# In[ ]:


train_df.info()


# # Preprocessing Data

# ### * Imputing Data

# In[ ]:


def null_percentage(column):
    df_name = column.name
    nans = np.count_nonzero(column.isnull().values)
    total = column.size
    frac = nans / total
    perc = int(frac * 100)
    print('%d%% or %d missing values from [ %s ] column.' % (perc, nans, df_name))

def check_nan(df):
    columns = df.columns
    for col in columns: null_percentage(df[col])


# In[ ]:


check_nan(train_df)


# In[ ]:


# Just one missing value exists, DROP it.
train_df = train_df.dropna()
train_df.reset_index(drop=True)
train_df.describe().T


# In[ ]:


# Drop columns
train_df2 = train_df.drop(columns=['Id', 'groupId', 'matchId'])
test_df2 = test_df.drop(columns=['Id', 'groupId', 'matchId'])


# ### * Reomving highly correlated features

# In[ ]:


corr_df = train_df2.drop(columns=['matchType'])


# In[ ]:


_, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(corr_df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


# In[ ]:


corr_matrix = corr_df.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]


# In[ ]:


train_df3 = train_df2.drop(to_drop, axis=1)
test_df3 = test_df2.drop(to_drop, axis=1)


# ### * Encoding Data

# In[ ]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train_df3['matchType'] = le.fit_transform(train_df3['matchType'].astype(str))
test_df3['matchType'] = le.fit_transform(test_df3['matchType'].astype(str))


# ### * Constructing Models

# In[ ]:


# Split training dataset into train/validation set (ratio = 7:3)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

X = train_df3.iloc[:, 0:-1]; y = train_df3.iloc[:, -1]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1040941203)


# In[ ]:


from catboost import CatBoostRegressor

cbr = CatBoostRegressor(iterations=1000,
                        learning_rate=0.05,
                        depth=16,
                        loss_function='MAE',
                        eval_metric='MAE',
                        bootstrap_type='Bayesian',
                        random_seed = 1040941203,
                        bagging_temperature = 0.4,
                        od_type='Iter',
                        od_wait=10,
                        use_best_model=True,
                        rsm = 0.2)


# In[ ]:


cbr.fit(X_train, y_train,
        eval_set=(X_val, y_val),
        cat_features=[X_train.columns.get_loc('matchType')],
        use_best_model=True,
        verbose=True)


# In[ ]:


y_pred_cbr = cbr.predict(test_df3)


# In[ ]:


submission_df['winPlacePerc'] = y_pred_cbr
submission_df.to_csv('submission_cbr.csv', index=False)

