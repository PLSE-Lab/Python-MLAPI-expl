#!/usr/bin/env python
# coding: utf-8

# # It is a starter kit kinda notebook to get everyone started in this competition, Hope you find it useful.

# ### Let's import what we may need

# In[1]:


import warnings

# not the best solution for suppressing occasional ugly warnings but it works
warnings.filterwarnings('ignore') 

import pandas
import numpy
import pandas_profiling

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsRegressor

# in case we may have forgot to set the psuedo random number generator's initial seed somewhere
numpy.random.seed(7)


# ### Loading the train, test and sample submission datasets.
# #### You may need to upload the train.csv and test.csv manually, It seems they are not where they ought to be.

# In[2]:


train = pandas.read_csv("../input/ubaar-dataset/train.csv", encoding="UTF-8", index_col="ID")
test = pandas.read_csv("../input/ubaar-dataset/test.csv", encoding="UTF-8", index_col="ID")
submission = pandas.read_csv("../input/ubaar-dataset/sampleSubmission15kRandom.csv", encoding="UTF-8")


# ### Let's have a peek into the train dataframe

# In[3]:


train.head(10)


# ### A very very very incomplete data analysis by calculating the typical statistical measures for every attribute in the dataset

# In[4]:


train_test_stackedup = pandas.concat([train, test], axis=0, sort=True)
profile = pandas_profiling.ProfileReport(train_test_stackedup)
rejected_variables = profile.get_rejected_variables(threshold=0.9)
rejected_variables


# In[5]:


# Let's see the datasets stats
profile


# ### Helper routines, OneHotEncoder and MAPE 

# In[6]:


def one_hot_encode(dataframe, columns, rem_original_cols=False):
    """
    @param dataframe pandas DataFrame
    @param columns a list of columns to encode 
    @param rem_original_cols if True remove the original column in the resulting dataframe
    @return a DataFrame with one-hot encoding
    """
    for column in columns:
        dummies = pandas.get_dummies(dataframe[column], prefix=column, drop_first=False)
        dataframe = pandas.concat([dataframe, dummies], axis=1)
        if rem_original_cols:
            dataframe.drop(columns=[column], inplace=True) # need pandas version 0.21.0 or above
    return dataframe

def mean_absolute_percentage_error(y_true, y_pred):
    """
    @param y_true array like object holding the real labels of the samples
    @param y_pred array like object holding the corresponfing predictions
    @return MAPE score for true values and predicted ones
    """
    y_true, y_pred = numpy.array(y_true), numpy.array(y_pred)
    return numpy.mean(numpy.abs((y_true - y_pred) / y_true)) * 100


# ### Let me see ... Hmm

# In[7]:


list(train_test_stackedup.columns)


# ### Let's select features we want in our model to be used, the target for predictions is of course the price

# In[11]:


features = ['SourceState', 'destinationLatitude', 'destinationLongitude',
            'destinationState', 'distanceKM', 'sourceLatitude', 'sourceLongitude',
            'vehicleOption', 'vehicleType', 'date', 'weight']
 
target = ['price']


# ### It maybe wise to make a copy of the dataset before doing anything that may casue altering the state of our dataset irreversibly, also let's find samples with  NaN for the feature values

# In[12]:


copy_of_train_test_stackedup = train_test_stackedup.copy()
copy_of_train_test_stackedup[copy_of_train_test_stackedup[features].isnull().any(axis=1)]


# ### Simplest way maybe to impute the NaNs is either by removing the samples containing them or by replacing them with some extreme constant ;0).

# In[13]:


copy_of_train_test_stackedup.fillna(-1, inplace=True)


# Let's turn all the categorical features into numerical ones via one-hot encoding scheme

# In[14]:


X, y = copy_of_train_test_stackedup[features], copy_of_train_test_stackedup[target]

categorical_features = []
for column in X.columns:
    if X[column].dtype == 'object':
        categorical_features.append(column)
        
X = one_hot_encode(X, categorical_features, rem_original_cols=True)
X.head(10)


# ### Let's build train, validation and the test data from the original dataset

# In[19]:


train_size = 50000
train_X = X[:train_size]
y_true = y[:train_size]
test_X = X[train_size:]


# In[20]:


# 20% of labeled data for validation and the rest for training the model
X_train, X_validation, y_train, y_validation = train_test_split(train_X, y_true, test_size=0.2, random_state=7)


# ### Building a very simple model here, nothing especial yet

# In[22]:


base_regressor = KNeighborsRegressor(n_neighbors=10, weights='uniform', algorithm='auto', leaf_size=30,
                                     p=2, metric='minkowski', metric_params=None, n_jobs=-1)

meta_regressor = make_pipeline(MinMaxScaler(), base_regressor)

meta_regressor.fit(X_train, y_train)


# ### Calculating the validation error ...

# In[18]:


validation_y_pred = meta_regressor.predict(X_validation)
validation_error = mean_absolute_percentage_error(y_validation, validation_y_pred)
print("MAPE on validation is: %f" % validation_error)


# ### Making the predictions for the actual test data, just need to submit the **submission.csv** file hereonafter

# In[75]:


submission.loc[:, 'price'] = meta_regressor.predict(test_X)
submission.to_csv("./submission.csv", index=False, encoding="UTF-8")

