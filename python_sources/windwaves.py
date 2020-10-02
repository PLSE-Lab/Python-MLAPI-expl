#!/usr/bin/env python
# coding: utf-8

# # **K-folk Cross Validation Logistic Regression on NOAA ICOADS**
# 
# Using NOAA ICOADS global marine meteorlogical dataset, I would like to develop a Neural Network to predict oceanic waveheights using correlated features.
# 
# One major weakness in this dataset is quantifying the uncertainties and error. Many features are given an indicator code which is assigned to each measurement indicating the precision - however this is done coarsely. 
# 
# This initial run only utilizes the 2017 dataset due to size and training time constraints. 
# 
# Please comment if you have suggestions for improvements!

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing
from bq_helper import BigQueryHelper # Safely navigate the giant dataset
import matplotlib.pyplot as plt # plotting library
import seaborn as sns # plotting parameters

import warnings
warnings.filterwarnings('ignore') # I was having some annoying scipy deprecation warnings so I'm going to ignore them

sns.set_style("whitegrid")
sns.set_context("talk")
sns.set(rc={'figure.figsize':(11.7,8.27)})

bq_assistant = BigQueryHelper("bigquery-public-data", "noaa_icoads") 
# Get the data and loaf into BigQuery Helper object


# Check all tables available

# In[ ]:


get_ipython().run_cell_magic('time', '', 'bq_assistant.list_tables()')


# Time the loading of 2017, view available columns

# In[ ]:


get_ipython().run_cell_magic('time', '', 'bq_assistant.table_schema("icoads_core_2017")')


# Display first 10 rows for all columns in table

# In[ ]:


get_ipython().run_cell_magic('time', '', 'bq_assistant.head("icoads_core_2017", num_rows=10)')


# Here I select the data based on the descriptions of the NOAA ICOADS table [here](https://www.kaggle.com/noaa/noaa-icoads://)
# 
# QUERY is an SQL query. 
# 
# I am mainly interested in the Pacific Northwest off of British Columbia, Canada hence the lattitude and longitude selectors.**

# In[ ]:


QUERY = """
        SELECT latitude, longitude, wind_direction_true, amt_pressure_tend,  air_temperature, sea_level_pressure, wave_height, timestamp
        FROM `bigquery-public-data.noaa_icoads.icoads_core_2017`
        WHERE longitude > -130 AND longitude <= -110 AND latitude > 45 AND latitude <= 60 AND wind_direction_true <= 360
        """


# Time the estimate of the query. 

# In[ ]:


get_ipython().run_cell_magic('time', '', 'bq_assistant.estimate_query_size(QUERY)')


# Load the query results into pandas as a dataframe and display its shape.

# In[ ]:


df_bq = bq_assistant.query_to_pandas(QUERY)
print(df_bq.shape)
df = df_bq.dropna() # remove NaN values
print(df.shape)


# Augment data by applying a 10-realization monte-carlo simulation of the data

# In[ ]:


def monte_carlo(df):
    """
    Takes a pandas dataframe and runs a monte-carlo simulation on the target 
    labels.
    ====================================================================
    Inputs: Pandas df
    
    Outputs: Monte-carlo'd pandas df
    """
    
    nb_increase=10 # Number of montecarlo sample 
    size = df.shape[0]
    df_old = df.copy()
    for feature_name in df.columns:
        for nb in range(1,nb_increase):
            if feature_name!='timestamp':
                df_n = df_old.copy()
                df_n[feature_name] = df_old[feature_name].values + np.random.normal(0.0, 1.0, size)*df_old[feature_name].std()
                df = pd.concat([df, df_n], ignore_index=True)
    return df.dropna()
#df=monte_carlo(df).copy()


# In[ ]:


print(df.shape)


# Now that the data is loaded, we can start to visualize what we're up against.

# In[ ]:


# Compute the correlation matrix
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots()

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# Start visualizing the parameter space

# In[ ]:


sns.scatterplot(df['air_temperature'].values, df['sea_level_pressure'].values)


# In[ ]:


sns.scatterplot(df['longitude'].values, df['latitude'].values)


# In[ ]:


sns.distplot(df['amt_pressure_tend'])


# In[ ]:


sns.distplot(df['sea_level_pressure'])


# In[ ]:


sns.distplot(df['wave_height'])


# In[ ]:


sns.distplot(df.latitude)


# In[ ]:


sns.distplot(df.longitude)


# In[ ]:


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

normalized_df=normalize(df)


# In[ ]:


X = normalized_df[['wind_direction_true', 'amt_pressure_tend', 'air_temperature', 'sea_level_pressure', 'timestamp']].copy()
y = normalized_df[['wave_height']].copy()


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.optimizers import SGD


# In[ ]:


# define base model
def baseline_model():
    # create model
    droprate = 0.1
    model = Sequential()
    
    model.add(Dense(32, input_dim=5, kernel_initializer='normal', activation='relu')) #input_dim=13,
    model.add(BatchNormalization())
    model.add(Dropout(droprate))#3    #model.add(Dense(5096, kernel_initializer='normal', activation='relu'))

    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(droprate))#3
    
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(droprate))#3
    
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# In[ ]:


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, epochs=30, batch_size=5000, verbose=0)


# In[ ]:


kfold = KFold(n_splits=3, random_state=seed)
results = cross_val_score(estimator, X, y, cv=kfold)

print("Results: %.5f (%.5f) MSE" % (results.mean(), results.std()))


# In[ ]:


test_loss, test_acc, train_loss, train_acc = [], [], [], []

for train, test in kfold.split(X.values, y.values):
    history = estimator.fit(X.values[train], y.values[train], validation_data=(X.values[test], y.values[test]), epochs=30, batch_size=5000, verbose=1)

    test_loss.append(history.history['val_loss'])    
    train_loss.append(history.history['loss'])

test_loss = np.asarray(test_loss)
train_loss = np.asarray(train_loss)

test_loss_max = np.amax(test_loss, axis=0)
test_loss_min = np.amin(test_loss, axis=0)
train_loss_max = np.amax(train_loss, axis=0)
train_loss_min = np.amin(train_loss, axis=0)


# In[ ]:


fig, ax = plt.subplots(figsize=(8, 6))

ax.fill_between(range(0,len(test_loss_max)), test_loss_min, test_loss_max, alpha=0.5, color='red')
ax.fill_between(range(0,len(train_loss_max)), train_loss_min, train_loss_max, alpha=0.5)
ax.set_yscale('log')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
ax.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[ ]:




