#!/usr/bin/env python
# coding: utf-8

# Hello everyone. This is my first ever kernel on kaggle and is pretty rudimentary. Any criticism, positive or negative is highly appreciated.

# In[ ]:


# Importing libraries
from fastai.imports import *
from fastai.structured import *

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from IPython.display import display


# In[ ]:


# importing data (only 10 million data points)
PATH = '../input'
df_raw = pd.read_csv(f'{PATH}/train.csv', nrows=10000000)


# Let's have a look at the dataset.

# In[ ]:


# Function to set display options
def display_all(df):
    with pd.option_context('display.max_rows',1000):
        with pd.option_context('display.max_columns',1000):
            display(df)


# In[ ]:


display_all(df_raw.head(5))


# ## Feature Engineering

# Now, let's do some feature engineering based on the pickup_datetime column.

# In[ ]:


add_datepart(df_raw,'pickup_datetime',drop=True, time=True)


# In[ ]:


display_all(df_raw.head(5))


# Let's define a function which creates some more new features based on the pickup and dropoff lat. and long.

# In[ ]:


def distance(data):
    data['longitutde_traversed'] = (data.dropoff_longitude - data.pickup_longitude).abs()
    data['latitude_traversed'] = (data.dropoff_latitude - data.pickup_latitude).abs()


# In[ ]:


distance(df_raw)


# In[ ]:


display_all(df_raw.head(2).T)


# ## Data Quality check and Outlier Detection

# Next, let's see if there are any missing values in the dataset.

# In[ ]:


df_raw.isnull().sum()


# Since we have ample amount of data, let's just drop these few missing lines of data. (Note: We have 45 million more data rows in the training set)

# In[ ]:


df_raw.dropna(axis=0, how='any', inplace=True)


# In[ ]:


df_raw.shape


# Let's drop the key column, as it is same as the pickup_datetime columns.

# In[ ]:


key = df_raw.key
df_raw.drop('key', axis=1, inplace = True)


# Now, let's have a look at the passenger_count column

# In[ ]:


df_raw.passenger_count.value_counts()


# So there are taxis with 208, 129, 51 and 49 passengers as well, which is quite far fetched. Moreover, there are 35263 records of taxis with no passengers at all, let's remove all these data points as well.

# In[ ]:


df_raw = df_raw[(df_raw.passenger_count>0)&(df_raw.passenger_count<10)]


# In[ ]:


len(df_raw)


# In[ ]:


df_raw.reset_index(drop=True, inplace=True)


# Next, let's remove any outliers in the data.

# In[ ]:


outliers = []
# For each feature find the data points with extreme high or low values
for feature in df_raw.keys():
    Q1 = np.percentile(df_raw[feature],25,axis=0)
    Q3 = np.percentile(df_raw[feature],75,axis=0)
    step = 2*(Q3-Q1)
    feature_outlier = df_raw[~((df_raw[feature] >= Q1 - step) & (df_raw[feature] <= Q3 + step))]
    outliers += feature_outlier.index.tolist()


# In[ ]:


len(outliers)/len(df_raw)


# Woah, this doesn't seem right! More than 50% of points are considered outliers according to this condition. I guess, we can't use datetime and basic lat./long. features for outlier detection. Moreover, let's increase the step size to see what we consider outliers.

# In[ ]:


outliers = []
# For each feature find the data points with extreme high or low values
for feature in ['longitutde_traversed','latitude_traversed']:
    Q1 = np.percentile(df_raw[feature],25,axis=0)
    Q3 = np.percentile(df_raw[feature],75,axis=0)
    step = 10*(Q3-Q1)
    feature_outlier = df_raw[~((df_raw[feature] >= Q1 - step) & (df_raw[feature] <= Q3 + step))]
    outliers += feature_outlier.index.tolist()


# In[ ]:


len(outliers)/len(df_raw)


# Even after using 10 times the IQR, about 1% of the data is considered outliers. Let's remove it. (Note: We can have a look at these points to confirm if they are outliers or not)

# In[ ]:


df = df_raw.drop(df_raw.index[outliers]).reset_index(drop = True)


# In[ ]:


len(df)


# Let's split our data into training and validation sets. We will have a validation set of about 10000 rows (Same as test set).

# In[ ]:


y = df_raw.fare_amount
df_raw.drop('fare_amount', axis=1, inplace = True)


# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(df_raw, y, test_size = 10000)


# Next, let's create a scorer function

# In[ ]:


def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    print(res)


# ## Naive Random Forest

# Now, instead of using all the data for training the model, let's only use the first 10000 rows for each tree in the forest. This will save us the hassle of waiting for long time to see our result/changes.

# In[ ]:


set_rf_samples(10000)


# In[ ]:


m = RandomForestRegressor(n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# ## Feature Importance

# Based on the simple model above, let's have a look at the important features for our prediction.

# In[ ]:


fi = rf_feat_importance(m,X_train)
fi[:10]


# In[ ]:


def plot_fi(fi): return fi.plot('cols','imp','barh',figsize=(12,8),legend=False)
plot_fi(fi)


# The plot shows that the fare is highly dependent on the distance traversed features(longitude_traversed, latitude_traversed). Whereas, there are a few features which have almost a non-existent relationship with the model and may not be needed at all.

# ## Predictions on test set

# So with a little feature engineering, even an untuned naive Random Forest seems to be performing better than a Linear Regression. Let's make a submission based on this model.

# In[ ]:


test_set = pd.read_csv(f'{PATH}/test.csv')


# In[ ]:


test_key = test_set.key
test_set.drop('key', axis = 1, inplace = True)


# Let's perform the same feature engineering functions as training set to the test set as well

# In[ ]:


add_datepart(test_set,'pickup_datetime',drop=True, time=True)
distance(test_set)


# In[ ]:


test_predictions = m.predict(test_set)


# ### Submission

# In[ ]:


submission = pd.DataFrame({'key': test_key, 
                           'fare_amount': test_predictions})
submission.to_csv('submissions.csv', index=False)


# ## Improvements

# There are a lot of improvements that can be done:
# 
#     1) The obvious one, tuning the model itself rather than using the default Random Forest settings.
#     2) Using the feature importance chart to explore relevant features further and have better pre-processing (almost non-existent here)
#     3) Creating more features, such as the whether the pickup time is at night or day, distance traversed(rather than lat./long. traversed), etc.
#     4) If we really want to dig deep, we can use external data sources to come up with features as well.
