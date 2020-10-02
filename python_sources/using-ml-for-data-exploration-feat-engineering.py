#!/usr/bin/env python
# coding: utf-8

# In this kernel I'll try something different from the standard Exploratory Data Analysis. I will use Machine Learning for my EDA and base my Feature Engineering on the findings. I will use a Random Forest as it is fast to implement and provides Feature Importances. I can then use the Feature Importances to focus my time on the most important features. If a feature turns out to be 100 times more important than another feature, then I will spend 100 times more time on it.
# 
# The steps I will take are:
# 
# 1. Read in a subset of the entire dataset
# 2. Do some initial data processing, feature engineering and handle missing data using the fast AI library
# 3. Run a simple Random Forest
# 4. Get the Feature Importances
# 5. Create a cluster analysis to find similarities between features and plot them in a dendrogram

# ## Load the Libraries

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

from fastai.imports import *
from fastai.structured import *
from sklearn.ensemble import RandomForestRegressor
from IPython.display import display
from sklearn import metrics


# In[ ]:


# Set the plot sizes
set_plot_sizes(12,14,16)


# ## Read in the Data

# I am reading in 100k rows from the training set. This will be enough to do a fast, initial EDA. I am also not reading in 'key' of the training set as it is just an id and I predefine the datatypes to save memory space. It is probably also neceassary to reduce the longitude and lattitude data to float32 from float64, which will slightly affect their values but reduce memory significantly. This is however only important if you read in the whole dataset.

# In[ ]:


PATH = "../input/"
df_raw = pd.read_csv(f'{PATH}train.csv', nrows=100000, parse_dates=['pickup_datetime'], dtype={'passenger_count': 'int8', 'fare_amount': 'float16'}, 
                     usecols=['fare_amount', 'pickup_datetime','pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude','passenger_count'])
df_raw_test = pd.read_csv(f'{PATH}test.csv', parse_dates=['pickup_datetime'], dtype={'passenger_count': 'int8'})


# In[ ]:


# Expands the summary tables if there are a lot of columns androws
def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)


# In[ ]:


# Shows the last 5 rows of the traning set
display_all(df_raw.tail().T)


# In[ ]:


# Shows summary of training set
display_all(df_raw.describe(include='all').T)


# In[ ]:


# Shows summary of test set
display_all(df_raw_test.describe(include='all').T)


# The summaries show some interesting things:
# 
# 1. Training and test set have the same start and end dates - let's look at this in more detail in a bit
# 2. In the training set there seem to be some outliers in latitudes and longitudes. There aren't any obvious outliers in the test set.
# 3. The training set can have negative fare mounts
# 4. The training set has rides with 0 passengers
# 5. The means of both datasets are fairly similar but the standard deviation of training set is way bigger than that of the test set, also indicating outliers

# ## Comparing Number of Taxi Rides in Training and Test Set

# Interestingly the number of rides per month and year is fairly constant in the training set but there are marked spikes in the test set. It is quite possible that this may affect prediction errors if time plays an important role. It's too early to say but certainly something to keep in mind.

# In[ ]:


plt.figure(figsize=(20, 4))
df_raw['pickup_datetime'].groupby([df_raw["pickup_datetime"].dt.year, df_raw["pickup_datetime"].dt.month]).count().plot(kind="bar")
plt.title('Traing Set Rides per Month and Year')
plt.show()


# In[ ]:


plt.figure(figsize=(20, 4))
df_raw_test['pickup_datetime'].groupby([df_raw_test["pickup_datetime"].dt.year, df_raw_test["pickup_datetime"].dt.month]).count().plot(kind="bar")
plt.title('Test Set Rides per Month and Year')
plt.show()


# ## Outliers

# ### Negative Fare Amount

# In[ ]:


df_raw[df_raw['fare_amount'] < 0]


# ### Pickup Longitude Outliers

# * **Negative longitude outliers: ** The -736 might be -73.6 instead
# * **Positive longitude outliers:** There are a lot of cases with incorrect longitudes and lattitudes in fact because they are all set to 0. These rows need to go.

# In[ ]:


# Large negative longitudes
df_raw[df_raw['pickup_longitude'] < -75]


# In[ ]:


# Large positive longitudes
df_raw[df_raw['pickup_longitude'] > -73]


# ### Pickup Latitude Outliers

# * **Small Positive Latitudes:** This is a similar story to the **Large Positive Longitudes**
# * **Large Positive Latitudes:** There seem to be two outliers. The 401 might be 40.1 in fact.

# In[ ]:


# Small positive latitudes
df_raw[df_raw['pickup_latitude'] < 40]


# In[ ]:


# Large positive latitudes
df_raw[df_raw['pickup_latitude'] > 42]


# ## Initial Processing

# ### Delete Outliers

# I will delete all outliers to keep things simple, even those that could potentially be amended by changing the decmal place.

# In[ ]:


df_raw.shape


# In[ ]:


df_raw = df_raw[df_raw['pickup_longitude'] > -76]
df_raw = df_raw[df_raw['pickup_longitude'] < -73]
df_raw = df_raw[df_raw['pickup_latitude'] > 40]
df_raw = df_raw[df_raw['pickup_latitude'] < 44]


# In[ ]:


df_raw.shape


# I deleted 2008 rows.

# ### Off-the-Shelf Data Type Conversion and Feature Engineering

# In[ ]:


# Converts all strings to categorical features
train_cats(df_raw)


# In[ ]:


# Splits dates into subcomponenets
add_datepart(df_raw, 'pickup_datetime')


# In[ ]:


df_raw.info()


# In[ ]:


# Splits the data into independent and dependent features and keeps a column 'nas' that keeps track of features that had missing values and had to be imputed
df, y, nas = proc_df(df_raw, 'fare_amount')


# ## Initial Model

# In[ ]:


m = RandomForestRegressor(n_estimators=30, min_samples_leaf=3, oob_score=True, n_jobs=-1)
m.fit(df, y)


# ## Feature Importance

# Now let's find the most important features. This will give us an idea on which features to focus on the most. And what we find is that longitude and latitude data matters the most. In fact there is a big gap between those four features and the next one, which is Time Elapsed and then there is another big gap.

# In[ ]:


# Calculate Feature Importance
fi = rf_feat_importance(m, df); fi[:10]


# In[ ]:


# Shows the table above visually
fi.plot('cols', 'imp', figsize=(10,6), legend=False)
plt.title('Feature Importance by Feature');


# In[ ]:


def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)


# In[ ]:


# Same visualisation but easier to see which feature contributes how much
plot_fi(fi[:30])
plt.title('Feature Importance by Feature');


# ## Similar Features

# The only features that are quite similar to each other are Day of Year, Month and Week as well as Time Elapsed and Year. This doesn't seem very useful as the numbers are probably just very similar but I will need to confirm this.
# 
# The way to read the plot is the following. The x-axis shows the similarity or distance between the features. The longer the lines are the less similar two features are.

# In[ ]:


from scipy.cluster import hierarchy as hc


# In[ ]:


corr = np.round(scipy.stats.spearmanr(df).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=df.columns, orientation='left', leaf_font_size=16)
plt.title('Feature Similarities')
plt.show()


# ## Conclusion and Next Steps

# In conclusion I would say the following:
# 
# 1. Geographical features were by far the most important and some additional features based on them should be built such as Manhattan distance
# 2. Additional data sources including weather or holidays would be a useful complement
# 3. Do a similar analysis on test and train data that was done on months and years but on longitudes and latitudes - this can also be done on the entire dataset if you read it in in chunks
# 4. Look at missing data in more detail. I have skimmed over this here and simply imputed them using the mean
# 5. Do an actual prediction. Random Forest gives good intial results but other algorithms will be necessary to increase the accuracy. Neural networks have the advantage that they can extrapolate, i.e. make a prediction on data (like longitudes outside the training set longitudes) they haven't seen before, which is something a Random Forest cannot do. RF will do a prediction but it won't be very good.
# 6. Remove features with low feature importance and check the results. That may help with overfitting.
#     
