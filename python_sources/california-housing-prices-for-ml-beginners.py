#!/usr/bin/env python
# coding: utf-8

# Hi Everyone! Today I'll like to explain the popular California Housing Price Prediction that is easy to understand for 100% beginners.  (https://www.kaggle.com/camnugent/california-housing-prices) If you find that there is any error here or anything that can be better explained, please feel free to comment. Otherwise, let me go ahead and get started. 
# 
# # The Data
# 
# The dataset pertains to the houses found in a given California district and some summary stats about them based on the 1990 census data. We will be using this dataset in order to predict what housing prices could be in the future. Obviously the data is outdated but it will be a good way to help learn machine learning. This is also included in Kaggle Machine Learning course which I highly recommend you check out. Here are some information on the dataset that we are given: 
# 
# 1. **longitude**: A measure of how far west a house is; a higher value is farther west
# 
# 2. **latitude**: A measure of how far north a house is; a higher value is farther north
# 
# 3. **housingMedianAge**: Median age of a house within a block; a lower number is a newer building
# 
# 4. **totalRooms**: Total number of rooms within a block
# 
# 5. **totalBedrooms**: Total number of bedrooms within a block
# 
# 6. **population**: Total number of people residing within a block
# 
# 7. **households**: Total number of households, a group of people residing within a home unit, for a block
# 
# 8. **medianIncome**: Median income for households within a block of houses (measured in tens of thousands of US Dollars)
# 
# 9. **medianHouseValue**: Median house value for households within a block (measured in US Dollars)
# 
# 10. **oceanProximity**: Location of the house w.r.t ocean/sea
# 

# # Importing Libraries 
# Let's get started by importing the relevant packages. For those who are curious, here's what we are importing: 
# 1. **Pandas**: one of the most popular libraries on python which is used for data analysis
# 2. **NumPy**: a library used for working with arrays, which is much faster than lists
# 3. **%matplotlib inline**: used in interactive environments like Jupyter or this Kaggle kernal in order to make your plot outputs appear and be stored within the notebook
# 4. **Matplotlib**: a plotting library for the Python
# 5. **matplotlib.pyplot**: Supports logarithmic and logit scales
# 
# 

# In[ ]:


import pandas as pd
import numpy as np 
np.random.seed(42)

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl 
import matplotlib.pyplot as plt
mpl.rc("axes", labelsize = 14)
mpl.rc("xtick", labelsize = 12)
mpl.rc("ytick", labelsize = 12) 


# In[ ]:


import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)


# # Loading & Understanding The Data 
# Next, let us import the file and take a brief look at the data using .info() 
# There are two things to look out for here:
# 
# **1. Are there null values?**
# 
# Most machine learning algorithms cannot deal with Null values. These null values must be either filled or dropped. 
# 
# **2. The type of data** 
# 
# If they are non numeric, categorical data, we need to transform them to numeric data as well.   
# ***
# 
# 
# How we do so will be discussed later. Let's take a look at the data first! 
# 

# In[ ]:


housing = pd.read_csv('../input/california-housing-prices/housing.csv')
housing.info()


# We note that total_bedrooms have some null values and ocean_proximity is an object. Let's check out what does categories does ocean_proximity consist of. 

# In[ ]:


housing.head()


# In[ ]:


housing.ocean_proximity.unique()


# We find that ocean_proximity contains 5 categories, which we can work with later on. But for now, let's do a bit more data exploration. 

# # Data Visualisation 
# Let's plot out all the data using the longitude and latitude on a scatter plot. This will give us a clearer ideas of where the houses are clustered at. 

# In[ ]:


housing.plot(kind='scatter',x='longitude', y='latitude', alpha=0.1)


# Aside from plotting the longitude and latitude, we can also plot other data points to get a better understanding of the data. Here, we have decided to use the population as shown by size of circles (s here is used to represent shape) and median house value as shown by the colour (c here is used to represent the colour). 
# 
# 
# * cmap: used to determine the colourmap we want. In this case, I used the one called "OrRd". You can find more here (https://matplotlib.org/3.2.1/tutorials/colors/colormaps.html) 
# 
# * sharex: controls sharing of properties among x (sharex) or y (sharey) axes. In this case, we don't want to share the axes among all the other data. 
# 
# 

# In[ ]:


housing.plot.scatter(x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("OrRd"), colorbar=True,
    sharex=False)
plt.legend()


# # Exploring Correlation Between Features
# Aside from looking at it visually, we can also calculate the correlation between all the features and the median house values using the correlation matrix. The .corr() function is used to compute pairwise correlation of columns in a dataframe, excluding NA/null values.

# In[ ]:


corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending = False)


# # Alright! Let's Start Working On The Data 
# 
# We start by splitting the data into the training dataset and the testing dataset. A good mix would be 80% for training and 20% for testing. 

# In[ ]:


from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=1)


# Selecting the data we need to work with. We drop median_house_value as that is what we are trying to predict and ocean proximity (for now) as it is categorical. 

# In[ ]:


chosen_features = ['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income']
housing_train_set = train_set[chosen_features]
housing = train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = train_set["median_house_value"].copy()


# Back to the null data! Remember some of total_bedrooms data was null. To fix that, we can easily replace that with the median. However, before you replace it with the median, it needs to first calculate the median. This step of calculating is done using the .fit(). Next, we want to replace the dataset with the newly calculated value and return the new one. That is what the .transform() function does. 
# 
# After that we build the data frame with the newly transformed values. Now total_bedrooms have no null values. 

# In[ ]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(housing_train_set)
X = imputer.transform(housing_train_set)
housing_tr = pd.DataFrame(X, columns=housing_train_set.columns, index=housing_train_set.index)
housing_tr.info()


# Back to the categorical data, ocean_proximity. In order to analyse it, we can use the OneHotEncoder function. 

# In[ ]:


from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
cat_ocean_proximity = housing[['ocean_proximity']]
housing_cat = cat_encoder.fit_transform(cat_ocean_proximity)
#Change it to an array 
housing_cat.toarray()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])


from sklearn.compose import ColumnTransformer

num_attribs = list(housing_train_set)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)


# # Train Your Model 
# Finally we train our model. Here I used the RandomForestRegressor model.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)


# After that, we can evaluate the model by calculating the root mean squared error. 

# In[ ]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# Alright, I've come to the end of it! Over here, I didn't validate the model using the test data, but I figured it was a good start. For this practice, I really have to credit this video: https://www.youtube.com/watch?v=z1gibQGgbdg&list=PLnWSiezypm5xwNXilf31sRP_30wtf3i6j&index=6&t=2234s for helping me tons in understanding machine learning better along with the Kaggle Machine Learning course.
