#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
import os


# This is my first notebook to play a little bit with different ml-algorithms. The goal is to clean and preprocess the data, also to compare <b>LinearRegression, DecissionTree and RandomForest</b>. The structure is as following: <br>
# 
#     
#     
# <ol>
# <li><b>Loading the Data and getting first intuitions</b>
# <ol>
# <li>Identifiying missing values
# <li>Plausibility check of numerical attributes
# <li>Checking values of categorical attributes
# </ol>
#   
# <li><b>Visualization / Scatter-Matrix and Histogram of "prices"</b>
# <ol>
# <li> Rough Data Cleansing for plausible Visualization
# <li> Checking the amount of data after rough cleansing
# </ol> 
# 
# <li><b>Buidling Custom-Transformers and Preprocessing-Pipelines</b>
# <ol>
# <li>Data Cleansing (Filtering, Replacing NaN-Values, OneHotEncoding)
# <li>Custom-Transformers & Pipelines
# <li>Training and comparing Models
# <li>Cross-Validation
# <li>Feature Importance (<font color=red>Note:</font> I can only show numerical features. Help is appreciated if you could help me out with categorical features.)
# </ol>
# 
# <li><b>Final prediction and Conclusion</b>
#         
# </ol>

# # 1. Loading the Data and getting first intuitions

# In[ ]:


cars = pd.read_csv("../input/autos.csv", encoding = "latin-1")

cars.head()


# ### A. Identifiying missing values
# At the first glance we can see, that "vehicleType" and "model" contains <font color =red>NaN-Values</font>. Lets check, how many missing values are there for each column.

# In[ ]:


missing_values = cars.isnull().sum()
missing_values


# It seems that there are 5 columns which contain <font color = red>NaN-Values</font>. We will take care of them later. Now lets get more intuition by describing the dataset.

# ### B. Plausibility check of numerical attributes

# In[ ]:


cars.describe()


# We can clearly see, that we need to do some preprocessing. As example, the max value for <font color = purple>yearOfRegistration</font> is 9999. The max value of <font color = purple>PowerPS </font> is 20.000. The fastest car which i am aware of is "Arash AF10 Hybrid" and has about 2080 Horsepower :). <br>Now lets have a look onto the categorical data.

# ### C. Checking values of categorical attributes

# In[ ]:


cat_val = ["seller", "offerType", "abtest", "gearbox","fuelType", "notRepairedDamage", "nrOfPictures"]

for col in cat_val:
    print ([col]," : ",cars[col].unique())


# When it comes to the categorical values, we can delete <font color = "purple">nrOfPictures</font> (due to single value "0") and <font color = "purple">abtest</font> (assumption that these values with produce noise). Furthermore we will set <font color = "purple">offerType</font> to "Angebot" (offers) and discard "Gesuch" (searches). I assume that people who are looking for an used car will enter a lower price. Moreover there are only 12 search entries, so the filtering will hardly effect the remaining amount of data.

# # 2. Visualization / Scatter-Matrix and Histogram of "prices"

# In this section we will clean the data slighty to make the Visualization more plausible for insights.
# 
# <font color = red>Note:</font> Having a look into the data revealed, that many cars have over 1000 Horsepower which can not be true due to car car brand, model etc. For the sake of practice i will go with my gutfeeling when it comes to choose parameters. Usually, an exhaustive analysis and data cleansing must be done for a suitable data quality.

# ### A. Rough cleansing

# In[ ]:




# Filter bad data
cars_c = cars.copy()
cars_c = cars_c[
    (cars_c["yearOfRegistration"].between(1945, 2017, inclusive=True)) &
    (cars_c["powerPS"].between(100, 500, inclusive=True)) &
    (cars_c["price"].between(100, 200000, inclusive=True))
]


# In[ ]:


num_attributes = ["price", "yearOfRegistration", "powerPS", "kilometer"]
get_ipython().run_line_magic('matplotlib', 'inline')
pd.plotting.scatter_matrix(cars_c[num_attributes], figsize = (12,8), alpha = 0.1)


# Seems like that the most cars haven been registered after 1990 and have lower horsepower than 180. Many cars have more than approximately 130.000 kilometers. <font color = purple> yearOfRegistration </font> and <font color = purple> powerPS </font> show an effect onto the price. We will find this out later by "feature importance".

# In[ ]:


cars_c["price"].hist(bins = 50, log = True)


# ### B. Discarded amount

# In[ ]:


# Discarded amount of the Data

print("Current Data Amount : ", cars_c.shape[0]/cars.shape[0] * cars.shape[0], "\n","Current Data Amount %: ", cars_c.shape[0]/cars.shape[0])


# Wow, half of the Data was discarded due to previous filtering.

# # 3. Buidling Custom-Transformers and Preprocessing-Pipelines

# ### A. Data Cleansing
# 
# First, we will do some data cleansing:<br>
# - Filtering "bad data"
# - Replacing NaN-Values, options:
# <ol>
# <li>Delete the whole NaN-Columns (obviously bad idead)
# <li>Delete the NaN-Rows (better, but not wanted, since we already filtered almost the half of the dataset)
# <li>Replace the NaN-Values with dummies (Best option in my opinion)
# </ol>
# - Assigning codes to categorical attributes for OneHotEncoder
# 
# 
# 

# In[ ]:


# Fresh copy
cars_clean = cars.copy()

# Filter bad data
cars_clean = cars_clean[
    (cars_clean["yearOfRegistration"].between(1945, 2017, inclusive=True)) &
    (cars_clean["powerPS"].between(100, 500, inclusive=True)) &
    (cars_clean["price"].between(100, 200000, inclusive=True)) &
    (cars_clean["offerType"] == "Angebot") 
]

# Replace the NaN-Values
cars_clean['vehicleType'].fillna(value='blank', inplace=True)
cars_clean['gearbox'].fillna(value='blank', inplace=True)
cars_clean['model'].fillna(value='blank', inplace=True)
cars_clean['fuelType'].fillna(value='blank', inplace=True)
cars_clean['notRepairedDamage'].fillna(value='blank', inplace=True)

# Change categorical attributes dtype to category

for col in cars_clean:
    if cars_clean[col].dtype == "object":
        cars_clean[col] = cars_clean[col].astype('category')
        
# Assign codes to categorical attribues instead of strings

cat_columns = cars_clean.select_dtypes(['category']).columns

cars_clean[cat_columns] = cars_clean[cat_columns].apply(lambda x: x.cat.codes)
        
    
# Drop probably unuseful columns

drop_cols = ["dateCrawled", "abtest", "dateCreated", "nrOfPictures", "lastSeen"]

cars_clean = cars_clean.drop(drop_cols, axis=1)


# In[ ]:


cars_clean.head()


# In[ ]:


# Getting the train and test sets
train_set, test_set = train_test_split(cars_clean, test_size = 0.2, random_state = 42)

# Seperation of Predcitors (Features) and the Labes (Targets)

cars_price = train_set["price"].copy()
cars = train_set.drop("price", axis=1)


# ### B. Custom-Transformers and Pipelines

# In[ ]:


# Since Scikit-Learn doesn't hanldes DataFrame, we build a class for it

# Create a class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
class DFSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


# In[ ]:


# Setting categorical and numerical attributes

cat_attribs = ["name", "seller", "offerType", "vehicleType", "fuelType", "brand", "notRepairedDamage"]
num_attribs = list(cars.drop(cat_attribs, axis=1))

# Building the Pipelines

num_pipeline = Pipeline([
    ("selector", DFSelector(num_attribs)),
    ("std_scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("selector", DFSelector(cat_attribs)),
    ("encoder", OneHotEncoder(sparse=True))
])

full_pipeline = FeatureUnion(transformer_list =[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline)
])


# In[ ]:


cars_prepared = full_pipeline.fit_transform(cars)


# ### C. Training and Comparing Models

# In[ ]:


lin_reg = LinearRegression()
lin_reg.fit(cars_prepared, cars_price)


# In[ ]:


from sklearn.metrics import mean_squared_error


# In[ ]:


cars_predictions = lin_reg.predict(cars_prepared)
lin_mse = mean_squared_error(cars_price, cars_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[ ]:


cars_predictions[0:4]


# In[ ]:


list(cars_price[0:4])


# In[ ]:


from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(cars_prepared, cars_price)


# In[ ]:


cars_predictions = tree_reg.predict(cars_prepared)
tree_mse = mean_squared_error(cars_price, cars_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(random_state=42, n_jobs =-1, max_depth = 30 )
forest_reg.fit(cars_prepared, cars_price)


# In[ ]:


cars_predictions = forest_reg.predict(cars_prepared)
forest_mse = mean_squared_error(cars_price, cars_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# LinearRegression does very well. DecisionTree is either perfect or it overfitted badly. CrossValidation will show us the truth. RandomFroest is slightly better than LinearRegression. I played a little bith with the "max_depth" to have a good comprise between result and computation power.

# ### D. Cross-Validation

# In[ ]:


from sklearn.model_selection import cross_val_score

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# LinReg - CrossValidation

# In[ ]:


# Offline i used CV=10

scores = cross_val_score(lin_reg, cars_prepared, cars_price,
                         scoring="neg_mean_squared_error", cv=4)
lin_rmse_scores = np.sqrt(-scores)

display_scores(lin_rmse_scores)


# DecissionTree - CrossValidation

# In[ ]:


# Offline i used CV=10

scores = cross_val_score(tree_reg, cars_prepared, cars_price,
                         scoring="neg_mean_squared_error", cv=4)
tree_rmse_scores = np.sqrt(-scores)

display_scores(tree_rmse_scores)


# RandomForest - CrossValidation

# In[ ]:


# Offline i used CV=8

from sklearn.model_selection import cross_val_score

scores = cross_val_score(forest_reg, cars_prepared, cars_price,
                         scoring="neg_mean_squared_error", cv=2)
forest_rmse_scores = np.sqrt(-scores)

display_scores(forest_rmse_scores)


# The Cross-Validation reveals, that LinearRegression is doing way worse, especially the DecisionTree. RandomForest seems to be the best alternative.

# ### E. Feature Importance
# 
# As mentioned before, i am struggling with displaying the names of the categorical features. Help would be great :)

# In[ ]:


feature_importances = forest_reg.feature_importances_
feature_importances


# In[ ]:


cat_encoder = cat_pipeline.named_steps["encoder"]
#cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs #+ cat_encoder
sorted(zip(feature_importances, attributes), reverse=True)


# # 4. Final Prediction and conclusion

# In[ ]:


final_model = forest_reg

cars_test = test_set.drop("price", axis = 1)
cars_price_test = test_set["price"].copy()

cars_test_prepared = full_pipeline.transform(cars_test) ## call transform NOT fit_transform


from sklearn.metrics import mean_squared_error
final_predictions = final_model.predict(cars_test_prepared)

final_mse = mean_squared_error(cars_price_test, final_predictions)

final_rmse = np.sqrt(final_mse)


# In[ ]:


final_rmse


# In[ ]:


final_model.score(cars_test_prepared, cars_price_test)


# This Notebook was my first touch with machine learning. It was meant as a first hands on, instead of realizing the best model. But i think a score of 0.83 is not that bad at all. I've learned that the predictions of the training set are vague and cross validation is undispensable. Furthermore RandomForest seems to be a good choice for this dataset. Playing with a little bit more hyperparamters and better data cleasing should squeeze better results.
