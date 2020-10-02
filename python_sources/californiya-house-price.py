#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


house = pd.read_csv('../input/californiya-housing/Housing_californiya.csv')
house.head(3)


# In[ ]:


house.info()


# In[ ]:


house.describe()


# In[ ]:


house.shape


# In[ ]:


#Discover and visualize the data to gain insights
house.hist(bins=50, figsize=(9,9))


# In[ ]:


house1=house.copy()


# In[ ]:


#Visualizing Geographical Data
#We had geographical information (latitude and longitude) for a district. So lets plot all the districts
#and visualize the data.
house1.plot(kind="scatter", x="longitude", y="latitude")


# In[ ]:


house1.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=house["population"]/100, label="population", figsize=(10,7),
    c="median_house_value",colormap='plasma', colorbar=True,
    sharex=False)
#Housing prices are very much related to the location and to the population density find by this graph.


# In[ ]:


corr_matrix=house1.corr()
corr_matrix
#Median house values tends to go up when the median income goes up. (0.68)-find.


# In[ ]:


house1.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)
#here line show that there are problem in data collection.


#  ****Attributes Combinations****

# In[ ]:


house1["rooms_per_household"] = house1["total_rooms"]/house1["households"]
house1["bedrooms_per_room"] = house1["total_bedrooms"]/house1["total_rooms"]
house1["population_per_household"]=house1["population"]/house1["households"]


# In[ ]:


house1.shape


# In[ ]:


house1.corr()


# In[ ]:





# Prepare the data for Machine Learning algorithms
# 1. Data Cleaning

# In[ ]:


housing_features = house1.drop('median_house_value', axis = 1)
print(housing_features.columns)
housing_features.head()


# In[ ]:


housing_target = house1['median_house_value']
housing_target.shape


# In[ ]:


x=housing_features
y=housing_target


# In[ ]:


x.isnull().sum()


# In[ ]:


# housing1.dropna()                                                                 # method 1
# housing1.drop("total_bedrooms", axis = 1)                                         # method 2
# housing1['total_bedrooms'].fillna(housing1['total_bedrooms'].median(), inplace = True)   # method 3
# scikit learn provides a handy class to take care of missing values: SimpleImputer
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")


# In[ ]:


housing_num = x.drop("ocean_proximity", axis=1)
housing_num.columns


# In[ ]:


imputer.fit(housing_num)


# In[ ]:


imputer.statistics_


# In[ ]:


transformed_values = imputer.transform(housing_num)
transformed_values


# In[ ]:


housing_transformed = pd.DataFrame(transformed_values)
housing_transformed.head()


# In[ ]:


housing_transformed.columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income',
       'rooms_per_household', 'bedrooms_per_room', 'population_per_household']
housing_transformed.head()


# In[ ]:


housing_transformed.isnull().sum()


# 2. Handling Text and Categorical attribute

# In[ ]:


from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()


# In[ ]:


housing_cat = house[['ocean_proximity']]


# In[ ]:


dummy_values=cat_encoder.fit_transform(housing_cat)
dummy_values


# In[ ]:


dummy_values.toarray()


# In[ ]:


cat_encoder.categories_


# In[ ]:


housing_cat = pd.DataFrame(dummy_values.toarray())
housing_cat.head()


# In[ ]:


housing_cat.columns = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
housing_cat.head()


# 3. Feature Scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()


# In[ ]:


std_scaler.fit(housing_num)


# In[ ]:


housing_num.describe()


# In[ ]:


scaled_values=std_scaler.transform(housing_num)


# In[ ]:


housing_scaled = pd.DataFrame(scaled_values)


# In[ ]:


housing_scaled.columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income',
       'rooms_per_household', 'bedrooms_per_room', 'population_per_household']


# In[ ]:


housing_scaled.describe()


# 4. Transformations Pipeline

# In[ ]:


from sklearn.pipeline import Pipeline
num_pipeline = Pipeline([
('imputer', SimpleImputer(strategy="median")),
('std_scaler', StandardScaler()),
])
housing_num_tr = num_pipeline.fit_transform(housing_num)


# In[ ]:


housing_num_tr


# In[ ]:


housing_pipeline = pd.DataFrame(housing_num_tr)


# In[ ]:


housing_pipeline.columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income',
       'rooms_per_household', 'bedrooms_per_room', 'population_per_household']


# In[ ]:


housing_pipeline.head()


# ColumnTransformer : to apply all the transformation together on the housing dataset

# In[ ]:


from sklearn.compose import ColumnTransformer


# In[ ]:


num_attribs = list(housing_num)
print(num_attribs)
cat_attribs = ["ocean_proximity"]
print(cat_attribs)


# In[ ]:


housing_features.columns


# In[ ]:


full_pipeline = ColumnTransformer([
("num", num_pipeline, num_attribs),
("cat", OneHotEncoder(), cat_attribs),
])
housing_prepared = full_pipeline.fit_transform(housing_features)


# In[ ]:


housing_prepared = pd.DataFrame(housing_prepared)
housing_prepared.shape


# In[ ]:


housing_prepared.head()


# In[ ]:


housing_prepared.columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income','rooms_per_household', 'bedrooms_per_room',
       'population_per_household','<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']


# Create a train_Test spliting

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train,y_test = train_test_split(housing_prepared,house['median_house_value'], test_size=0.2, random_state=42)
X_train.shape, X_test.shape,y_train.shape, y_test.shape


# Select a model and train it

# In[ ]:


from sklearn.linear_model import LinearRegression
model_reg = LinearRegression()


# In[ ]:


model_reg.fit(X_train, y_train)


# In[ ]:


model_reg.score(X_train, y_train)


# In[ ]:


y_pred_train = model_reg.predict(X_train)
y_pred_train


# In[ ]:


from sklearn.metrics import mean_squared_error


# In[ ]:


lin_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
lin_rmse


# check with test data

# In[ ]:


y_pred_test = model_reg.predict(X_test)


# In[ ]:


SSE = np.sum((y_pred_test-y_test)**2)
SSE


# In[ ]:


SST = np.sum((y_test-np.mean(y_train))**2)


# In[ ]:


r_square= 1 - SSE/SST
r_square


# In[ ]:





# In[ ]:




