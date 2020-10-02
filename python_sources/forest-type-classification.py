#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # Classify Forest Types
# One of Kaggle competitions. The task is to predict types of trees based on geographic features. The dataset was provided by Jock A. Blackard and Colorado State University.

# ## References
# *   This [Note Book](https://www.kaggle.com/evimarp/top-6-roosevelt-national-forest-competition/notebook#Distances-analysis) inspired me how to explore each feature more precisely.
# *   This [Course](https://www.kaggle.com/learn/intro-to-machine-learning) is a great source to learn about basics of machine learning.

# In[ ]:


# import libraries
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 
from scipy.stats import norm 


# In[ ]:


forest_data = pd.read_csv("../input/learn-together/train.csv")
forest_test = pd.read_csv("../input/learn-together/test.csv")


# ## Data Preprocessing

# Investigave different aspect of data, like data type, missing data, data distribution and etc.
# <br>Using `.info()` we can see that there is no missing data and all data types are `int64`, so we don't need further process to deal with these two.

# In[ ]:


forest_data.info()


# In[ ]:


forest_data.describe()


# In[ ]:


forest_data.head()


# In[ ]:


forest_data.columns


# In[ ]:


forest_data.dtypes


# In[ ]:


forest_data.shape


# ### Interactive Visualization

# In[ ]:


# yet I didn't use any


# ###Correlation Analysis
# This step is required to investigate possible relation and dependency between different features.

# In[ ]:


#corrMatt = X_train[["","","","","","",""]].corr()
corrmat = forest_data[[ 'Elevation', 'Aspect', 'Slope',
       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
       'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',
       'Cover_Type']].corr()

f, ax = plt.subplots(figsize =(11, 10)) 
sns.heatmap(corrmat, annot=True, ax = ax, cmap ="YlGnBu", linewidths = 0.1)


# Reviewing these the plot, we can identify that there are relations between ( I only considered features with corr > 0.5)
# 
# 
# *   `Elevtion`  and   `Horizontal_Distance_To_Roadways`
# *   `Aspect` and `Hillshade_3pm`
# 
# 
# *   `Horizontal_Distance_To_Hydrology` and `Vertical_Distance_To_Hydrology`
# *   `Hillshade_noon` and `Hillshade_3pm`
# 
# 
# *   `Horizontal_Distance_To_Fire_Points` and `Horizontal_Distance_To_Roadways`
# 
# 
# 
# 
# 
# 

# Now using histogram we investigate distribution of trees based on different features to see what pattern we can find.
# <br> `Soil_Types` and `Wilderness_Area` are categorical variables and shows us absence or presence of each soil type or wilderness area type.

# #### Elevation

# In[ ]:


# Elevation in meters
sns.distplot(forest_data.Elevation, color="b")


# It shows that must of the forest cover are in elevation between 2050 ~ 3450 m.

# Investigate change in other features with `Elevetion` .

# In[ ]:


f, axes = plt.subplots(3, 1, figsize=(15, 15), sharex=True, sharey=True)
sns.scatterplot(y='Hillshade_9am', x='Elevation', 
                 data=forest_data, ax=axes[0])
sns.scatterplot(y='Hillshade_Noon', x='Elevation', 
                 data=forest_data, ax=axes[1])
sns.scatterplot(y='Hillshade_3pm', x='Elevation', 
                 data=forest_data, ax=axes[2])


# #### Hillshade
# `Hillshade`: 0 to 255 index, obtains the hypothetical illumination of a surface by determining illumination values for each cell.
# <br>Most of the areas have high sunlight in 9 am and noon (in expected elevation with forest cover). This should be obvious in Hillshade histograms. 
# <br> The plot related to 3 pm is distributed normaly but the two other is more pushed toward higher values.

# In[ ]:


f, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

sns.distplot(forest_data.Hillshade_9am, color="y", ax=axes[0])
sns.distplot(forest_data.Hillshade_Noon, color="b", ax=axes[1])
sns.distplot(forest_data.Hillshade_3pm, color="g", ax=axes[2])


# #### Aspect and Slope
# `Aspect`: indicates the direction in which the cell's slope faces, measures counterclockwise in degrees from 0 (due north) to 360.

# In[ ]:


f, axes = plt.subplots(2, 1, figsize=(15, 15), sharex=True, sharey=False)

sns.distplot(forest_data.Slope, color="y", ax=axes[0])
sns.distplot(forest_data.Aspect, color="b", ax=axes[1])


# These two show that most of forest cover are concentrated in areas with 0~60 degree slope and are facing toward ( `Aspect` ): North, North-West, West and North-East.

# In[ ]:


f, axes = plt.subplots(2, 1, figsize=(15, 15), sharex=True, sharey=False)
sns.scatterplot(y='Slope', x='Aspect', 
                 data=forest_data, ax=axes[0])
sns.scatterplot(y='Aspect', x='Slope', 
                 data=forest_data, ax=axes[1])


# # Data Processing
# Simple analysis on data using Random Forest.

# In[ ]:


forest_train = forest_data.drop(["Id"], axis = 1)

forest_test_id = forest_test["Id"]
forest_test = forest_test.drop(["Id"], axis = 1)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Split X and y in train and validation sets
X_train, X_val, y_train, y_val = train_test_split(forest_train.drop(['Cover_Type'], axis=1), forest_train['Cover_Type'], test_size=0.2, random_state = 50)

# Define model
forest_model = RandomForestClassifier(n_estimators=100, random_state=50)
# Fit the model to train data
forest_model.fit(X_train, y_train)


# In[ ]:


# Check the model accuracy
from sklearn.metrics import classification_report, accuracy_score
forest_model.score(X_train, y_train)


# In[ ]:


# Make prediction
forest_preds = forest_model.predict(X_val)

accuracy_score(y_val, forest_preds)


# 

# In[ ]:


# Select features
features = ['Elevation', 'Aspect', 'Slope',
       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm','Horizontal_Distance_To_Fire_Points']

forest_data_reduced = forest_data[features]


# Seperate features and label (Cover_Type) and save them into `X` and `y`.

# In[ ]:


X = forest_data_reduced.copy()
y = forest_data['Cover_Type']


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Split X and y in train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state = 2)

# Define model
forest_model = RandomForestClassifier(n_estimators=100, random_state=2)
# Fit the model to train data
forest_model.fit(X_train, y_train)


# In[ ]:


# Check the model accuracy
from sklearn.metrics import classification_report, accuracy_score
forest_model.score(X_train, y_train)


# In[ ]:


# Make prediction
forest_preds = forest_model.predict(X_val)
accuracy_score(y_val, forest_preds)


# Try different numbers of leaf for Random Forest model to find out which gives us better accuracy.

# In[ ]:


# Define a function to calculate accuracy_score
def acc_calculate(max_leaf_nodes, X_train, X_val, y_train, y_val):
    model = RandomForestClassifier(n_estimators=100,max_leaf_nodes=max_leaf_nodes, random_state=50)
    model.fit(X_train, y_train)
    val_preds = model.predict(X_val)
    acc = accuracy_score(y_val, val_preds)
    return(acc)


# In[ ]:


# compare accuracy_score with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000, 10000]:
    my_acc = acc_calculate(max_leaf_nodes, X_train, X_val, y_train, y_val)
    print("Max leaf nodes: %d  \t\t accuracy_score:  %f" %(max_leaf_nodes, my_acc))


# ## Prediction
# Use test data to make prediction.

# In[ ]:


# Run the best model to be used for prediction
X_train, X_val, y_train, y_val = train_test_split(forest_train.drop(['Cover_Type'], axis=1), forest_train['Cover_Type'], test_size=0.2, random_state = 50)

# Define model
forest_model = RandomForestClassifier(n_estimators=100, random_state=50)
# Fit the model to train data
forest_model.fit(X_train, y_train)


# In[ ]:


test_preds = forest_model.predict(forest_test)


# In[ ]:


# To submit on kaggle
output = pd.DataFrame({'Id': forest_test_id,
                       'Cover_Type': test_preds})
output.to_csv('submission.csv', index=False)

