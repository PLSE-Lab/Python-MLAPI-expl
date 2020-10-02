#!/usr/bin/env python
# coding: utf-8

# **UNT INFO 5731 Term Project on Waze Historial Data **
# * Members: Moneerah Alboulayan, Schenita Floyd, Cary Jim

# In[ ]:


#File to be loaded from the directory in Kaggle and importing library
import os
print(os.listdir("../input"))
import pickle
import pandas as pd


# In[ ]:


#Load the dataset in pandas dataframe and print the first 5 columns 
dataset = pd.read_pickle("../input/time_converted_df.pickle")
print(dataset.head())


# In[ ]:


#Printing the dataset information of the datatype
print(dataset.info())


# In[ ]:


#Identify unique values in the EVENT_STATE column to see which type of event status 
dataset.EVENT_STATE.unique()


# **Exploring and Cleaning of Data**
# * After we examine the basic information in the dataset we proceed to data exploration and data cleaning for the modelling in this project. 

# In[ ]:


# Create new DF of dataset with only Dallas county 
Dallasdataset = dataset[dataset["COUNTY"].isin(["Dallas"])]


# In[ ]:


#Exploring Dallas Dataset to see what needs to be cleaned
print(Dallasdataset.shape)


# In[ ]:


#Listing of unique values of Dallas County data
print(Dallasdataset.nunique())


# In[ ]:


# Adding Day of Week column 
Dallasdataset['dayofweek'] = Dallasdataset['time'].dt.dayofweek 
Dallasdataset.head()


# In[ ]:


# Checking information on the dataset again after adding the column
print(Dallasdataset.shape)


# In[ ]:


# Duplicates Check
Dallasdataset = Dallasdataset.drop_duplicates(subset='EVENT_ID', keep='first')


# In[ ]:


# Making sure duplicates were removed
# The result rows are now down to 1204434 
print(Dallasdataset.shape)


# In[ ]:


# Splitting date and time
# date format for time field 2018-11-01 00:00:26
from datetime import datetime   
Dallasdataset['Dates'] = pd.to_datetime(Dallasdataset['time']).dt.date
Dallasdataset['Time'] = pd.to_datetime(Dallasdataset['time']).dt.time


# In[ ]:


#Checking the dataset after the manipulation of date and time
#Two more columns are added to the dataframe
print(Dallasdataset.shape)


# In[ ]:


#Taking a look at the dataframe 
Dallasdataset.head()


# In[ ]:


# Cleaning out extra cities that is not part of the Dallas county
# Printing the dataset shape, it is down to 1204422 rows 
Dallasdataset = Dallasdataset[~Dallasdataset['CITY'].isin(["Arlington" , "Fate" , "Lacy-Lakeview" , "McKinney" , "Plano" , "Rice" , "Royse City" , "Talty" , "Terrell" , "Waco"])]
print (Dallasdataset.shape)


# **Summary for Exploring and Cleaning **
# * EVENT ID is based on date and time ranging from 11-01-2018 through 1-31-2019 
# * EVENT State has only two categories closed or updated and only two records were labled updated 
# * EVENT TYPE has 32 unique values with no need to clean. We may want to transform and or combine variables
# * FACILITY_NAME has over 9000 names and we should consider removing facilities less than one or focusing on US Highways or Interstates 
# * DIRECTION is evenly across the board and looks clean 
# * EVENT_DESCRIPTION is a combination of the Facility Name and Event Type in a descriptive format 
# * Split out 2018-11-01 00:00:26 to two fields date and time
# * Added the day of the week
# * CITY looks good but needed to delete records in the following listed cities that are not in Dallas County: Arlington, Fate, Lacy-Lakeview, McKinney, Plano, Rice, Royse City, Talty, Terrell , Waco
# * There were 140,631 duplicate records that I removed. The final dataset after this process contains 1204422 rows and 29 columns. 

# **Further preparation for visualization and exploration**

# In[ ]:


#Checking null values in dataframe
Dallasdataset.isnull().sum()


# In[ ]:


#There are still some missing value that needs to be handled. 
#Check for missing value in City Column
Dallasdataset.CITY.isnull().sum()


# In[ ]:


#Remove NaN value in the CITY Column
Dallasdataset = Dallasdataset[pd.notnull(Dallasdataset['CITY'])]
Dallasdataset.head()


# In[ ]:


#Checking how many missig values (NaN) in the whole dataframe
Dallasdataset.isnull().sum().sum()


# In[ ]:


#Drop columns if they only contain missing values
#Notice City was removed
Dallas2 = Dallasdataset.dropna(axis=1)
Dallas2.head()


# In[ ]:


#Checking again on missig values in the whole dataframe, doesn't count "0" values as NaN
Dallas2.isnull().sum().sum()


# In[ ]:


#Check new dataframe Dallas2 for information
print(Dallas2.info())


# In[ ]:


#Check unique values of each column
print(Dallas2.nunique())


# **Graphs and Figures**

# In[ ]:


# Importing libraries that are needed for graphing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Verify Counts before visualization
#Categories count of EVENT_TYPE
#Count of categories in EVENT_TYPE

print(Dallas2['EVENT_TYPE'].value_counts())
print(Dallas2['EVENT_TYPE'].value_counts().count())


# In[ ]:


#Figure 1 - Frequency of Event Type
event_count = Dallas2['EVENT_TYPE'].value_counts()
sns.set(style="darkgrid")
plt.figure(figsize=(40, 20))
sns.barplot(event_count.index, event_count.values, alpha=0.9)
plt.title('Frequency Distribution of Event Types in Dallas County', fontsize = 35)
plt.xticks(rotation = 90, fontsize = 20)
plt.yticks(fontsize = 20)
plt.ylabel('Number of Occurrences', fontsize=35)
plt.xlabel('Event Types', fontsize=35)  
plt.show()


# In[ ]:


# Figure 2 - Event Type by Percentages 
# Pie Chart to display event type by percentages
labels = Dallas2['EVENT_TYPE'].astype('category').cat.categories.tolist()
counts = Dallas2['EVENT_TYPE'].value_counts()
sizes = [counts[var_cat] for var_cat in labels]
fig1, ax1 = plt.subplots(figsize=(30,30))
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=False) 
ax1.axis('equal')
plt.show()


# In[ ]:


#Checking Percentage of Incidents in Event Type 
Dallas2.EVENT_TYPE.value_counts(normalize=True)


# In[ ]:


# Figure 3 - Distribution of Events by Days in Dallas County
# Distribution by dayofweek
day_count = Dallas2['dayofweek'].value_counts()
plt.figure(figsize=(40, 20))
sns.barplot(day_count.index, day_count.values, alpha=0.9)
plt.title('Frequency of Reported Event Days in Dallas County', fontsize = 35)
plt.xticks(rotation = 0, fontsize = 25)
plt.yticks(fontsize = 25)
plt.ylabel('Number of Occurrences', fontsize=35)
plt.xlabel('Day', fontsize=35)  
plt.show()


# **Additional Steps to Process the Data before Modeling**

# In[ ]:


#Converting Dates to "datetime" format
Dallas2["Dates"] = pd.to_datetime(Dallas2["Dates"])
Dallas2.info()


# In[ ]:


#Adding a target value for our models
Dallas2.loc[(Dallas2.EVENT_TYPE == 'accident') | (Dallas2.EVENT_TYPE == 'minor accident'), 'target'] = 1
Dallas2.loc[(Dallas2.EVENT_TYPE != 'accident') & (Dallas2.EVENT_TYPE != 'minor accident'), 'target'] = 0
Dallas2.head()


# In[ ]:


#Convert target to integers
Dallas2["target"] = Dallas2["target"].astype('int')
Dallas2.info()


# **Model 1:
# Decision Tree**

# In[ ]:


# Adding the libraries and packages needed for the Decision Tree and Visualization of the Tree as Output
get_ipython().system('pip install pydotplus')
get_ipython().system('pip install -U scikit-learn')
from sklearn import metrics
import pydotplus as pdot
import graphviz
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split


# In[ ]:


# Verifying data before modeling 
Dallas2.info()


# Decision Tree Version 1 
# * This is experimental of using decision tree with the following columns: 'dayofweek','LAT','LON', 'Event Type' and 'City' 

# In[ ]:


#Transform categorical variables and check the data types and count 
Dallas3 = pd.get_dummies(Dallas2, columns=["EVENT_TYPE","CITY"])
Dallas3.info()


# In[ ]:


#Setting up the feature columns for the model
feature_cols =['dayofweek','LAT','LON', 'EVENT_TYPE_accident', 'EVENT_TYPE_animal on the shoulder',
       'EVENT_TYPE_animal struck', 'EVENT_TYPE_flooding', 'EVENT_TYPE_fog',
       'EVENT_TYPE_hail', 'EVENT_TYPE_hazard on road',
       'EVENT_TYPE_hazard on the shoulder', 'EVENT_TYPE_heavy traffic',
       'EVENT_TYPE_huge traffic jam', 'EVENT_TYPE_ice on roadway',
       'EVENT_TYPE_large traffic jam', 'EVENT_TYPE_major event',
       'EVENT_TYPE_malfunctioning traffic light',
       'EVENT_TYPE_medium traffic jam', 'EVENT_TYPE_minor accident',
       'EVENT_TYPE_missing sign on the shoulder',
       'EVENT_TYPE_object on roadway', 'EVENT_TYPE_other',
       'EVENT_TYPE_pothole', 'EVENT_TYPE_road closed',
       'EVENT_TYPE_road closed due to construction',
       'EVENT_TYPE_road closed due to hazard', 'EVENT_TYPE_road construction',
       'EVENT_TYPE_slowdown', 'EVENT_TYPE_small traffic jam',
       'EVENT_TYPE_stopped car', 'EVENT_TYPE_stopped car on the shoulder',
       'EVENT_TYPE_stopped traffic', 'EVENT_TYPE_traffic heavier than normal',
       'EVENT_TYPE_traffic jam', 'EVENT_TYPE_weather hazard', 'CITY_Addison',
       'CITY_Balch Springs', 'CITY_Carrollton', 'CITY_Cedar Hill',
       'CITY_Cockrell Hill', 'CITY_Combine', 'CITY_Coppell',
       'CITY_DFW Airport', 'CITY_Dallas', 'CITY_DeSoto', 'CITY_Duncanville',
       'CITY_Farmers Branch', 'CITY_Ferris', 'CITY_Flower Mound',
       'CITY_Fort Worth', 'CITY_Garland', 'CITY_Glenn Heights',
       'CITY_Grand Prairie', 'CITY_Grapevine', 'CITY_Highland Park',
       'CITY_Hutchins', 'CITY_Irving', 'CITY_Lancaster', 'CITY_Lewisville',
       'CITY_Mesquite', 'CITY_Ovilla', 'CITY_Red Oak', 'CITY_Richardson',
       'CITY_Rockwall', 'CITY_Rowlett', 'CITY_Sachse', 'CITY_Seagoville',
       'CITY_Sunnyvale', 'CITY_University Park', 'CITY_Wilmer', 'CITY_Wylie']


# In[ ]:


#Spilting the data
train_x, test_x, train_y, test_y = train_test_split(Dallas3[feature_cols], Dallas3['target'],test_size = 0.3, random_state = 1)


# In[ ]:


#Set up tree branches and depth
depths_list = [2,3,4,5,6,7,8]

for depth in depths_list:
    clf_tree = tree.DecisionTreeClassifier(max_depth = depth)
    clf_tree.fit(train_x, train_y)


# In[ ]:


#Specify the number of branches
clf_tree = tree.DecisionTreeClassifier(max_depth = 8)


# In[ ]:


#Fit the training data
clf_tree.fit(train_x, train_y)


# In[ ]:


#Apply test data to model
tree_predict = clf_tree.predict(test_x)


# In[ ]:


#Display the tree and export the file in odt and jpg format
tree.export_graphviz(clf_tree, 
               out_file = "model_tree.odt",
               feature_names = train_x.columns)

model_tree_graph = pdot.graphviz.graph_from_dot_file("model_tree.odt")
model_tree_graph.write_jpg("model_tree.jpg")

from IPython.display import Image
Image(filename= "model_tree.jpg")


# In[ ]:


# Using sklearn library and module to produce metrics reports 
from sklearn.metrics import accuracy_score
pred_y = tree_predict
accuracy_score(test_y,pred_y)


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(test_y, pred_y)


# In[ ]:


from sklearn.metrics import classification_report
classification_report(test_y, pred_y)


# Decision Tree Version 2
# * This is experimenting with a different set up for the decision tree with only 'dayofweek','LAT','LON'

# In[ ]:


#Set Up predictors 
feature_cols =['dayofweek','LAT','LON']

#Spilting the data
train_x, test_x, train_y, test_y = train_test_split(Dallas3[feature_cols], Dallas3['target'],test_size = 0.3, random_state = 1)

#Set up tree branches and depth
depths_list = [2,3,4,5,6,7,8]

for depth in depths_list:
    clf_tree = tree.DecisionTreeClassifier(max_depth = depth)
    clf_tree.fit(train_x, train_y)

#Specify the number of branches
clf_tree = tree.DecisionTreeClassifier(max_depth = 8)

#Fit the training data
clf_tree.fit(train_x, train_y)

#Apply test data to model
tree_predict = clf_tree.predict(test_x)

#Display the tree and 
tree.export_graphviz(clf_tree, 
               out_file = "model_tree2.odt",
               feature_names = train_x.columns)

model_tree_graph = pdot.graphviz.graph_from_dot_file("model_tree2.odt")
model_tree_graph.write_jpg("model_tree2.jpg")

#Export the image of the decision tree
from IPython.display import Image
Image(filename= "model_tree2.jpg")


# In[ ]:


#Utilize sklearn library and module to produce metrics report
from sklearn.metrics import accuracy_score
pred_y = tree_predict
accuracy_score(test_y,pred_y)


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(test_y, pred_y)


# In[ ]:


from sklearn.metrics import classification_report
classification_report(test_y, pred_y)


# To summarize the two versions of the decision tree:
# * The target variable is binary and decision tree can handle both numerical and categorical data. 
# * Choosing a decision tree to create a model by learning decision rules and its ability to visualize the decision process.
# * The first attempt was not realistic because the exact match of True/True and False/False in the confuison matrix is indicating that all the prediction matched to the test set. 
# * The second attempt is a bit better, but is still overfitting. This is also one of the disadvantages of using decision trees. Looking at the spilts and the number of support indicates there is bias based on the dataset. 

# **Linear Regression Model**
# * The first part is to prepare the dataset for this modeling and is named as new_dataframe
# * Then, the linear regression models are implemented 

# In[ ]:


# Checking unique value in City and Event Type in the dataset
len(Dallasdataset['CITY'].unique())


# In[ ]:


Dallasdataset['EVENT_TYPE'].unique()


# In[ ]:


#Adding a new column target to predict accidents
Dallasdataset.loc[(Dallasdataset.EVENT_TYPE == 'accident') | (Dallasdataset.EVENT_TYPE == 'minor accident'), 'target'] = 1
Dallasdataset.loc[(Dallasdataset.EVENT_TYPE != 'accident') & (Dallasdataset.EVENT_TYPE != 'minor accident'), 'target'] = 0


# In[ ]:


#Setting the target value to int
Dallasdataset["target"] = Dallasdataset["target"].astype('int')


# In[ ]:


#Seperating the values of hour, minute and month
Dallasdataset["hours"] = Dallasdataset["time"].dt.hour
Dallasdataset["minutes"]=Dallasdataset["time"].dt.minute
Dallasdataset["month"]=Dallasdataset["time"].dt.month


# In[ ]:


#Preparing categorical values for the columns city, event_type, direction
new_dataframe=pd.get_dummies(Dallasdataset,columns=['CITY','DIRECTION','EVENT_STATE'])
new_dataframe["LAT"] = new_dataframe["LAT"].astype('float64')
new_dataframe["LON"] = new_dataframe["LON"].astype('float64')


# In[ ]:


#Checking the new data frame
new_dataframe.head()


# In[ ]:


# Getting information on the new_dataframe
new_dataframe.info()


# In[ ]:


#Ploting heatmap to view the correlation between the variables
import seaborn as sns
corr1=new_dataframe.corr()
sns.heatmap(corr1, mask=np.zeros_like(corr1, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True)


# In[ ]:


#Selecting the inputs and predictors
X=new_dataframe[['dayofweek','LAT','LON','CITY_Addison',
       'CITY_Balch Springs', 'CITY_Carrollton', 'CITY_Cedar Hill',
       'CITY_Cockrell Hill', 'CITY_Combine', 'CITY_Coppell',
       'CITY_DFW Airport', 'CITY_Dallas', 'CITY_DeSoto', 'CITY_Duncanville',
       'CITY_Farmers Branch', 'CITY_Ferris', 'CITY_Flower Mound',
       'CITY_Fort Worth', 'CITY_Garland', 'CITY_Glenn Heights',
       'CITY_Grand Prairie', 'CITY_Grapevine', 'CITY_Highland Park',
       'CITY_Hutchins', 'CITY_Irving', 'CITY_Lancaster', 'CITY_Lewisville',
       'CITY_Mesquite', 'CITY_Ovilla', 'CITY_Red Oak', 'CITY_Richardson',
       'CITY_Rockwall', 'CITY_Rowlett', 'CITY_Sachse', 'CITY_Seagoville',
       'CITY_Sunnyvale', 'CITY_University Park', 'CITY_Wilmer', 'CITY_Wylie','DIRECTION_Southbound','DIRECTION_Westbound','DIRECTION_Eastbound','DIRECTION_Northbound']]
y=new_dataframe['target']


# In[ ]:


#Divinding the dataset into training and testing data set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[ ]:


#Performing the linear regression on train model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)


# In[ ]:


#Applying the model to the testing dataset
predictions = lm.predict(X_test)


# In[ ]:


#Plotting the output for the test dataset
import matplotlib.pyplot as plt
plt.scatter(y_test,predictions)
plt.xlabel("Test Values")
plt.ylabel("Predictions of target")


# In[ ]:


#Calculating the R-squared value for the output
import numpy as np
from sklearn.metrics import mean_squared_error
lin_mse = mean_squared_error(predictions, y_test)
print(lin_mse)
lin_rmse = np.sqrt(lin_mse)
print('The rmse is %.4f' % lin_rmse)


# In[ ]:


#Calculating the mean error for the model
from sklearn.metrics import mean_absolute_error
lin_mae = mean_absolute_error(predictions, y_test)
print('Linear Regression MAE: %.4f' % lin_mae)


# In[ ]:


#Comparing the original and the predicted values
df1 = pd.DataFrame({'Actual': y_test, 'Predicted': predictions}) 
df1


# The receiving operating curve is used to measure the true positives and negatives. As the regression models don't have a parameter to detect the true positives and negatives as in decision trees and other classification models. So the auc_roc which is area under the ROC curve. This shows the percentage of the samples classified as per the test data.

# In[ ]:


from sklearn import metrics
auc_roc=metrics.roc_auc_score(y_test,predictions)
auc_roc


# Linear Regression Model 2 using Ordinary least square regression

# In[ ]:


#Selecting the inputs and predictors
X=new_dataframe[['dayofweek','minutes','LAT','LON','DIRECTION_Southbound','DIRECTION_Westbound','DIRECTION_Eastbound','DIRECTION_Northbound']]
y=new_dataframe['target']


# In[ ]:


#Dividing the dataset into training and testing data set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[ ]:


#Performing the linear regression to train the model
import statsmodels.formula.api as smf
reg1=smf.OLS(endog=new_dataframe['target'], exog=new_dataframe[['DIRECTION_Southbound','DIRECTION_Westbound','DIRECTION_Eastbound','DIRECTION_Northbound','LAT','LON']], missing='drop')


# In[ ]:


results = reg1.fit()


# In[ ]:


#Displays the summary of the linear regression output
results.summary()


# To summarize the above results of the linear regression. 
# * First I have considered the categorical data for the event_state, direction and the city columns. One of the basic necessity for the linear regression that it is applicable only to the numerical data or boolean values. The target variable is used to predict the accident
# * That variable as a boolean value of 1 and 0. 1 for the accident or minor accident and 0 for others. 
# * First regression model is just a  simple linear regression. The R-squared for this is 11.83%. Since it is a categorical data with just 1 and 0 as the output predictors. So due to this the R-squared value is less. All these are depicted as part of the scatter plot.
# * The ROC curve shows that there is 79% of the values are a part of the true positives and negatives. This shows a good area under the curve.
# 
# The second regression model uses the ordinal least square regression. This regression produces a further less R-squared value than the normal linear regression.
# * The outcomes of this model is shown using the summary method.
# * In order to study the correlation between several columns in the dataset I have plotted the heatmap. Which shows the columns those are strongly associated.
# * The inputs considered for predicting the target variable in the first model is day of the week, direction such as Northbound, Southbound, Westbound, Eastbound and the city.
# * The inputs considered for the second model are the direction, lat and long.
# * I chose these inputs for the second model. Since when I re run the model, only by considering the direction and lat long I was getting the R-squared value. It was not changing even if I considered any other columns.
# * From this we can state that the direction and lat, long are inputs that add value to the training of the model.
# * As per the model summary for the second model we find that the R-squared is lesser than the first model. So we are not using the second model for prediciting since the first model is more efficient.

# **Neural Networks using TensorFlow Model**

# In[ ]:


# Installing Tensor Flow
get_ipython().system('pip install tensorflow==2.0.0-alpha0')


# In[ ]:


# Importing TensorFlow
import tensorflow as tf


# In[ ]:


from tensorflow import keras


# In[ ]:


# Creating a Dataframe for Neural Network model
NNDallas = Dallasdataset[['CITY','LAT', 'LON','dayofweek','target','hours', 'minutes', 'month']].copy()
NNDallas["LAT"] = NNDallas["LAT"].astype('float64')
NNDallas["LON"] = NNDallas["LON"].astype('float64')
NNDallas.head()


# In[ ]:


NNDallas.info()


# In[ ]:


# Checking for the counts in the imbalanced datasets
import numpy as np

target_count = NNDallas.target.value_counts()
print('Accident 1:', target_count[1])
print('Accident 0:', target_count[0])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

target_count.plot(kind='bar', title='Count (target)');


# In[ ]:


# Resampling to address the unbalanced data problem. Undersampling with Random Samples
# Class count
count_class_0, count_class_1 = NNDallas.target.value_counts()
# Divide by class
NNDallas_class_0 = NNDallas[NNDallas['target'] == 0]
NNDallas_class_1 = NNDallas[NNDallas['target'] == 1]
print(NNDallas_class_0.head())
print(NNDallas_class_1.head())


# In[ ]:


NNDallas_class_0_under = NNDallas_class_0.sample(count_class_1)
NNDallas_test_under = pd.concat([NNDallas_class_0_under, NNDallas_class_1], axis=0)

print('Random under-sampling:')
print(NNDallas_test_under.target.value_counts())


# In[ ]:


#Rename the dataframe
NNDallas = NNDallas_test_under
NNDallas.head()


# In[ ]:


# Creating training and testing dataframes
train, test = train_test_split(NNDallas, test_size=0.2, random_state=1)


# In[ ]:


#Length of training set
len(train)


# In[ ]:


#Length of test set
len(test)


# In[ ]:


#Creating training and validation set
train, valid = train_test_split(train, test_size=0.2, random_state=1)


# In[ ]:


#Length of training set
len(train)


# In[ ]:


#Length of validation set
len(valid)


# In[ ]:


#Ensuring validation set has correct columns
valid.head()


# In[ ]:


#Importing the feature_column and viewing columns for assignment
from tensorflow import feature_column
tf.random.set_seed(1)
NNDallas.columns


# In[ ]:


#Assigning columns to numeric or categorical 
numeric_columns = ['LAT', 'LON','hours', 'minutes', 'month']
categorical_columns = ['CITY','dayofweek']
feature_columns = []


# In[ ]:


#Assigning the numeric columns to the feature columns
for header in numeric_columns:
  feature_columns.append(feature_column.numeric_column(header))
feature_columns


# In[ ]:


def get_one_hot_rom_categorical(colname):
  categorical = feature_column.categorical_column_with_vocabulary_list(
  colname,
  train[colname].unique().tolist())
  return feature_column.indicator_column(categorical)


# In[ ]:


# Identifying the cities to categorize them
train["CITY"].unique().tolist()


# In[ ]:


get_one_hot_rom_categorical("CITY")


# In[ ]:


# Categorizing the other categories
for col in categorical_columns:
  feature_columns.append(get_one_hot_rom_categorical(col))


# In[ ]:


print (feature_columns)


# In[ ]:


from tensorflow.keras import layers


# In[ ]:


feature_layer = layers.DenseFeatures(feature_columns)
feature_layer


# In[ ]:


model = keras.Sequential()
model.add(feature_layer)
model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# In[ ]:


model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


def NNDallas_to_tfdata(NNDallas, shuffle=True, bs=32):
  NNDallas = NNDallas.copy()
  labels = NNDallas.pop('target')
  ds = tf.data.Dataset.from_tensor_slices((dict(NNDallas), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(NNDallas), seed=1)
  ds = ds.batch(bs)
  return ds


# In[ ]:


train_ds = NNDallas_to_tfdata(train)


# In[ ]:


valid_ds = NNDallas_to_tfdata(valid, shuffle=False)
test_ds = NNDallas_to_tfdata(test, shuffle=False)


# In[ ]:


model.fit(train_ds,
          validation_data = valid_ds,
          epochs=5)


# In[ ]:


model.evaluate(test_ds)


# In[ ]:


train.target.value_counts()


# In[ ]:


test.target.value_counts()


# To build the neural network model we used Tensorflow and Keras. We identified features for the model which included the city, longitude, latitude, day of the week,
# hours, and minutes. Our results were slightly higher than flipping a coin.
# 

# **In conclusion:** 
# Three models were utilized to make prediction on the Waze dataset focused on Dallas County, accidents data. Each model yields different results of accuracy/metrics. 
# Studying Waze data gave our team incredibly insight to traffic data. If we had more time and resources, we could create more visualizations and attempts other approaches to help solve and plan for road conditions.
