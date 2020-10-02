#!/usr/bin/env python
# coding: utf-8

# # Hello People! Here we will look for the solution of the problem using K-nearest neighbor algorithm
# 
# This notebook has the following structure 
# 
# 1. Read and explore the train data
# 2. Understanding the Data
# 3. Target Variable
# 4. Read the test data
# 5. Understanding the features and the problems
# 2. Clean the Data
# 3. Visualize the Data
# 4. Model the Data using KNN

# In[ ]:


#Load the important libraries neede
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.

train_data = pd.read_csv("../input/train.csv")

# Display the first few rows of the data
print(train_data.head())

# Get the shape of the data just to know how many rows and columns it contains
print(train_data.shape)

# Any results you write to the current directory are saved as output.


# # Understanding the data
# 
# It's important to understand all the columns before we move further. Train data has the following columns:
# 
# 1. Dates - timestamp of the crime incident
# 2. **Category** - category of the crime incident (only in train.csv). This is the target variable you are going to predict.
# 3. Descript - detailed description of the crime incident (only in train.csv)
# 4. DayOfWeek - the day of the week
# 5. PdDistrict - name of the Police Department District
# 6. Resolution - how the crime incident was resolved (only in train.csv)
# 7. Address - the approximate street address of the crime incident 
# 8. X - Longitude
# 9. Y - Latitude

# In[ ]:


#target variable

target = train_data["Category"].unique()
print(target)

# There are multiple categorical values. It looks like a multi class classification problem.


# In[ ]:


#Let's read the test data now

test_data = pd.read_csv("../input/test.csv")
print(test_data.head())
print(test_data.shape)

#Test data does not have the target variable and the resolution


# In[ ]:


data_dict = {}
count = 1
for data in target:
    data_dict[data] = count
    count+=1
train_data["Category"] = train_data["Category"].replace(data_dict)

#Replacing the day of weeks
data_week_dict = {
    "Monday": 1,
    "Tuesday":2,
    "Wednesday":3,
    "Thursday":4,
    "Friday":5,
    "Saturday":6,
    "Sunday":7
}
train_data["DayOfWeek"] = train_data["DayOfWeek"].replace(data_week_dict)
test_data["DayOfWeek"] = test_data["DayOfWeek"].replace(data_week_dict)
#District
district = train_data["PdDistrict"].unique()
data_dict_district = {}
count = 1
for data in district:
    data_dict_district[data] = count
    count+=1 
train_data["PdDistrict"] = train_data["PdDistrict"].replace(data_dict_district)
test_data["PdDistrict"] = test_data["PdDistrict"].replace(data_dict_district)


# In[ ]:


print(train_data.head())


# Let's try to find some correlations between the target variable and the numeric variables but before that first remove the resolution column and describe the numeric columns to look for any missing values in the data.

# In[ ]:


columns_train = train_data.columns
print(columns_train)
columns_test = test_data.columns
print(columns_test)


# In[ ]:


cols = columns_train.drop("Resolution")
print(cols)


# In[ ]:


train_data_new = train_data[cols]
print(train_data_new.head())


# In[ ]:


print(train_data_new.describe())

# All the numeric columns have no missing values.


# In[ ]:


corr = train_data_new.corr()
print(corr["Category"])
 
# There is no strong correlation of category with any numeric value


# In[ ]:


#Calculate the skew

skew = train_data_new.skew()
print(skew)


# Modeling the Data - KNN Algorithm

# In[ ]:


#Let's use knn algorithm on numeric columns

features = ["DayOfWeek", "PdDistrict",  "X", "Y"]
X_train = train_data[features]
y_train = train_data["Category"]
X_test = test_data[features]

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)


# In[ ]:


from collections import OrderedDict
data_dict_new = OrderedDict(sorted(data_dict.items()))
print(data_dict_new)
                


# In[ ]:


#print(type(predictions))
result_dataframe = pd.DataFrame({
    "Id": test_data["Id"]
})
for key,value in data_dict_new.items():
    result_dataframe[key] = 0
count = 0
for item in predictions:
    for key,value in data_dict.items():
        if(value == item):
            result_dataframe[key][count] = 1
    count+=1
result_dataframe.to_csv("submission_knn.csv", index=False) 


# In[ ]:


#Logistic Regression


# In[ ]:


from sklearn.linear_model import LogisticRegression
lgr = LogisticRegression()
lgr.fit(X_train, y_train)
predictions = knn.predict(X_test)

#print(type(predictions))
result_dataframe = pd.DataFrame({
    "Id": test_data["Id"]
})
for key,value in data_dict_new.items():
    result_dataframe[key] = 0
count = 0
for item in predictions:
    for key,value in data_dict.items():
        if(value == item):
            result_dataframe[key][count] = 1
    count+=1
result_dataframe.to_csv("submission_logistic.csv", index=False) 


# In[ ]:


from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train, y_train)
predictions = log.predict(X_test)

for key,value in data_dict_new.items():
    result_dataframe[key] = 0
count = 0
for item in predictions:
    for key,value in data_dict.items():
        if(value == item):
            result_dataframe[key][count] = 1
    count+=1
result_dataframe.to_csv("submission_logistic.csv", index=False) 

