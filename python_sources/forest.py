#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize

import os
print(os.listdir("../input/learn-together/"))


# In[ ]:


data = pd.read_csv('../input/learn-together/train.csv')


# In[ ]:


data.head()


# In[ ]:


data.columns


# # Cover Type

# In[ ]:


data_group = data.Cover_Type.value_counts().reset_index()


sns.barplot( y = 'Cover_Type', x = 'index', data = data_group, color="blue")

title_name = "Count"
xlabel_name = "Cover_Type"
ylabel_name = "Count"

fig=plt.gcf()
fig.set_size_inches(15,10)

plt.tick_params(labelsize=14)
plt.title(title_name,fontsize = 20)
plt.xlabel(xlabel_name,fontsize = 20)
plt.ylabel(ylabel_name,fontsize = 20)
plt.xticks(rotation=90)


# In[ ]:


data_group


# # Distribution of Elevation

# In[ ]:


fillColor = "#FFA07A"
fillColor2 = "#F1C40F"

sns.distplot(data.Elevation,kde = False, color = fillColor2)

title_name = "Elevation Distribution"
xlabel_name = "Elevation"
ylabel_name = "Count"

fig=plt.gcf()
fig.set_size_inches(15,10)

plt.tick_params(labelsize=14)
plt.title(title_name,fontsize = 20)
plt.xlabel(xlabel_name,fontsize = 20)
plt.ylabel(ylabel_name,fontsize = 20)


# In[ ]:


data.Elevation.describe()


# # Cover Type and Median Elevation

# In[ ]:


data_group = data.groupby('Cover_Type').Elevation.median().reset_index().sort_values( by = 'Elevation',ascending = False)
data_group


# In[ ]:


sns.barplot( y = 'Elevation', x = 'Cover_Type', data = data_group, color="red")

title_name = "Cover Type and Median Elevation"
ylabel_name = "Elevation"
xlabel_name = "Cover Type"

fig=plt.gcf()
fig.set_size_inches(15,10)

plt.tick_params(labelsize=14)
plt.title(title_name,fontsize = 20)
plt.xlabel(xlabel_name,fontsize = 20)
plt.ylabel(ylabel_name,fontsize = 20)


# # Horizontal_Distance_To_Hydrology

# In[ ]:


data.Horizontal_Distance_To_Hydrology.describe()


# In[ ]:


data_group = data.groupby('Cover_Type').Horizontal_Distance_To_Hydrology.median().reset_index(). sort_values( by = 'Horizontal_Distance_To_Hydrology',ascending = False)

data_group


# In[ ]:


sns.barplot( y = 'Horizontal_Distance_To_Hydrology', x = 'Cover_Type', data = data_group, color= fillColor2)

title_name = "Cover Type and Median Horizontal Distance To Hydrology"
ylabel_name = "Horizontal Distance To Hydrology"
xlabel_name = "Cover Type"

fig=plt.gcf()
fig.set_size_inches(15,10)

plt.tick_params(labelsize=14)
plt.title(title_name,fontsize = 20)
plt.xlabel(xlabel_name,fontsize = 20)
plt.ylabel(ylabel_name,fontsize = 20)


# # Soil Types Data Analysis

# In[ ]:


features = ['Soil_Type1', 'Soil_Type2', 'Soil_Type3',
       'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',
       'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
       'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',
       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']


# In[ ]:


data[features].dtypes


# # Wilderness Area Analysis

# In[ ]:


features = [ 'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',
       'Wilderness_Area4']


# In[ ]:


data[features].dtypes


# #  Vertical Distance To Hydrology

# In[ ]:


data.Vertical_Distance_To_Hydrology.describe()


# In[ ]:


data_group = data.groupby('Cover_Type').Vertical_Distance_To_Hydrology.median().reset_index(). sort_values( by = 'Vertical_Distance_To_Hydrology',ascending = False)

data_group


# In[ ]:


sns.barplot( y = 'Vertical_Distance_To_Hydrology', x = 'Cover_Type', data = data_group, color= 'blue')

title_name = "Cover Type and Median Vertical Distance To Hydrology"
ylabel_name = "Vertical Distance To Hydrology"
xlabel_name = "Cover Type"

fig=plt.gcf()
fig.set_size_inches(15,10)

plt.tick_params(labelsize=14)
plt.title(title_name,fontsize = 20)
plt.xlabel(xlabel_name,fontsize = 20)
plt.ylabel(ylabel_name,fontsize = 20)


# # Horizontal Distance To Roadways

# In[ ]:


data.Horizontal_Distance_To_Roadways.describe()


# In[ ]:


data_group = data.groupby('Cover_Type').Horizontal_Distance_To_Roadways.median().reset_index(). sort_values( by = 'Horizontal_Distance_To_Roadways',ascending = False)

data_group


# In[ ]:


sns.barplot( y = 'Horizontal_Distance_To_Roadways', x = 'Cover_Type', data = data_group, color= 'blue')

title_name = "Cover Type and Median Horizontal Distance To Roadways"
ylabel_name = "Horizontal Distance To Roadways"
xlabel_name = "Cover Type"

fig=plt.gcf()
fig.set_size_inches(15,10)

plt.tick_params(labelsize=14)
plt.title(title_name,fontsize = 20)
plt.xlabel(xlabel_name,fontsize = 20)
plt.ylabel(ylabel_name,fontsize = 20)


# # Model

# In[ ]:


features = data.columns


# In[ ]:


type(features)


# In[ ]:


features =  features.drop(['Id','Cover_Type'])


# In[ ]:


features


# In[ ]:


X = data[features]


# In[ ]:


y = data['Cover_Type']


# In[ ]:


from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


# ## DecisionTree Classifier

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

model = DecisionTreeClassifier()

model.fit(train_X,train_y)

predictions = model.predict(val_X)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(predictions,val_y)


# ## ExtraTrees Classifier

# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier


# In[ ]:


model = ExtraTreesClassifier(n_estimators = 100)


# In[ ]:


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


# In[ ]:


model.fit(train_X,train_y)


# In[ ]:


model


# In[ ]:


predictions = model.predict(val_X)


# In[ ]:


accuracy_score(predictions,val_y)


# In[ ]:


test = pd.read_csv('../input/learn-together/test.csv')


# In[ ]:


test.head()


# In[ ]:


features = test.columns


# In[ ]:


features =  features.drop(['Id'])


# In[ ]:


features


# In[ ]:


model.fit(X,y)


# In[ ]:


predictions = model.predict(test[features])


# In[ ]:


sub = pd.read_csv('../input/learn-together/sample_submission.csv')


# In[ ]:


sub.head()


# In[ ]:


sub_df = pd.DataFrame()


# In[ ]:


sub_df["Id"] = test["Id"]


# In[ ]:


sub_df["Cover_Type"] = predictions


# In[ ]:


sub_df.to_csv("Predictions03Sep2019.csv",index=False)


# ### Feature Importance of Extra Trees Classifier

# In[ ]:


importances = model.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X.shape[1]):
    print("%d. feature %s  (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))

features_df =  pd.DataFrame({'Feature':features[indices],'Importance':importances[indices]})

data_group  = features_df.head(10)
sns.barplot( y = 'Feature', x = 'Importance', data = data_group, color= 'blue')

title_name = "Extra Trees Classifier"
ylabel_name = "Features"
xlabel_name = "Importance"

fig=plt.gcf()
fig.set_size_inches(15,10)

plt.tick_params(labelsize=14)
plt.title(title_name,fontsize = 20)
plt.xlabel(xlabel_name,fontsize = 20)
plt.ylabel(ylabel_name,fontsize = 20)


# # Feature Engineering

# In[ ]:


#https://www.kaggle.com/codename007/forest-cover-type-eda-baseline-model

####################### data data #############################################
data['HF1'] = data['Horizontal_Distance_To_Hydrology']+data['Horizontal_Distance_To_Fire_Points']
data['HF2'] = abs(data['Horizontal_Distance_To_Hydrology']-data['Horizontal_Distance_To_Fire_Points'])
data['HR1'] = abs(data['Horizontal_Distance_To_Hydrology']+data['Horizontal_Distance_To_Roadways'])
data['HR2'] = abs(data['Horizontal_Distance_To_Hydrology']-data['Horizontal_Distance_To_Roadways'])
data['FR1'] = abs(data['Horizontal_Distance_To_Fire_Points']+data['Horizontal_Distance_To_Roadways'])
data['FR2'] = abs(data['Horizontal_Distance_To_Fire_Points']-data['Horizontal_Distance_To_Roadways'])
#data['ele_vert'] = data.Elevation-data.Vertical_Distance_To_Hydrology

data['slope_hyd'] = (data['Horizontal_Distance_To_Hydrology']**2+data['Vertical_Distance_To_Hydrology']**2)**0.5
data.slope_hyd=data.slope_hyd.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any

#Mean distance to Amenities 
data['Mean_Amenities']=(data.Horizontal_Distance_To_Fire_Points + data.Horizontal_Distance_To_Hydrology + data.Horizontal_Distance_To_Roadways) / 3 
#Mean Distance to Fire and Water 
data['Mean_Fire_Hyd']=(data.Horizontal_Distance_To_Fire_Points + data.Horizontal_Distance_To_Hydrology) / 2 

####################### Test data #############################################
test['HF1'] = test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Fire_Points']
test['HF2'] = abs(test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Fire_Points'])
test['HR1'] = abs(test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Roadways'])
test['HR2'] = abs(test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Roadways'])
test['FR1'] = abs(test['Horizontal_Distance_To_Fire_Points']+test['Horizontal_Distance_To_Roadways'])
test['FR2'] = abs(test['Horizontal_Distance_To_Fire_Points']-test['Horizontal_Distance_To_Roadways'])
#test['ele_vert'] = test.Elevation-test.Vertical_Distance_To_Hydrology

test['slope_hyd'] = (test['Horizontal_Distance_To_Hydrology']**2+test['Vertical_Distance_To_Hydrology']**2)**0.5
test.slope_hyd=test.slope_hyd.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any

#Mean distance to Amenities 
test['Mean_Amenities']=(test.Horizontal_Distance_To_Fire_Points + test.Horizontal_Distance_To_Hydrology + test.Horizontal_Distance_To_Roadways) / 3 
#Mean Distance to Fire and Water 
test['Mean_Fire_Hyd']=(test.Horizontal_Distance_To_Fire_Points + test.Horizontal_Distance_To_Hydrology) / 2


# In[ ]:


features = data.columns


# In[ ]:


features =  features.drop(['Id','Cover_Type'])


# In[ ]:


features


# In[ ]:


X = data[features]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
model.fit(train_X,train_y)

importances = model.feature_importances_

indices = np.argsort(importances)[::-1]

features_df =  pd.DataFrame({'Feature':features[indices],'Importance':importances[indices]})

data_group  = features_df.head(10)
sns.barplot( y = 'Feature', x = 'Importance', data = data_group, color= 'blue')

title_name = "Extra Trees Classifier"
ylabel_name = "Features"
xlabel_name = "Importance"

fig=plt.gcf()
fig.set_size_inches(10,10)

plt.tick_params(labelsize=14)
plt.title(title_name,fontsize = 20)
plt.xlabel(xlabel_name,fontsize = 20)
plt.ylabel(ylabel_name,fontsize = 20)


# In[ ]:


predictions = model.predict(test[features])


# In[ ]:


sub_df["Cover_Type"] = predictions

sub_df.to_csv("Predictions07Sep2019-1.csv",index=False)

