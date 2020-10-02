#!/usr/bin/env python
# coding: utf-8

# # Beginner-Friendly competence 
# ## Roosevelt National Forest Classification
# Classify forest types based on information about the area

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Imports

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

#from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split, GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report, accuracy_score


# ## 1. Load data

# In[ ]:


train_path = '../input/learn-together/train.csv'
test_path = '../input/learn-together/test.csv'
train_df = pd.read_csv(train_path, index_col='Id')
test_df = pd.read_csv(test_path, index_col='Id')
train_df.head()


# In[ ]:


test_df.head()


# ## 2. Exploratory Data Analysis (EDA)

# ### Get the list of categorical variables

# In[ ]:


# Find if there is any categorical data containing strings
s = (train_df.dtypes =='object')
objects_cols = list(s[s].index)

print('Categorical Variables \n', objects_cols)


# This means that all the categories are numeric:

# In[ ]:


# Print columns name
train_df.columns


# In[ ]:


# Count the number of unique values in each category
num_val = train_df.nunique()
print(num_val)


# ### The numeric variables are:
# * Elevation - Elevation in meters
# * Aspect - Aspect in degrees azimuth
# * Slope - Slope in degrees
# * Horizontal_Distance_To_Hydrology - Horz Dist to nearest surface water features
# * Vertical_Distance_To_Hydrology - Vert Dist to nearest surface water features
# * Horizontal_Distance_To_Roadways - Horz Dist to nearest roadway
# * Horizontal_Distance_To_Fire_Points    5826
# * Hillshade_9am (0 to 255 index) - Hillshade index at 9am, summer solstice
# * Hillshade_Noon (0 to 255 index) - Hillshade index at noon, summer solstice
# * Hillshade_3pm (0 to 255 index) - Hillshade index at 3pm, summer solstice

# ### The categorical variables are:
# 
# * Wilderness_Area (4 binary columns, 0 = absence or 1 = presence) - Wilderness area designation
# * Soil_Type (40 binary columns, 0 = absence or 1 = presence) - Soil Type designation
# * Cover_Type (7 types, integers 1 to 7) - Forest Cover Type designation

# In[ ]:


numerical_cols= ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
       'Horizontal_Distance_To_Fire_Points']
categorical_cols = ['Wilderness_Area1',
       'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
       'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5',
       'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10',
       'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14',
       'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18',
       'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22',
       'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26',
       'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30',
       'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34',
       'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38',
       'Soil_Type39', 'Soil_Type40']

Obj_type ='Cover_Type'

print('Features number = ', len(numerical_cols)  + len(categorical_cols))


# In[ ]:


f,ax = plt.subplots(figsize=(8,6))
Categ = numerical_cols.copy()
Categ.append(Obj_type)

sns.heatmap(train_df[Categ].corr(),annot=True, linewidths=.5, fmt='.1f', ax=ax)
plt.show()


# This means that the most important correlations are between:
# * Elevation and Horizontal_Distance_To_Roads
# * Horizontal_Distance_to_Hydrology and Vertical_Distance_To_Roadways
# * Hillshade_9am and Hillshade_3pm

# ## 3. Data visualization

# ## 3.1. Plot of most important correlations

# ### Plot of Elevation and Horizontal_Distance_To_Roads

# In[ ]:


train_df.plot(kind='scatter', x='Elevation', y='Horizontal_Distance_To_Roadways', alpha=0.5, color='blue', figsize = (12,9))
plt.title('Elevation And Horizontal Distance To Roadways')
plt.xlabel("Vertical Distance")
plt.ylabel("Horizontal Distance")
plt.show()


# In[ ]:


train_df['Elevation'].plot(kind='hist', alpha=0.5)


# In[ ]:


train_df['Horizontal_Distance_To_Roadways'].plot(kind='hist', alpha=0.5)


# ### Plot of Horizontal_Distance_to_Hydrology and Vertical_Distance_To_Roadways

# In[ ]:


train_df.plot(kind='scatter', x='Vertical_Distance_To_Hydrology', y='Horizontal_Distance_To_Hydrology', alpha=0.5, color='green', figsize = (12,9))
plt.title('Vertical And Horizontal Distance To Hydrology')
plt.xlabel("Vertical Distance")
plt.ylabel("Horizontal Distance")
plt.show()


# In[ ]:


train_df['Vertical_Distance_To_Hydrology'].plot(kind='hist', alpha=0.5)


# In[ ]:


train_df['Horizontal_Distance_To_Hydrology'].plot(kind='hist', alpha=0.5)


# ### Plot of Hillshade_9am and Hillshade_3pm

# In[ ]:


train_df.plot(kind='scatter', x='Hillshade_9am', y='Hillshade_3pm', alpha=0.5, color='black', figsize = (12,9))
plt.title('Vertical And Horizontal Distance To Hydrology')
plt.xlabel("Vertical Distance")
plt.ylabel("Horizontal Distance")
plt.show()


# In[ ]:


train_df['Hillshade_9am'].plot(kind='hist', alpha=0.5)


# In[ ]:


train_df['Hillshade_3pm'].plot(kind='hist', alpha=0.5)


# ## 3.2. Plot for the Features of the different Cover Types.
# 
# 1. Spruce/Fir
# 2. Lodgepole Pine
# 3. Ponderosa Pine
# 4. Cottonwood/Willow
# 5. Aspen
# 6. Douglas-fir
# 7. Krummholz

# In[ ]:


s = train_df.copy()
s = s.groupby('Cover_Type')
s['Elevation'].size()


# In[ ]:


s['Elevation'].plot(kind='hist', alpha=0.5)
plt.legend()


# In[ ]:


s['Aspect'].plot(kind='hist', alpha=0.5)
plt.legend()


# In[ ]:


s['Slope'].plot(kind='hist', alpha=0.5)
plt.legend()


# In[ ]:


s['Horizontal_Distance_To_Hydrology'].plot(kind='hist', alpha=0.5)
plt.legend()

plt.title("Histogram for Horizontal_Distance_To_Hydrology in different Cover Type")


# ### This means that for the cover type:
# * The cover type depends on Elevation. However, it seems not to depend on aspect, slope  or Horizontal_Distance_To_Hydrology

# ## 3.3. Here I plot the different features for Wilderness areas.
# 
# The wilderness areas are:
# 
# 1. Rawah Wilderness Area
# 2. Neota Wilderness Area
# 3. Comanche Peak Wilderness Area
# 4. Cache la Poudre Wilderness Area

# In[ ]:


# Select areas
Area1 = train_df[train_df['Wilderness_Area1']==1].copy()
Area2 = train_df[train_df['Wilderness_Area2']==1].copy()
Area3 = train_df[train_df['Wilderness_Area3']==1].copy()
Area4 = train_df[train_df['Wilderness_Area4']==1].copy()


# In[ ]:


def histogram_plot(feature):
    plt.hist(Area1[feature], bins='auto', alpha = 0.5)
    plt.hist(Area2[feature], bins='auto', alpha = 0.5)
    plt.hist(Area3[feature], bins='auto', alpha = 0.5)
    plt.hist(Area4[feature], bins='auto', alpha = 0.5)


# In[ ]:


feature1 = 'Cover_Type'
histogram_plot(feature1)


plt.title("Histogram for Cover Type in different Wilderness Area")
plt.legend(['Area 1', 'Area 2', 'Area 3', 'Area 4'])


# In[ ]:


feature1 = 'Elevation'
histogram_plot(feature1)


plt.title("Histogram for Elevation in different Wilderness Area")
plt.legend(['Area 1', 'Area 2', 'Area 3', 'Area 4'])


# ### Effect of Aspect

# In[ ]:


feature2 = 'Aspect'
histogram_plot(feature2)

plt.title("Histogram for Aspect in different Wilderness Area")
plt.legend(['Area 1', 'Area 2', 'Area 3', 'Area 4'])


# ### Effect of Slope
# 

# In[ ]:


feature3 = 'Slope'
histogram_plot(feature3)
plt.title("Histogram for Aspect in different Wilderness Area")
plt.legend(['Area 1', 'Area 2', 'Area 3', 'Area 4'])


# ### Effect of Horizontal Distance To Hydrology
# 

# In[ ]:


feature4 = 'Horizontal_Distance_To_Hydrology'
histogram_plot(feature4)
plt.title("Histogram for Horizontal Distance To Hydrology in different Wilderness Area")
plt.legend(['Area 1', 'Area 2', 'Area 3', 'Area 4'])


# In[ ]:


feature5 = 'Vertical_Distance_To_Hydrology'
histogram_plot(feature5)
plt.title("Histogram for Vertical_Distance_To_Hydrology in different Wilderness Area")
plt.legend(['Area 1', 'Area 2', 'Area 3', 'Area 4'])


# In[ ]:


feature6 = 'Horizontal_Distance_To_Roadways'
histogram_plot(feature6)
plt.title("Histogram for Horizontal_Distance_To_Roadways  in different Wilderness Area")
plt.legend(['Area 1', 'Area 2', 'Area 3', 'Area 4'])


# In[ ]:


feature7 = 'Hillshade_9am'
histogram_plot(feature7)
plt.title("Histogram for Hillshade_9am  in different Wilderness Area")
plt.legend(['Area 1', 'Area 2', 'Area 3', 'Area 4'])


# ### This means that for the Areas:
# * Elevation is an important parameter: Low elevations will be in Area 4
# * Aspect, area, horizontal and vertical distance to hydrology seems to be not a very relevant parameter
# * Cover type 4 is only present in area 4
# * Area 2 present cover types 1, 2 and 7

# # 4. Modelling
# ## Objective:
# ### I am interested predict an integer classification for the forest cover type.
# 
# ## 4.1. Ramdom Forest Classifier

# In[ ]:


Select_num = list(range(2,8))
Select_num.insert(0,0)

num_cols = numerical_cols.copy()

# Here I am removing the columns that seem to be irrelevant after the data visualization
num_cols.remove('Aspect')
num_cols.remove('Hillshade_Noon')

features2 = num_cols.copy() 
features2.extend(categorical_cols)

print("The number of features that are considered in this study are:", len(features2))


# In[ ]:


X_df = train_df.drop(['Cover_Type'], axis=1)
y_df = train_df['Cover_Type']
X_train, X_val, y_train, y_val = train_test_split(X_df[features2], y_df, test_size=0.2, random_state = 0)

print(X_train.shape)


# In[ ]:


"""
def scoreModel(n_estim, X_train, y_train):
    for n_est in n_estim:
        model = RandomForestClassifier(n_estimators=n_est, random_state = 0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mae_3 = mean_absolute_error(y_val, y_pred)
        print("\nFor n_estimators = {} \n Mean Absolute Error: {}".format(n_est, mae_3))
        #print(classification_report(y_val, y_pred))
        print('Accuracy = ', accuracy_score(y_val, y_pred))

n_estim = [20, 50, 100, 150, 200, 250, 300] 
scoreModel(n_estim, X_train[features2], y_train)
"""


# In[ ]:


"""
model = RandomForestClassifier(n_estimators=100, random_state = 0)
model.fit(X_train, y_train)
"""


# In[ ]:


"""
y_pred = model.predict(X_val)
y_pred.shape
"""


# In[ ]:


# print(classification_report(y_val, y_pred))


# In[ ]:


# print('Accuracy = ', accuracy_score(y_val, y_pred))
        


# In[ ]:


"""
# Calculate MAE
mae_3 = mean_absolute_error(y_val, y_pred)
print("Mean Absolute Error:" , mae_3)
"""


# In[ ]:


"""
# Preprocessing of validation data, get predictions
preds = model.predict(test_df[features2])
print(preds)
"""


# ## 4.2. XGBooost

# In[ ]:


from xgboost import XGBClassifier


# In[ ]:


"""
def scoreModel_xgb(n_estim, X_train, y_train):
    for n_est in n_estim:
        xgb = XGBClassifier( n_estimators = n_est,  #todo : search for good parameters
                    learning_rate= 0.5,  #todo : search for good parameters
                    objective= 'binary:logistic', #this outputs probability,not one/zero. should we use binary:hinge? is it better for the learning phase?
                    random_state= 1,
                    n_jobs=-1)
        xgb.fit(X_train, y_train)
        preds_xgb = xgb.predict(X_val)
        print('Accuracy for ', n_est , ' = ', accuracy_score(y_val, preds_xgb))
        
n_estim = [100, 500, 1000, 2000, 5000] 
scoreModel_xgb(n_estim, X_train[features2], y_train)
"""


# In[ ]:


"""
n_estimators_best = 2000
xgb = XGBClassifier( n_estimators=n_estimators_best,  
                    learning_rate= 0.5, 
                    objective= 'binary:logistic', 
                    random_state= 1,
                    n_jobs=-1)
"""


# In[ ]:


# xgb.fit(X_train, y_train)


# In[ ]:


"""
y_pred2 = xgb.predict(X_val)
y_pred2.shape
print('Accuracy = ', accuracy_score(y_val, y_pred2))
mae_3 = mean_absolute_error(y_val, y_pred2)
print("Mean Absolute Error:" , mae_3)
"""


# In[ ]:


"""
# Preprocessing of validation data, get predictions
preds2 = xgb.predict(test_df[features2])
print(preds2)
"""


# In[ ]:


"""
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

model_XGB = XGBClassifier(#silent=False, 
                      #scale_pos_weight=1,
                      learning_rate=0.01,  
                      colsample_bytree = 1,
                      subsample = 0.8,
                      objective='multi:softmax', 
                      n_estimators=2000, 
                      #reg_alpha = 0.3,
                      max_depth=3, 
                      earlystoppingrounds=5,
                      gamma=1)

# X_train, X_val, y_train, y_val
#eval_set = [(X_train, y_train), (X_val, y_val)]
eval_set = [ (X_val, y_val)]
eval_metric = ["rmse"]
model_XGB.fit(X_train, y_train, eval_metric=eval_metric, verbose=True)
y_pred_XGB = model_XGB.predict(X_val)

print('Accuracy = ', accuracy_score(y_val, y_pred_XGB))
mae_XGB = mean_absolute_error(y_val, y_pred_XGB)
print("Mean Absolute Error:" , mae_XGB)

"""


# In[ ]:


"""
# Preprocessing of validation data, get predictions
preds_XGB = model_XGB.predict(test_df[features2])
"""


# ## 4.3. AdaBoost Classifier

# In[ ]:


"""
from sklearn.ensemble import AdaBoostClassifier

# Train model
AdaCl = AdaBoostClassifier(n_estimators=1000)
AdaCl.fit(X_train, y_train)

# PRedictions
y_pred3 = AdaCl.predict(X_val)
y_pred3.shape

# Statistics
print('Accuracy = ', accuracy_score(y_val, y_pred3))
mae_3 = mean_absolute_error(y_val, y_pred3)
print("Mean Absolute Error:" , mae_3)

# Preprocessing of validation data, get predictions
preds3 = AdaCl.predict(test_df[features2])
print(preds3)
"""


# ## 4.4. Light GBM Classifier

# In[ ]:



from lightgbm import LGBMClassifier

# Elevation and Horizontal_Distance_To_Roads
# * Horizontal_Distance_to_Hydrology and Vertical_Distance_To_Roadways
# * Hillshade_9am and Hillshade_3pm

lgbm = LGBMClassifier(objective='multiclass',learning_rate=0.01, n_estimators=2000 , random_state=5, num_leaves = 500)
lgbm.fit(X_train, y_train)

# PRedictions
y_pred4 = lgbm.predict(X_val)

# Statistics
print('Accuracy = ', accuracy_score(y_val, y_pred4))
mae_4 = mean_absolute_error(y_val, y_pred4)
print("Mean Absolute Error:" , mae_4)


# In[ ]:



# Preprocessing of validation data, get predictions
preds4 = lgbm.predict(test_df[features2])
#print(preds4)


# ## Models summary
# 
# ### Random Forest classifier:
# - model = RandomForestClassifier(n_estimators=100, random_state = 0)
# - Accuracy =  0.8525132275132276
# - Mean Absolute Error: 0.3568121693121693
# 
# ### XGBoost Classifier
# - n_estimators_best = 2000
# - xgb = XGBClassifier( n_estimators=n_estimators_best, learning_rate= 0.5, objective= 'binary:logistic', random_state= 1, n_jobs=-1)
# - Accuracy =  0.8392857142857143
# - Mean Absolute Error: 0.3872354497354497
# 
# ### AdaBoost Classifier
# - AdaCl = AdaBoostClassifier(n_estimators=100)
# - Accuracy =  0.4176587301587302
# - Mean Absolute Error: 1.874669312169312
# 
# ### Light GBM Classifier
# - lgbm = LGBMClassifier(objective='multiclass', random_state=5)
# - Accuracy =  0.8353174603174603
# - Mean Absolute Error: 0.3968253968253968
# ---------
# - lgbm = LGBMClassifier(objective='multiclass', n_estimators=400, random_state=5)
# - Accuracy =  0.8558201058201058
# - Mean Absolute Error: 0.3482142857142857
# --------
# - lgbm = LGBMClassifier(objective='multiclass', n_estimators=700, random_state=5)
# - Accuracy =  0.8578042328042328
# - Mean Absolute Error: 0.34755291005291006
# ------
# - lgbm = LGBMClassifier(objective='multiclass', n_estimators=1000, random_state=5)
# - Accuracy =  0.8611111111111112
# - Mean Absolute Error: 0.3402777777777778
# -----
# - lgbm = LGBMClassifier(objective='multiclass', n_estimators=1500, random_state=5)
# - Accuracy =  0.8637566137566137
# - Mean Absolute Error: 0.32771164021164023

# # 5. Output

# In[ ]:


test_ids = test_df.index

output = pd.DataFrame({'Id': test_ids,
                       'Cover_Type': preds4})
output.to_csv('submission.csv', index=False)

output.head()

