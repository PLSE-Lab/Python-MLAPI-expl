#!/usr/bin/env python
# coding: utf-8

# ## Problem Statement

# Predict the forest cover type from the given cartographic variables. This study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado. These areas represent forests with minimal human-caused disturbances, so that existing forest cover types are more a result of ecological processes rather than forest management practices.

# In[ ]:


import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from scipy.stats import norm 
from matplotlib import cm
import seaborn as sns


# In[ ]:


import os
print(os.listdir("/kaggle/input/forest-cover-type-kernels-only"))


# In[ ]:


import zipfile
train_zip = zipfile.ZipFile('/kaggle/input/forest-cover-type-kernels-only/train.csv.zip')
test_zip = zipfile.ZipFile('/kaggle/input/forest-cover-type-kernels-only/test.csv.zip')

train = pd.read_csv(train_zip.open('train.csv'))
test = pd.read_csv(test_zip.open('test.csv'))

Id = test['Id']


# Data Exploration and analysis

# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.info()


# The given dataset cantains 56 features including the target variable Cover_Type, along with 15120 observations and following are the features
# 
# * Elevation - Elevation in meters
# * Aspect - Aspect in degrees azimuth
# * Slope - Slope in degrees
# * Horizontal_Distance_To_Hydrology - Horz Dist to nearest surface water features
# * Vertical_Distance_To_Hydrology - Vert Dist to nearest surface water features
# * Horizontal_Distance_To_Roadways - Horz Dist to nearest roadway
# * Hillshade_9am (0 to 255 index) - Hillshade index at 9am, summer solstice
# * Hillshade_Noon (0 to 255 index) - Hillshade index at noon, summer solstice
# * Hillshade_3pm (0 to 255 index) - Hillshade index at 3pm, summer solstice
# * Horizontal_Distance_To_Fire_Points - Horz Dist to nearest wildfire ignition points
# * Wilderness_Area (4 binary columns, 0 = absence or 1 = presence) - Wilderness area designation
# * Soil_Type (40 binary columns, 0 = absence or 1 = presence) - Soil Type designation
# * Cover_Type (7 types, integers 1 to 7) - Forest Cover Type designation
# 
# The wilderness areas are:
# 
# * 1 - Rawah Wilderness Area
# * 2 - Neota Wilderness Area
# * 3 - Comanche Peak Wilderness Area
# * 4 - Cache la Poudre Wilderness Area
# 
# The soil types are:
# 
# * 1 Cathedral family - Rock outcrop complex, extremely stony.
# * 2 Vanet - Ratake families complex, very stony.
# * 3 Haploborolis - Rock outcrop complex, rubbly.
# * 4 Ratake family - Rock outcrop complex, rubbly.
# * 5 Vanet family - Rock outcrop complex complex, rubbly.
# * 6 Vanet - Wetmore families - Rock outcrop complex, stony.
# * 7 Gothic family.
# * 8 Supervisor - Limber families complex.
# * 9 Troutville family, very stony.
# * 10 Bullwark - Catamount families - Rock outcrop complex, rubbly.
# * 11 Bullwark - Catamount families - Rock land complex, rubbly.
# * 12 Legault family - Rock land complex, stony.
# * 13 Catamount family - Rock land - Bullwark family complex, rubbly.
# * 14 Pachic Argiborolis - Aquolis complex.
# * 15 unspecified in the USFS Soil and ELU Survey.
# * 16 Cryaquolis - Cryoborolis complex.
# * 17 Gateview family - Cryaquolis complex.
# * 18 Rogert family, very stony.
# * 19 Typic Cryaquolis - Borohemists complex.
# * 20 Typic Cryaquepts - Typic Cryaquolls complex.
# * 21 Typic Cryaquolls - Leighcan family, till substratum complex.
# * 22 Leighcan family, till substratum, extremely bouldery.
# * 23 Leighcan family, till substratum - Typic Cryaquolls complex.
# * 24 Leighcan family, extremely stony.
# * 25 Leighcan family, warm, extremely stony.
# * 26 Granile - Catamount families complex, very stony.
# * 27 Leighcan family, warm - Rock outcrop complex, extremely stony.
# * 28 Leighcan family - Rock outcrop complex, extremely stony.
# * 29 Como - Legault families complex, extremely stony.
# * 30 Como family - Rock land - Legault family complex, extremely stony.
# * 31 Leighcan - Catamount families complex, extremely stony.
# * 32 Catamount family - Rock outcrop - Leighcan family complex, extremely stony.
# * 33 Leighcan - Catamount families - Rock outcrop complex, extremely stony.
# * 34 Cryorthents - Rock land complex, extremely stony.
# * 35 Cryumbrepts - Rock outcrop - Cryaquepts complex.
# * 36 Bross family - Rock land - Cryumbrepts complex, extremely stony.
# * 37 Rock outcrop - Cryumbrepts - Cryorthents complex, extremely stony.
# * 38 Leighcan - Moran families - Cryaquolls complex, extremely stony.
# * 39 Moran family - Cryorthents - Leighcan family complex, extremely stony.
# * 40 Moran family - Cryorthents - Rock land complex, extremely stony.

# In[ ]:


print("The number of traning examples(data points) = %i " % train.shape[0])
print("The number of features we have = %i " % train.shape[1])


# In[ ]:


print("The number of traning examples(data points) = %i " % test.shape[0])
print("The number of features we have = %i " % test.shape[1])


# Let's check if any of the columns contains NaNs or Nulls so that we can fill those values if they are insignificant or drop them. We may drop a whole column if most of its values are NaNs or fill its value according to its relation with other columns in the dataframe. Nones can also be 0 in some datasets and that is why i am going to use the describe of the train to see if the range of numbers is not reasonable or not. if you are dropping rows with NaNs and you notice that you need to drop a large portion of your dataset then you should think about filling the NaN values or drop a column that has most of its values missing.

# In[ ]:


train.describe()


# In[ ]:


train.isnull().sum()


# It seems we don't have any NaN or Null value among the dataset we are trying to classify. Let's now discover the correlation matrix for this dataset and see if we can combine features or drop some according to its correlation with the output labels.

# In[ ]:


f,ax = plt.subplots(figsize=(25, 25))
sns.heatmap(train.corr(), annot=True, linewidths=.5, fmt= '.3f',ax=ax)
plt.show()


# In[ ]:


train.corr()


# From the above results it seems that soil_Type7 and soil_Type15 doesn't haveany correlation with the output cover_Type so we can easily drop them from the data we have. Also Soil_Type9, Soil_Type36, Soil_Type27, Soil_Type25, Soil_Type8 have weak correlation, but when a feature has a weak correlation tht doesn't mean it is useful cuz combined with other feature it may make a good impact. I choose those columns after experimenting many times with the data i have from the Extratrees, correlation matrix and the heatmap.

# In[ ]:


#train.drop(['Id'], inplace = True, axis = 1 )
train.drop(['Id','Soil_Type15' , "Soil_Type7"], inplace = True, axis = 1 )
test.drop(['Id','Soil_Type15' , "Soil_Type7"], inplace = True, axis = 1 )


# In[ ]:


train['HorizontalHydrology_HorizontalFire'] = (train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Fire_Points'])
train['Neg_HorizontalHydrology_HorizontalFire'] = (train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Fire_Points'])
train['HorizontalHydrology_HorizontalRoadways'] = (train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Roadways'])
train['Neg_HorizontalHydrology_HorizontalRoadways'] = (train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Roadways'])
train['HorizontalFire_Points_HorizontalRoadways'] = (train['Horizontal_Distance_To_Fire_Points']+train['Horizontal_Distance_To_Roadways'])
train['Neg_HorizontalFire_Points_HorizontalRoadways'] = (train['Horizontal_Distance_To_Fire_Points']-train['Horizontal_Distance_To_Roadways'])

train['Neg_Elevation_Vertical'] = train['Elevation']-train['Vertical_Distance_To_Hydrology']
train['Elevation_Vertical'] = train['Elevation']+train['Vertical_Distance_To_Hydrology']

train['mean_hillshade'] =  (train['Hillshade_9am']  + train['Hillshade_Noon'] + train['Hillshade_3pm'] ) / 3

train['Mean_HorizontalHydrology_HorizontalFire'] = (train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Fire_Points'])/2
train['Mean_HorizontalHydrology_HorizontalRoadways'] = (train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Roadways'])/2
train['Mean_HorizontalFire_Points_HorizontalRoadways'] = (train['Horizontal_Distance_To_Fire_Points']+train['Horizontal_Distance_To_Roadways'])/2

train['MeanNeg_Mean_HorizontalHydrology_HorizontalFire'] = (train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Fire_Points'])/2
train['MeanNeg_HorizontalHydrology_HorizontalRoadways'] = (train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Roadways'])/2
train['MeanNeg_HorizontalFire_Points_HorizontalRoadways'] = (train['Horizontal_Distance_To_Fire_Points']-train['Horizontal_Distance_To_Roadways'])/2

train['Slope2'] = np.sqrt(train['Horizontal_Distance_To_Hydrology']**2+train['Vertical_Distance_To_Hydrology']**2)
train['Mean_Fire_Hydrology_Roadways']=(train['Horizontal_Distance_To_Fire_Points'] + train['Horizontal_Distance_To_Hydrology'] + train['Horizontal_Distance_To_Roadways']) / 3
train['Mean_Fire_Hyd']=(train['Horizontal_Distance_To_Fire_Points'] + train['Horizontal_Distance_To_Hydrology']) / 2 

train["Vertical_Distance_To_Hydrology"] = abs(train['Vertical_Distance_To_Hydrology'])

train['Neg_EHyd'] = train.Elevation-train.Horizontal_Distance_To_Hydrology*0.2


test['HorizontalHydrology_HorizontalFire'] = (test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Fire_Points'])
test['Neg_HorizontalHydrology_HorizontalFire'] = (test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Fire_Points'])
test['HorizontalHydrology_HorizontalRoadways'] = (test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Roadways'])
test['Neg_HorizontalHydrology_HorizontalRoadways'] = (test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Roadways'])
test['HorizontalFire_Points_HorizontalRoadways'] = (test['Horizontal_Distance_To_Fire_Points']+test['Horizontal_Distance_To_Roadways'])
test['Neg_HorizontalFire_Points_HorizontalRoadways'] = (test['Horizontal_Distance_To_Fire_Points']-test['Horizontal_Distance_To_Roadways'])

test['Neg_Elevation_Vertical'] = test['Elevation']-test['Vertical_Distance_To_Hydrology']
test['Elevation_Vertical'] = test['Elevation'] + test['Vertical_Distance_To_Hydrology']

test['mean_hillshade'] = (test['Hillshade_9am']  + test['Hillshade_Noon']  + test['Hillshade_3pm'] ) / 3

test['Mean_HorizontalHydrology_HorizontalFire'] = (test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Fire_Points'])/2
test['Mean_HorizontalHydrology_HorizontalRoadways'] = (test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Roadways'])/2
test['Mean_HorizontalFire_Points_HorizontalRoadways'] = (test['Horizontal_Distance_To_Fire_Points']+test['Horizontal_Distance_To_Roadways'])/2

test['MeanNeg_Mean_HorizontalHydrology_HorizontalFire'] = (test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Fire_Points'])/2
test['MeanNeg_HorizontalHydrology_HorizontalRoadways'] = (test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Roadways'])/2
test['MeanNeg_HorizontalFire_Points_HorizontalRoadways'] = (test['Horizontal_Distance_To_Fire_Points']-test['Horizontal_Distance_To_Roadways'])/2

test['Slope2'] = np.sqrt(test['Horizontal_Distance_To_Hydrology']**2+test['Vertical_Distance_To_Hydrology']**2)
test['Mean_Fire_Hydrology_Roadways']=(test['Horizontal_Distance_To_Fire_Points'] + test['Horizontal_Distance_To_Hydrology'] + test['Horizontal_Distance_To_Roadways']) / 3 
test['Mean_Fire_Hyd']=(test['Horizontal_Distance_To_Fire_Points'] + test['Horizontal_Distance_To_Hydrology']) / 2


test['Vertical_Distance_To_Hydrology'] = abs(test["Vertical_Distance_To_Hydrology"])

test['Neg_EHyd'] = test.Elevation-test.Horizontal_Distance_To_Hydrology*0.2


# Now we should seperate the training set from the labels and name them x and y then we will split them into training and test sets to be able to see how well it would do on unseen data which will give anestimate on how well it will do when testing on Kaggle test data. I will use the convention of using 80% of the data as training set and 20% for the test set.

# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


from sklearn.model_selection import train_test_split
x = train.drop(['Cover_Type'], axis = 1)
y = train['Cover_Type']


x_train, x_val, y_train, y_val = train_test_split( x.values, y.values, test_size=0.2, random_state=42 )
print(x_train.shape)
print(x_val.shape)
print(y_train.shape)
print(y_val.shape)


# It is important to know if the number of points in the classes are balanced. If the data is skewed then we will not be able to use accuracy as a performance metric since it will be misleading but if it is skewed we may use F-beta score or precision and recall. Precision or recall or F1 score. the choice depends on the problem itself. Where high recall means low number of false negatives , High precision means low number of false positives and F1 score is a trade off between them. You can refere to this article for more about precision and recall http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html

# In[ ]:


unique, count= np.unique(y_train, return_counts=True)
print("The number of occurances of each class in the dataset = %s " % dict (zip(unique, count) ), "\n" )


# It seems the data points in each class are almost balanced so it will be okay to use accuracy as a metric to measure how well the ML model performs

# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)

test = scaler.transform(test)


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier , ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression


# ## Select and Initialize Classifiers
# 
# I have tried to select various classifiers. The key is a good validation score and if possible the use of a diffferent method/classifier for the ensembling.

# In[ ]:


rf_1 = RandomForestClassifier(n_estimators = 200,criterion = 'entropy',random_state = 0)
rf_1.fit(X=x_train, y=y_train)

y_pred_train_rf_1 = rf_1.predict(x_train)
y_pred_val_rf_1 = rf_1.predict(x_val)

y_pred_test_rf_1 = rf_1.predict(test)


# In[ ]:


rf_2 = RandomForestClassifier(n_estimators = 200,criterion = 'gini',random_state = 0)
rf_2.fit(X=x_train, y=y_train)

y_pred_train_rf_2 = rf_2.predict(x_train)
y_pred_val_rf_2 = rf_2.predict(x_val)

y_pred_test_rf_2 = rf_2.predict(test)


# In[ ]:


et_1 = ExtraTreesClassifier(n_estimators = 200,criterion = 'entropy',random_state = 0)
et_1.fit(X=x_train, y=y_train)

y_pred_train_et_1 = et_1.predict(x_train)
y_pred_val_et_1 = et_1.predict(x_val)

y_pred_test_et_1 = et_1.predict(test)


# In[ ]:


et_2 = ExtraTreesClassifier(n_estimators = 200,criterion = 'gini',random_state = 0)
et_2.fit(X=x_train, y=y_train)

y_pred_train_et_2 = et_2.predict(x_train)
y_pred_val_et_2 = et_2.predict(x_val)

y_pred_test_et_2 = et_2.predict(test)


# In[ ]:


lgb = LGBMClassifier(n_estimators = 200,learning_rate = 0.1)
lgb.fit(X=x_train, y=y_train)

y_pred_train_lgb = lgb.predict(x_train)
y_pred_val_lgb = lgb.predict(x_val)

y_pred_test_lgb = lgb.predict(test)


# In[ ]:


lr_1 = LogisticRegression(solver = 'liblinear',multi_class = 'ovr',C = 1,random_state = 0)
lr_1.fit(X=x_train, y=y_train)

y_pred_train_lr_1 = lr_1.predict(x_train)
y_pred_val_lr_1 = lr_1.predict(x_val)

y_pred_test_lr_1 = lr_1.predict(test)


# In[ ]:


xgb_1 = XGBClassifier(seed = 0,colsample_bytree = 0.7, silent = 1, subsample = 0.7, learning_rate = 0.1, objective = 'multi:softprob',
                      num_class = 7,max_depth = 4, min_child_weight = 1, eval_metric = 'mlogloss', nrounds = 200)
xgb_1.fit(X=x_train, y=y_train)

y_pred_train_xgb_1 = xgb_1.predict(x_train)
y_pred_val_xgb_1 = xgb_1.predict(x_val)

y_pred_test_xgb_1 = xgb_1.predict(test)


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=5)  

knn.fit(x_train, y_train)

y_pred_train_knn = knn.predict(x_train)
y_pred_val_knn = knn.predict(x_val)

y_pred_test_knn = knn.predict(test)


# In[ ]:


## Creating DF from Predections

stack_train = pd.DataFrame([y_pred_train_rf_1,y_pred_train_rf_2,y_pred_train_et_1,y_pred_train_et_2,y_pred_train_lgb,
                            y_pred_train_lr_1,y_pred_train_xgb_1,y_pred_train_knn])

stack_val = pd.DataFrame([y_pred_val_rf_1,y_pred_val_rf_2,y_pred_val_et_1,y_pred_val_et_2,y_pred_val_lgb,
                            y_pred_val_lr_1,y_pred_val_xgb_1,y_pred_val_knn])

stack_test = pd.DataFrame([y_pred_test_rf_1,y_pred_test_rf_2,y_pred_test_et_1,y_pred_test_et_2,y_pred_test_lgb,
                            y_pred_test_lr_1,y_pred_test_xgb_1,y_pred_test_knn])


# In[ ]:


print(stack_train.head())
print(stack_val.head())

print(stack_test.head())


# In[ ]:


## Transpose - it will change row into columns and columns into rows

stack_train = stack_train.T
stack_val = stack_val.T

stack_test = stack_test.T


# In[ ]:


print(stack_train.head())
print(stack_val.head())
print(stack_test.head())


# In[ ]:


print(stack_train.shape)
print(stack_val.shape)
print(stack_test.shape)


# In[ ]:


stack_test.isnull().sum()


# In[ ]:


lr_2 = LogisticRegression(solver = 'liblinear',multi_class = 'ovr',C = 5,random_state = 0)
lr_2.fit(X=stack_train, y=y_train)

stacked_pred_train = lr_2.predict(stack_train)
stacked_pred_val = lr_2.predict(stack_val)

stacked_pred_test = lr_2.predict(stack_test)


# In[ ]:


#Id = test['Id']
#test.drop(['Id'], inplace = True, axis = 1 )
#-final_pred = lr_2.predict(test)


submission_1 = pd.DataFrame()
submission_1['Id'] = Id
submission_1['Cover_Type'] = stacked_pred_test
submission_1.to_csv('submission_stack.csv', index=False)
submission_1.head(5)


# ## CatBoostClassifier

# In[ ]:


from catboost import Pool, CatBoostClassifier

cat = CatBoostClassifier()

cat.fit(x_train, y_train)


# In[ ]:


print('Accuracy of classifier on training set: {:.2f}'.format(cat.score(x_train, y_train) * 100))
print('Accuracy of classifier on test set: {:.2f}'.format(cat.score(x_val, y_val) * 100))


# In[ ]:


cat_predictions = cat.predict(test)


# In[ ]:


submission_2 = pd.DataFrame()
submission_2['Id'] = Id
submission_2['Cover_Type'] = cat_predictions
submission_2.to_csv('submission.csv', index=False)
submission_2.head(5)


# ## XGBoostClassifier

# In[ ]:


XGB = XGBClassifier()

XGB.fit(x_train, y_train)


# In[ ]:


print('Accuracy of classifier on training set: {:.2f}'.format(XGB.score(x_train, y_train) * 100))
print('Accuracy of classifier on test set: {:.2f}'.format(XGB.score(x_val, y_val) * 100))


# In[ ]:


XGB_predictions = XGB.predict(test)


# In[ ]:


submission_3 = pd.DataFrame()
submission_3['Id'] = Id
submission_3['Cover_Type'] = XGB_predictions
submission_3.to_csv('submission_XGB.csv', index=False)
submission_3.head(5)


# ## RandomForestClassifier

# In[ ]:


RFC = RandomForestClassifier()

RFC.fit(x_train, y_train)


# In[ ]:


print('Accuracy of classifier on training set: {:.2f}'.format(RFC.score(x_train, y_train) * 100))
print('Accuracy of classifier on test set: {:.2f}'.format(RFC.score(x_val, y_val) * 100))


# In[ ]:


RFC_predictions = RFC.predict(test)


# In[ ]:


submission_4 = pd.DataFrame()
submission_4['Id'] = Id
submission_4['Cover_Type'] = RFC_predictions
submission_4.to_csv('submission_RFC.csv', index=False)
submission_4.head(5)


# In[ ]:




