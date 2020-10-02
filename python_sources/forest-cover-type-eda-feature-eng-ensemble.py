#!/usr/bin/env python
# coding: utf-8

# ## Problem Statement

# Predict the forest cover type from the given cartographic variables. 
# This study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado. These areas represent forests with minimal human-caused disturbances, so that existing forest cover types are more a result of ecological processes rather than forest management practices.

# ### Importing Libraries

# In[ ]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from scipy.stats import norm 
from matplotlib import cm
import seaborn as sns


# #### Readding the given files - Train data, Test data & Sample Submission data

# In[ ]:


df_train = pd.read_csv('../input/given-data/train.csv')


# In[ ]:


df_test = pd.read_csv('../input/given-data/test.csv')


# In[ ]:


df_sample = pd.read_csv('../input/given-data/sample_submission.csv')


# #### Data Exploration and analysis

# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


df_sample.head()


# In[ ]:


df_sample.nunique()


# In[ ]:


df_train.nunique()


# The given dataset cantains **56** features including the target variable Cover_Type, along with **15120** observations and following are the features
# 
# 
# - Elevation - Elevation in meters
# - Aspect - Aspect in degrees azimuth
# - Slope - Slope in degrees
# - Horizontal_Distance_To_Hydrology - Horz Dist to nearest surface water features
# - Vertical_Distance_To_Hydrology - Vert Dist to nearest surface water features
# - Horizontal_Distance_To_Roadways - Horz Dist to nearest roadway
# - Hillshade_9am (0 to 255 index) - Hillshade index at 9am, summer solstice
# - Hillshade_Noon (0 to 255 index) - Hillshade index at noon, summer solstice
# - Hillshade_3pm (0 to 255 index) - Hillshade index at 3pm, summer solstice
# - Horizontal_Distance_To_Fire_Points - Horz Dist to nearest wildfire ignition points
# - Wilderness_Area (4 binary columns, 0 = absence or 1 = presence) - Wilderness area designation
# - Soil_Type (40 binary columns, 0 = absence or 1 = presence) - Soil Type designation
# - Cover_Type (7 types, integers 1 to 7) - Forest Cover Type designation
# 
# The wilderness areas are:
# 
# - 1 - Rawah Wilderness Area
# - 2 - Neota Wilderness Area
# - 3 - Comanche Peak Wilderness Area
# - 4 - Cache la Poudre Wilderness Area
# 
# The soil types are:
# 
# - 1 Cathedral family - Rock outcrop complex, extremely stony.
# - 2 Vanet - Ratake families complex, very stony.
# - 3 Haploborolis - Rock outcrop complex, rubbly.
# - 4 Ratake family - Rock outcrop complex, rubbly.
# - 5 Vanet family - Rock outcrop complex complex, rubbly.
# - 6 Vanet - Wetmore families - Rock outcrop complex, stony.
# - 7 Gothic family.
# - 8 Supervisor - Limber families complex.
# - 9 Troutville family, very stony.
# - 10 Bullwark - Catamount families - Rock outcrop complex, rubbly.
# - 11 Bullwark - Catamount families - Rock land complex, rubbly.
# - 12 Legault family - Rock land complex, stony.
# - 13 Catamount family - Rock land - Bullwark family complex, rubbly.
# - 14 Pachic Argiborolis - Aquolis complex.
# - 15 unspecified in the USFS Soil and ELU Survey.
# - 16 Cryaquolis - Cryoborolis complex.
# - 17 Gateview family - Cryaquolis complex.
# - 18 Rogert family, very stony.
# - 19 Typic Cryaquolis - Borohemists complex.
# - 20 Typic Cryaquepts - Typic Cryaquolls complex.
# - 21 Typic Cryaquolls - Leighcan family, till substratum complex.
# - 22 Leighcan family, till substratum, extremely bouldery.
# - 23 Leighcan family, till substratum - Typic Cryaquolls complex.
# - 24 Leighcan family, extremely stony.
# - 25 Leighcan family, warm, extremely stony.
# - 26 Granile - Catamount families complex, very stony.
# - 27 Leighcan family, warm - Rock outcrop complex, extremely stony.
# - 28 Leighcan family - Rock outcrop complex, extremely stony.
# - 29 Como - Legault families complex, extremely stony.
# - 30 Como family - Rock land - Legault family complex, extremely stony.
# - 31 Leighcan - Catamount families complex, extremely stony.
# - 32 Catamount family - Rock outcrop - Leighcan family complex, extremely stony.
# - 33 Leighcan - Catamount families - Rock outcrop complex, extremely stony.
# - 34 Cryorthents - Rock land complex, extremely stony.
# - 35 Cryumbrepts - Rock outcrop - Cryaquepts complex.
# - 36 Bross family - Rock land - Cryumbrepts complex, extremely stony.
# - 37 Rock outcrop - Cryumbrepts - Cryorthents complex, extremely stony.
# - 38 Leighcan - Moran families - Cryaquolls complex, extremely stony.
# - 39 Moran family - Cryorthents - Leighcan family complex, extremely stony.
# - 40 Moran family - Cryorthents - Rock land complex, extremely stony.

# In[ ]:


df_train.isnull().sum()


# There is no null values in the given data

# ##### Checking the correlation between the given features and the target vaiables

# In[ ]:


f,ax = plt.subplots(figsize=(25, 25))
sns.heatmap(df_train.corr(), annot=True, linewidths=.5, fmt= '.3f',ax=ax)
plt.show()


# In[ ]:


df_train.corr()


# From the above correlation map and matrix results, we can see that soil_Type7 and soil_Type15 doesn't have any correlation with the target variable Cover_Type so we can drop them from the data. Also, we can see that Soil_Type9, Soil_Type36, Soil_Type27, Soil_Type25, Soil_Type8 have weak correlation but when a feature has a weak correlation that doesn't mean it is not useful because it may make a good impact if we do interactions.

# In[ ]:


Id = df_test['Id']
df_train.drop(['Id','Soil_Type15' , "Soil_Type7"], inplace = True, axis = 1 )
df_test.drop(['Id','Soil_Type15' , "Soil_Type7"], inplace = True, axis = 1 )


# Also it seems that the vertical distance contain some negative number, so we are going to make them absolute. Tree based models only fits vertical and horizontal lines so it is very important to engineer features like slope. So in the next we are introducing some new features and as well as we are making the negative values absolute.

# In[ ]:


df_train['HorizontalHydrology_HorizontalFire'] = (df_train['Horizontal_Distance_To_Hydrology']+df_train['Horizontal_Distance_To_Fire_Points'])
df_train['Neg_HorizontalHydrology_HorizontalFire'] = (df_train['Horizontal_Distance_To_Hydrology']-df_train['Horizontal_Distance_To_Fire_Points'])
df_train['HorizontalHydrology_HorizontalRoadways'] = (df_train['Horizontal_Distance_To_Hydrology']+df_train['Horizontal_Distance_To_Roadways'])
df_train['Neg_HorizontalHydrology_HorizontalRoadways'] = (df_train['Horizontal_Distance_To_Hydrology']-df_train['Horizontal_Distance_To_Roadways'])
df_train['HorizontalFire_Points_HorizontalRoadways'] = (df_train['Horizontal_Distance_To_Fire_Points']+df_train['Horizontal_Distance_To_Roadways'])
df_train['Neg_HorizontalFire_Points_HorizontalRoadways'] = (df_train['Horizontal_Distance_To_Fire_Points']-df_train['Horizontal_Distance_To_Roadways'])

df_train['Neg_Elevation_Vertical'] = df_train['Elevation']-df_train['Vertical_Distance_To_Hydrology']
df_train['Elevation_Vertical'] = df_train['Elevation']+df_train['Vertical_Distance_To_Hydrology']

df_train['mean_hillshade'] =  (df_train['Hillshade_9am']  + df_train['Hillshade_Noon'] + df_train['Hillshade_3pm'] ) / 3

df_train['Mean_HorizontalHydrology_HorizontalFire'] = (df_train['Horizontal_Distance_To_Hydrology']+df_train['Horizontal_Distance_To_Fire_Points'])/2
df_train['Mean_HorizontalHydrology_HorizontalRoadways'] = (df_train['Horizontal_Distance_To_Hydrology']+df_train['Horizontal_Distance_To_Roadways'])/2
df_train['Mean_HorizontalFire_Points_HorizontalRoadways'] = (df_train['Horizontal_Distance_To_Fire_Points']+df_train['Horizontal_Distance_To_Roadways'])/2

df_train['MeanNeg_Mean_HorizontalHydrology_HorizontalFire'] = (df_train['Horizontal_Distance_To_Hydrology']-df_train['Horizontal_Distance_To_Fire_Points'])/2
df_train['MeanNeg_HorizontalHydrology_HorizontalRoadways'] = (df_train['Horizontal_Distance_To_Hydrology']-df_train['Horizontal_Distance_To_Roadways'])/2
df_train['MeanNeg_HorizontalFire_Points_HorizontalRoadways'] = (df_train['Horizontal_Distance_To_Fire_Points']-df_train['Horizontal_Distance_To_Roadways'])/2

df_train['Slope2'] = np.sqrt(df_train['Horizontal_Distance_To_Hydrology']**2+df_train['Vertical_Distance_To_Hydrology']**2)
df_train['Mean_Fire_Hydrology_Roadways']=(df_train['Horizontal_Distance_To_Fire_Points'] + df_train['Horizontal_Distance_To_Hydrology'] + df_train['Horizontal_Distance_To_Roadways']) / 3
df_train['Mean_Fire_Hyd']=(df_train['Horizontal_Distance_To_Fire_Points'] + df_train['Horizontal_Distance_To_Hydrology']) / 2 

df_train["Vertical_Distance_To_Hydrology"] = abs(df_train['Vertical_Distance_To_Hydrology'])

df_train['Neg_EHyd'] = df_train.Elevation-df_train.Horizontal_Distance_To_Hydrology*0.2


df_test['HorizontalHydrology_HorizontalFire'] = (df_test['Horizontal_Distance_To_Hydrology']+df_test['Horizontal_Distance_To_Fire_Points'])
df_test['Neg_HorizontalHydrology_HorizontalFire'] = (df_test['Horizontal_Distance_To_Hydrology']-df_test['Horizontal_Distance_To_Fire_Points'])
df_test['HorizontalHydrology_HorizontalRoadways'] = (df_test['Horizontal_Distance_To_Hydrology']+df_test['Horizontal_Distance_To_Roadways'])
df_test['Neg_HorizontalHydrology_HorizontalRoadways'] = (df_test['Horizontal_Distance_To_Hydrology']-df_test['Horizontal_Distance_To_Roadways'])
df_test['HorizontalFire_Points_HorizontalRoadways'] = (df_test['Horizontal_Distance_To_Fire_Points']+df_test['Horizontal_Distance_To_Roadways'])
df_test['Neg_HorizontalFire_Points_HorizontalRoadways'] = (df_test['Horizontal_Distance_To_Fire_Points']-df_test['Horizontal_Distance_To_Roadways'])

df_test['Neg_Elevation_Vertical'] = df_test['Elevation']-df_test['Vertical_Distance_To_Hydrology']
df_test['Elevation_Vertical'] = df_test['Elevation'] + df_test['Vertical_Distance_To_Hydrology']

df_test['mean_hillshade'] = (df_test['Hillshade_9am']  + df_test['Hillshade_Noon']  + df_test['Hillshade_3pm'] ) / 3

df_test['Mean_HorizontalHydrology_HorizontalFire'] = (df_test['Horizontal_Distance_To_Hydrology']+df_test['Horizontal_Distance_To_Fire_Points'])/2
df_test['Mean_HorizontalHydrology_HorizontalRoadways'] = (df_test['Horizontal_Distance_To_Hydrology']+df_test['Horizontal_Distance_To_Roadways'])/2
df_test['Mean_HorizontalFire_Points_HorizontalRoadways'] = (df_test['Horizontal_Distance_To_Fire_Points']+df_test['Horizontal_Distance_To_Roadways'])/2

df_test['MeanNeg_Mean_HorizontalHydrology_HorizontalFire'] = (df_test['Horizontal_Distance_To_Hydrology']-df_test['Horizontal_Distance_To_Fire_Points'])/2
df_test['MeanNeg_HorizontalHydrology_HorizontalRoadways'] = (df_test['Horizontal_Distance_To_Hydrology']-df_test['Horizontal_Distance_To_Roadways'])/2
df_test['MeanNeg_HorizontalFire_Points_HorizontalRoadways'] = (df_test['Horizontal_Distance_To_Fire_Points']-df_test['Horizontal_Distance_To_Roadways'])/2

df_test['Slope2'] = np.sqrt(df_test['Horizontal_Distance_To_Hydrology']**2+df_test['Vertical_Distance_To_Hydrology']**2)
df_test['Mean_Fire_Hydrology_Roadways']=(df_test['Horizontal_Distance_To_Fire_Points'] + df_test['Horizontal_Distance_To_Hydrology'] + df_test['Horizontal_Distance_To_Roadways']) / 3 
df_test['Mean_Fire_Hyd']=(df_test['Horizontal_Distance_To_Fire_Points'] + df_test['Horizontal_Distance_To_Hydrology']) / 2


df_test['Vertical_Distance_To_Hydrology'] = abs(df_test["Vertical_Distance_To_Hydrology"])

df_test['Neg_EHyd'] = df_test.Elevation-df_test.Horizontal_Distance_To_Hydrology*0.2


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# ### Model building

# In[ ]:


from sklearn.model_selection import train_test_split
x = df_train.drop(['Cover_Type'], axis = 1)
y = df_train['Cover_Type']

x_train, x_test, y_train, y_test = train_test_split( x.values, y.values, test_size=0.2, random_state=116214 )


# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# In[ ]:


df_test.head()


# In[ ]:


df_test.head()
df_test = scaler.transform(df_test)


# #### Decision Tree Model

# In[ ]:


DT = DecisionTreeClassifier()


# In[ ]:


DT.fit(x_train, y_train)


# In[ ]:


print('Accuracy of classifier on training set: {:.2f}'.format(DT.score(x_train, y_train) * 100))
print('Accuracy of classifier on test set: {:.2f}'.format(DT.score(x_test, y_test) * 100))


# In[ ]:


prediction_DT = DT.predict(df_test)


# In[ ]:


submission_DT = pd.DataFrame()
submission_DT['Id'] = Id
submission_DT['Cover_Type'] = prediction_DT
submission_DT.to_csv('submission_DT.csv', index=False)
submission_DT.head(5)


# #### Random Forest model

# In[ ]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
RF = RandomForestClassifier()


# In[ ]:


param_grid = {'n_estimators' :[100,150,200,250,300]}


# In[ ]:


RF_R = RandomizedSearchCV(estimator = RF, param_distributions = param_grid, n_jobs = -1, cv = 10)


# In[ ]:


RF_R.fit(x_train, y_train)


# In[ ]:


print('Accuracy of classifier on training set: {:.2f}'.format(RF_R.score(x_train, y_train) * 100))
print('Accuracy of classifier on test set: {:.2f}'.format(RF_R.score(x_test, y_test) * 100))


# In[ ]:


prediction_RF = RF_R.predict(df_test)


# In[ ]:


submission_RF = pd.DataFrame()
submission_RF['Id'] = Id
submission_RF['Cover_Type'] = prediction_RF
submission_RF.to_csv('submission_RF.csv', index=False)
submission_RF.head(5)


# #### XGBoost Model

# In[ ]:


XGB = XGBClassifier()


# In[ ]:


XGB.fit(x_train, y_train)


# In[ ]:


print('Accuracy of classifier on training set: {:.2f}'.format(XGB.score(x_train, y_train) * 100))
print('Accuracy of classifier on test set: {:.2f}'.format(XGB.score(x_test, y_test) * 100))


# In[ ]:


prediction_XGB = XGB.predict(df_test)


# In[ ]:


submission_XGB = pd.DataFrame()
submission_XGB['Id'] = Id
submission_XGB['Cover_Type'] = prediction_XGB
submission_XGB.to_csv('submission_XGB.csv', index=False)
submission_XGB.head(5)


# #### CatBoost Model

# In[ ]:


from catboost import Pool, CatBoostClassifier


# In[ ]:


CBR = CatBoostClassifier(iterations=100)


# In[ ]:


CBR.fit(x_train, y_train)


# In[ ]:


print('Accuracy of classifier on training set: {:.2f}'.format(CBR.score(x_train, y_train) * 100))
print('Accuracy of classifier on test set: {:.2f}'.format(CBR.score(x_test, y_test) * 100))


# In[ ]:


prediction_CBR = CBR.predict(df_test)


# In[ ]:


submission_CBR = pd.DataFrame()
submission_CBR['Id'] = Id
submission_CBR['Cover_Type'] = prediction_CBR
submission_CBR.to_csv('submission_CBR.csv', index=False)
submission_CBR.head(5)


# In[ ]:


submission_RF.to_csv('submission.csv', index=None)

