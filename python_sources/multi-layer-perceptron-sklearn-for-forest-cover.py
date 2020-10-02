#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.neural_network import MLPClassifier


# In[ ]:


train=pd.read_csv('../input/train.csv')
train.drop(['Id'],axis=1,inplace=True)
test=pd.read_csv('../input/test.csv')

Id=test['Id']
test.drop(['Id'],axis=1,inplace=True)
train.head()


# In[ ]:


train.shape


# In[ ]:


train['HF1'] = train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Fire_Points']
train['HF2'] = abs(train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Fire_Points'])
train['HR1'] = abs(train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Roadways'])
train['HR2'] = abs(train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Roadways'])
train['FR1'] = abs(train['Horizontal_Distance_To_Fire_Points']+train['Horizontal_Distance_To_Roadways'])
train['FR2'] = abs(train['Horizontal_Distance_To_Fire_Points']-train['Horizontal_Distance_To_Roadways'])
train['neg_ele_vert'] = train.Elevation-train.Vertical_Distance_To_Hydrology
train['ele_vert'] = train['Elevation']+train['Vertical_Distance_To_Hydrology']
#Amenities
train['Mean_Amenities']=(train.Horizontal_Distance_To_Fire_Points + train.Horizontal_Distance_To_Hydrology + train.Horizontal_Distance_To_Roadways) / 3 
#Mean Distance to Fire and Water 
train['Mean_Fire_Hyd']=(train.Horizontal_Distance_To_Fire_Points + train.Horizontal_Distance_To_Hydrology) / 2 
#Slope calc
train['slope2'] = np.sqrt(train['Horizontal_Distance_To_Hydrology']**2+train['Vertical_Distance_To_Hydrology']**2)
#Mean Hillside
train['mean_hillshade'] =  (train['Hillshade_9am']  + train['Hillshade_Noon'] + train['Hillshade_3pm'] ) / 3
#Absolute value
train["Vertical_Distance_To_Hydrology"] = abs(train['Vertical_Distance_To_Hydrology'])
train['Neg_EHyd'] = train.Elevation-train.Horizontal_Distance_To_Hydrology*0.2

#Added recently
train['Mean_HorizontalHydrology_HorizontalFire'] = (train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Fire_Points'])/2
train['Mean_HorizontalHydrology_HorizontalRoadways'] = (train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Roadways'])/2
train['Mean_HorizontalFire_Points_HorizontalRoadways'] = (train['Horizontal_Distance_To_Fire_Points']+train['Horizontal_Distance_To_Roadways'])/2

train['MeanNeg_Mean_HorizontalHydrology_HorizontalFire'] = (train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Fire_Points'])/2
train['MeanNeg_HorizontalHydrology_HorizontalRoadways'] = (train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Roadways'])/2
train['MeanNeg_HorizontalFire_Points_HorizontalRoadways'] = (train['Horizontal_Distance_To_Fire_Points']-train['Horizontal_Distance_To_Roadways'])/2
train['Mean_Fire_Hydrology_Roadways']=(train['Horizontal_Distance_To_Fire_Points'] + train['Horizontal_Distance_To_Hydrology'] + train['Horizontal_Distance_To_Roadways']) / 3
train['slope_hyd'] = np.sqrt(train.Vertical_Distance_To_Hydrology**2 + train.Horizontal_Distance_To_Hydrology**2) 
train.slope_hyd=train.slope_hyd.map(lambda x: 0 if np.isinf(x) else x) 
train['Elev_to_HD_Hyd']=train.Elevation - 0.2 * train.Horizontal_Distance_To_Hydrology
train['Elev_to_HD_Road']=train.Elevation - 0.05 * train.Horizontal_Distance_To_Roadways
train['Elev_to_VD_Hyd']=train.Elevation - train.Vertical_Distance_To_Hydrology

train['Mean_Fire_Hyd']=(train.Horizontal_Distance_To_Fire_Points + train.Horizontal_Distance_To_Hydrology) / 2


# In[ ]:


#Same for test
test['HF1'] = test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Fire_Points']
test['HF2'] = abs(test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Fire_Points'])
test['HR1'] = abs(test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Roadways'])
test['HR2'] = abs(test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Roadways'])
test['FR1'] = abs(test['Horizontal_Distance_To_Fire_Points']+test['Horizontal_Distance_To_Roadways'])
test['FR2'] = abs(test['Horizontal_Distance_To_Fire_Points']-test['Horizontal_Distance_To_Roadways'])
test['neg_ele_vert'] = test.Elevation-test.Vertical_Distance_To_Hydrology
test['ele_vert'] = test['Elevation']+test['Vertical_Distance_To_Hydrology']
#Amenities
test['Mean_Amenities']=(test.Horizontal_Distance_To_Fire_Points + test.Horizontal_Distance_To_Hydrology + test.Horizontal_Distance_To_Roadways) / 3 
#Mean Distance to Fire and Water 
test['Mean_Fire_Hyd']=(test.Horizontal_Distance_To_Fire_Points + test.Horizontal_Distance_To_Hydrology) / 2 
#Slope calc
test['slope2'] = np.sqrt(test['Horizontal_Distance_To_Hydrology']**2+test['Vertical_Distance_To_Hydrology']**2)
#Mean Hillside
test['mean_hillshade'] =  (test['Hillshade_9am']  + test['Hillshade_Noon'] + test['Hillshade_3pm'] ) / 3
#Absolute value
test["Vertical_Distance_To_Hydrology"] = abs(test['Vertical_Distance_To_Hydrology'])
test['Neg_EHyd'] = test.Elevation-test.Horizontal_Distance_To_Hydrology*0.2

#Added Recently
test['Mean_HorizontalHydrology_HorizontalFire'] = (test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Fire_Points'])/2
test['Mean_HorizontalHydrology_HorizontalRoadways'] = (test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Roadways'])/2
test['Mean_HorizontalFire_Points_HorizontalRoadways'] = (test['Horizontal_Distance_To_Fire_Points']+test['Horizontal_Distance_To_Roadways'])/2

test['MeanNeg_Mean_HorizontalHydrology_HorizontalFire'] = (test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Fire_Points'])/2
test['MeanNeg_HorizontalHydrology_HorizontalRoadways'] = (test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Roadways'])/2
test['MeanNeg_HorizontalFire_Points_HorizontalRoadways'] = (test['Horizontal_Distance_To_Fire_Points']-test['Horizontal_Distance_To_Roadways'])/2
test['Mean_Fire_Hydrology_Roadways']=(test['Horizontal_Distance_To_Fire_Points'] + test['Horizontal_Distance_To_Hydrology'] + test['Horizontal_Distance_To_Roadways']) / 3

test['slope_hyd'] = np.sqrt(test.Vertical_Distance_To_Hydrology**2 + test.Horizontal_Distance_To_Hydrology**2) 
test.slope_hyd=test.slope_hyd.map(lambda x: 0 if np.isinf(x) else x) 
test['Elev_to_HD_Hyd']=test.Elevation - 0.2 * test.Horizontal_Distance_To_Hydrology
test['Elev_to_HD_Road']=test.Elevation - 0.05 * test.Horizontal_Distance_To_Roadways
test['Elev_to_VD_Hyd']=test.Elevation - test.Vertical_Distance_To_Hydrology

test['Mean_Fire_Hyd']=(test.Horizontal_Distance_To_Fire_Points + test.Horizontal_Distance_To_Hydrology) / 2


# In[ ]:


train.drop(['Soil_Type15' , "Soil_Type7"], inplace = True, axis = 1 )
test.drop(['Soil_Type15' , "Soil_Type7"], inplace = True, axis = 1 )


# In[ ]:


cols=[i for i in train.columns if i not in ['Id', 'Cover_Type' ]]


# In[ ]:


X_train = train[cols]
y_train = train['Cover_Type']
X_test = test[cols]


# In[ ]:


#from sklearn.cross_validation import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.20, random_state=2)


# In[ ]:


#from sklearn.metrics import accuracy_score


# In[ ]:


from sklearn.preprocessing import StandardScaler
standard_X = StandardScaler()

X_std_train = standard_X.fit_transform(X_train)
X_std_test = standard_X.transform(X_test)


# In[ ]:


#X_train, X_test, y_train, y_test = train_test_split(X_std_train, y_train, test_size=0.20, random_state=2)


# In[ ]:


mlp = MLPClassifier(activation='tanh', alpha=3, solver='lbfgs', hidden_layer_sizes=(300,150,75, 40), learning_rate='adaptive', random_state=1, tol=0.000001)
#SVClassifier = svm.SVC(C=100)
mlp.fit(X_std_train,y_train)


# In[ ]:


y_pred = mlp.predict(X_std_test)
#accuracy_score(y_pred, y_test)


# In[ ]:


submission = pd.DataFrame({'Id':Id,'Cover_Type':y_pred})
submission.to_csv('MLP_submission1.csv',index=False)


# In[ ]:




