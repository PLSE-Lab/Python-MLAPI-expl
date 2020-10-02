#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Contents of feature engineering have been taken from Lathwal's kernel
import numpy as np
import pandas as pd
from sklearn import ensemble 
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier

df = pd.read_csv('../input/train.csv')
df.head()


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


####################### Train data #############################################
train['HF1'] = train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Fire_Points']
train['HF2'] = abs(train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Fire_Points'])
train['HR1'] = abs(train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Roadways'])
train['HR2'] = abs(train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Roadways'])
train['FR1'] = abs(train['Horizontal_Distance_To_Fire_Points']+train['Horizontal_Distance_To_Roadways'])
train['FR2'] = abs(train['Horizontal_Distance_To_Fire_Points']-train['Horizontal_Distance_To_Roadways'])
train['ele_vert'] = train.Elevation-train.Vertical_Distance_To_Hydrology

train['slope_hyd'] = (train['Horizontal_Distance_To_Hydrology']**2+train['Vertical_Distance_To_Hydrology']**2)**0.5
train.slope_hyd=train.slope_hyd.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any

#Mean distance to Amenities 
train['Mean_Amenities']=(train.Horizontal_Distance_To_Fire_Points + train.Horizontal_Distance_To_Hydrology + train.Horizontal_Distance_To_Roadways) / 3 
#Mean Distance to Fire and Water 
train['Mean_Fire_Hyd']=(train.Horizontal_Distance_To_Fire_Points + train.Horizontal_Distance_To_Hydrology) / 2 

####################### Test data #############################################
test['HF1'] = test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Fire_Points']
test['HF2'] = abs(test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Fire_Points'])
test['HR1'] = abs(test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Roadways'])
test['HR2'] = abs(test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Roadways'])
test['FR1'] = abs(test['Horizontal_Distance_To_Fire_Points']+test['Horizontal_Distance_To_Roadways'])
test['FR2'] = abs(test['Horizontal_Distance_To_Fire_Points']-test['Horizontal_Distance_To_Roadways'])
test['ele_vert'] = test.Elevation-test.Vertical_Distance_To_Hydrology

test['slope_hyd'] = (test['Horizontal_Distance_To_Hydrology']**2+test['Vertical_Distance_To_Hydrology']**2)**0.5
test.slope_hyd=test.slope_hyd.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any

#Mean distance to Amenities 
test['Mean_Amenities']=(test.Horizontal_Distance_To_Fire_Points + test.Horizontal_Distance_To_Hydrology + test.Horizontal_Distance_To_Roadways) / 3 
#Mean Distance to Fire and Water 
test['Mean_Fire_Hyd']=(test.Horizontal_Distance_To_Fire_Points + test.Horizontal_Distance_To_Hydrology) / 2


# In[ ]:


feature = [col for col in train.columns if col not in ['Cover_Type','Id']]
X_train = train[feature]
X_test = test[feature]
preds = pd.DataFrame()


# In[ ]:


m1 = ensemble.AdaBoostClassifier(ensemble.ExtraTreesClassifier(n_estimators=500), n_estimators=250, learning_rate=0.01, algorithm='SAMME')  
m1.fit(X_train, train['Cover_Type']) 
preds["Model1"] = m1.predict(X_test)


# In[ ]:


m2 = ensemble.ExtraTreesClassifier(n_estimators=550)  
m2.fit(X_train, train['Cover_Type'])
preds["Model2"] = m2.predict(X_test)


# In[ ]:


m3 = XGBClassifier(max_depth=20, n_estimators=1000)  
m3.fit(X_train, train['Cover_Type'])
preds["Model3"] = m3.predict(X_test)


# In[ ]:


m4 = LGBMClassifier(n_estimators=2000, max_depth=15)
m4.fit(X_train, train['Cover_Type'])
preds["Model4"] = m4.predict(X_test)


# In[ ]:


m5 = ensemble.AdaBoostClassifier(ensemble.GradientBoostingClassifier(n_estimators=1000, max_depth=10), n_estimators=1000, learning_rate=0.01, algorithm="SAMME")
m5.fit(X_train, train['Cover_Type'])
preds["Model5"] = m5.predict(X_test)


# In[ ]:


preds["Model7"] = m1.predict(X_test)


# In[ ]:


m6 = SGDClassifier(loss='hinge')
m6.fit(X_train, train['Cover_Type'])
preds["Model6"] = m6.predict(X_test)


# In[ ]:


preds.head()


# In[ ]:


pred = preds.mode(axis=1)
pred


# In[ ]:


sub = pd.DataFrame({"Id": test['Id'],"Cover_Type": pred[0].astype('int').values})
sub.to_csv("sub.csv", index=False)

