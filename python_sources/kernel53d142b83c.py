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
train=pd.read_csv('../input/learn-together/train.csv', index_col='Id')
test=pd.read_csv('../input/learn-together/test.csv', index_col='Id')

# combining Hillshade_9, Noon & 3pm into one -Hillshade_mean
train['Hillshade_mean']=train[['Hillshade_3pm','Hillshade_9am','Hillshade_Noon']].mean(axis=1)
test['Hillshade_mean']=test[['Hillshade_3pm','Hillshade_9am','Hillshade_Noon']].mean(axis=1)

# 'Aspect'- creating new variable 'Compass'
def Compass(row):
    if row.Aspect < 22.5:
        return 'N'
    elif row.Aspect < 67.5:
        return 'NE'
    elif row.Aspect < 112.5:
        return 'E'
    elif row.Aspect < 157.5:
        return 'SE'
    elif row.Aspect < 202.5:
        return 'S'
    elif row.Aspect < 247.5:
        return 'SW'
    elif row.Aspect < 292.5:
        return 'W'
    elif row.Aspect < 337.5:
        return 'NW'
    else:
        return 'N'
    
train['Compass']=train.apply(Compass,axis=1)
test['Compass']=test.apply(Compass,axis=1)

#creating dummy variable from 'Compass'
train=pd.get_dummies(data=train, columns=['Compass'])
test=pd.get_dummies(data=test, columns=['Compass'])

#Soil_Type 7 and 15 need to be deleted from train data as they only have 1 unique value vs 2
train.drop(['Soil_Type7','Soil_Type15'],axis=1, inplace =True)

features=['Elevation','Slope', 'Horizontal_Distance_To_Hydrology',
       'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1',
       'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
       'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5',
       'Soil_Type6', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11',
       'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type16',
       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40','Hillshade_mean', 'Compass_E', 'Compass_N', 'Compass_NE',
       'Compass_NW', 'Compass_S', 'Compass_SE', 'Compass_SW', 'Compass_W']

X_train=train[features]
X_test=test[features]
y_train=train['Cover_Type']

from sklearn.ensemble import RandomForestClassifier
rfc_g=RandomForestClassifier(n_estimators=100)
rfc_g.fit(X_train,y_train)
pred_rfc_g=rfc_g.predict(X_test)

submission = pd.DataFrame({'Id': X_test.index, 'Cover_Type': pred_rfc_g})
submission.to_csv('submission.csv', index=False)