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

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#reading input data
df = pd.read_csv('../input/webclubrecruitment2019/TRAIN_DATA.csv')
df_test = pd.read_csv('../input/webclubrecruitment2019/TEST_DATA.csv')

test_index=df_test['Unnamed: 0'] 

#spliting data into test and train
X = df.loc[:, 'V1':'V16']
y = df.loc[:, 'Class']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=42)

#one hot encoding
X_train['V2_1'] = pd.get_dummies(X_train['V2'])[0]
X_train['V2_2'] = pd.get_dummies(X_train['V2'])[1]
X_train['V2_3'] = pd.get_dummies(X_train['V2'])[2]
X_train['V2_4'] = pd.get_dummies(X_train['V2'])[3]
X_train['V2_5'] = pd.get_dummies(X_train['V2'])[4]
X_train['V2_6'] = pd.get_dummies(X_train['V2'])[5]
X_train['V2_7'] = pd.get_dummies(X_train['V2'])[6]
X_train['V2_8'] = pd.get_dummies(X_train['V2'])[7]  
X_train['V2_9'] = pd.get_dummies(X_train['V2'])[8]
X_train['V2_10'] = pd.get_dummies(X_train['V2'])[9]
X_train['V2_11'] = pd.get_dummies(X_train['V2'])[10]
X_train['V2_12'] = pd.get_dummies(X_train['V2'])[11]

X_train['V3_1'] = pd.get_dummies(X_train['V3'])[0]
X_train['V3_2'] = pd.get_dummies(X_train['V3'])[1]
X_train['V3_3'] = pd.get_dummies(X_train['V3'])[2]

X_train['V4_1'] = pd.get_dummies(X_train['V4'])[0]
X_train['V4_2'] = pd.get_dummies(X_train['V4'])[1]
X_train['V4_3'] = pd.get_dummies(X_train['V4'])[2]
X_train['V4_4'] = pd.get_dummies(X_train['V4'])[3]

X_train['V5_1'] = pd.get_dummies(X_train['V5'])[0]
X_train['V5_2'] = pd.get_dummies(X_train['V5'])[1]

X_train['V7_1'] = pd.get_dummies(X_train['V7'])[0]
X_train['V7_2'] = pd.get_dummies(X_train['V7'])[1]

X_train['V16_1'] = pd.get_dummies(X_train['V16'])[0]
X_train['V16_2'] = pd.get_dummies(X_train['V16'])[1]
X_train['V16_3'] = pd.get_dummies(X_train['V16'])[2]
X_train['V16_4'] = pd.get_dummies(X_train['V16'])[3]

#dropping the columns
X_train.drop(columns=['V2','V3','V4','V5','V7','V16'],axis = 1, inplace = True)

#one hot encoding
df_test['V2_1'] = pd.get_dummies(df_test['V2'])[0]
df_test['V2_2'] = pd.get_dummies(df_test['V2'])[1]
df_test['V2_3'] = pd.get_dummies(df_test['V2'])[2]
df_test['V2_4'] = pd.get_dummies(df_test['V2'])[3]
df_test['V2_5'] = pd.get_dummies(df_test['V2'])[4]
df_test['V2_6'] = pd.get_dummies(df_test['V2'])[5]
df_test['V2_7'] = pd.get_dummies(df_test['V2'])[6]
df_test['V2_8'] = pd.get_dummies(df_test['V2'])[7]  
df_test['V2_9'] = pd.get_dummies(df_test['V2'])[8]
df_test['V2_10'] = pd.get_dummies(df_test['V2'])[9]
df_test['V2_11'] = pd.get_dummies(df_test['V2'])[10]
df_test['V2_12'] = pd.get_dummies(df_test['V2'])[11]  

df_test['V3_1'] = pd.get_dummies(df_test['V3'])[0]
df_test['V3_2'] = pd.get_dummies(df_test['V3'])[1]
df_test['V3_3'] = pd.get_dummies(df_test['V3'])[2]

df_test['V4_1'] = pd.get_dummies(df_test['V4'])[0]
df_test['V4_2'] = pd.get_dummies(df_test['V4'])[1]
df_test['V4_3'] = pd.get_dummies(df_test['V4'])[2]
df_test['V4_4'] = pd.get_dummies(df_test['V4'])[3] 
                                          
df_test['V5_1'] = pd.get_dummies(df_test['V5'])[0]
df_test['V5_2'] = pd.get_dummies(df_test['V5'])[1]
    
df_test['V7_1'] = pd.get_dummies(df_test['V7'])[0]
df_test['V7_2'] = pd.get_dummies(df_test['V7'])[1]

df_test['V16_1'] = pd.get_dummies(df_test['V16'])[0]
df_test['V16_2'] = pd.get_dummies(df_test['V16'])[1]
df_test['V16_3'] = pd.get_dummies(df_test['V16'])[2]
df_test['V16_4'] = pd.get_dummies(df_test['V16'])[3]

#dropping the columns
df_test.drop(columns=['V2', 'V3', 'V4','V5','V7','V16'], axis = 1, inplace = True)
df_test = df_test.loc[:, 'V1':'V16_4']

#standerdization of the data
X_train['V6'] = StandardScaler().fit_transform(X_train['V6'].values.reshape(-1, 1))
df_test['V6'] = StandardScaler().fit_transform(df_test['V6'].values.reshape(-1, 1))

#creating and training the model
model = XGBClassifier(learning_rate=0.12,n_estimators=300,max_depth=2,min_child_weight=1,colsample_bytree=1,subsample=0.8)
model.fit(X_train,y_train)

#predicitng results
pred=model.predict_proba(df_test)

#output handling
result=pd.DataFrame()
result['Id'] = test_index
result['PredictedValue'] = pd.DataFrame(pred[:,1])
result.head()

result.to_csv('output.csv',index=False)
print(result.head(10))

