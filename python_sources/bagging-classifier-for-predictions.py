import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression

traindata=pd.read_csv('../input/train.csv')
testdata=pd.read_csv('../input/test.csv')

import os
print(os.listdir("../input"))

traindata.drop("Id", axis = 1, inplace = True)
test_id=testdata['Id']

traindata['slope_hyd'] = (traindata['Horizontal_Distance_To_Hydrology']**2+traindata['Vertical_Distance_To_Hydrology']**2)**0.5
testdata['slope_hyd'] = (testdata['Horizontal_Distance_To_Hydrology']**2+testdata['Vertical_Distance_To_Hydrology']**2)**0.5

feature = [col for col in traindata.columns if col not in ['Cover_Type']]
X=traindata[feature]
y=traindata['Cover_Type']

test_feature = [col for col in testdata.columns if col not in ['Cover_Type','Id']]
Xtest=testdata[test_feature]


model=BaggingClassifier(n_estimators=1000, bootstrap_features=True)

model.fit(X,np.ravel(y))
preds=model.predict(Xtest)

data_submission=pd.DataFrame()
data_submission['Id']=test_id
data_submission["Cover_Type"]=preds
data_submission.to_csv("submission_anj.csv",index=False)


