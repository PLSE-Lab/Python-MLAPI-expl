# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

file_path = '../input/'
train_df = pd.read_json(file_path+'train.json')
test_df = pd.read_json(file_path+'test.json')
fts = ['bathrooms','bedrooms','latitude','longitude','price']
X_train = train_df[fts]
X_test = test_df[fts]

rf = RandomForestClassifier(n_estimators=2000)

rf.fit(X_train, train_df['interest_level'])

Y_pred = rf.predict_proba(X_test)

submission = pd.DataFrame({
        "listing_id": test_df["listing_id"],
        "high": Y_pred[:,0],
        "medium":Y_pred[:,1],
        "low":Y_pred[:,2]
    })
    
columnsTitles=["listing_id","high","medium","low"]
submission=submission.reindex(columns=columnsTitles)
submission.to_csv('submission.csv', index=False)