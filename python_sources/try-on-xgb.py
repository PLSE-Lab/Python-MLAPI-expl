import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import xgboost as xgb

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

train = pd.read_csv('../input/train.csv',header=0)
test = pd.read_csv('../input/test.csv',header=0)

train_x = train.values[:,1:].astype(float)
train_y = train.values[:,0]

test_data = test.values[:,:]
gbm = xgb.XGBClassifier(max_depth=3,n_estimators=150,learning_rate=0.05).fit(train_x,train_y)
predictions = gbm.predict(test_data)

# Output
results = pd.DataFrame({'ImageId':range(1,28001)})
results['Label'] = predictions
results.to_csv('xgb.csv',index=False)