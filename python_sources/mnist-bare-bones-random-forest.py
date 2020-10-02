import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load data

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.head()
train.tail()
train.describe()

X_train = train.values[:, 1:].astype(float)
y_train = train.values[:, 0]

rf = RandomForestClassifier(200)
rf.fit(X_train, y_train)

rf_predictions = rf.predict(test)
submission= pd.DataFrame(rf_predictions)
submission.columns = ["Label"]
submission.insert(0, 'ImageID', range(1, 1 + len(submission)))
submission.to_csv("submission.csv", index=False)