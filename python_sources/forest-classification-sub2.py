
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

SEED = 0

# Load data...
data_path = '../input/learn-together/train.csv' 
test_path = '../input/learn-together/test.csv' 

data = pd.read_csv(data_path)
test_data = pd.read_csv(test_path)


X = data.drop(['Cover_Type'], axis = 1)
y = data['Cover_Type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED)


numeric_transformer = Pipeline(steps=[( 'imputer', SimpleImputer(strategy = 'mean') ),
                                      ( 'scaler', StandardScaler() )])



# Create the pipeline for the Random Forest.
rf_pipe = Pipeline(
    steps = [
        ( 'num', numeric_transformer ),
        ( 'classifier', RandomForestClassifier(
            n_estimators = 3003,
            min_samples_split = 2,
            min_samples_leaf = 1,
            max_features = 'auto',
            max_depth = 50,
            bootstrap = False,
            class_weight=None,
            criterion='gini',
            max_leaf_nodes=None,
            oob_score=False,
            verbose=0, 
            warm_start=False
        ) )
    ]
)

# NB: Optimum parameters for the pipeline above were got from Random Search and Grid Search.

_ = rf_pipe.fit(X_train, y_train)

y_pred_rf = rf_pipe.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print( f'RF Accuracy = {accuracy_rf}' )


_ = rf_pipe.fit(X, y)
test_data['Cover_Type'] = rf_pipe.predict(test_data)

comp1_submission = test_data[['Id', 'Cover_Type']]
comp1_submission.to_csv('comp1_final.csv', index=False)
