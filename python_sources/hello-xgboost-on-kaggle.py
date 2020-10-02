# Absolutely simple Kernel that uses XGBoost and gets 0.77033 on the entry

import pandas as pd
import xgboost as xgb

# Load the data
train_df = pd.read_csv('../input/train.csv', header=0)
test_df = pd.read_csv('../input/test.csv', header=0)

# Select features
features_to_use = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
categorical_features = ['Sex', 'Cabin', 'Embarked']

# Handle both sets as one
big_X = train_df.loc[:,features_to_use].append(test_df.loc[:,features_to_use])

# Some munging
big_X.loc[:,'Cabin'] = big_X['Cabin'].fillna('X').str[0]
big_X = pd.get_dummies(big_X, columns=categorical_features)

# Prepare the inputs for the model
train_X = big_X[0:train_df.shape[0]].as_matrix()
test_X = big_X[train_df.shape[0]::].as_matrix()
train_y = train_df['Survived']

# Train and predict
gbm = xgb.XGBClassifier(max_depth=7, n_estimators=500, learning_rate=0.01).fit(train_X, train_y)
predictions = gbm.predict(test_X)

# Write results
submission = pd.DataFrame({ 'PassengerId': test_df['PassengerId'],
                            'Survived': predictions })
submission.to_csv("submission.csv", index=False)