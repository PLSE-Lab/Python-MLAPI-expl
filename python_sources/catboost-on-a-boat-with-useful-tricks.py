import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import Pool, CatBoostClassifier, cv

rnd_state = 42

# read data
df = pd.read_csv('../input/train.csv', index_col='PassengerId')

df.fillna(-999, inplace=True)

X = df.drop('Survived', axis=1) 
y = df.Survived

# make train val split to try out-of-the-box
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=rnd_state)

categorical_features_indices = np.where(X.dtypes != np.float)[0]
clf = CatBoostClassifier(random_seed=rnd_state, custom_metric='Accuracy')
clf.fit(X_train, y_train, cat_features=categorical_features_indices)
clf.score(X_val, y_val)

# Submission 1: catboost submission with all training data and early stopping on Accuracy
test_df = pd.read_csv('../input/test.csv', index_col='PassengerId')
test_df.fillna(-999, inplace=True)
clf_od = CatBoostClassifier(random_seed=rnd_state, od_type='Iter', od_wait=20, eval_metric='Accuracy')
clf_od.fit(X, y, cat_features=categorical_features_indices)

## cross validation score
cv_data = cv(Pool(X, label=y, cat_features=categorical_features_indices), clf_od.get_params())
print(f"Best validation accuracy score: {np.max(cv_data['Accuracy_test_avg'])}±{cv_data['Accuracy_test_stddev'][np.argmax(cv_data['Accuracy_test_avg'])]} on step {np.argmax(cv_data['Accuracy_test_avg'])}")

submission = pd.DataFrame()
submission['PassengerId'] = test_df.index
submission['Survived'] = clf_od.predict(test_df).astype('int')
submission.to_csv('submission_early_stopping.csv', index=False)
