import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

df_train = pd.DataFrame(train)
df_test = pd.DataFrame(test)

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
#print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())

#Any files you save will be available in the output tab below
train.to_csv('copy_of_the_training_data.csv', index=False)

# Feature Engineering
# Data cleanup
median_age = df_train.Age.median()
df_train.loc[df_train.Age.isnull(), 'Age'] = median_age

df_train['Sex'] = pd.factorize(df_train.loc[:,'Sex'])[0]
df_train['Embarked'] = pd.factorize(df_train['Embarked'])[0]
if len(df_train.Embarked[df_train.Embarked.isnull()]) > 0:
    df_train.Embarked.fillna(df_train.Embarked.mode())
Pclass = pd.factorize(df_train['Pclass'])[0]


features = df_train.iloc[:, [2,4,5,6,11]]
print(features.head())

clf = RandomForestClassifier(n_jobs=2)
y, _ = pd.factorize(df_train['Survived'])


clf.fit(features, y)

median_age = df_test.Age.median()
df_test.loc[df_test.Age.isnull(), 'Age'] = median_age
df_test['Sex'] = pd.factorize(df_test['Sex'])[0]
df_test['Pclass'] = pd.factorize(df_test['Pclass'])[0]
df_test['Embarked'] = pd.factorize(df_test['Embarked'])[0]
if len(df_test.Embarked[df_test.Embarked.isnull()]) > 0:
    df_test.Embarked.fillna(df_test.Embarked.mode())


test_features = df_test.iloc[:,[1,3,4,5,10]]
preds = clf.predict(test_features)
print(preds)
#pd.crosstab(df_test['Survived'], preds, rownames=['actual'], colnames=['preds'])