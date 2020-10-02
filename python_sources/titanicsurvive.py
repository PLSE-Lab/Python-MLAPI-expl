from collections import Counter
import warnings

import pandas as pd
from pylab import *
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier

warnings.simplefilter("ignore")

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

ethnicity = pd.read_csv("../input/ethnicity/ethnicity.csv")
print(ethnicity)

#Print to standard output, and see the results in the "log" section below after running your script
#print("\n\nTop of the training data:")
#print(train.head())

#print("\n\nSummary statistics of training data")
#print(train.describe())

columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Name']

train_target = train['Survived']
train_sample = train[columns]

le = LabelEncoder()

def regular_data(df_data):
    df_data['Fare'] = df_data['Fare'].fillna(test['Fare'].mean()).astype(float)
    df_data['Age'] = df_data['Age'].fillna(test['Age'].mean()).astype(int)
    le.fit(df_data.Sex.tolist())
    df_data.Sex = le.transform(df_data.Sex.tolist())
    df_data['Embarked'] = df_data['Embarked'].fillna('S')
    le.fit(df_data.Embarked.tolist())
    df_data.Embarked = le.transform(df_data.Embarked.tolist())
    return df_data

train_sample = regular_data(train_sample)
test = regular_data(test)
train_name = train['Name']
test_name = train['Name']

title_set = pd.Series(Counter(train_name.map(lambda x: x.split(',')[1].split('.')[0])))
name_titles = pd.Series(train_name.map(lambda x: x.split(',')[1].split('.')[0]))
test_name_titles = pd.Series(test_name.map(lambda x: x.split(',')[1].split('.')[0]))
#diff_set = name_titles.index.difference(test_name_titles.index)
name_col = pd.Series(np.arange(len(title_set)), index=title_set.index)
name_titles = name_titles.map(lambda x: name_col[x])
test_name_titles = test_name_titles.map(lambda x: name_col[x])
train_sample['Name'] = name_titles
test['Name'] = test_name_titles

#alg = RandomForestClassifier(random_state=1, n_estimators=15, min_samples_leaf=11)

# kf = cross_validation.KFold(train_sample.shape[0], n_folds=5, random_state=1)
# scores = cross_validation.cross_val_score(alg, train_sample, train_target, cv=kf)
# print(scores.mean())
clf = XGBClassifier()

clf.fit(train_sample, train_target)
predict_results = clf.predict(test[columns])
submission = pd.DataFrame({"PassengerId": test['PassengerId'],"Survived": predict_results})
submission.to_csv('submission.csv', index=False)

#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)