import numpy as np
import pandas as pd
import numpy as np
import re as re

train_set = pd.read_csv('../input/train.csv')
test_set = pd.read_csv('../input/test.csv')
full_data = [train_set, test_set]

train_set.info()
for dataset in full_data:
    dataset['Family_Size'] = dataset['SibSp'] + dataset['Parch'] + 1
print(train_set[['Family_Size', 'Survived']].groupby(['Family_Size'], as_index=False).mean())
for dataset in full_data:
    dataset['isAlone'] = 0
    dataset.loc[dataset['Family_Size'] == 1, 'isAlone'] = 1
print(train_set[['isAlone', 'Survived']].groupby(['isAlone'], as_index=False).mean())
for dataset in full_data:
    dataset['Embarked'].fillna('S')
print(train_set[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())
train_set['Fare'].fillna(dataset['Fare'].median())
train_set['Cat_Fare'] = pd.qcut(dataset['Fare'], 4)
print(train_set[['Cat_Fare', 'Survived']].groupby(['Cat_Fare'], as_index=False).mean())
for dataset in full_data:
    mean = dataset['Age'].mean()
    std = dataset['Age'].std()
    null_count = dataset['Age'].isnull().sum()

    age_null_list = np.random.randint(mean - std, mean + std, size=null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_list
    dataset['Age'] = dataset['Age'].astype(int)

train_set['CatAge'] = pd.qcut(train_set['Age'], 5)
print(train_set[['CatAge', 'Survived']].groupby(['CatAge'], as_index=False).mean())


def get_title(name):
    search_t = re.search(' ([A-Za-z]+)\.', name)
    if (search_t):
        return search_t.group(1)
    return ""


for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
print(pd.crosstab(train_set['Title'], train_set['Sex']))
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print(train_set[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).fillna(0).astype(int)

    # Mapping Fare
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].fillna(0).astype(int)

    # Mapping Age
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4

# Feature Selection
drop_elements = ['Name', 'SibSp', 'Ticket', 'Cabin', 'Parch', 'Family_Size']
train_set = train_set.drop(drop_elements, axis=1)
train_set = train_set.drop(['PassengerId'], axis=1)
train_set = train_set.drop(['CatAge', 'Cat_Fare'], axis=1)

test_set = test_set.drop(drop_elements, axis=1)

print(train_set.head(10))
print(test_set.head(10))

train = train_set.values
test = test_set.drop(['PassengerId'], axis=1).values
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression


from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

classifiers = [
    KNeighborsClassifier(3),
    SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression()]
log_cols = ["Classifier", "Accuracy"]
log = pd.DataFrame(columns=log_cols)

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

X = train[0::, 1::]
y = train[0::, 0]

acc_dict = {}

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    for clf in classifiers:
        name = clf.__class__.__name__
        clf.fit(X_train, y_train)
        train_predictions = clf.predict(X_test)
        acc = accuracy_score(y_test, train_predictions)
        if name in acc_dict:
            acc_dict[name] += acc
        else:
            acc_dict[name] = acc

for clf in acc_dict:
    acc_dict[clf] = acc_dict[clf] / 10.0
    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
    log = log.append(log_entry)

plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')

sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
#candidate_classifier = SVC(probability=True)
candidate_classifier = XGBClassifier()
candidate_classifier.fit(train[0::, 1::], train[0::, 0])
y_result = candidate_classifier.predict(test)
submission = pd.DataFrame({
    "PassengerId": test_set["PassengerId"],
    "Survived": y_result
})
submission.to_csv('titanic.csv', index=False)