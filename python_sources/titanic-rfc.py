import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from numpy import median

#########
# Datenimport & Merging
data_train = pd.read_csv('../input/titanic/train.csv')
data_test = pd.read_csv('../input/titanic/test.csv')

#########
# Data Wrangling methods
def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df
def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0]) # Nur erstes Zeichen behalten; apply führt Funktion in ganzer Achse/Spalte aus
    return df
def simplify_embarked(df):
    df.Embarked = df.Embarked.fillna('N')
    return df
def simplify_fare(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 513)
#     bins = (-1, 0, 8, 15, 31, 1000) # keine Veränderung, da Fare.max = 512.4
    group_names = ["Unknown", "quart1", "quart2", "quart3", "quart4"]
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df
def format_name(df):
    df['LName'] = df.Name.apply(lambda x: x.split(' ')[0])
    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
    return df
def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)
#     return df.drop(['Ticket', 'Name'], axis=1) # Bedeutung von Embarked prüfen
def transform_feature(df):
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fare(df)
    df = simplify_embarked(df)
    df = format_name(df)
    df = drop_features(df)
    return df
from sklearn import preprocessing
def encode_features(df_train, df_test):
    '''Wandelt Features in Zahlen um, die sonst nur in Strings vorliegen. Ermöglichst Verwendung div. Algorithmen.'''
    features = ['Fare', 'Cabin', 'Age', 'Sex', 'LName', 'NamePrefix']
#     features = ['Fare', 'Cabin', 'Embarked', 'Age', 'Sex', 'LName', 'NamePrefix'] # Bedeutung von Embarked prüfen
    df_combined = pd.concat([df_train[features], df_test[features]])
    for feature in features:
        le = preprocessing.LabelEncoder() # OneHotEncoder bessere Wahl bei kategorischen und nicht skalaren Größen?
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test

#########
# Data Exploration & Data Wrangling execution

data_train = transform_feature(data_train)
data_test = transform_feature(data_test)

# sns.barplot(x="Embarked", y="Survived", hue="Sex", data=data_train)
# sns.barplot(x="SibSp", y="Survived", hue="Sex", data=data_train)
# sns.barplot(x="Pclass", y="Survived", hue="Sex", data=data_train)
# sns.barplot(x="Age", y="Survived", hue="Sex", data=data_train)
# sns.barplot(x="Fare", y="Survived", hue="Sex", data=data_train)
# sns.barplot(x="Cabin", y="Survived", hue="Sex", data=data_train)
# sns.barplot(x="Pclass", y="Survived", hue="Sex", data=data_train)

data_train, data_test = encode_features(data_train, data_test)

#########
# Machine Learning
from sklearn.model_selection import train_test_split
X_all = data_train.drop(['Survived', 'PassengerId'], axis=1) # axis=1 notwendig, da default =0 -> Zeile statt Spalte gesucht
X_test_all = data_test
Y_all = data_train['Survived']
num_test = 0.20 # 80% zum Trainieren nutzen, Testen gegen 20 %
X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y_all, test_size=num_test, random_state=69) # 23 im Tutorial

## Algorithmus
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

clf = RandomForestClassifier()
# clf = # Alternative classifier testen

parameters = {'n_estimators': [4, 6, 9], 'max_features': ['log2', 'sqrt', 'auto'],
             'criterion': ['entropy', 'gini'], 'max_depth': [2, 3, 5, 10],
             'min_samples_split': [2, 3, 5], 'min_samples_leaf': [1, 5, 8]
             } # Parameter variieren

# Typ des scoring Mechanismus
acc_scorer = make_scorer(accuracy_score) 
# Grid Search
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer, cv=5) # cv in zukünftigen Releases default 5 statt 3
grid_obj = grid_obj.fit(X_train, Y_train) # mit Features in X Vorhersage des Features Y trainieren
clf = grid_obj.best_estimator_ # Beste Parameterkombination verwenden
clf.fit(X_train, Y_train) # Besten Algorithmus auf Daten anwenden
predictions = clf.predict(X_test)
print(accuracy_score(Y_test, predictions))

#########
# Algorithmen vergleichen mit KFold
#from sklearn.model_selection import KFold # Tutorial hat veraltetes Untermodul cross_validation

#def run_kfold(clf):
##    kf = KFold(891, n_folds=10) # 891 wegen Datensatzlänge? - veraltete Parameter!
#    kf = KFold(n_splits=10)
#    outcomes = []
#    fold = 0
#    for train_index, test_index in kf.split(X_all):
#        fold += 1
#        X_train, X_test = X_all.values[train_index], X_all.values[test_index]
#        Y_train, Y_test = Y_all.values[train_index], Y_all.values[test_index]
#        clf.fit(X_train, Y_train)
#        predictions = clf.predict(X_test)
#        accuracy = accuracy_score(Y_test, predictions)
#        outcomes.append(accuracy)
#        print('Fold {0} accuracy: {1}'.format(fold, accuracy))
#    mean_outcome = np.mean(outcomes)
#    std_outcome = np.std(outcomes)
#    print('Mean Accuracy: {0}'.format(mean_outcome))
#    print('Standard Deviation: {0}'.format(std_outcome))
#    return
#
#run_kfold(clf)

X_test_submit = X_test_all.drop('PassengerId', axis=1)
Sub_PassengerIds = X_test_all[['PassengerId']]
submission = clf.predict(X_test_submit)

sub_df = pd.DataFrame(data=Sub_PassengerIds, columns=['PassengerId'])
sub_df['Survived'] = submission
sub_df.to_csv('titanic_submit_RFC_DATE.csv',index=False)