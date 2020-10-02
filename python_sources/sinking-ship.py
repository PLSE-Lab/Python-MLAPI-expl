# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------

# Links

# Titanic: Machine Learning from Disaster
# https://www.kaggle.com/c/titanic

# Kaggle Python Tutorial on Machine Learning
# https://www.datacamp.com/community/open-courses/kaggle-python-tutorial-on-machine-learning#gs.7vaQ7NY

# Auto-scaling scikit-learn with Apache Spark
# https://databricks.com/blog/2016/02/08/auto-scaling-scikit-learn-with-apache-spark.html



# ----------------------------------------------------------------------------
# Import libraries

import numpy as np
import pandas as pd

#import subprocess as sp

import sklearn as skl
import sklearn.ensemble as skle
import sklearn.model_selection as sklms


# ----------------------------------------------------------------------------
# Load data

def LoadData():

    # print(sp.check_output(['ls', '../input']).decode('utf8'))

    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')

    print('\nThe source data has been loaded')
    print('Train DS shape: ', str(train.shape))
    print('Test DS shape: ' + str(test.shape))

    return (train, test)


# ----------------------------------------------------------------------------
# Show data
"""
def ShowData(data):

    print(data.head())

    return


# ----------------------------------------------------------------------------
# Review data

def ReviewData(data):

    d = data.describe()

    print('\nColumns:')
    print("All: " + str(data.columns.get_values()))
    print("Numeric: " + str(d.columns.get_values()))
    print("Non-null: ")
    print(data.count())

    print('\nDescribe:')
    print(d)

    return


# ----------------------------------------------------------------------------
# Print column values

def PrintColumnValues(data, columns):

    data = data[columns]

    for c in data.columns:
        print('\nColumn ' + c)
        print(data[c].value_counts())

    return


# ----------------------------------------------------------------------------
# Preprocess data

def PreprocessData(data):

    data = data.copy()

    # Sex
    data['Sex_'] = 0
    data.set_value(data['Sex'] == 'male', 'Sex_', 1)
    data.set_value(data['Sex'] == 'female', 'Sex_', -1)

    # Embarked
    data['Embarked_S'] = 0
    data['Embarked_C'] = 0
    data['Embarked_Q'] = 0
    data.set_value(data['Embarked'] == 'S', 'Embarked_S', 1)
    data.set_value(data['Embarked'] == 'C', 'Embarked_C', 1)
    data.set_value(data['Embarked'] == 'Q', 'Embarked_Q', 1)

    # Age
    data['Age_'] = data['Age']
    data.set_value(data['Age_'].isnull(), 'Age_', 28.0) 
    # = train['Age'].median()

    # Family
    data['Family_'] = data['SibSp'] + data['Parch']
    
    # Fare
    data['Fare_'] = data['Fare']
    data.set_value(data['Fare_'].isnull(), 'Fare_', 14.4542) 
    # = train['Fare'].median()
    

    # Drop preprocessed columns
    data.drop(['Sex', 'Embarked', 'Age', 'SibSp', 'Parch', 'Fare'], 
              axis=1, inplace=True)

    # Drop unused columns
    data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    return data


# ----------------------------------------------------------------------------
# Split train data

def SplitTrainData(data, train_fraction, val_fraction):

    # Normalize fractions
    fraction_sum = train_fraction + val_fraction
    train_fraction /= fraction_sum
    val_fraction /= fraction_sum

    # Reorder data
    n = len(data)
    ix = np.random.permutation(n)
    data = data.iloc[ix]

    # Split data
    cut_ix = int(n * train_fraction)
    train_data = data[:cut_ix]
    val_data = data[cut_ix:]

    return (train_data, val_data)


# ----------------------------------------------------------------------------
# Exctract features

def ExtractFeatures(data):

    data_ix = data['PassengerId'].values

    if 'Survived' in data.columns:
        data_X = data.drop(['Survived', 'PassengerId'], axis=1).values
    else:
        data_X = data.drop(['PassengerId'], axis=1).values

    return (data_ix, data_X)


# ----------------------------------------------------------------------------
# Exctract features and labels

def ExtractFeaturesAndLabels(data):

    (data_ix, data_X) = ExtractFeatures(data)
    data_y = data['Survived'].values

    return (data_ix, data_X, data_y)


# ----------------------------------------------------------------------------
# Create a classifier

def CreateClassifierByType(classifier_type):

    classifier = None

    if classifier_type == 'Decision Tree':
        classifier = skl.tree.DecisionTreeClassifier(max_depth=10,
                                                     min_samples_split=5)
    elif classifier_type == 'Random Forest':
        classifier = skle.RandomForestClassifier(max_depth=10,
                                                 min_samples_split=5,
                                                 n_estimators=100)

    elif classifier_type == 'Random Forest Grid Search':
        param_grid = {
                      'max_depth': [3, 5, 10, None],
                      'max_features': [1, 3, 5, None],
                      'min_samples_split': [2, 3, 5, 10],
                      'min_samples_leaf': [1, 3, 5, 10],
                      'bootstrap': [True, False],
                      'criterion': ['gini', 'entropy'],
                      'n_estimators': [10, 20, 40, 80]
                      }
                      
#        classifier = skl.grid_search.GridSearchCV(skle.RandomForestClassifier(),
#                                                  param_grid=param_grid)
        classifier = sklms.GridSearchCV(skle.RandomForestClassifier(),
                                        param_grid=param_grid)

    return classifier


# ----------------------------------------------------------------------------
# Fit and validate

def FitAndValidate(pretrain_X, pretrain_y,
                   val_X, val_y, classifier_type):

    classifier = CreateClassifierByType(classifier_type)
    classifier.fit(pretrain_X, pretrain_y)

    val_score = classifier.score(val_X, val_y)

    return val_score


# ----------------------------------------------------------------------------
# Validate classifiers

def ValidateClassifiers(pretrain_X, pretrain_y, val_X, val_y):

#    classifier_types = ['Decision Tree', 'Random Forest',
#                        'Random Forest Grid Search']
    classifier_types = ['Decision Tree', 'Random Forest']
    score_best = -1
    ct_best = 'None'
    print('\nValidate classifiers')
    for ct in classifier_types:
        val_score = FitAndValidate(pretrain_X, pretrain_y,
                                   val_X, val_y, ct)
        print('Score of ' + ct + ': ' + str(val_score))

        if val_score > score_best:
            score_best = val_score
            ct_best = ct

    return ct_best


# ----------------------------------------------------------------------------
# Fit and predict

def CreateAndFit(train_X, train_y, classifier_type):

    print('\nFinal learning begins')

    classifier = CreateClassifierByType(classifier_type)
    classifier.fit(train_X, train_y)

    print('Final learning ends')
    
    return classifier;



# ----------------------------------------------------------------------------
# Main function

if __name__ == '__main__':

    # 1. Load and show data
    (train, test) = LoadData()
#    ShowData(train)

    # 2. Review data before preprocessing
#    ReviewData(train)
#    PrintColumnValues(train, ['Sex', 'Cabin', 'Embarked'])

    # 3. Preprocess data and review it again
    train_dat = PreprocessData(train)
#    ReviewData(train_dat)
#    PrintColumnValues(train_dat,
#                      ['Sex_', 'Embarked_S', 'Embarked_C', 'Embarked_Q'])

    # 4. Split train data
    (pretrain_dat, val_dat) = SplitTrainData(train_dat, 80, 20)
    print('\nSplit of train dataset')
    print('Pretrain DS shape: ' + str(pretrain_dat.shape))
    print('Validate DS shape: ' + str(val_dat.shape))

    # 5. Extract features and labels for validation
    (pretrain_ix, pretrain_X,
     pretrain_y) = ExtractFeaturesAndLabels(pretrain_dat)
    (val_ix, val_X, val_y) = ExtractFeaturesAndLabels(val_dat)

    # 6.Validate classifiers
    ctype_best = ValidateClassifiers(pretrain_X, pretrain_y, val_X, val_y)
    print('\nThe best classifier is ' + ctype_best)

    # 7. Final learning
    (train_ix, train_X, train_y) = ExtractFeaturesAndLabels(train_dat)
    cl_best = CreateAndFit(train_X, train_y, ctype_best)

    # 8. Final predicting
#    ReviewData(test)
    test_dat = PreprocessData(test)
    ReviewData(test_dat)
    (test_ix, test_X) = ExtractFeatures(test_dat)
    test_predictions = cl_best.predict(test_X)
    
    # 9. Create and save the final prediction file
    prediction_ds = pd.DataFrame(index=test_ix, data=test_predictions,
                                  columns=['Survived'])
    print('\nFinal prediction distribution:')
    print(prediction_ds['Survived'].value_counts())
    prediction_ds.to_csv('MyFirstPenTest.csv', index_label='PassengerId')

"""
# ----------------------------------------------------------------------------

