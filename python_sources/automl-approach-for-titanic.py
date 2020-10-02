import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier

#read training set into a pandas dataframe
train = pd.read_csv('../input/train.csv')

#get median age & fare to impute missing values
median_age = train['Age'].median()
median_fare = train['Fare'].median()

#create a function to do prepare data for modelling
#following features are adopted from many other kaggle notebooks
def prep_data(df):
    to_be_dropped = ['Name', 'Cabin', 'Ticket']
    
    #handle missing values
    #fill missing values of embarked with most repeated value
    df['Embarked'] = df['Embarked'].fillna('S')
    #fill missing values of age with mean
    df['Age'] = df['Age'].fillna(median_age)
    #fill missing values for age and create bands
    df['Fare'] = df['Fare'].fillna(median_fare)
    
    #create bands from companion size
    df['Companions'] = df['Parch'] + df['SibSp']

    #add parch, sibsp to the list of columns to be dropped
    to_be_dropped.extend(['Parch', 'SibSp'])
    
    #create bands for age
    df.loc[ df['Age'] <= 16, 'Age'] = 0
    df.loc[ (df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
    df.loc[ (df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
    df.loc[ (df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
    df.loc[ df['Age'] > 64, 'Age'] = 4
    df['Age'] = df['Age'].astype(int)
    
    #create bands for fare
    df.loc[ df['Fare'] <= 7.91, 'Fare'] = 0
    df.loc[ (df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
    df.loc[ (df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare'] = 2
    df.loc[ df['Fare'] > 31, 'Fare'] = 3
    df['Fare'] = df['Fare'].astype(int)
    
    #find titles within name field
    df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=False)
    #assign a value for missing titles
    df['Title'] = df['Title'].fillna('NoTitle')
    #Unify titles
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
                       
    #mapping values
    df['Embarked'] = df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    df['Sex'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    df['Title'] = df['Title'].map({'NoTitle': 0, 'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}).astype(int)
    df['CabinCode'] = df['Cabin'].astype(str).str[0]
    df['CabinCode'] = df['CabinCode'].map({'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E':1, 'F':1, 'G': 1, 'T': 0, 'n' : 0}).astype(int)
    
    df = df.set_index('PassengerId')
    df = df.drop(to_be_dropped, axis = 1)
             
    return df

train = prep_data(train)

#split dataframe into X & Y
y = train['Survived']
X = train.drop('Survived', axis = 1)

#create traning & test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=7)

tpot_config = {

    # Classifiers
    'sklearn.ensemble.ExtraTreesClassifier': {
        'n_estimators': [100],
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },

    'sklearn.ensemble.RandomForestClassifier': {
        'n_estimators': [100],
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf':  range(1, 21),
        'bootstrap': [True, False]
    },

    'sklearn.svm.LinearSVC': {
        'penalty': ["l1", "l2"],
        'loss': ["hinge", "squared_hinge"],
        'dual': [True, False],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.]
    },

    'xgboost.XGBClassifier': {
        'n_estimators': [100],
        'max_depth': range(1, 11),
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'subsample': np.arange(0.05, 1.01, 0.05),
        'min_child_weight': range(1, 21),
        'nthread': [1]
    },

    # Selectors
    'sklearn.feature_selection.SelectFwe': {
        'alpha': np.arange(0, 0.05, 0.001),
        'score_func': {
            'sklearn.feature_selection.f_classif': None
        }
    },

    'sklearn.feature_selection.SelectPercentile': {
        'percentile': range(1, 100),
        'score_func': {
            'sklearn.feature_selection.f_classif': None
        }
    },

    'sklearn.feature_selection.VarianceThreshold': {
        'threshold': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    },

    'sklearn.feature_selection.RFE': {
        'step': np.arange(0.05, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': [100],
                'criterion': ['gini', 'entropy'],
                'max_features': np.arange(0.05, 1.01, 0.05)
            }
        }
    },

    'sklearn.feature_selection.SelectFromModel': {
        'threshold': np.arange(0, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': [100],
                'criterion': ['gini', 'entropy'],
                'max_features': np.arange(0.05, 1.01, 0.05)
            }
        }
    }

}

#create a TPOT Classifier
tpot =  TPOTClassifier(verbosity = 2, n_jobs = -1, generations = 9999, offspring_size = 200, max_time_mins = 9999, config_dict = tpot_config, early_stop = 25)

#fit training data
tpot.fit(X, y)

#print (tpot)
#print (tpot.score(X_test, y_test))
    
#export 
tpot.export('tpot_export.py')

#read test data & prepare it to fit 
test = pd.read_csv('../input/test.csv')
test = prep_data(test)

#make predictions
predictions = tpot.predict(test)

#create a submission df from predictions
submission = pd.DataFrame({
        "PassengerId": test.index,
        "Survived": predictions
    })

#output submissions df
submission.to_csv('submission.csv', index=False)