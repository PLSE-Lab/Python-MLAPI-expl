import pandas as pd
import numpy as np
import re

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score
from sklearn.svm import SVC
from sklearn.model_selection import KFold



def get_title(name):
    title_pattern = re.search(' ([A-Za-z]+)\.', name)
    if title_pattern:
        return title_pattern.group(1)
    return ''

class SklearnHelper(object):
    def __init__(self, clf, seed=0, x_train=None, x_test=None, y_train=None, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)
        self.ntrain = x_train.shape[0]
        self.ntest = x_test.shape[0]
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.SEED = seed  # for reproducibility
        self.NFOLDS = 5  # set folds for out-of-fold prediction
        self.kf = KFold(n_splits=self.NFOLDS, random_state=self.SEED, shuffle=False)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):
        print(self.clf.fit(x, y).feature_importances_)

    def get_oof(self, clf):
        oof_train = np.zeros((self.ntrain,))
        oof_test = np.zeros((self.ntest,))
        oof_test_skf = np.empty((self.NFOLDS, self.ntest))

        for i, (train_index, test_index) in enumerate(self.kf.split(self.x_train)):
            x_tr = self.x_train[train_index]
            y_tr = self.y_train[train_index]
            x_te = self.x_train[test_index]

            clf.train(x_tr, y_tr)

            oof_train[test_index] = clf.predict(x_te)
            oof_test_skf[i, :] = clf.predict(self.x_test)

        oof_test[:] = oof_test_skf.mean(axis=0)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
        
class TitanicSolution:
    def __init__(self):
        self.train = None
        self.test = None
        self.SEED = 0
        self.load_data()
        self.train, self.test = self.feature_engineering()
        self.feature_selection()

    def load_data(self):
        self.train = pd.read_csv('../input/train.csv')
        self.test = pd.read_csv('../input/test.csv')

    def feature_engineering(self):
        full_data = [self.train, self.test]

        # add some feature into dataset
        self.train['Name_length'] = self.train['Name'].apply(len)
        self.test['Name_length'] = self.test['Name'].apply(len)

        self.train['Has_Cabin'] = self.train['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
        self.test['Has_Cabin'] = self.test['Cabin'].apply(lambda x: 0 if type(x) == float else 1)

        # Create new feature FamilySize as a combination of SibSp and Parch
        for data_set in full_data:
            data_set['FamilySize'] = data_set['SibSp'] + data_set['Parch'] + 1
        # Create new feature IsAlone from FamilySize
        for data_set in full_data:
            data_set['IsAlone'] = 0
            data_set.loc[data_set['FamilySize'] == 1, 'IsAlone'] = 1
        # Remove all NULLS in Embarked column
        for data_set in full_data:
            data_set['Embarked'] = data_set['Embarked'].fillna('S')
        # Remove all Nulls in Fare column and create new feature CategoricalFare for train
        for data_set in full_data:
            data_set['Fare'] = data_set['Fare'].fillna(self.train['Fare'].median())
        self.train['CategoricalFare'] = pd.qcut(self.train['Fare'], 4)

        # Create a New feature CategoricalAge
        for data_set in full_data:
            age_avg = data_set['Age'].mean()
            age_std = data_set['Age'].std()
            age_null_count = data_set['Age'].isnull().sum()
            age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
            data_set['Age'][np.isnan(data_set['Age'])] = age_null_random_list
            data_set['Age'] = data_set['Age'].astype(int)
        self.train['CategoricalAge'] = pd.qcut(self.train['Age'], 5)
        # Create a new feature Title
        for data_set in full_data:
            data_set['Title'] = data_set['Name'].apply(get_title)

        for data_set in full_data:
            data_set['Title'] = data_set['Title'].replace(
                ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
            data_set['Title'] = data_set['Title'].replace('Mlle', 'Miss')
            data_set['Title'] = data_set['Title'].replace('Ms', 'Miss')
            data_set['Title'] = data_set['Title'].replace('Mme', 'Mrs')

        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        for data_set in full_data:
            data_set['Sex'] = data_set['Sex'].map({'female': 0, 'male': 1}).astype(int)
            data_set['Title'] = data_set['Title'].map(title_mapping).astype(int)
            data_set['Title'] = data_set['Title'].fillna(0)
            data_set['Embarked'] = data_set['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
            data_set.loc[data_set['Fare'] <= 7.91, 'Fare'] = 0
            data_set.loc[(data_set['Fare'] > 7.91) & (data_set['Fare'] <= 14.454), 'Fare'] = 1
            data_set.loc[(data_set['Fare'] > 14.454) & (data_set['Fare'] <= 31), 'Fare'] = 2
            data_set.loc[data_set['Fare'] > 31, 'Fare'] = 3
            data_set['Fare'] = data_set['Fare'].astype(int)
            data_set.loc[data_set['Age'] <= 16, 'Age'] = 0
            data_set.loc[(data_set['Age'] > 16) & (data_set['Age'] <= 32), 'Age'] = 1
            data_set.loc[(data_set['Age'] > 32) & (data_set['Age'] <= 48), 'Age'] = 2
            data_set.loc[(data_set['Age'] > 48) & (data_set['Age'] <= 64), 'Age'] = 3
            data_set.loc[data_set['Age'] > 64, 'Age'] = 4
            data_set['Age'] = data_set['Age'].astype(int)
        return full_data[0], full_data[1]

    def feature_selection(self):
        drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
        self.train = self.train.drop(drop_elements, axis=1)
        self.train = self.train.drop(['CategoricalAge', 'CategoricalFare'], axis=1)
        self.test = self.test.drop(drop_elements, axis=1)

    def train_model(self):
        # Random forest
        rf_params = {
            'n_jobs': -1,
            'n_estimators': 500,
            'warm_start': True,
            'max_depth': 6,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'verbose': 0
        }
        # Gradient Boosting parameters
        gb_params = {
            'n_estimators': 500,
            # 'max_features': 0.2,
            'max_depth': 6,
            'min_samples_leaf': 2,
            'verbose': 0
        }

        # Support Vector Classifier parameters
        svc_params = {
            'kernel': 'linear',
            'C': 0.025
        }
        y_train = self.train['Survived'].ravel()
        self.train = self.train.drop(['Survived'], axis=1)
        x_train = self.train.values
        x_test = self.test.values

        rf = SklearnHelper(clf=RandomForestClassifier, seed=self.SEED, x_train=x_train, x_test=x_test, y_train=y_train,
                           params=rf_params)
        rf_oof_train, rf_oof_test = rf.get_oof(rf)

        rf_fpr, rf_tpr, rf_thresholds = metrics.roc_curve(y_train[:len(x_train)], rf_oof_train, pos_label=1)
        print('rf auc:{}'.format(metrics.auc(rf_fpr, rf_tpr)))
        print('rf precision:{}'.format(precision_score(y_train, rf_oof_train, average='macro')))

        gbt = SklearnHelper(clf=GradientBoostingClassifier, seed=self.SEED, x_train=x_train, x_test=x_test,
                            y_train=y_train,
                            params=gb_params)
        gbt_oof_train, gbt_oof_test = gbt.get_oof(gbt)

        gbt_fpr, gbt_tpr, gbt_thresholds = metrics.roc_curve(y_train[:len(x_train)], gbt_oof_train, pos_label=1)
        print('gbdt auc:{}'.format(metrics.auc(gbt_fpr, gbt_tpr)))
        print('gbdt precision:{}'.format(precision_score(y_train, gbt_oof_train, average='macro')))
        #
        svc = SklearnHelper(clf=SVC, seed=self.SEED, x_train=x_train, x_test=x_test, y_train=y_train,
                            params=svc_params)

        svc_oof_train, svc_oof_test = svc.get_oof(svc)

        svc_fpr, svc_tpr, svc_thresholds = metrics.roc_curve(y_train[:len(x_train)], svc_oof_train, pos_label=1)
        print('svc auc:{}'.format(metrics.auc(svc_fpr, svc_tpr)))
        print('svc precision:{}'.format(precision_score(y_train, svc_oof_train, average='macro')))

        print('Training is completed')


ts = TitanicSolution()
ts.train_model()
