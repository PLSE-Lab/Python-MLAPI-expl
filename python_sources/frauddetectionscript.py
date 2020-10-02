import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import catboost
from catboost import Pool, CatBoostClassifier, cv #CatBoost
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import KFold

def splitData(X, y):
    return train_test_split(X, y, test_size=0.1)

if __name__ == '__main__':
        m_name = sys.argv[1]
        base_path = './output/'

        train_transaction = pd.read_csv('./input/train_transaction.csv', index_col='TransactionID')
        test_transaction = pd.read_csv('./input/test_transaction.csv', index_col='TransactionID')

        train_identity = pd.read_csv('./input/train_identity.csv', index_col='TransactionID')
        test_identity = pd.read_csv('./input/test_identity.csv', index_col='TransactionID')

        sample_submission = pd.read_csv('./input/sample_submission.csv', index_col='TransactionID')

        train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
        test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)
        del train_transaction, train_identity, test_transaction, test_identity

        '''
        Data Augmentation
        '''
        train['TransactionAmt_to_mean_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('mean')
        train['TransactionAmt_to_mean_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('mean')
        train['TransactionAmt_to_std_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('std')
        train['TransactionAmt_to_std_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('std')

        test['TransactionAmt_to_mean_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('mean')
        test['TransactionAmt_to_mean_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('mean')
        test['TransactionAmt_to_std_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('std')
        test['TransactionAmt_to_std_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('std')

        train['id_02_to_mean_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('mean')
        train['id_02_to_mean_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('mean')
        train['id_02_to_std_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('std')
        train['id_02_to_std_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('std')

        test['id_02_to_mean_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('mean')
        test['id_02_to_mean_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('mean')
        test['id_02_to_std_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('std')
        test['id_02_to_std_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('std')

        train['D15_to_mean_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('mean')
        train['D15_to_mean_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('mean')
        train['D15_to_std_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('std')
        train['D15_to_std_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('std')

        test['D15_to_mean_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('mean')
        test['D15_to_mean_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('mean')
        test['D15_to_std_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('std')
        test['D15_to_std_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('std')

        train['D15_to_mean_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('mean')
        train['D15_to_mean_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('mean')
        train['D15_to_std_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('std')
        train['D15_to_std_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('std')

        test['D15_to_mean_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('mean')
        test['D15_to_mean_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('mean')
        test['D15_to_std_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('std')
        test['D15_to_std_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('std')

        train[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = train['P_emaildomain'].str.split('.', expand=True)
        train[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = train['R_emaildomain'].str.split('.', expand=True)
        test[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = test['P_emaildomain'].str.split('.', expand=True)
        test[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = test['R_emaildomain'].str.split('.', expand=True)

        '''
        Drop Columns
        '''
        many_null_cols = [col for col in train.columns if train[col].isnull().sum() / train.shape[0] > 0.9]
        many_null_cols_test = [col for col in test.columns if test[col].isnull().sum() / test.shape[0] > 0.9]
        big_top_value_cols = [col for col in train.columns if train[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
        big_top_value_cols_test = [col for col in test.columns if test[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
        one_value_cols = [col for col in train.columns if train[col].nunique() <= 1]
        one_value_cols_test = [col for col in test.columns if test[col].nunique() <= 1]
        cols_to_drop = list(set(many_null_cols + many_null_cols_test + big_top_value_cols + big_top_value_cols_test + one_value_cols+ one_value_cols_test))
        cols_to_drop.remove('isFraud')
        train = train.drop(cols_to_drop, axis=1)
        test = test.drop(cols_to_drop, axis=1)
        train = train.drop(['TransactionDT'], axis=1)
        test = test.drop(['TransactionDT'], axis=1)

        print(train.head())
        print(train.shape)

        '''
        Target Column
        '''
        y_train = train['isFraud'].copy()
        X_train = train.drop('isFraud', axis=1)
        X_test = test.copy()
        del train, test


        X_train = X_train.fillna(-999)
        X_test = X_test.fillna(-999)

        # Label Encoding
        for f in X_train.columns:
                if '_pred' in f: continue
                if (X_train[f].dtypes == 'object') or (X_test[f].dtypes == 'object'): 
                        lbl = preprocessing.LabelEncoder()
                        lbl.fit(list(X_train[f].values) + list(X_test[f].values))
                        X_train[f] = lbl.transform(list(X_train[f].values))
                        X_test[f] = lbl.transform(list(X_test[f].values))   

        version = 'aug-drop09_%s_non_non_manual1'%m_name
        X_train.to_csv('./output/train_%s.csv'%version)
        X_test.to_csv('./output/test_%s.csv'%version)

        if m_name == 'xgboost':
                classifier = xgb.XGBClassifier(
                        n_estimators=500,
                        max_depth=9,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        missing=-999,
                        random_state=2019,
                        tree_method='gpu_hist'  # THE MAGICAL PARAMETER
                )
        elif m_name == 'catboost':
                classifier = CatBoostClassifier(
                        task_type='GPU',
                        eval_metric='AUC',
                        loss_function='Logloss', 
                        learning_rate=0.1,
                        iterations=10000, 
                        max_leaves=48,
                        od_wait=100,
                        max_depth=6,
                        class_weights=[0.05, 0.95],
                        metric_period = 100,
                )
        elif m_name == 'svm':
                classifier = LinearSVC()
        elif m_name == 'random_forest':
                classifier = RandomForestClassifier()
        elif m_name == 'logistic':
                classifier = LogisticRegression()

        kf = KFold(n_splits = 5, shuffle = True)
        for train_index, test_index in kf.split(X_train):
                X = X_train.values[train_index]
                y = y_train.values[train_index]
                classifier.fit(X, y)
                predict = classifier.predict(X_train.values[test_index])
                print(classification_report(y_train.values[test_index], predict))

                if m_name != 'svm':
                        prob = classifier.predict_proba(X_train.values[test_index])[:,1]
                        print(roc_auc_score(y_train.values[test_index], prob))

        classifier.fit(X_train, y_train)
        predict = classifier.predict(X_train)
        if m_name != 'svm':
                prob = classifier.predict_proba(X_train)[:,1]
                print(roc_auc_score(y_train, prob))

        if m_name == 'xgboost' or m_name == 'random_forest' or m_name == 'logistic' or m_name == 'catboost':
                pred = classifier.predict_proba(X_test)[:,1]
        elif m_name == 'svm':
                pred = classifier.predict(X_test)

        print(pred)
        sample_submission['isFraud'] = pred
        sample_submission.to_csv('./output/submission_%s.csv'%m_name)


