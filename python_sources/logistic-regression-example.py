import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score


# edit the data_root to reflect the location of your data
DATA_ROOT = '../input/'


def load_train_data(path):
    '''
    loads and returns pandas dataframe with training data
    :param path: path to the location of the training data
    :return: DataFrame
    '''
    df = pd.read_csv(path, skipinitialspace=True)
    return df


def load_test_data(path):
    '''
    loads and returns pandas dataframe with testing data (submission)
    :param path: path to the location of the testing data
    :return: DataFrame
    '''
    df = pd.read_csv(path, skipinitialspace=True)
    df['id'] = df['id'].astype(int)
    return df


def random_submission(df):
    '''
    appends a target column to the pandas dataframe
    :param df: testing dataframe
    :return: DataFrame
    '''
    df['target'] = 0.5
    return df


def make_random_submission():
    '''
    creates and saves the random submission
    :return: None
    '''
    df = load_test_data(DATA_ROOT + 'adsse_test.csv')
    df = random_submission(df)
    # extracting id and target column for the submission
    df = pd.concat([df['id'], df['target']], axis=1, keys=['id', 'target'])
    # saves id and target column to a csv file
    df.to_csv(DATA_ROOT + 'submission.csv', columns=['id', 'target'], index=False)


def logistic_regression_submission():
    '''
    creates and saves the logistic regression submission
    :return: None
    '''
    # loads training and testing data
    train_data = load_train_data(DATA_ROOT + 'adsse_train.csv')
    test_data = load_test_data(DATA_ROOT + 'adsse_test.csv')

    # separates features from target
    X = train_data.ix[:, train_data.columns != 'target'].values
    X_test = test_data.ix[:, test_data.columns != 'id'].values
    y = train_data['target'].values

    # trains classifier
    clf = LogisticRegression()
    # CV
    kfold = KFold(n_splits=5, shuffle=True, random_state=1337)
    score = cross_val_score(estimator=clf, X=X, y=y, scoring='neg_log_loss', cv=kfold)
    print("---" * 20)
    print('kfold score: {}'.format(-score.mean()))
    print("---" * 20)
    
    clf.fit(X, y)

    # makes predictions and saves them to a csv file
    test_data['target'] = clf.predict_proba(X_test)[:, 1]
    test_data = pd.concat([test_data['id'], test_data['target']], axis=1, keys=['id', 'target'])
    test_data.to_csv(DATA_ROOT + 'submission.csv', columns=['id', 'target'], index=False)

if __name__ == '__main__':
    # create a submission from simple logistic regression
    logistic_regression_submission()
    # create a submission with p=0.5 confidence in the prediction
    # uncomment below to create a submission with only 0.5
    # make_random_submission()
