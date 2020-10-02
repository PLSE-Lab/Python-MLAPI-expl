# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

def fit_num_imputer(df):
    num_imputer = Imputer(strategy='mean', axis=1)
    num_imputer.fit(df)
    return num_imputer

def fit_cat_imputer(df):
    cat_imputer = Imputer(strategy='most_frequent', axis=1)
    cat_imputer.fit(df)
    return cat_imputer

def fit_x_vectorizer(df):
    """Transform categorical data into values in dictionary"""
    cat_dict = df.T.to_dict().values()
    x_vectorizer = DictVectorizer(sparse=False)
    x_vectorizer.fit(cat_dict)
    return x_vectorizer


def fit_x_normalizer(df):
    """Decide values to normalize numerical data on a scale of 0 to 1"""
    x_normalizer = MinMaxScaler()
    x_normalizer.fit(df.as_matrix())
    return x_normalizer


def fit_y_vectorizer(df):
    """Transform categorical data into values in dictionary"""
    cat_dict = df.T.to_dict().values()
    y_vectorizer = DictVectorizer(sparse=False)
    y_vectorizer.fit(cat_dict)
    return y_vectorizer


def fit_y_normalizer(df):
    """Decide values to normalize numerical data on a scale of 0 to 1"""
    y_normalizer = MinMaxScaler()
    y_normalizer.fit(df.as_matrix())
    return y_normalizer


def x_vectorize(df, x_vectorizer) -> np.ndarray:
    """Converts dictionary of vectorized values to a numpy array"""
    t = x_vectorizer.transform(df.T.to_dict().values())
    return np.array(t)


def x_normalize(df, x_normalizer) -> np.ndarray:
    """Transform numeric data to fitted normalized scale in a numpy array"""
    t = x_normalizer.transform(df.as_matrix())
    return np.array(t)


def y_vectorize(df, y_vectorizer) -> np.ndarray:
    """Converts dictionary of vectorized values to a numpy array"""
    t = y_vectorizer.transform(df.T.to_dict().values())
    return np.array(t)


def y_normalize(df, y_normalizer) -> np.ndarray:
    """Transform numeric data to fitted normalized scale in a numpy array"""
    t = y_normalizer.transform(df.as_matrix())
    return np.array(t)


def denormalize(matrix, y_normalizer, y_num_cols) -> pd.DataFrame:
    """Converts output predicted values to usable output based on normalized scale determined previously"""
    data = y_normalizer.inverse_transform(matrix)
    df = pd.DataFrame(data)
    df.columns = y_num_cols
    return df


def train_remove_nulls(df):
    """"create lists of missing data and how much of it is missing in each column. then deals with nulls."""
    null_total = df.isnull().sum().sort_values(ascending=False)
    null_percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
    missing_data_df = pd.DataFrame(pd.concat(
            [null_total, null_percent],
            axis=1,
            keys=['Total', 'Percent']))
    print(missing_data_df[missing_data_df['Total'] > 0])
    # deal with missing data
    # drop columns with more than 1 missing record
    df = df.drop((missing_data_df[missing_data_df['Total'] > 4]).index, 1)
    null_rows = missing_data_df[missing_data_df['Total'] <= 4]
    df.dropna(axis='rows', subset=null_rows.index.tolist(), inplace=True)
    print(df.isnull().sum().max())
    return df


def test_fill_nulls(df, cat_list, num_list, cat_imputer, num_imputer):
    """"create lists of missing data and how much of it is missing in each column. then deals with nulls."""
    null_total = df.isnull().sum().sort_values(ascending=False)
    null_percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
    missing_data_df = pd.DataFrame(pd.concat(
            [null_total, null_percent],
            axis=1,
            keys=['Total', 'Percent']))
    print(missing_data_df[missing_data_df['Total'] > 0])
    df = df.drop((missing_data_df[missing_data_df['Total'] > 4]).index, 1)

    # df[cat_list] = cat_imputer.transform(df[cat_list])
    df[num_list] = num_imputer.transform(df[num_list])

    print(df.isnull().sum().max())
    return df


def write_output_data(df_id, df_y):
    """formats output"""
    y_format_df = pd.DataFrame(pd.concat(
        [df_id.reset_index(), df_y.reset_index()],
        axis=1,
        join='inner',
        ignore_index=True))
    y_format_df.columns = ["Ix", "Id", "NewIx", "SalePrice"]
    pd.DataFrame(y_format_df.set_index('Id').drop(['Ix', 'NewIx'], axis=1).to_csv('submission.csv'))
    print(y_format_df.shape)


train = train_remove_nulls(train)

x_string = []
x_num = []
y_num = [
    'SalePrice'
]


for t in train:
    if t == 'Id' or t == 'SalePrice':
        pass
    elif train[t].dtype == 'object':
        x_string.append(t)
    elif train[t].dtype == 'int64':
        x_num.append(t)


num_impute = fit_num_imputer(train[x_num])
cat_impute = fit_cat_imputer(train[x_string])

test = test_fill_nulls(test, x_string, x_num, cat_impute, num_impute)

# clean up the categorical data
x_cat_vec = fit_x_vectorizer(train[x_string])
x_cat_np_train = x_vectorize(train[x_string], x_cat_vec)

# clean up the numeric data
x_num_norm = fit_x_normalizer(train[x_num])
y_num_norm = fit_y_normalizer(train[y_num])
x_num_np_train = x_normalize(train[x_num], x_num_norm)
y_num_np_train = y_normalize(train[y_num], y_num_norm)
fit_y = np.ravel(y_num_np_train)

# do the same for the test data set for later
x_cat_np_test = x_vectorize(test[x_string], x_cat_vec)
x_num_np_test = x_normalize(test[x_num], x_num_norm)

# put humpty dumpty back together
x_train_stack = np.hstack((x_num_np_train, x_cat_np_train))
x_test_stack = np.hstack((x_num_np_test, x_cat_np_test))

# Building and fitting my_forest
forest = RandomForestRegressor(max_depth=10, min_samples_split=2, n_estimators=100, random_state=1)
my_forest = forest.fit(x_train_stack, fit_y)

# Print the score of the fitted random forest
print(my_forest.score(x_train_stack, fit_y))

# Compute predictions on our test set features then print the length of the prediction vector
y_np_test = my_forest.predict(x_test_stack)
y_np_test_reshape = np.reshape(y_np_test, (-1, 1))
test_output = denormalize(y_np_test_reshape, y_num_norm, y_num)

write_output_data(test['Id'], test_output)

