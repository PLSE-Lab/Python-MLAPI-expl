# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.linear_model import Ridge
import pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def get_data(path):
    df = pd.read_csv(path)
    return df


def process_data(df):
    df.drop('origin', axis=1, inplace=True)
    df = df[df['horsepower'] != '?']
    df = df.sample(frac=1).reset_index(drop=True)
    #dummy_data = pd.get_dummies(df['origin'])
    #df.drop('origin', axis=1, inplace=True)
    #dummy_data.drop(dummy_data.columns[0], axis=1, inplace=True)
    #df = pd.concat([df, dummy_data], axis=1)
    return df


def get_train_test_data(df):
    data_x = df.drop(['car name', 'mpg'], axis=1)
    data_y = df['mpg']
    car_names = df['car name']
    return data_x, data_y, car_names


def train_model(x, y, deg):
    lm = make_pipeline(PolynomialFeatures(degree=deg), Ridge())
    lm.fit(x, y)
    return lm


def serialize_object(obj, path):
    pickle.dump(obj, path)


def deserialize_object(path):
    obj = pickle.load(path)
    return obj


def create_cv_sets(x, y):
    train_x = x.values[:-40]
    train_y = y.values[:-40]
    test_x = x.values[-40:]
    test_y = y.values[-40:]
    return train_x, train_y, test_x, test_y


def get_predictions(model, x):
    predictions = model.predict(x)
    return predictions


def get_score(model, x, y):
    score = model.score(x, y)
    return score


if __name__ == '__main__':
    p = "../input/auto-mpg.csv"
    data = get_data(p)
    data = process_data(data)
    x_data, y_data, cars = get_train_test_data(data)
    train_x, train_y, test_x, test_y = create_cv_sets(x_data, y_data)
    degree_list = [1, 2, 3]
    for degree in degree_list:
        system = train_model(train_x, train_y, degree)
        s = get_score(system, test_x, test_y)
        print(s)
