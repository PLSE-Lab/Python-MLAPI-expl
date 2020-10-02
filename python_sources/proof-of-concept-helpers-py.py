# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from scipy.stats import pearsonr

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error

def create_pipe_data(num_days):

    '''
    Let's create data that replicates the history of flow of water through a pipe.
    '''

    pipe = pd.DataFrame(np.random.randint(19,101,size=(num_days, 1)), columns=['percent_flow'])
    pipe = pipe.sort_values(by=['percent_flow'], ascending=False)
    pipe.reset_index(drop=True, inplace=True)

    pipe_days = pd.Series(range(1,(num_days + 1)))
    pipe_days = pd.DataFrame(pipe_days, columns=['days'])

    pipe_df = pipe.merge(pipe_days, left_index=True, right_index=True)

    return pipe_df

def poly_regression(pipe_name, train_size):

    train = pipe_name[:int(pipe_name.shape[0] * train_size)]
    test = pipe_name[int(pipe_name.shape[0] * train_size):]

    X_train = train[['days']]
    y_train = train[['percent_flow']]

    X_test = test[['days']]
    y_test = test[['percent_flow']]

    poly_features = PolynomialFeatures(degree=2)

    ### transforms the existing features to higher degree features.
    X_train_poly = poly_features.fit_transform(X_train)

    ### fit the transformed features to Linear Regression
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
        
    ### predicting on training data-set
    y_train_predicted = poly_model.predict(X_train_poly)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_predicted))
    r2_train = r2_score(y_train, y_train_predicted)

    print("The model performance for the training set")
    print("-------------------------------------------")
    print("RMSE of training set is {}".format(rmse_train))
    print("R2 score of training set is {}".format(r2_train))

    y_test_predicted = poly_model.predict(poly_features.fit_transform(X_test))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_predicted))
    r2_test = r2_score(y_test, y_test_predicted)
    print("The model performance for the test set")
    print("-------------------------------------------")
    print("RMSE of test set is {}".format(rmse_test))
    print("R2 score of test set is {}".format(r2_test))

    pipe_days_pred =  pd.DataFrame(y_test_predicted, columns=['percent_flow_pred'])
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    clean_day = X_test.merge(y_test, left_index=True, right_index=True)
    clean_day = clean_day.merge(pipe_days_pred, left_index=True, right_index=True)
    # clean_day = pipe_days_pred[pipe_days_pred.percent_flow <= 20][:1]

    print(clean_day)