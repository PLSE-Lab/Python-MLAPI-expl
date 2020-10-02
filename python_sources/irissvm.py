# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



def read_datasets():
    global iris_dataset
    iris_dataset = pd.read_csv('../input/Iris.csv',skiprows=1,names=['id','sepal_length','sepal_width','petal_length','petal_width','class'],index_col='id')
    print(iris_dataset.head())


def show_visuals():
    global iris_read_datasets
    fig = plt.figure()
    ax= fig.add_subplot(111)
    plt.legend()
    plt.show()

def split_params():

    global validation_size
    global seed
    validation_size = 0.2
    seed=7

def split_dataset():
    split_params()
    global iris_dataset
    global train_predictors,test_predictors,train_targets,test_targets
    train,test = train_test_split(iris_dataset,test_size=validation_size,random_state=seed)
    train_predictors = train.drop('class',axis=1)
    train_targets= train['class']
    test_predictors= test.drop('class',axis=1)
    test_targets= test['class']

def fit_to_model_and_test(model):
    global train_predictors,test_predictors,train_targets,test_targets
    model.fit(train_predictors,train_targets)
    print(model.score(test_predictors,test_targets)*100)

def main():
    read_datasets()
    global iris_dataset
    print(iris_dataset.shape)
    print(iris_dataset.describe())
    print(iris_dataset.isnull().any())
    split_dataset()
    model= SVC()
    fit_to_model_and_test(model)

if __name__ == "__main__":
    main()