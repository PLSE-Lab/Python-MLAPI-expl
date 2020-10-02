import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())

#Any files you save will be available in the output tab below
train.to_csv('copy_of_the_training_data.csv', index=False)

#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn import tree
import time
from functools import wraps

def monitor_time(func):

    @wraps(func)
    def calculate_time(*args, **kwargs ):
        start_time = time.time()
        result=func(*args, **kwargs)
        end_time=time.time()
        cost_time=end_time-start_time
        print(cost_time)
        return result

    return calculate_time


def preprocess_train_data(file):
    df = pd.read_csv(file, dtype={"Age": np.float64}, )
    
    print (df.info())
    print (df.head(2))

    cols = ['Name','Ticket','Cabin']

    df = df.drop(cols,axis=1)
    df = df.dropna()

    dummies = []
    cols = ['Pclass','Sex','Embarked']
    for col in cols:
        dummies.append(pd.get_dummies(df[col]))

    titanic_dummies = pd.concat(dummies, axis=1)

    df = pd.concat((df,titanic_dummies),axis=1)
    y = df['Survived'].values
    df=df.drop(['Survived'], axis=1)

    df = df.drop(['Pclass','Sex','Embarked'],axis=1)

    df['Age'] = df['Age'].interpolate()

    X = df.values


    X = np.delete(X,1,axis=1)

    fit_value=[X, y]
    return fit_value

def preprocess_test_data(file):
    df = pd.read_csv(file, dtype={"Age": np.float64}, )

    cols = ['Name','Ticket','Cabin']

    df = df.drop(cols,axis=1)
    df = df.dropna()

    dummies = []
    cols = ['Pclass','Sex','Embarked']
    for col in cols:
        dummies.append(pd.get_dummies(df[col]))

    titanic_dummies = pd.concat(dummies, axis=1)

    df = pd.concat((df,titanic_dummies),axis=1)

    df = df.drop(['Pclass','Sex','Embarked'],axis=1)

    df['Age'] = df['Age'].interpolate()

    X = df.values


    X = np.delete(X,1,axis=1)
    return X

def normalizing():
    pass


def save_result(results,file):  
    this_file=open(file,'w')
    this_file.write("PassengerId, Survived\n")
    for i,r in enumerate(results):
        this_file.write(str(i+1)+","+str(int(r))+"\n")

    this_file.close()

@monitor_time
def decision_tree_classify(train_data,train_result=None,test_data=None):

    train_fit_value=preprocess_train_data(train_data)
    X_train=train_fit_value[0]
    y_train=train_fit_value[1]


    test_fit_value=preprocess_test_data(test_data)
    X_test=test_fit_value

    clf = tree.DecisionTreeClassifier(max_depth=5)
    clf.fit(X_train, y_train)

    y_results = clf.predict(X_test)


    output = np.column_stack((X_test[:,0],y_results))
    df_results = pd.DataFrame(output.astype('int'),columns=['PassengerID','Survived'])
    df_results.to_csv('decision_tree_classify_titanic_results.csv',index=False)


def main():

    decision_tree_classify("../input/train.csv",train_result=None,test_data="../input/test.csv")

if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()