# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import svm,neighbors,cross_validation
from sklearn.linear_model import LinearRegression

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os
print(os.listdir("../input"))


data = pd.read_csv('../input/insurance.csv')
bmi = data['bmi']


# to recode strings to integers
def recode():
    data.replace('female',1,inplace=True)
    data.replace('male', 0, inplace=True)
    data.replace('yes', 1, inplace=True)
    data.replace('no',0,inplace=True)
    data.replace('southwest',1,inplace=True)
    data.replace('southeast', 2, inplace=True)
    data.replace('northwest', 3, inplace=True)
    data.replace('northeast', 4, inplace=True)


recode()


# recoding bmi
for i in bmi:
    if i > 25 :
        data['bmi'] = 2
    elif 18<i <25:
        data['bmi'] = 1
    else:
        data['bmi'] = 0

# data for prediction
# age = input('AGE : ')
# sex = input('\n[Male - 0 , Female - 1 ]---SEX : ')
# bmi = input('\n< 18 = 0 , 18 -25 = 1, > 25 = 2 ----BMI : ')
# children = input('\nNo. Children : ')
# reg = input('\n [ Southwest - 1 , Southeast - 2,Northwest - 3 , Northeast - 4] ---REGION YOU ARE IN :')

# print('\nTo check whether he is a smoker or not Answer this ......! ')





# Using k-Nearest neighbours to predict a smoker
def smokers_():
    # X and y data
    X = np.array(data.drop(['smoker'],1))
    y = np.array(data['smoker'])
    # char is the insurance amount [prediction]
# char = input('\nWhat was the cost of insurance ! ')

# x = [[age,sex,bmi,children,reg,char]]
    # # passing the x values as float
# pred = np.array(x,float)
    # spliting the data for testing and training
    X_train , X_test,y_train , y_test = cross_validation.train_test_split(X,y,test_size=0.2)
    # the KNN function
    clf = neighbors.KNeighborsClassifier()
    # Training
    clf.fit(X_train,y_train)
    # Testing
    acc = clf.score(X_test,y_test)
    acc = acc * 100
    #########[prediction]############
# predictor = clf.predict(pred)
# if predictor == 0:
#     print('with 90 % accuracy I can say that the person is not a Smoker !\n')
# else:
#     print('Unfortunately the data says that he is a smoker ! \n')
    #################################
    print( str(acc) + '% is the accuracy for predicting a smoker  \n')

smokers_()

print('Now for Amount Charged by the Insurance company .....!\n')


#######################################################################################################################

# Using Linear Regression for predicting the Insurance amount
def amount_charged():
    X = np.array(data.drop(['charges'],1))
    y = np.array(data['charges'])


# smoke = input('Please state whether he is a smoker or not !...[Enter 1 if YES! ,Enter 0 if NO!] ')

# x = np.array([[age, sex, bmi, children, smoke, reg]])
# x = np.asfarray(x,float)


    X_train , X_test,y_train , y_test = cross_validation.train_test_split(X,y,test_size=0.2)


    clf = LinearRegression()
    # training
    clf.fit(X_train,y_train)
    # Testing
    acc = clf.score(X_test,y_test)
    acc = acc * 100
    ###############Prediction#################
# predictor2 = clf.predict(x)
# print(predictor2)
    # ##########################################
    print(str(acc) + '% is the accuracy for predicting the amount charged by the insurance company \n')
amount_charged()
