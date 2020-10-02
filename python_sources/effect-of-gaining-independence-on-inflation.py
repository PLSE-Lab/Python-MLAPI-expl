# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
'''
Problem Statement: To understand the effect of 'how gaining its independence' has affected the country's inflation rate


Assumption: For this exercise, I assume that other than inflation (which is my dependent variable), 'independence' which
is the variable I am studying ,the other columns are control units which are not broadly independent of one another. 

Using linear regression,I will fit the data in each case for each country.

'''
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def train_test(train,test):
    clf=LinearRegression()
    train_set=train.drop(columns=['inflation_annual_cpi','cc3','banking_crisis','country','independence','year'],axis=1)
    test_set=test.drop(columns=['inflation_annual_cpi','cc3','banking_crisis','country','independence','year'],axis=1)
    for columns in train_set.columns:
        train_set[columns]=train_set[columns].astype('float')
        test_set[columns]=test_set[columns].astype('float')
    clf.fit(train_set,train['inflation_annual_cpi'].astype('float'))
    print("The fit on train_set of  is {} and the fit on test set is {}".format(clf.score(train_set,train['inflation_annual_cpi'].astype('float')),clf.score(test_set,test['inflation_annual_cpi'].astype('float'))))
    new_set=train_set.copy()
    print(len(new_set))
    new_set=pd.concat([new_set,test_set])
    print(len(new_set))
    predictions=clf.predict(new_set)
    print(len(predictions))
    return list(predictions)

def UnderstandingTrends(file_path,country,data):
    try:
        processed_data={}
        groupa,groupb=data.groupby('independence')
    except:
        print(country,"no split")
        return
    processed_data[country+' - '+str(groupa[0])]=train_test(train=groupa[1],test=groupb[1])   
def GroupingByCountries(file_path,input_data):
    Different_countries=np.unique(input_data.country)
    data=input_data.groupby('country')
    for country,data in data:
        print(country)
        UnderstandingTrends(file_path,country,data)
    
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        file_path=os.path.join(dirname, filename)
        input_data=pd.read_csv(file_path)
        GroupingByCountries(dirname,input_data)


# Any results you write to the current directory are saved as output.
