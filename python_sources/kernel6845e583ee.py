# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
# Path of the file to read
gp_file_path = '../input/googleplaystore.csv'

gp_data = pd.read_csv(gp_file_path)


train, test = train_test_split(gp_data, test_size=0.2)

#print(len(train)) #8672
#print(len(test)) #2169

#print(train.columns)
# we want to predict price of app so we take y as price
train = train[np.isfinite(train['Rating'])] #droping  nan rows
train = train[~train['Reviews'].isin(['3.0M'])] #droping a particular row which had 3.0 Milion reviews 

test = test[np.isfinite(test['Rating'])] #droping  nan rows
test = test[~test['Reviews'].isin(['3.0M'])]

y=train.Reviews
test_y = test.Reviews

features = ['Rating']
X = train[features]
#X = X[np.isfinite(X['Rating'])]

#print(X)
#print(y)
test_X = test[features]

gp_model = DecisionTreeRegressor(random_state=1)
gp_model.fit(X,y)

predictions = gp_model.predict(test_X)
print(predictions)
print(test_y.head())













