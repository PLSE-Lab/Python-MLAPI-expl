# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, model_selection, linear_model
import pickle

# read the dataset, parse the dates and set the index
df = pd.read_csv('/kaggle/input/googlestockpricing/Google.csv', parse_dates = ['Date'])
df.set_index('Date', inplace=True)

# feature selection
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# visualize the adjusted close values
plt.plot(df['Adj. Close'])
plt.show()

# apply a rolling mean to the adjusted close column
df['Adj. Close'] = df['Adj. Close'].rolling(25).mean()

# visualize the adjusted close values after apply rolling mean
plt.plot(df['Adj. Close'])
plt.show()

# create a new label column by shifting the adjusted close column by some values
df['Label'] = df['Adj. Close'].shift(-10)

# save the values with no labels for predicting and inferencing purposes
check = df[np.isnan(df['Label'])][['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
check.dropna(inplace=True)

# handle the missing values
df.dropna(inplace=True)

# create features and labels from the dataframe
X = np.array(df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']])
X = preprocessing.scale(X)

y = np.array(df['Label'])

# split the dataset into training and testing data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# initialize the linear regression algorithm, train the model and check the accuracy
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
accuracy = reg.score(X_test, y_test)

# save the model in a seperate file
pickle_out = open("regressionstocks.pickle","wb")
pickle.dump(reg, pickle_out)
pickle_out.close()

# load the saved model back in our script
pickle_in = open('regressionstocks.pickle', 'rb')
model = pickle.load(pickle_in)

# preprocess data before passing it for predictions
check = preprocessing.scale(np.array(check))

# predict values using trained model
pred = reg.predict(check)

# predict values using saved model
pred1 = model.predict(check)

# visualize the predicted values
plt.plot(pred)
plt.show()