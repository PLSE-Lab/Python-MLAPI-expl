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

print('Importing data ...')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

print('Data prep ...')
train_x = train.iloc[:,1:785]  # all rows and only the pixel columns
train_y = train.iloc[:,0] # all rows and only the label column
test_x = test.iloc[:,0:784] # all rows and only the pixel columns

# build model
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(train_x,train_y)

print('Creating submission file ...')
submit = pd.DataFrame({'ImageId': range(1, 28001), 'Label': model.predict(test_x)}) # create submission file
submit.to_csv('submit.csv', index = False)  # export the submit file