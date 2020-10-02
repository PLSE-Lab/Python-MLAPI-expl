# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# full = train.append( test , ignore_index = True )
# titanic = full[ :len(train) ]

# print ('Datasets:' , 'train + test:' , full.shape , 'titanic:' , titanic.shape)

def predictSurvivedIfFemale(x):
  if x == 'female':
      return 1
  else:
      return 0

# print (test.head())
passenger_id = test.PassengerId
survived = test.Sex.apply(predictSurvivedIfFemale)
test_sex = test.Sex

# test_pred = pd.DataFrame( { 'PassengerId': passenger_id, 'Sex': test_sex, 'Survived': survived })
test_pred = pd.DataFrame( { 'PassengerId': passenger_id, 'Survived': survived })

test_pred.to_csv( 'titanic_pred.csv' , index = False )
print (test_pred.head())

# del train , test




# Any results you write to the current directory are saved as output.