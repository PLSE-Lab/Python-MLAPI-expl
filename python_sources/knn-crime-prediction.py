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

import math

from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


	
def hr_func(ts):
    return (float)(ts[11:13])
	
def llfun(act, pred):
    """ Logloss function for 1/0 probability
    """
    return (-(~(act == pred)).astype(int) * math.log(1e-15)).sum() / len(act)


train = pd.read_csv("../input/train.csv")[['X', 'Y','Dates','Address', 'Category','PdDistrict']]
train['street_corner'] = train['Address'].apply(lambda x: 1 if '/' in x else 0)
train['Hour'] = train['Dates'].apply(hr_func)
train['Year']=(float)(train['Dates'][0][0:4])
train['Month']=(float)(train['Dates'][0][5:7])
train['Day']=(float)(train['Dates'][0][8:10])
PdDistrict = sorted(train['PdDistrict'].unique())
PdDistrict_mapping = dict(zip(PdDistrict, range(0, len(PdDistrict) + 1)))
train['District'] = train['PdDistrict'].map(PdDistrict_mapping).astype(int)

std_scaleX = preprocessing.StandardScaler().fit(train['X'])
std_scaleY = preprocessing.StandardScaler().fit(train['Y'])
std_scaleHour = preprocessing.StandardScaler().fit(train['Hour'])
std_scaleCorner = preprocessing.StandardScaler().fit(train['street_corner'])
std_scaleDay = preprocessing.StandardScaler().fit(train['Day'])
std_scaleMonth = preprocessing.StandardScaler().fit(train['Month'])
std_scaleYear = preprocessing.StandardScaler().fit(train['Year'])
std_scaleDistrict = preprocessing.StandardScaler().fit(train['District'])

train['normalized_X'] = std_scaleX.transform(train['X'])
train['normalized_Y'] = std_scaleY.transform(train['Y'])
train['normalized_Hour'] = std_scaleHour.transform(train['Hour'])
train['normalized_Corner'] = std_scaleCorner.transform(train['street_corner'])
train['normalized_Day'] = std_scaleDay.transform(train['Day'])
train['normalized_Month'] = std_scaleMonth.transform(train['Month'])
train['normalized_Year'] = std_scaleCorner.transform(train['Year'])
train['normalized_District'] = std_scaleCorner.transform(train['District'])
# Separate test and train set out of orignal train set.
msk = np.random.rand(len(train)) < 0.8
knn_train = train[msk]
knn_test = train[~msk]
n = len(knn_test)


# Prepare data sets

x = knn_train[['normalized_X', 'normalized_Y','normalized_Hour','normalized_Corner','normalized_Day','normalized_Month','normalized_Year','normalized_District']]
y = knn_train['Category'].astype('category')
actual = knn_test['Category'].astype('category')


# Fit
logloss = []
for i in range(5, 50, 1):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x, y)
    
    # Predict on test set
    outcome = knn.predict(knn_test[['normalized_X', 'normalized_Y','normalized_Hour','normalized_Corner','normalized_Day','normalized_Month','normalized_Year','normalized_District']])
    print(i)
    # Logloss
    logloss.append(llfun(actual, outcome))

plt.plot(logloss)
plt.savefig('n_neighbors_vs_logloss.png')

# Submit for K=40

test = pd.read_csv("../input/test.csv")
test['Hour'] = test['Dates'].apply(hr_func)
test['street_corner'] = test['Address'].apply(lambda x: 1 if '/' in x else 0)
test['District']=test['PdDistrict'].map(PdDistrict_mapping).astype(int)

test['normalized_X'] = std_scaleX.transform(test['X'])
test['normalized_Y'] = std_scaleY.transform(test['Y'])
test['normalized_Hour'] = std_scaleHour.transform(test['Hour'])
test['normalized_Corner'] = std_scaleCorner.transform(test['street_corner'])
test['normalized_Day'] = std_scaleDay.transform(test['Day'])
test['normalized_Month'] = std_scaleMonth.transform(test['Month'])
test['normalized_Year'] = std_scaleCorner.transform(test['Year'])
test['normalized_District'] = std_scaleDistrict.transform(test['District'])

x_test = test[['normalized_X', 'normalized_Y','normalized_Hour','normalized_Corner','normalized_Day','normalized_Month','normalized_Year','normalized_District']]
knn = KNeighborsClassifier(n_neighbors=40)
knn.fit(x, y)
outcomes = knn.predict(x_test)

submit = pd.DataFrame({'Id': test.Id.tolist()})
for category in y.cat.categories:
	submit[category] = np.where(outcomes == category, 1, 0)
    
submit.to_csv('k_nearest_neigbour.csv', index = False)