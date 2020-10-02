# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
train_size = train.shape

# Output/result is the price
y = train['SalePrice']

column_names = train.columns.values[:-1]
# Let's explore each column and its distribution on price


test = pd.read_csv('../input/test.csv')

test_id = test['Id']
all_ = pd.concat([train.loc[:,'MSSubClass':'SaleCondition'], test.loc[:,'MSSubClass':'SaleCondition']])
r,c = all_.shape
# print(r,c)

# Count number of NA for each column present in the dataframe
print(all_.isnull().sum())

# Dummy varialbe for categorical variable
all_ = pd.get_dummies(all_)

# For simplicity, replace na with the mean
all_ = all_.fillna(all_.mean())

#print(all_.head)
from sklearn.linear_model import Ridge, LassoCV
from sklearn.model_selection import cross_val_score

train = all_[:train_size[0]]
test = all_[train_size[0]:]

def rsme_(m):
    return np.sqrt(-cross_val_score(m, train, y, scoring="neg_mean_squared_error")).mean()

# Ridge
a = [0.01*(i**3) for i in range(1,20)]
rid = [rsme_(Ridge(alpha=aa)) for aa in a]
print(rid)

cv_ridge = pd.Series(rid, index = a)
cv_ridge.plot()
plt.xlabel("alpha")
plt.ylabel("rmse")
'''
# Lasso
a = [0.01*(i**2) for i in range(1,10)] + [1]
lasso = rsme_(LassoCV(alphas = a).fit(train, y))
print(lasso)
'''
# Lasso does not converge well without normalizing the house price

from sklearn.ensemble import RandomForestRegressor

max_depth = 30
regr_rf = RandomForestRegressor(max_depth=max_depth, random_state=2)
regr_rf.fit(train, y)

score1 = regr_rf.score(train,y)
print(regr_rf.score(train,y))
y_rf = regr_rf.predict(test)

from sklearn import svm
clf = svm.SVC()
clf.fit(train, y) 

score2 = clf.score(train,y)
print(clf.score(train,y))
y_regr = clf.predict(test)


output = {'Id':test_id,'SalePrice':y_rf}
output = pd.DataFrame.from_dict(output)
output.to_csv('submission_bae.csv')