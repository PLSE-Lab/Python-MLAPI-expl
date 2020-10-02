# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

df = pd.read_csv('../input/HR_comma_sep.csv', header=0)
salary_mapping = {'high':3,'medium':2,'low':1}
df['salary'] = df['salary'].map(salary_mapping)

X = df[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','sales','salary']]


X = pd.get_dummies(X, drop_first=True)


y = df[['left']]

y = pd.DataFrame.as_matrix(y).ravel()





from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.svm import SVC


pipe_SVM = Pipeline([('scl', StandardScaler()),
                   ('clf', SVC(kernel='rbf',C=100,gamma=1))])

#clf = SVC(kernel='rbf',C=1)

X_train, X_test, y_train, y_test = \
    train_test_split(X,y, test_size=0.20, random_state=1)
##from sklearn import svm
#from sklearn.cross_validation import StratifiedKFold

pipe_lr = Pipeline([('scl', StandardScaler()),
                      ('clf', SVC(kernel='rbf',C=100,gamma=1))])

pipe_lr.fit(X_train,y_train)
pipe_lr.score(X_test,y_test)