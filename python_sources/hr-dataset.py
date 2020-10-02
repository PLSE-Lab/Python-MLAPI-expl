# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.

hr = pd.read_csv('../input/HR_comma_sep.csv', sep=',')
from sklearn.neural_network import multilayer_perceptron as mp
import sklearn.preprocessing as preprocessing
import sklearn.metrics as m
import matplotlib.pyplot as plt
import numpy as np

for j in ['sales','salary']:
    z = hr['left'].groupby(hr[j]).sum() / hr[j].groupby(hr[j]).count()

    for i in z.keys():
        hr.loc[hr[j] == i, j] = z[i]
x = hr[['satisfaction_level','last_evaluation','number_project',
        'average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','sales','salary']]
y =  hr[['left']]
x_train=x[:int(len(x)/2)]

x_for_sample=[]

cls_1=hr.loc[hr.left==1, ['satisfaction_level', 'last_evaluation',
                'number_project', 'average_montly_hours', 'time_spend_company',
                'Work_accident', 'left', 'promotion_last_5years', 'sales', 'salary']]
cls_0=hr.loc[hr.left==0, ['satisfaction_level', 'last_evaluation',
                'number_project', 'average_montly_hours', 'time_spend_company',
                'Work_accident', 'left', 'promotion_last_5years', 'sales', 'salary']]
train=[cls_1[:int(len(cls_1)/2)],cls_0[:int(len(cls_1)/2)]]
train =  pd.concat(train)

test=[cls_1[int(len(cls_1)/2):],cls_0[int(len(cls_1)/2):]]
test =  pd.concat(test)

x = hr['satisfaction_level']+hr['last_evaluation']+hr['number_project']+\
    hr['average_montly_hours']+hr['time_spend_company']+hr['Work_accident']+hr['promotion_last_5years']+hr['sales']+hr['salary']
y =  hr[['left']]

x_train=train[['satisfaction_level','last_evaluation','number_project',
        'average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','sales','salary']]
#scaler = preprocessing.StandardScaler()
#x_train = scaler.fit_transform(x_train)


std = preprocessing.StandardScaler().fit(x_train[ ['satisfaction_level','last_evaluation','number_project',
        'average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','sales','salary']])
x_train.is_copy = False
x_train.loc[:,( 'satisfaction_level','last_evaluation','number_project',
        'average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','sales','salary')] \
    = std.transform(x_train[ ['satisfaction_level','last_evaluation','number_project',
        'average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','sales','salary']])


y_train=train['left']

x_test=test[['satisfaction_level','last_evaluation','number_project',
        'average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','sales','salary']]
y_test=test['left']
std = preprocessing.StandardScaler().fit(x_test[ ['satisfaction_level','last_evaluation','number_project',
        'average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','sales','salary']])
x_test.is_copy = False
x_test.loc[:,( 'satisfaction_level','last_evaluation','number_project',
        'average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','sales','salary')] \
    = std.transform(x_test[ ['satisfaction_level','last_evaluation','number_project',
        'average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','sales','salary']])
clf = mp.MLPClassifier(solver='lbfgs', hidden_layer_sizes=(9, 17,
                                                           2), random_state=1)

y_train  = y_train.as_matrix().astype(np.float)
#x_train = x_train.as_matrix().astype(np.float)
y_test  = y_test.as_matrix().astype(np.float)
#x_test = x_test.as_matrix().astype(np.float)


clf.fit(x_train, y_train)
y_train_c = clf.predict(x_train)
y_test_c = clf.predict(x_test)


print (m.accuracy_score(y_train, y_train_c), m.accuracy_score(y_test, y_test_c))