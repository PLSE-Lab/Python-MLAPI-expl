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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import  seaborn as sns
from sklearn.model_selection import train_test_split
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# 2. (30 pts) Carry out an exploratory data analysis on the provided Defaulters Dataset. Identify how
# to optimally build a Logistic Regression classifier for the given dataset. Tune the model and
# explain your process and the results.

desired_width=320

pd.set_option('display.width', desired_width)



pd.set_option('display.max_columns',10)
data = r'/kaggle/input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv'
df = pd.read_csv(data,index_col=0)
print(df.shape)
print(df.info())
print(df.describe().transpose())
df = df.rename(columns={'default.payment.next.month':'default_payment_next_month'})
print(df.head())
print(df.columns)

print(df['default_payment_next_month'].value_counts())

ax1 = sns.countplot(x='default_payment_next_month', data=df)
ax1.set_xticklabels(['No default', 'Default'])
plt.show()

'''we se imbalance class here'''

plt.title('Education dist')
ax2 = sns.countplot(x='EDUCATION', hue='default_payment_next_month', data=df)
plt.show()
'''we see that most of the defaulters are from education label 2'''

plt.title('Sex Dist')
ax3 = sns.countplot(x='SEX', hue='default_payment_next_month', data=df)
ax3.set_xticklabels(['Female', 'Male'])
plt.show()
'''most defaulters are male'''

plt.title('Age dist with default/no_default')
agedist = df[df['default_payment_next_month']==0]['AGE']
agedist_1 = df[df['default_payment_next_month']==1]['AGE']
sns.distplot(agedist, bins=100, color='blue')
sns.distplot(agedist_1, bins=100, color='orange')
plt.show()

plt.title('Credit amount dist w/ default/no_default')
cred0 = df[df['default_payment_next_month']==0]['LIMIT_BAL']
cred1 = df[df['default_payment_next_month']==1]['LIMIT_BAL']
sns.distplot(cred0, bins=100, color='blue')
sns.distplot(cred1, bins=100, color='orange')
plt.show()
'''here we can see clients with lower credit amounts tend to default,
 as evident by the positive skewness of the curve '''


'''one hot encode all catergories '''
cat = ['MARRIAGE', 'EDUCATION', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
df_sex = pd.get_dummies(data=df['SEX'], prefix='SEX_')

df = pd.get_dummies(data=df, columns=cat, drop_first=True)
df.reset_index(drop=True, inplace=True)
df_sex.reset_index(drop=True, inplace=True)
df = pd.concat([df, df_sex], axis=1)
df.drop(columns=['SEX'], inplace=True,  axis=1)
print(df.shape)
print(df.isnull().sum())

df_x = df.drop(columns='default_payment_next_month')
df_y = df['default_payment_next_month']
'''lets scale'''
print([col for col in df.columns])
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""


scalable_vars = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
df_x_scaled = df_x
print(df_x_scaled.columns)
df_x_scaled[scalable_vars] = ss.fit_transform(df_x_scaled[scalable_vars])

x_train, x_test, y_train, y_test = train_test_split(df_x_scaled, df_y, random_state=0, shuffle=True,  test_size=0.3)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
model = LogisticRegression(solver='lbfgs',
                           max_iter=500,
                           random_state=0)

print(np.isnan(x_train).sum())

model.fit(x_train, y_train)
pred = model.predict(x_test)

print(np.unique(pred, return_counts=True))

from sklearn.metrics import precision_score, classification_report, recall_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score

print('accuracy is : %.2f'%accuracy_score(y_test, pred))
'''we're getting a accuracy of 81% is fairly decent but can be deceiving as there is imbalanced class
lets check precision...'''

print('precision is : %.2f'%precision_score(y_test, pred))

''' Precision of 66% '''

'''lets check the confusion matrix to reconfirm that...'''

print(confusion_matrix(y_test, pred, labels=[1, 0]))
print(classification_report(y_test, pred, labels=[1, 0]))
print('recall is : %.2f'%recall_score(y_test, pred))
print('f1_score is : %.2f'%f1_score(y_test, pred))

'''so we see that our ability to predict 1's or defaulters is bad because our recall and f1_scores are low 

this sort of means our model can classify 0's correctly but misinterprets 1's(defaulters) due to uneven classes' widths.'''

from sklearn.model_selection import cross_val_score
# from sklearn import metrics

scores = cross_val_score(model, x_test, y_test, cv=10)
print('Cross_validation score on test: %.2f'%scores.mean())
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""


'''lets balance the class widths using under sampling'''

no_of_fraud = len(df[df['default_payment_next_month']==1])
no_fraud_index = df[df['default_payment_next_month']==0].index
random_indices = np.random.choice(no_fraud_index, no_of_fraud, replace=False)
fraud_index = df[df['default_payment_next_month']==1].index
under_sampling_index = np.concatenate([fraud_index, random_indices])
under_sampling =df.loc[under_sampling_index]
print('even class is  {}'.format(under_sampling['default_payment_next_month'].value_counts()))

df_x = under_sampling.drop(columns='default_payment_next_month')
df_x_scaled = df_x
df_x_scaled[scalable_vars] = ss.fit_transform(df_x_scaled[scalable_vars])
df_y = under_sampling['default_payment_next_month']
print(df_y.describe())
x_train, x_test, y_train, y_test = train_test_split(df_x_scaled, df_y, random_state=10, shuffle=True,  test_size=0.3)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
model = LogisticRegression(solver='lbfgs',
                           max_iter=500,
                           random_state=10)

model.fit(x_train, y_train)
pred = model.predict(x_test)

print('accuracy is : %.2f'%accuracy_score(y_test, pred))
print('precision is : %.2f'%precision_score(y_test, pred))
print(confusion_matrix(y_test, pred, labels=[1, 0]))

print(classification_report(y_test, pred, labels=[1, 0]))
print(recall_score(y_test, pred))
print(f1_score(y_test, pred))
scores = cross_val_score(model, x_test, y_test, cv=10)
print('Cross_validation score on test: %.2f'%scores.mean())

'''here our accuracy has gone low but our precision increased and so did our recall and f1_score'''
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

'''lets try oversampling this time..'''
from imblearn.over_sampling import SMOTE

df_x = df.drop(columns='default_payment_next_month')
df_x_scaled = df_x
df_x_scaled[scalable_vars] = ss.fit_transform(df_x_scaled[scalable_vars])
df_y = df['default_payment_next_month']
x_train, x_test, y_train, y_test = train_test_split(df_x_scaled, df_y, random_state=10, shuffle=True,  test_size=0.3)

sm = SMOTE(random_state=10)
x_train_res, y_train_res = sm.fit_sample(x_train, y_train.ravel())
print('the balanced classes are {}'.format(np.unique(y_train_res, return_counts=True)))
model= LogisticRegression(max_iter=500, solver='lbfgs', random_state=10, class_weight='balanced')
model.fit(x_train_res, y_train_res)
pred = model.predict(x_test)
print('accuracy is : %.2f'%accuracy_score(y_test, pred))
print('precision is : %.2f'%precision_score(y_test, pred))
print(classification_report(y_test, pred))
scores = cross_val_score(model, x_test, y_test, cv=10)
'''here we maybe overfitting by oversampling using SMOTE as we are making defaulters prediction by populating defaulters case based on our training data'''
print('recall is : %.2f'%recall_score(y_test, pred))
print('f1_score is : %.2f'%f1_score(y_test, pred))

print('Cross_validation score on test: %.2f'%scores.mean())


''' here our accuracy has increased but precision is less that 0.5 and recall and f1_score for defaulters is also low '''

'''So finally the best method is to use underspampling and then apply logistic_regression for uneven class width of the the dataset.'''