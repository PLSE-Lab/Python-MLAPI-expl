# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

pd.set_option('display.max_colwidth', -1)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

#import data
df = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')
#print(df.describe())
#print(df.info())
print("# of NaN in each columns:", df.isnull().sum(), sep='\n')
print(df.columns)
df.is_canceled.mean()

#create new columns
df['hotel'] = np.where(df['hotel'] == 'Resort Hotel', 1, 0)
df['children'].fillna(0, inplace=True)
df['country'].fillna('NA', inplace=True)
print("# of NaN in each columns:", df.isnull().sum(), sep='\n')
df['family'] = np.where((df.adults * df.children != 0)| (df.adults * df.babies != 0), 1, 0)
print(df[df.adr <= 0].is_canceled.mean())
# # of ADR <= 0 : 1960
df['total_nights'] = df.stays_in_weekend_nights + df.stays_in_week_nights
df['living_expense'] = df['adr'] * df['total_nights']
df['is_free'] = np.where(df['living_expense'] <= 0, 1, 0)
df['by_agent'] = np.where(df['agent'].isnull(), 0, 1)
df['deposit'] = np.where((df['deposit_type'] == 'No Deposit') | (df['deposit_type'] == 'Refundable'), 0, 1)
df['arrival_date_month'] = df['arrival_date_month'].map({'January':1, 'February': 2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7,
                                                            'August':8, 'September':9, 'October':10, 'November':11, 'December':12})

#encoding
le = LabelEncoder()
df['country'] = le.fit_transform(df['country'])
df['country'] = pd.Categorical(df['country'])
df['agent'] = pd.Categorical(df['agent'])
df['arrival_date_week_number'] = pd.Categorical(df['arrival_date_week_number'])
df['arrival_date_day_of_month'] = pd.Categorical(df['arrival_date_day_of_month'])
df['arrival_date_year'] = pd.Categorical(df['arrival_date_year'])
df['arrival_date_month'] = pd.Categorical(df['arrival_date_month'])


df = pd.get_dummies(data = df, columns = ['meal', 'market_segment', 'distribution_channel','reserved_room_type', 'assigned_room_type', 'customer_type'])

#correlation
pd.set_option('display.max_rows', None)
pd.DataFrame(df.corr(method = 'pearson').is_canceled.sort_values(ascending=False).head(100))
df_m = df.copy()
df_m = df_m.drop(columns = ['adr', 'total_nights', 'agent', 'deposit_type', 'reservation_status_date', 'reservation_status', 'company'])
df_corr = pd.DataFrame(df_m.corr())
df_corr.head(72)

df_corr.is_canceled.sort_values()


#modeling: logistic Regression
y = df_m['is_canceled']
X = df_m.drop(columns = ['is_canceled'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 33)

lr = LogisticRegression(solver='liblinear', penalty='l1')
cv_score = cross_val_score(lr, X, y, cv = 5, scoring='accuracy')
#cv_score.mean()
lr_model = lr.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
print(roc_auc_score(y_test, y_pred))
feature_importance = abs(lr_model.coef_[0])
pd.DataFrame(data = lr_model.coef_[0]*100,
                   columns = ["Importances"],
                   index = X_train.columns).sort_values("Importances", ascending = False)[:20]

#modeling - Random Forest
rf = RandomForestClassifier()
cv_score_rf = cross_val_score(rf, X, y, cv = 5, scoring='accuracy')
#cv_score_rf.mean()
rf_model = rf.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print(roc_auc_score(y_test, y_pred_rf))
pd.DataFrame(data = rf_model.feature_importances_*100,
                   columns = ["Importances"],
                   index = X_train.columns).sort_values("Importances", ascending = False)[:20]

#modeling - SVM
svm = SVC(kernel = 'linear')
cv_score_svm = cross_val_score(svm, X, y, cv = 5, scoring='accuracy')
#cv_score_svm.mean()
svm_model = svm.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
print(roc_auc_score(y_test, y_pred_svm))
pd.DataFrame(data = svm_model.feature_importances_*100,
                   columns = ["Importances"],
                   index = X_train.columns).sort_values("Importances", ascending = False)[:20]

#modeling - xgboost
xgb = XGBClassifier()
cv_score_xgb = cross_val_score(xgb, X, y, cv = 5, scoring='accuracy')
cv_score_xgb.mean()
xgb_model = rf.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print(roc_auc_score(y_test, y_pred_xgb))
pd.DataFrame(data = xgb_model.feature_importances_*100,
                   columns = ["Importances"],
                   index = X_train.columns).sort_values("Importances", ascending = False)[:20]

#submission
X_sol = X_test.copy()
X_sol['hotel'] = np.where(X_sol['hotel'] == 1, 'Resort Hotel', 'City Hotel')

sol = pd.DataFrame(data = {'hotel':X_sol['hotel'], 'booking':y_pred_xgb})
sol.to_csv('solution.csv')