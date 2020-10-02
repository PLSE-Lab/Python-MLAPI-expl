# %% [code]
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% [code]
train_dataset = pd.read_csv('../input/titanic/train.csv')
train_dataset

# %% [code]
original_train_dataset = train_dataset.copy()

# %% [code]
print('# of passengers in original train_data : ' + str(len(train_dataset.index)))

# %% [code]
test_dataset = pd.read_csv('../input/titanic/test.csv')
test_dataset

# %% [code]
print('# of passengers in original test_data : ' + str(len(test_dataset.index)))

# %% [code]
original_test_dataset = test_dataset.copy()

# %% [code]
train_dataset.describe()

# %% [code]
test_dataset.describe()

# %% [code]
train_dataset.info()

# %% [code]
test_dataset.info()

# %% [code]
train_dataset.isnull().sum()

# %% [code]
test_dataset.isnull().sum()

# %% [markdown]
# ### Imputing the missing values

# %% [code]
train_dataset['Embarked'].fillna(train_dataset['Embarked'].mode(), inplace=True)
test_dataset['Fare'].fillna(test_dataset['Fare'].mean(), inplace = True)

# %% [code]
train_dataset.drop(['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)
train_dataset.info()

# %% [code]
test_dataset.drop(['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)
test_dataset.info()

# %% [code]
train_omit = train_dataset.dropna(axis=0)
train_omit.info()

# %% [code]
train_omit = pd.get_dummies(train_omit,drop_first=True)
train_omit

# %% [code]
test_set = pd.get_dummies(test_dataset, drop_first=True)
test_set

# %% [code]
test_set.info()

# %% [code]
### Imputing values for Age in test dataset
test_set['Age'].fillna(round(test_set['Age'].mean()), inplace = True)

# %% [markdown]
# ### Getting Dummies

# %% [code]
#sex=pd.get_dummies(omit_data['Sex'], drop_first=True)
#embark = pd.get_dummies(omit_data['Embarked'], drop_first=True)
#pcl = pd.get_dummies(omit_data['Pclass'], drop_first=True)

# %% [code]
X1 = train_omit.drop(['Survived'],axis='columns', inplace=False)
X1

# %% [code]
Y1 =train_omit['Survived']
Y1.head()

# %% [code]
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# %% [code]
X_train, X_valid, y_train, y_valid = train_test_split(X1,Y1, test_size=0.30, random_state=21)

# %% [code]
X_valid.head()

# %% [code]
X_train.info()

# %% [code]
logmodel1  = LogisticRegression().fit(X_train,y_train)
logmodel1

# %% [code]
lr_predictions1 = logmodel1.predict(X_valid)

# %% [code]
log_mse1 = mean_squared_error(y_valid,lr_predictions1)
log_mse1

# %% [code]
r2_log_train1 = logmodel1.score(X_train, y_train)
r2_log_test1 = logmodel1.score(X_valid, y_valid)
print(r2_log_train1,r2_log_test1)

# %% [code]
residuals1 = y_valid - lr_predictions1
sns.regplot(x=lr_predictions1, y=residuals1, scatter=True, fit_reg=False)
residuals1.describe()

# %% [code]
from sklearn.metrics import classification_report
print(classification_report(y_valid, lr_predictions1))

# %% [code]
from sklearn.metrics import accuracy_score
accuracy_score(y_valid,lr_predictions1)*100

# %% [code]
lr_final_predictions = logmodel1.predict(test_set)

# %% [code]
lr_final_predictions

# %% [code]
survivors = pd.DataFrame(lr_final_predictions, columns = ['Survived'])
len(survivors)
survivors.insert(0, 'PassengerId', original_test_dataset['PassengerId'], True)
survivors

# %% [code]
survivors.to_csv('Titanic_submission.csv', index=False)

# %% [markdown]
# # Random Forest Regressor Trial

# %% [code]
rf = RandomForestRegressor(n_estimators=100,max_features='auto',max_depth=100,min_samples_split=10,min_samples_leaf=4,random_state=1)
rf_model1 = rf.fit(X_train,y_train)

# %% [code]
rf_predictions1 = rf_model1.predict(X_valid)

# %% [code]
rfr_predictions1 = rf_predictions1.round(0).astype(int)
rfr_predictions1

# %% [code]
rf_mse1 = mean_squared_error(y_valid,rfr_predictions1)
rf_mse1

# %% [code]
r2_rf_train1 = rf_model1.score(X_train, y_train)
r2_rf_test1 = rf_model1.score(X_valid, y_valid)
print(r2_rf_train1,r2_rf_test1)

# %% [code]
residuals2 = y_valid - rf_predictions1
sns.regplot(x=rf_predictions1, y=residuals2, scatter=True, fit_reg=False)
residuals1.describe()

# %% [code]
print(classification_report(y_valid, rfr_predictions1))

# %% [code]
accuracy_score(y_valid,rfr_predictions1)*100

# %% [code]
rf_final_predictions = rf_model1.predict(test_set)
rfr_final_predictions = rf_final_predictions.round(0).astype(int)
rfr_final_predictions

# %% [code]
survivors_list = pd.DataFrame(rfr_final_predictions, columns = ['Survived'])
len(survivors_list)
survivors_list.insert(0, 'PassengerId', original_test_dataset['PassengerId'], True)
survivors_list

# %% [code]
survivors_list.to_csv('Titanic__Submission.csv', index=False)
