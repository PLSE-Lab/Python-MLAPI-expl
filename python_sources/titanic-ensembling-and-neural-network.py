#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


X_full_path = '../input/titanic/train.csv'
X_full = pd.read_csv(X_full_path, index_col='PassengerId')
X = X_full.copy()


# In[ ]:


X_full.describe()


# # **DATA PREPROCESSING**

# In[ ]:


X_full.columns


# In[ ]:


#Identifying columns with missing values
cols_with_missing = [col for col in X_full if X_full[col].isnull().any()]
cols_with_missing


# In[ ]:


#Counting the number of missing values for each column with missing values
for col in cols_with_missing:
    missing_values = X_full[col].isnull().sum()
    print(col+' : '+str(missing_values))


# In[ ]:


print(X_full['Name'].nunique())
print(X_full['Ticket'].nunique())


# In[ ]:


#Cabin has a lot of missing values, Name has all unique values, Ticket has most of the values unique.
#Hence it is highly unlikely that these would have any considerable effect on the target.
#Thus dropping these columns seems better.
X_full.drop(['Cabin','Name','Ticket'], axis=1, inplace=True)


# In[ ]:


#Identifying columns with categorical data 
object_cols = [col for col in X_full if X_full[col].dtypes=='object']
object_cols


# In[ ]:


pd.set_option('display.max_columns', None)


# In[ ]:


X_full.head(20)


# In[ ]:


#Transforming the Sex column values into numerical values
X_full=X_full.assign(is_male=(X_full['Sex']=='male').astype(int))
X_full.drop(['Sex'], axis=1, inplace=True)
X_full.head()


# In[ ]:


#Identifying columns with missing values post preprocessing
cols_with_missing_postpre = [col for col in X_full if X_full[col].isnull().any()]
cols_with_missing_postpre


# In[ ]:


#Separating target from features
y = X_full.Survived
X_full.drop(['Survived'], axis=1, inplace=True)


# In[ ]:


#Splitting data into training and validation set
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, test_size=0.2, random_state=0)


# In[ ]:


#Imputing missing values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='most_frequent')

X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train))
X_valid_imputed = pd.DataFrame(imputer.transform(X_valid))

X_train_imputed.index = X_train.index
X_valid_imputed.index = X_valid.index

X_train_imputed.columns = X_train.columns
X_valid_imputed.columns = X_valid.columns

X_train_imputed.head()


# In[ ]:


#Applying CatBoost to Embarked column
import category_encoders as ce

cat_feature = ['Embarked']

cat_boost = ce.CatBoostEncoder(cols=cat_feature)
cat_boost.fit(X_train_imputed[cat_feature], y_train)

X_train_imputed = X_train_imputed.join(cat_boost.transform(X_train_imputed[cat_feature]).add_suffix('_cb'))
X_valid_imputed = X_valid_imputed.join(cat_boost.transform(X_valid_imputed[cat_feature]).add_suffix('_cb'))


# In[ ]:


X_train_imputed.drop(['Embarked'], axis=1, inplace=True)
X_train_imputed.head()


# In[ ]:


X_valid_imputed.drop(['Embarked'], axis=1, inplace=True)
X_valid_imputed.head()


# In[ ]:


#Converting Datatypes to int,float and bool
int_feature = ['Pclass', 'Age', 'SibSp', 'Parch']
float_feature = ['Fare', 'Embarked_cb']
bool_feature = ['is_male']

X_train_imputed[int_feature] = X_train_imputed[int_feature].astype(int)
X_valid_imputed[int_feature] = X_valid_imputed[int_feature].astype(int)

X_train_imputed[float_feature] = X_train_imputed[float_feature].astype(float)
X_valid_imputed[float_feature] = X_valid_imputed[float_feature].astype(float)

X_train_imputed[bool_feature] = X_train_imputed[bool_feature].astype(bool)
X_valid_imputed[bool_feature] = X_valid_imputed[bool_feature].astype(bool)


# In[ ]:


#Loading test data
X_test_path = '../input/titanic/test.csv'
X_test = pd.read_csv(X_test_path, index_col='PassengerId')


# In[ ]:


#Preprocessing test data
X_test.drop(['Cabin','Name','Ticket'], axis=1, inplace=True)
X_test = X_test.assign(is_male = (X_test['Sex']=='male').astype(int))
X_test.drop(['Sex'], axis=1, inplace=True)
X_test_imputed = pd.DataFrame(imputer.transform(X_test))
X_test_imputed.columns = X_test.columns
X_test_imputed.index = X_test.index
X_test_imputed=X_test_imputed.join(cat_boost.transform(X_test_imputed[cat_feature]).add_suffix('_cb'))
X_test_imputed.drop(['Embarked'], axis=1, inplace=True)

int_feature = ['Pclass', 'Age', 'SibSp', 'Parch']
float_feature = ['Fare', 'Embarked_cb']
bool_feature = ['is_male']

X_test_imputed[int_feature] = X_test_imputed[int_feature].astype(int)

X_test_imputed[float_feature] = X_test_imputed[float_feature].astype(float)

X_test_imputed[bool_feature] = X_test_imputed[bool_feature].astype(bool)

cols_in_test = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'is_male', 'Embarked_cb']
for col in cols_in_test:
    print(X_test_imputed[col].dtypes)
    
X_test_imputed.head()


# # **DATA VISUALISATION**

# In[ ]:


pd.plotting.register_matplotlib_converters
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


sns.barplot(x=X['Pclass'], y=X['Survived'])
plt.title('Relationship Between Passenger Class and Survival')
plt.xlabel('Passenger Class')


# In[ ]:


sns.barplot(x=X['Embarked'], y=X['Survived'])
plt.title('Relationship Between Port of Embarkation and Survival')
plt.xlabel('Port of Embarkation')


# In[ ]:


sns.barplot(x=X['Sex'], y=X['Survived'])
plt.title('Relationship Between Gender and Survival')
plt.xlabel('Gender')


# In[ ]:


X['Embarked'].unique()


# In[ ]:


def percentSurvived(total, survived):
    return (len(survived)/len(total))*100


# In[ ]:


S_total = X[X['Embarked']=='S']
S_survived = X[(X['Embarked']=='S') & (X['Survived']==1)]

C_total = X[X['Embarked']=='C']
C_survived = X[(X['Embarked']=='C') & (X['Survived']==1)]

Q_total = X[X['Embarked']=='Q']
Q_survived = X[(X['Embarked']=='Q') & (X['Survived']==1)]

print(percentSurvived(S_total, S_survived))
print(percentSurvived(C_total, C_survived))
print(percentSurvived(Q_total, Q_survived))


# # **Ensembling**

# In[ ]:


from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

lgb_model = LGBMClassifier(n_estimators=800, learning_rate=0.05)
xgb_model = XGBClassifier(n_estimators=1000, learning_rate=0.01)
rf_model = RandomForestClassifier(n_estimators=500)

lgb_model.fit(X_train_imputed, y_train)
xgb_model.fit(X_train_imputed, y_train)
rf_model.fit(X_train_imputed, y_train)

print("LightGBM Score     : ",lgb_model.score(X_valid_imputed, y_valid))
print("XGBoost Score      : ",xgb_model.score(X_valid_imputed, y_valid))
print("RandomForests Score: ",rf_model.score(X_valid_imputed, y_valid))


# In[ ]:


#Stacking Classifiers
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

estimators=[('lgb', lgb_model), ('xgb', xgb_model), ('rf', rf_model)]
clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
clf.fit(X_train_imputed, y_train)
print("Ensemble Score: ", clf.score(X_valid_imputed, y_valid))


# #### Note:- The Ensemble scored 0.77511 in the competition.

# # **Neural Network**

# In[ ]:


#Scaling the data for MLPClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train_imputed)

X_train_transformed = scaler.transform(X_train_imputed)
X_valid_transformed = scaler.transform(X_valid_imputed)


# In[ ]:


#Training the model
from sklearn.neural_network import MLPClassifier

mlp_model = MLPClassifier(hidden_layer_sizes=(150, 100, 50), max_iter=1000)
mlp_model.fit(X_train_transformed, y_train)
print("MLP Score: ", mlp_model.score(X_valid_transformed, y_valid))


# #### Note:- The MLPClassifier scored 0.76555 in the competition.

# # **MAKING PREDICTIONS WITH TEST DATA**

# In[ ]:


test_preds = clf.predict(X_test_imputed)


# In[ ]:


test_preds


# In[ ]:


output = pd.DataFrame({
        "PassengerId": X_test_imputed.index,
        "Survived": test_preds
    })


# In[ ]:


output


# In[ ]:


output.to_csv('submission.csv', index=False)

