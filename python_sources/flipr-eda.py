#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df_train = pd.read_excel('../input/flipr-hiring-challenge/Train_dataset.xlsx')
df_test = pd.read_excel('../input/flipr-hiring-challenge/Test_dataset.xlsx')
df_train.head(14)


# In[ ]:


df_train.describe()


# In[ ]:


df_train.info()


# In[ ]:


df_train.isna().sum()


# In[ ]:


for col in df_train.columns:
    print("col-name: ", col, " | no_of_unique_values: ", df_train[col].nunique(dropna=True))


# There are many columns having empty values, so we will impute them accordingly. These columns are - name, children, occupation, mode_transport, comorbidity, cardiological pressure, Diuresis, Platelets, HBB, d-dimer, Heart rate, HDL_cholesterol, Insurance, FT/month

# But first lets drop some unneccessary columns that we know will not contribute to our model selection like name, etc. The followings columns are removed: name, designation because irrespective of the values, it just cannot decide the infection probability

# In[ ]:


df_train = df_train.drop(['Designation', 'Name'], axis = 1)
df_train.head(10)


# In[ ]:


# import pandas_profiling
# df_train.profile_report()


# In[ ]:


df_train.Infect_Prob[(df_train['Infect_Prob']>45.0) & (df_train['Infect_Prob']<55.0)].count()


# 80% of probability resides between 45 and 55 probability

# ## Encoding non-null categorical features before imputation
# 
# For gender, married and region- gender and married have binary values, we will encode with LabelEncoder, while one-hot encoding for region

# In[ ]:


from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

lb = LabelBinarizer()
df_train['Gender'] = lb.fit_transform(df_train.Gender)
df_train['Married'] = lb.fit_transform(df_train.Married)


df_train.head(5)


# ## Imputing missing values

# Now, we have to impute the missing values in several columns.
# 
# The method of imputation will vary according to the feature. So we need to visualize how the 

# In[ ]:


import missingno as msno
msno.matrix(df_train)


# In[ ]:


msno.heatmap(df_train)


# This is map showing nullity correlation:
# 
# It can be seen that individuals having children have some affinity towards having an insurance and have an occupation.
# There is high relation between d-dimer and heart-rate, and also platelets and d-dimer | heart-rate.
# 
# So, we will use certain viz for imputations

# ### Imputing feature 'children'

# In[ ]:


df_train.Children.isna().sum()


# In[ ]:


df_train.Married[(df_train['Married']==0.0) & (df_train['Children'].isna()==1)].count()


# It means that there is no row such that married column is 1 for missing values of Children. Therefore we can atleast place value in children as 0 for the individuals who are no married 

# In[ ]:


def impute_children(cols):
    children = cols[0]
    married = cols[1]
    
    if np.isnan(children):
        if married == 0.0:
            return 0
        else:
            return children
    else:
        return children

df_train['Children'] = df_train[['Children', 'Married']].apply(impute_children, axis=1)          
    


# In[ ]:


df_train.Children.isna().sum()


# ### Imputing feature 'Occupation'

# In[ ]:


df_train.salary[df_train['Occupation'].isna()].head()


# So there are persons whose occupation is not known but salary is non-empty.

# In[ ]:


sns.boxplot(x = "Occupation", y = "salary", hue='Gender', data = df_train)


# In[ ]:


sns.countplot(x = "Occupation", hue = "Gender", data = df_train)


# In[ ]:


sns.countplot(x = "Occupation", data = df_train)


# All Occupations have moraless equal number of males and females.
# Encoding the occupation feature.

# In[ ]:


sns.boxplot(x='Occupation', y='salary', data=df_train)


# In[ ]:


sns.scatterplot(x='Occupation', y='salary', data=df_train)


# All occupations have same range of income

# In[ ]:


sns.boxplot(x='Occupation', y='Infect_Prob', data=df_train)


# In[ ]:


sns.boxplot(x='Occupation', y='Charlson Index',hue='Gender', data=df_train)


# In[ ]:


sns.boxplot(x='Occupation', y='HDL cholesterol',hue='Gender', data=df_train)


# It seems that all occupation have been eually affected and no particular occupation is contributing to the prediction

# ### Imputing mode_transport

# In[ ]:


df_train[df_train.Mode_transport.isna()].head()


# In[ ]:


sns.countplot(x='Mode_transport', data=df_train)


# since public transport is the mode of the class, we fill 3 empty cells with Public

# In[ ]:


df_train['Mode_transport'] = df_train.Mode_transport.fillna('Public')
df_train.Mode_transport.isna().sum()


# ### Imputing feature comorbidity

# In[ ]:


sns.countplot(x='comorbidity', data=df_train)


# In[ ]:


# sns.catplot(x="comorbidity", y='salary', hue="Gender", kind='swarm', data=df_train)


# Unknown comorbidity means that they didn't know about their illness. Replacing with None

# In[ ]:


df_train['comorbidity'] = df_train.comorbidity.fillna('None')
df_train.comorbidity.isna().sum()


# ### Imputing feature 'cardiological-pressure'

# In[ ]:


sns.countplot(x='cardiological pressure', data=df_train)


# In[ ]:


sns.boxplot(x='cardiological pressure', y='HDL cholesterol',hue='Gender', data=df_train)


# In[ ]:


sns.boxenplot(x='Gender', y='Age',hue='cardiological pressure', data=df_train)


# In[ ]:


sns.boxplot(x='cardiological pressure', y='Infect_Prob', data=df_train)


# It seems that all four comorbidity equally contribute to the infection probability. Replacing with Normal

# In[ ]:


df_train['cardiological pressure'] = df_train['cardiological pressure'].fillna('Normal')
df_train['cardiological pressure'].isna().sum()


# ### Imputing Diuresis, Platelets, HBB, d-dimer,	Heart rate,	HDL cholesterol
# 

# Since they are continuous values we will replace them by their mean

# In[ ]:


df_train['Diuresis'] = df_train['Diuresis'].fillna(df_train['Diuresis'].mean())
df_train['Diuresis'].isna().sum()


# In[ ]:


df_train['Platelets'] = df_train['Platelets'].fillna(df_train['Platelets'].mean())
df_train['Platelets'].isna().sum()


# In[ ]:


df_train['HBB'] = df_train['HBB'].fillna(df_train['HBB'].mean())
df_train['HBB'].isna().sum()


# In[ ]:


df_train['d-dimer'] = df_train['d-dimer'].fillna(df_train['d-dimer'].mean())
df_train['d-dimer'].isna().sum()


# In[ ]:


df_train['Heart rate'] = df_train['Heart rate'].fillna(df_train['Heart rate'].mean())
df_train['Heart rate'].isna().sum()


# In[ ]:


df_train['HDL cholesterol'] = df_train['HDL cholesterol'].fillna(df_train['HDL cholesterol'].mean())
df_train['HDL cholesterol'].isna().sum()


# In[ ]:


df_train.head()


# ### Imputing Insurance

# In[ ]:


df_train[(df_train.Occupation.isna()) & (df_train.Insurance.isna())].count()


# In[ ]:


df_train[~(df_train.Occupation.isna()) & (df_train.Insurance.isna())].count()


# It seems that both occupation and insurance are not null at the same time. So, we can impute insurance with the mean of each ocuupation

# In[ ]:


means_ins = df_train.groupby('Occupation')['Insurance'].mean()
mean_ins = df_train.Insurance.mean()
print(mean_ins)
means_ins


# In[ ]:


means_ins['Cleaner']


# In[ ]:


def impute_insurance(cols):
    occ = cols[0]
    ins = cols[1]

    if pd.isnull(ins):
        if not pd.isnull(occ):
            ins = means_ins[str(occ)]
            return ins 
        else:
            return mean_ins
    else:
        return ins

    
df_train['Insurance'] = df_train[['Occupation', 'Insurance']].apply(impute_insurance, axis=1)
df_train.Insurance.isna().sum()


# ### Imputing feature FT/month

# In[ ]:


sns.boxenplot(x='FT/month', y='salary', data=df_train)


# In[ ]:


df_train['FT/month'] = df_train['FT/month'].fillna(df_train['FT/month'].median())
df_train['FT/month'].isna().sum()


# imputings are done Lets vizualize correlation and do hypothesis testing

# ## Categorical encoding
# encoding region, occupation, mode_transport, comorbidity

# In[ ]:


sns.countplot(x='Region', data=df_train)


# In[ ]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df_train['Region']= le.fit_transform(df_train['Region'])
# df_train['Occupation']= le.fit_transform(df_train['Occupation'])
df_train['Mode_transport']= le.fit_transform(df_train['Mode_transport'])
df_train['comorbidity']= le.fit_transform(df_train['comorbidity'])

df_train.head()


# In[ ]:


sns.boxenplot(x='Diuresis', y='Infect_Prob', hue='Gender', data=df_train)


# In[ ]:


sns.boxenplot(x='Region', y='Infect_Prob', data=df_train)


# In[ ]:


sns.boxenplot(x='Occupation', y='Mode_transport', data=df_train)


# In[ ]:


sns.countplot(x='Occupation', hue='Region', data=df_train)


# In[ ]:


sns.countplot(x='Occupation', hue='Mode_transport', data=df_train)


# In[ ]:


# sns.pairplot(df_train)


# In[ ]:


# for i, col in enumerate(df_train.columns):
#     if not col in ['Occupation', 'people_ID','Pulmonary score','cardiological pressure','Diuresis','Platelets','HBB','d-dimer','Heart rate','HDL cholesterol','Charlson Index','Blood Glucose', 'Infect_Prob']:
#         plt.figure(i)
#         sns.countplot(x=col, hue='Occupation', data=df_train)


# Dropping Occupation column

# In[ ]:


df_train = df_train.drop(['Occupation'], axis=1)
df_train.head()


# Processing pulmonary score

# In[ ]:


df_train['Pulmonary score'] = df_train['Pulmonary score'].str.replace('<', '')
df_train['Pulmonary score'].head()


# In[ ]:


df_train.head()


# Since Region has high collinearity with cases/1M and Deaths/1M and cases and death is too linearly related (conclusion drawn from paiwise plot), we will drop Region and Deaths/1M

# In[ ]:


df_train.drop(['Region', 'Deaths/1M'],axis=1, inplace=True)


# In[ ]:


df_train = pd.concat([df_train,pd.get_dummies(df_train['Mode_transport'], prefix='Mode_transport')],axis=1)
df_train = pd.concat([df_train,pd.get_dummies(df_train['comorbidity'], prefix='comorbidity')],axis=1)
df_train.drop(['Mode_transport', 'comorbidity'],axis=1, inplace=True)
df_train.head()


# In[ ]:


col_list = df_train.columns
col_list = ['people_ID','Mode_transport_0','Mode_transport_1','Mode_transport_2', 'Gender', 'Married', 'Children',
       'cases/1M','comorbidity_0','comorbidity_1','comorbidity_2','comorbidity_3', 'Age',
       'Coma score', 'Pulmonary score', 'cardiological pressure', 'Diuresis',
       'Platelets', 'HBB', 'd-dimer', 'Heart rate', 'HDL cholesterol',
       'Charlson Index', 'Blood Glucose', 'Insurance', 'salary', 'FT/month',
       'Infect_Prob']
df_train = df_train[col_list]
df_train.head()


# In[ ]:


df_train = pd.concat([df_train,pd.get_dummies(df_train['cardiological pressure'], prefix='cardiological pressure')],axis=1)
df_train.drop(['cardiological pressure'],axis=1, inplace=True)
df_train.head()


# In[ ]:


col_list = df_train.columns
col_list = ['people_ID','Mode_transport_0','Mode_transport_1','Mode_transport_2', 'Gender', 'Married', 'Children',
       'cases/1M','comorbidity_0','comorbidity_1','comorbidity_2','comorbidity_3', 'Age',
       'Coma score', 'Pulmonary score', 'cardiological pressure_Elevated','cardiological pressure_Normal','cardiological pressure_Stage-01',
       'cardiological pressure_Stage-02', 'Diuresis',
       'Platelets', 'HBB', 'd-dimer', 'Heart rate', 'HDL cholesterol',
       'Charlson Index', 'Blood Glucose', 'Insurance', 'salary', 'FT/month',
       'Infect_Prob']
df_train = df_train[col_list]
df_train.head()


# ### Normalizing or Standardizing the features

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
column_names_to_normalize = ['Age','Coma score', 'Pulmonary score', 'Diuresis',
       'Platelets', 'HBB', 'd-dimer', 'Heart rate', 'HDL cholesterol',
       'Charlson Index', 'Blood Glucose', 'Insurance', 'salary']
x = df_train[column_names_to_normalize].values
min_max_scaler=MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize, index = df_train.index)
df_train[column_names_to_normalize] = df_temp

df_train.head()


# ## Benchmark Model

# In[ ]:


df_train['Target_norm'] = df_train["Infect_Prob"]/100.0
df_train.head()


# In[ ]:


from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor 


# In[ ]:


X = df_train.iloc[:, 1:-2]
Y = df_train.iloc[:, -2]
Y_norm = df_train.iloc[:, -1]
Y.count()


# In[ ]:


X.head()


# In[ ]:


Y_norm.dtype


# In[ ]:


sns.distplot(Y)


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.02, random_state = 42)


# In[ ]:


X_train, X_test, Y_n_train, Y_n_test = train_test_split(X, Y_norm, test_size = 0.02, random_state = 42)


# In[ ]:


rf = RandomForestRegressor(criterion='mse', 
                             n_estimators=500,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
rf.fit(X_train, Y_n_train)
print("%.4f" % rf.oob_score_)


# In[ ]:


pd.concat((pd.DataFrame(X_train.columns, columns = ['variable']), 
           pd.DataFrame(rf.feature_importances_, columns = ['importance'])), 
          axis = 1).sort_values(by='importance', ascending = False)


# In[ ]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)


# In[ ]:


regressor.predict(X_test)
regressor.score(X_test,Y_test)


# In[ ]:


import statsmodels.api as sm
X_opt = X.iloc[:,:]
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_OLS.summary()


# Implementing XG_Boost

# In[ ]:


X_train = df_train.iloc[:,1:-2]
X_train.head()


# In[ ]:


Y_n_train = df_train.iloc[:,-1]
Y_n_train.head()


# In[ ]:


# def algorithm_pipeline(X_train_data, X_test_data, y_train_data, y_test_data, 
#                        model, param_grid, cv=10, scoring_fit='neg_mean_squared_error',
#                        do_probabilities = False):
#     gs = GridSearchCV(
#         estimator=model,
#         param_grid=param_grid, 
#         cv=cv, 
#         n_jobs=-1, 
#         scoring=scoring_fit,
#         verbose=2
#     )
#     fitted_model = gs.fit(X_train_data, y_train_data)
    
#     if do_probabilities:
#       pred = fitted_model.predict_proba(X_test_data)
#     else:
#       pred = fitted_model.predict(X_test_data)
    
#     return fitted_model, pred


# In[ ]:


import xgboost
from sklearn.model_selection import GridSearchCV

# # Let's try XGboost algorithm to see if we can get better results
# xgb = xgboost.XGBRegressor()
# param_grid = {
#     'n_estimators': [400, 500, 600,700,800],
#     'learning_rate': [0.002,0.008, 0.02, 0.4, 0.2],
#     'colsample_bytree': [0.3, 0.4,0.5,0.6,0.7],
#     'max_depth': [30,40,50,70,100],
#     'reg_alpha': [1.1, 1.2, 1.3],
#     'reg_lambda': [1.1, 1.2, 1.3],
#     'subsample': [0.7, 0.8, 0.9]
# }


# In[ ]:


xgb = xgboost.XGBRegressor(n_estimators=500, learning_rate=0.02, gamma=0, subsample=0.75,alpha=10,
                           colsample_bytree=0.3, max_depth=100)


# In[ ]:


# xgb, pred = algorithm_pipeline(X_train, X_test, Y_n_train, Y_n_test, xgb, 
#                                  param_grid, cv=5)


# In[ ]:


# print(np.sqrt(-xgb.best_score_))
# print(xgb.best_params_)


# In[ ]:


xgb.fit(X_train,Y_n_train,eval_metric='rmsle')


# In[ ]:


from sklearn.metrics import mean_squared_error
predictions = xgb.predict(X_test)
mse = mean_squared_error(predictions,Y_n_test)
print(np.sqrt(mse))


# In[ ]:


predictions = predictions*100.0


# In[ ]:


sns.distplot(predictions)


# In[ ]:


predictions


# # prediction time 

# In[ ]:


df_test = pd.read_excel('../input/flipr-hiring-challenge/Test_dataset.xlsx')
df_test.head(14)


# In[ ]:


df_test.describe()


# In[ ]:


df_test.info()


# In[ ]:


df_test.isna().sum()


# In[ ]:


for col in df_test.columns:
    print("col-name: ", col, " | no_of_unique_values: ", df_test[col].nunique(dropna=True))


# In[ ]:


df_test = df_test.drop(['Designation', 'Name'], axis = 1)
df_test.head(10)


# In[ ]:


from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

lb = LabelBinarizer()
df_test['Gender'] = lb.fit_transform(df_test.Gender)
df_test['Married'] = lb.fit_transform(df_test.Married)

df_test.head(5)


# In[ ]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df_test['Region']= le.fit_transform(df_test['Region'])
# df_train['Occupation']= le.fit_transform(df_train['Occupation'])
df_test['Mode_transport']= le.fit_transform(df_test['Mode_transport'])
df_test['comorbidity']= le.fit_transform(df_test['comorbidity'])

df_test.head()


# In[ ]:


df_test = df_test.drop(['Occupation'], axis=1)
df_test.head()


# In[ ]:


df_test['Pulmonary score'] = df_test['Pulmonary score'].str.replace('<', '')
df_test['Pulmonary score'].head()


# In[ ]:


df_test.drop(['Region', 'Deaths/1M'],axis=1, inplace=True)


# In[ ]:


df_test = pd.concat([df_test,pd.get_dummies(df_test['Mode_transport'], prefix='Mode_transport')],axis=1)
df_test = pd.concat([df_test,pd.get_dummies(df_test['comorbidity'], prefix='comorbidity')],axis=1)
df_test.drop(['Mode_transport', 'comorbidity'],axis=1, inplace=True)
df_test.head()


# In[ ]:


col_list = df_test.columns
col_list = ['people_ID','Mode_transport_0','Mode_transport_1','Mode_transport_2', 'Gender', 'Married', 'Children',
       'cases/1M','comorbidity_0','comorbidity_1','comorbidity_2','comorbidity_3', 'Age',
       'Coma score', 'Pulmonary score', 'cardiological pressure', 'Diuresis',
       'Platelets', 'HBB', 'd-dimer', 'Heart rate', 'HDL cholesterol',
       'Charlson Index', 'Blood Glucose', 'Insurance', 'salary', 'FT/month']
df_test = df_test[col_list]
df_test.head()


# In[ ]:


df_test = pd.concat([df_test,pd.get_dummies(df_test['cardiological pressure'], prefix='cardiological pressure')],axis=1)
df_test.drop(['cardiological pressure'],axis=1, inplace=True)
df_test.head()


# In[ ]:


col_list = df_test.columns
col_list = ['people_ID','Mode_transport_0','Mode_transport_1','Mode_transport_2', 'Gender', 'Married', 'Children',
       'cases/1M','comorbidity_0','comorbidity_1','comorbidity_2','comorbidity_3', 'Age',
       'Coma score', 'Pulmonary score', 'cardiological pressure_Elevated','cardiological pressure_Normal','cardiological pressure_Stage-01',
       'cardiological pressure_Stage-02', 'Diuresis',
       'Platelets', 'HBB', 'd-dimer', 'Heart rate', 'HDL cholesterol',
       'Charlson Index', 'Blood Glucose', 'Insurance', 'salary', 'FT/month']
df_test = df_test[col_list]
df_test.head()


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
column_names_to_normalize = ['Age','Coma score', 'Pulmonary score', 'Diuresis',
       'Platelets', 'HBB', 'd-dimer', 'Heart rate', 'HDL cholesterol',
       'Charlson Index', 'Blood Glucose', 'Insurance', 'salary']
x = df_test[column_names_to_normalize].values
min_max_scaler=MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize, index = df_test.index)
df_test[column_names_to_normalize] = df_temp

df_test.head()


# In[ ]:


test = df_test.iloc[:,1:]
test.head()


# In[ ]:


pred = xgb.predict(test)


# In[ ]:


pred = pred*100.0


# In[ ]:


pred


# In[ ]:


sns.distplot(pred)


# In[ ]:


submission = pd.read_excel('../input/flipr-hiring-challenge/Test_dataset.xlsx')
submission['infect_prob_20'] = pd.Series(pred)
submission.head()


# In[ ]:


pd.DataFrame(submission, columns=['people_ID', 'infect_prob_20']).to_csv('submission.csv', index = False)


# In[ ]:


sns.scatterplot(x='Diuresis', y='Infect_Prob', data=df_train)


# In[ ]:




