#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns               # Provides a high level interface for drawing attractive and informative statistical graphics
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()
from subprocess import check_output

import warnings                                            # Ignore warning related to pandas_profiling
warnings.filterwarnings('ignore') 

def annot_plot(ax,w,h):                                    # function to add data to plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for p in ax.patches:
         ax.annotate(f"{p.get_height() * 100 / df.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
         ha='center', va='center', fontsize=11, color='black', rotation=0, xytext=(0, 10),
         textcoords='offset points')             
def annot_plot_num(ax,w,h):                                    # function to add data to plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for p in ax.patches:
        ax.annotate('{0:.1f}'.format(p.get_height()), (p.get_x()+w, p.get_height()+h))

import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv('../input/income_evaluation.csv')


# In[ ]:


df.head()


# In[ ]:


df.nunique()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.info()


# In[ ]:


df.columns


# In[ ]:


df.columns = df.columns.str.replace(' ', '')


# In[ ]:


new_df = df.copy()


# In[ ]:


new_df.columns


# In[ ]:


new_df.head()


# In[ ]:


ax = sns.countplot(new_df['income'])
plt.xticks(rotation = 30, ha='right')
annot_plot(ax,0.08,1)


# In[ ]:


plt.figure(figsize=(12,6))
ax = sns.countplot(new_df['marital-status'], hue = new_df.income)
plt.xticks(rotation = 30, ha='right')
annot_plot(ax,0.08,1)


# In[ ]:


plt.figure(figsize=(12,6))
ax = sns.countplot(new_df['relationship'], hue = new_df.income)
plt.xticks(rotation = 30, ha='right')
annot_plot(ax,0.08,1)


# In[ ]:


plt.figure(figsize=(12,6))
ax = sns.countplot(new_df['occupation'], hue = new_df.income)
plt.xticks(rotation = 30, ha='right')
annot_plot(ax,0.08,1)


# In[ ]:


plt.figure(figsize=(10,6))
native_country_count = new_df['native-country'].value_counts()[:10].plot(kind='bar')
annot_plot(native_country_count, 0.08,1)


# ### Native_country column

# In[ ]:


native_country_col = pd.get_dummies(new_df['native-country'])
native_country_col.head()


# In[ ]:


native_country_col.columns.values


# In[ ]:


new_df = new_df.drop('native-country', axis = 1)


# In[ ]:


native_country_col.columns.values


# In[ ]:


#reorder_column
cols = [' ?', ' Cambodia', ' Canada', ' China', ' Columbia', ' Cuba',
       ' Dominican-Republic', ' Ecuador', ' El-Salvador', ' England',
       ' France', ' Germany', ' Greece', ' Guatemala', ' Haiti',
       ' Holand-Netherlands', ' Honduras', ' Hong', ' Hungary', ' India',
       ' Iran', ' Ireland', ' Italy', ' Jamaica', ' Japan', ' Laos',
       ' Mexico', ' Nicaragua', ' Outlying-US(Guam-USVI-etc)', ' Peru',
       ' Philippines', ' Poland', ' Portugal', ' Puerto-Rico',
       ' Scotland', ' South', ' Taiwan', ' Thailand', ' Trinadad&Tobago',
       ' Vietnam', ' Yugoslavia',' United-States']


# In[ ]:


country = native_country_col[cols]


# In[ ]:


other_country = country.loc[:,' ?':' Yugoslavia'].max(axis=1)
us_country = country.loc[:,' United-States':].max(axis=1)


# In[ ]:


new_df = pd.concat([new_df,us_country,other_country], axis =1)


# In[ ]:


new_df.head()


# In[ ]:


new_df = new_df.rename(columns = {0:"United_state",1:"Other_country"})


# In[ ]:


new_df.head(2)


# In[ ]:


country_mod = new_df.copy()


# In[ ]:


country_mod.head()


# In[ ]:


new_df.columns


# In[ ]:


reorder_cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
       'marital-status', 'occupation', 'relationship', 'race', 'sex',
       'capital-gain', 'capital-loss', 'hours-per-week','United_state', 
                'Other_country', 'income']


# In[ ]:


country_mod = country_mod[reorder_cols]


# In[ ]:


country_mod.head()


# # Workclass column:

# In[ ]:


country_mod.workclass.unique()


# In[ ]:


country_mod = country_mod.replace(' ?','unknown')


# In[ ]:


plt.figure(figsize=(10,6))
workclass_count = country_mod['workclass'].value_counts().plot(kind='bar')
annot_plot(workclass_count, 0.08,1)


# In[ ]:


workclass_dummies = pd.get_dummies(country_mod['workclass'])


# In[ ]:


workclass_dummies.head()


# In[ ]:


workclass_mod_df = country_mod.copy()


# In[ ]:


workclass_mod_df = workclass_mod_df.drop('workclass', axis = 1)


# In[ ]:


workclass_mod_df.head()


# In[ ]:


workclass_mod_df = pd.concat([workclass_mod_df,workclass_dummies], axis = 1)


# In[ ]:


workclass_mod_df.head()


# In[ ]:


workclass_mod_df.columns


# In[ ]:


#reorder_col
col = ['age', 'fnlwgt', 'education', 'education-num', 'marital-status',
       'occupation', 'relationship', 'race', 'sex', 'capital-gain',
       'capital-loss', 'hours-per-week', 'United_state', 'Other_country',
       ' Federal-gov', ' Local-gov', ' Never-worked', ' Private',
       ' Self-emp-inc', ' Self-emp-not-inc', ' State-gov', ' Without-pay',
       'unknown','income']


# In[ ]:


workclass_mod_df = workclass_mod_df[col]


# In[ ]:


workclass_mod_df.head()


# In[ ]:


workclass_mod_df.nunique()


# # Race Column:

# In[ ]:


plt.figure(figsize=(10,6))
workclass_count = country_mod['race'].value_counts().plot(kind='bar')
annot_plot(workclass_count, 0.08,1)


# In[ ]:


race_dummies = pd.get_dummies(workclass_mod_df['race'])


# In[ ]:


workclass_mod_df = workclass_mod_df.drop('race', axis = 1)


# In[ ]:


race_mod_df = workclass_mod_df.copy()


# In[ ]:


race_dummies.head()


# In[ ]:


race_dummies.columns


# In[ ]:


race_cols = [' Amer-Indian-Eskimo', ' Asian-Pac-Islander', ' Other', ' Black',' White']


# In[ ]:


race_dummies = race_dummies[race_cols]


# In[ ]:


other_country = country.loc[:,' ?':' Yugoslavia'].max(axis=1)
us_country = country.loc[:,' United-States':].max(axis=1)


# In[ ]:


race_dummies.head(15)


# In[ ]:


Others = race_dummies.loc[:,' Amer-Indian-Eskimo':' Other'].max(axis=1)


# In[ ]:


race_dummies = race_dummies.drop([' Amer-Indian-Eskimo',' Asian-Pac-Islander',' Other'], axis = 1)


# In[ ]:


race_dummies = pd.concat([race_dummies,Others], axis = 1)


# In[ ]:


race_dummies.head()


# In[ ]:


race_dummies = race_dummies.rename(columns = {0:"Other_race"})


# In[ ]:


race_dummies.head()


# In[ ]:


race_mod_df = pd.concat([race_mod_df, race_dummies], axis = 1)


# In[ ]:


race_mod_df.head()


# In[ ]:


race_mod_df.columns


# In[ ]:


race_col_reorder = ['age', 'fnlwgt', 'education', 'education-num', 'marital-status',
       'occupation', 'relationship', 'sex', 'capital-gain', 'capital-loss',
       'hours-per-week', 'United_state', 'Other_country', ' Federal-gov',
       ' Local-gov', ' Never-worked', ' Private', ' Self-emp-inc',
       ' Self-emp-not-inc', ' State-gov', ' Without-pay', 'unknown',
       ' Black', ' White', 'Other_race', 'income']


# In[ ]:


race_mod_df = race_mod_df[race_col_reorder]


# In[ ]:


race_mod_df.head()


# # Education Columns

# In[ ]:


Education_df = race_mod_df.copy()


# In[ ]:


Education_df.education.unique()


# In[ ]:


plt.figure(figsize=(10,6))
education_count = Education_df['education'].value_counts().plot(kind='bar')
annot_plot(education_count, 0.08,1)


# In[ ]:


education_dummies = pd.get_dummies(Education_df.education)


# In[ ]:


education_dummies.columns


# In[ ]:


edu_cols_reorder = [' Preschool', ' 1st-4th', ' 5th-6th', ' 7th-8th', ' 9th', ' 10th',
       ' 11th', ' 12th', ' Prof-school', ' Assoc-acdm', ' Assoc-voc',
       ' Bachelors', ' Some-college', ' Doctorate', ' Masters', ' HS-grad']


# In[ ]:


education_dummies = education_dummies[edu_cols_reorder]


# In[ ]:


education_dummies.head()


# In[ ]:


#Others = race_dummies.loc[:,' Amer-Indian-Eskimo':' Other'].max(axis=1)
pre_prof_school = education_dummies.loc[:,' Preschool':' Prof-school'].max(axis=1)


# In[ ]:


assosiate_edu = education_dummies.loc[:,' Assoc-acdm':' Assoc-voc'].max(axis=1)


# In[ ]:


masters_doc = education_dummies.loc[:,' Doctorate':' Masters'].max(axis=1)


# In[ ]:


college = education_dummies.loc[:,' Bachelors':' Some-college'].max(axis=1)


# In[ ]:


Education_df['pre_prof_school'] = pre_prof_school
Education_df['college'] = college
Education_df['masters_doc'] = masters_doc
Education_df['assosiate_edu'] = assosiate_edu


# In[ ]:


Education_df.head()


# In[ ]:


Education_df = Education_df.drop(['education'], axis = 1)


# In[ ]:


Education_df.head()


# # Marital-status col:

# In[ ]:


marital_status_df = Education_df.copy()


# In[ ]:


plt.figure(figsize=(10,6))
marital_status_count = marital_status_df['marital-status'].value_counts().plot(kind='bar')
annot_plot(marital_status_count, 0.08,1)


# In[ ]:


marital_status_reorder = [' Divorced', ' Married-spouse-absent',' Separated', ' Widowed',
                          ' Married-AF-spouse', ' Married-civ-spouse',
                          ' Never-married']


# In[ ]:


marital_status_dummies = pd.get_dummies(marital_status_df['marital-status'])


# In[ ]:


marital_status_dummies = marital_status_dummies[marital_status_reorder]


# In[ ]:


marital_status_dummies.head()


# In[ ]:


marital_status_df['Divorced'] = marital_status_dummies.loc[:, ' Divorced':' Widowed'].max(axis=1)
marital_status_df['Married' ] = marital_status_dummies.loc[:, ' Married-AF-spouse':' Married-civ-spouse'].max(axis=1)
marital_status_df['Never_Married'] = marital_status_dummies[' Never-married']


# In[ ]:


marital_status_df.head()


# In[ ]:


marital_status_df = marital_status_df.drop(['marital-status'], axis = 1)


# In[ ]:


marital_status_df.head()


# # Occupation:

# In[ ]:


occupation_df = marital_status_df.copy()


# In[ ]:


plt.figure(figsize=(10,6))
occupation_count = occupation_df['occupation'].value_counts().plot(kind='bar')
annot_plot(occupation_count, 0.08,1)


# In[ ]:


occupation_dummies = pd.get_dummies(occupation_df['occupation'])


# In[ ]:


occupation_dummies.columns


# In[ ]:


occupation_reorder = [' Adm-clerical', 
                      ' Armed-Forces',' Farming-fishing',' Handlers-cleaners',' Other-service',
                      ' Priv-house-serv',' Protective-serv','unknown',
                      ' Craft-repair', ' Machine-op-inspct',' Tech-support', ' Transport-moving',
                      ' Exec-managerial',' Prof-specialty',
                      ' Sales' ]


# In[ ]:


occupation_dummies = occupation_dummies[occupation_reorder]


# In[ ]:


occupation_dummies.head(2)


# In[ ]:


occupation_df['Group_1_ocp'] = occupation_dummies[' Adm-clerical']
occupation_df['Group_2_ocp'] = occupation_dummies.loc[:,' Armed-Forces':'unknown'].max(axis=1)
occupation_df['Group_3_ocp'] = occupation_dummies.loc[:,' Craft-repair':' Transport-moving'].max(axis=1)
occupation_df['Group_4_ocp'] = occupation_dummies.loc[:,' Exec-managerial':' Prof-specialty'].max(axis=1)
occupation_df['Group_5_ocp'] = occupation_dummies[' Sales']


# In[ ]:


occupation_df = occupation_df.drop(['occupation'], axis = 1)


# In[ ]:


occupation_df.head()


# In[ ]:


occupation_df.columns = occupation_df.columns.str.replace(' ', '')


# In[ ]:


occupation_df.columns


# # Relationship cols:

# In[ ]:


relationship_df = occupation_df.copy()


# In[ ]:


relationship_df.head()


# In[ ]:


plt.figure(figsize=(10,6))
relationship_count = relationship_df['relationship'].value_counts().plot(kind='bar')
annot_plot(relationship_count, 0.08,1)


# In[ ]:


relationship_df = relationship_df.drop(['relationship'], axis=1)


# In[ ]:


relationship_df.head()


# In[ ]:


relationship_df['sex'] = relationship_df['sex'].map({' Male':1, ' Female':0})


# In[ ]:


relationship_df.head()


# # Income col:

# In[ ]:


income_df = relationship_df.copy()


# In[ ]:


income_df.income.unique()


# In[ ]:


income_df['income'] = income_df['income'].map({' <=50K':0, ' >50K':1})


# In[ ]:


plt.figure(figsize=(10,6))
income_count = income_df['income'].value_counts().plot(kind='bar')
annot_plot(income_count, 0.08,1)


# In[ ]:





# # Final DataFrame:

# In[ ]:


final_df = income_df.copy()


# In[ ]:


final_df.head()


# In[ ]:


final_df.columns


# In[ ]:


final_col_reorder = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
       'hours-per-week', 'sex', 'United_state', 'Other_country', 'Federal-gov',
       'Local-gov', 'Never-worked', 'Private', 'Self-emp-inc',
       'Self-emp-not-inc', 'State-gov', 'Without-pay', 'unknown', 'Black',
       'White', 'Other_race', 'pre_prof_school', 'college',
       'masters_doc', 'assosiate_edu', 'Divorced', 'Married', 'Never_Married',
       'Group_1_ocp', 'Group_2_ocp', 'Group_3_ocp', 'Group_4_ocp',
       'Group_5_ocp', 'income']


# In[ ]:


final_df = final_df[final_col_reorder]


# In[ ]:


final_df.head()


# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

# the custom scaler class
class CustomScaler(BaseEstimator,TransformerMixin):
    
    def __init__(self,columns,copy=True,with_mean=True,with_std=True):
        self.scaler = StandardScaler(copy,with_mean,with_std)
        self.columns = columns
        self.mean_ = None
        self.var_ = None
        
    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self
    
    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


# In[ ]:


final_df.columns


# In[ ]:


columns_to_scale = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
       'hours-per-week']


# In[ ]:


income_scaler = CustomScaler(columns_to_scale)


# In[ ]:


unscaled_inputs = final_df.iloc[:,:-1]


# In[ ]:


unscaled_inputs.columns.values


# In[ ]:


income_scaler.fit(unscaled_inputs)


# In[ ]:


scaled_inputs = income_scaler.transform(unscaled_inputs)


# In[ ]:


scaled_inputs.head()


# In[ ]:


scaled_inputs.shape


# In[ ]:


targets = final_df.income


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, targets, train_size = 0.8, random_state = 20)


# In[ ]:


print(x_train.shape, y_train.shape)


# In[ ]:


print(x_test.shape, y_test.shape)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[ ]:


reg = LogisticRegression()


# In[ ]:


reg.fit(x_train, y_train)


# In[ ]:


reg.score(x_train, y_train)


# In[ ]:


pred_y = reg.predict(x_test)


# In[ ]:


metrics.confusion_matrix(pred_y,y_test)


# In[ ]:


reg.score(x_test,y_test)


# # Finding the intercept and coefficients:****

# In[ ]:


reg.intercept_


# In[ ]:


reg.coef_


# In[ ]:


unscaled_inputs.columns.values


# In[ ]:


feature_name = unscaled_inputs.columns.values


# In[ ]:


summary_table = pd.DataFrame(columns=['Feature_name'], data = feature_name)
summary_table['Coefficient'] = np.transpose(reg.coef_)


# In[ ]:


summary_table


# In[ ]:


summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['intercept', reg.intercept_[0]]
summary_table = summary_table.sort_index()
summary_table


# # Interpreting the Coefficients:

# In[ ]:


summary_table['Odds_ratio'] = np.exp(summary_table.Coefficient)


# In[ ]:


summary_table


# In[ ]:


summary_table.sort_values('Odds_ratio', ascending = False)


# if its coeff is around 0 and Odds ratio is around 1, means feature isn't particularly important.
# 
# **A weight(coeff) of 0 implies that no matter the featyure value, we will multiply it by 0(in the model)**
# 
# **For a unit change in the standardized feature, the odds increase by a multiple equal to the odds ratio(1=no change)**
# 
# Daily work load Average, Day of the week, Distance to work columns coeff is alomost 0 and odds ratio is almost 1, so it means that feature isn't important.

# ### Backward Elimination
# The idea is that we can simplify our model by removing all features which have close to no contribution to the model.
# 
# When we have the p-value, we get rid of all coeff with p-values>0.05.
# 
# if the weights is small enough. it won't make a diff anyway....

# In[ ]:


corr = final_df.corr()


# In[ ]:


plt.figure(figsize=(25,12))
sns.heatmap(corr, annot=True)
plt.show()


# # Random Forest Classifier.
# **Let's choose the best estimator and parameters :GridSearchCV**

# In[ ]:


#let's check what params will be best suitable for random forest classification.
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

rfc_clf = RandomForestClassifier()
params = {'n_estimators':[25,50,100,150,200,500],'max_depth':[0.5,1,5,10],'random_state':[1,10,20,42],
          'n_jobs':[1,2]}
grid_search_cv = GridSearchCV(rfc_clf, params, scoring='precision')
grid_search_cv.fit(x_train, y_train)


# In[ ]:


print(grid_search_cv.best_estimator_)
print(grid_search_cv.best_params_)


# In[ ]:


rfc_clf = grid_search_cv.best_estimator_
rfc_clf.fit(x_train,y_train)
rfc_clf_pred = rfc_clf.predict(x_test)
print('Accuracy:',accuracy_score(rfc_clf_pred,y_test) )
print('Confusion Matrix:', confusion_matrix(rfc_clf_pred,y_test).ravel()) #tn,fp,fn,tp
print('Classification report:')
print(classification_report(rfc_clf_pred,y_test))

# Let's make sure the data is not overfitting
score_rfc = cross_val_score(rfc_clf,x_train,y_train,cv = 10).mean()
print('cross val score:', score_rfc)


# In[ ]:


feature_imp = rfc_clf.feature_importances_.round(3)
ser_rank = pd.Series(feature_imp, index=scaled_inputs.columns).sort_values(ascending = False)

plt.figure(figsize=(12,7))
sns.barplot(x= ser_rank.values, y = ser_rank.index, palette='deep')
plt.xlabel('relative importance')
plt.show()


# # SupportVectorClassifier:

# In[ ]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
# Implement gridsearchcv to see which are our best p

params = {'C': [0.75, 0.85, 0.95, 1], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 
          'degree': [3, 4, 5]}

svc_clf = SVC(random_state=42)
grid_search_cv = GridSearchCV(svc_clf, params)
grid_search_cv.fit(x_train, y_train)


# In[ ]:


print(grid_search_cv.best_estimator_)
print(grid_search_cv.best_params_)


# In[ ]:


svc_clf = grid_search_cv.best_estimator_
svc_clf.fit(x_train,y_train)
svc_pred = svc_clf.predict(x_test)

print('Accuracy:',accuracy_score(svc_pred,y_test) )
print('Confusion Matrix:', confusion_matrix(svc_pred,y_test,labels=[0,1])) #tn,fp,fn,tp
print('Classification report:')
print(classification_report(svc_pred,y_test))

# Let's make sure the data is not overfitting
score_svc = cross_val_score(svc_clf,x_train,y_train, cv = 10).mean()
print('cross val score:', score_svc)


# In[ ]:




