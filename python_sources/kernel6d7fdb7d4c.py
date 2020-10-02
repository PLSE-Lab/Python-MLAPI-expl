#!/usr/bin/env python
# coding: utf-8

# NOTE: In order to run EDA start from the next cell.
# To run the model search for "Pre-processing" and start from that cell

# In[ ]:


'''
# bank client data:
1 - age (numeric)
2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5 - default: has credit in default? (categorical: 'no','yes','unknown')
6 - housing: has housing loan? (categorical: 'no','yes','unknown')
7 - loan: has personal loan? (categorical: 'no','yes','unknown')
  - balance: 
# related with the last contact of the current campaign:
8 - contact: contact communication type (categorical: 'cellular','telephone')
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10 - day: last contact day 
11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# other attributes:
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14 - previous: number of contacts performed before this campaign and for this client (numeric)
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')

Output variable (desired target):
21 - y - has the client subscribed a term deposit? (binary: 'yes','no')
'''


# In[ ]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from time import time
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import log_loss
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# For the tree visualization
import pydot
from IPython.display import Image
from sklearn.externals.six import StringIO


# In[ ]:


# Importing the dataset
# Input: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
filename='/kaggle/input/bank-marketing/bank-full.csv'
df = pd.read_csv(filename,sep=',')
orig_df_size = len(df)
print(df.info())


# Conclusion: there are no null cells

# In[ ]:


#helper function
def check(df, feature):
  df_new = df.groupby(feature)['y'].apply(lambda x: x.sum()/len(x))
  return df_new


# In[ ]:


df.loc[1:10]


# In[ ]:


# Convert the result and two other columns to a number
df['y'] = df['y'].apply(lambda x: 0 if 'no' in x else 1)
df['housing'] = df['housing'].apply(lambda x: 0 if 'no' in x else 1)
df['loan'] = df['loan'].apply(lambda x: 0 if 'no' in x else 1)
df['default'] = df['default'].apply(lambda x: 0 if 'no' in x else 1)


# In[ ]:


#check if data is balanced
print(f"the percentage of y out of all df {(df['y'].value_counts()[1]/df['y'].value_counts())[0]}")


# Coclusion: the data is not balance, only ~13% are 'y', the rest are 'n'

# In[ ]:


# drop of unknown values
orig_len = len(df)
df = df[(df.age !='unknown') & (df.job !='unknown') & (df.marital !='unknown') & (df.housing !='unknown') & (df.loan !='unknown') & (df.education != 'unknown')]
updated_len = len(df)
print(f'df length reduced by {(orig_len-updated_len)/orig_len*100} from {orig_len} to {updated_len}')


# Conclusion: the are about 5% rows with 'unknown' cells, it's OK to remove them

# In[ ]:


df.nunique()


# In[ ]:


df['marital'].value_counts()


# In[ ]:


axis = check(df,'marital').plot(kind="bar")
fig = axis.get_figure()


# Conclusion: Seems thar matiral is a requiered feature

# In[ ]:


df['default'].value_counts()


# In[ ]:


axis = check(df,'default').plot(kind="bar")
fig = axis.get_figure()


# * Conclusion: since there are too few 1s drop this feature

# In[ ]:


print(df['balance'].min())
print(df['balance'].max())


# In[ ]:


df['balance_'] = df['balance'] // 1000
axis = check(df,'balance_').plot(kind="bar",figsize=(20,5))


# Conclusion: the balance has no affect on the result, drop this feature

# In[ ]:


#convert duration to minutes
df['durationMin'] = df['duration'] // 60


# In[ ]:


#check the numbr of 'y' vs. duration
x1 = (df.groupby('durationMin')['y'].sum()) / (df.groupby('durationMin')['y'].count()) * 100
x1[0:80].plot(kind="bar", figsize=(15, 5))


# In[ ]:


x2 = df.groupby('durationMin')['y'].count()
x2.plot(kind="bar", figsize=(15, 5))


# In[ ]:


len(df[df['durationMin'] > 38]) / len(df) * 100


# Conclusion: we can remove rows with duration > 38 (outliers), these are 0.08% of the data

# In[ ]:


df['age'].hist()


# In[ ]:


axis = check(df,'age').plot(kind="bar", figsize=(20, 5))
fig = axis.get_figure()


# Conclusion: strange, we would expect higher results for ages 30-50. Remove ages age > 85

# In[ ]:


df['job'].value_counts()


# In[ ]:


axis = check(df,'job').plot(kind="bar", figsize=(15, 5))
fig = axis.get_figure()


# Conclusion: (We would like to union into fewer categories: employees, retired, self-employed, student, unemployed.)
# It will be better to convert the jobs to percentage of y

# In[ ]:


df['education'].value_counts()


# In[ ]:


df['education'].unique()


# In[ ]:


axis = check(df,'education').plot(kind="bar", figsize=(5, 5))
fig = axis.get_figure()


# In[ ]:


axis = check(df,'housing').plot(kind="bar")
fig = axis.get_figure()


# In[ ]:


axis = check(df,'loan').plot(kind="bar")
fig = axis.get_figure()


# In[ ]:


df['housing_loan'] = df.apply(lambda x: 1 if (x['loan']==1) | (x['housing']==1) else 0 ,axis=1)
df['housing_loan'] = np.where((df['loan']=='yes') | (df['housing']=='no'), 1,0)


# In[ ]:


df['month_num'] = df['month']
df['month_num'].replace({"jan":1,
                   "feb":2, 
                   "mar":3, 
                   "apr":4, 
                   "may":5, 
                   "jun":6, 
                   "jul":7, 
                   "aug":8, 
                   "sep":9, 
                   "oct":10, 
                   "nov":11, 
                   "dec":12}, inplace=True)


# In[ ]:


axis = check(df,'month_num').plot(kind="bar")
#fig = axis.get_figure()


# In[ ]:


df['month_num'].value_counts().sort_index()


# Conclusion: it seems that the month effects the percentage of y
# use the statistics approach

# In[ ]:


axis = check(df,'day').plot(kind="bar", figsize=(20,5))
fig = axis.get_figure()


# Conclusion: we don't see why day in the month effects, should be dropped

# In[ ]:


axis = check(df,'previous').plot(kind="bar",figsize=(15, 5))


# In[ ]:


df['previous'].value_counts().sort_index()


# Conclusion: we can remove previous >= 17

# In[ ]:


axis = check(df,'poutcome').plot(kind="bar")
fig = axis.get_figure()


# In[ ]:


df['poutcome'].value_counts()


# Conclusion: this feature effects the result

# In[ ]:


df_ct = pd.crosstab(columns=df[(df['previous'] != 0) & (df['poutcome'] != 'nonexistent')].previous, 
                          index=df[(df['previous'] != 0) & (df['poutcome'] != 'nonexistent')].poutcome, 
                          normalize='index')
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(ax=ax, data=df_ct, cmap='coolwarm')
#There is no immidiate conclusion therfore these two feature will remain to be tested later by the model


# Conclusion: there is no correlaiton betwen these feature for previous>1 so we need both features

# In[ ]:


print (len(df.loc[df.pdays==-1]))


# Conclusion: we should drop this feature

# In[ ]:


axis = check(df,'campaign').plot(kind="bar", figsize=(15, 5))
fig = axis.get_figure()


# In[ ]:


df['campaign'].value_counts()


# In[ ]:


len(df[df['campaign'] >= 17])


# Conclusion: it seems that campaign >= 17 are outliers

# In[ ]:


'''
Conclusions:
0   age             41188 non-null  int64 - use  
1   job             41188 non-null  object - change to percetage
2   marital         41188 non-null  object - use, get_dummies
3   education       41188 non-null  object - reduced, used (get_dummies)
4   default         41188 non-null  object - dropped
5   housing         41188 non-null  object - use 
6   loan            41188 non-null  object - use
7   contact         41188 non-null  object - dropped
8   month           41188 non-null  object - use as percentage
9   day_of_week     41188 non-null  object - dropped
10  duration        41188 non-null  int64 - use  
11  campaign        41188 non-null  int64 - use, remove outliers 
12  pdays           41188 non-null  int64 - Too many unknown values, dropped
13  previous        41188 non-null  int64 - use (consider reduce to 0,1,2,>=3) (correlate to poutcome?)
14  poutcome        41188 non-null  object - get dummies, use (correlate to previous?)
15  balance                                - drop
'''


# # Pre-processing

# In[ ]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from time import time
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import log_loss
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# For the tree visualization
import pydot
from IPython.display import Image
from sklearn.externals.six import StringIO


# In[ ]:


# Importing the dataset
# Input: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
filename='/kaggle/input/bank-marketing/bank-full.csv'
df = pd.read_csv(filename,sep=',')
orig_df_size = len(df)
print(df.info())


# In[ ]:


#helper function
def check(df, feature):
  df_new = df.groupby(feature)['y'].apply(lambda x: x.sum()/len(x))
  return df_new


# In[ ]:


#save the original dataframe
df1 = df.copy()
df1.head()


# In[ ]:


df1 = df1[(df.age !='unknown') & (df1.job !='unknown') & (df1.marital !='unknown') & (df1.housing !='unknown') & (df1.loan !='unknown') & (df1.education != 'unknown')]


# In[ ]:


#convert duration to minutes
df1['durationMin'] = df1['duration'] // 60
df1.drop(['duration'], axis=1, inplace=True)


# In[ ]:


# Convert the result and two other columns to a number
df1['y'] = df1['y'].apply(lambda x: 0 if 'no' in x else 1)
df1['housing'] = df1['housing'].apply(lambda x: 0 if 'no' in x else 1)
df1['loan'] = df1['loan'].apply(lambda x: 0 if 'no' in x else 1)
df1['default'] = df1['default'].apply(lambda x: 0 if 'no' in x else 1)


# In[ ]:


df1.drop(['contact','balance','pdays','day','default','poutcome'], axis=1, inplace=True)


# In[ ]:


#create getdummies 
df1 = pd.get_dummies(df1, columns=['marital'],drop_first=False)
df1 = pd.get_dummies(df1, columns=['education'],drop_first=False)


# # Splitting the data

# In[ ]:


train, test = train_test_split(df1, 
                              train_size=0.75, 
                              shuffle=True, 
                              stratify=df1['y'])


# In[ ]:


#Remove outliers
orig_len = len(train)
train = train[train['campaign'] <= 17]
train = train[train['durationMin'] < 38]
train = train[train['age'] < 85]
train = train[train['previous'] < 17]
print(f'removed {orig_len - len(train)} out of {len(train)}')


# In[ ]:


#covert job to percentage
df_job = pd.DataFrame(check(train,'job'))
print(list(df_job.columns))
df_job.rename(columns={'y':'job_percent'},inplace=True)
print('df_job ',list(df_job.columns))
print('train ',list(train.columns))
train = train.merge(df_job, how='inner', left_on='job',right_index=True)
train.drop(['job'], axis=1, inplace=True)
train


# In[ ]:


#covert month to percentage
df_month = pd.DataFrame(check(train,'month'))
df_month.rename(columns={'y':'month_percent'},inplace=True)
print('df_month ',list(df_month.columns))
print('train ',list(train.columns))
train = train.merge(df_month, how='inner', left_on='month',right_index=True)
train.drop(['month'], axis=1, inplace=True)


# In[ ]:


X_train = train.drop(['y'], axis=1)
y_train = train['y']
print(y_train)


# In[ ]:


X_train.info()


# # Models

# In[ ]:


def print_cm_metrics(cm):
    total1=sum(sum(cm))
    accuracy1=(cm[0,0]+cm[1,1])/total1
    print ('Accuracy : ', accuracy1)

    specificity = cm[0,0]/(cm[0,0]+cm[0,1])
    print('specificity : ', specificity )
    
    # sensitivity == recall
    sensitivity = cm[1,1]/(cm[1,0]+cm[1,1])
    print('sensitivity : ', sensitivity)


# In[ ]:


my_cv = StratifiedShuffleSplit(n_splits=10, train_size=0.7, test_size=0.3)


# In[ ]:


test = test.merge(df_month, how='inner', left_on='month',right_index=True)
test = test.merge(df_job, how='inner', left_on='job',right_index=True)
test.drop(['month'], axis=1, inplace=True)
test.drop(['job'], axis=1, inplace=True)


# In[ ]:


X_test = test.drop(['y'], axis=1)
y_test = test['y']


# # **Decision Tree**

# In[ ]:


dt_model_dt = DecisionTreeClassifier(class_weight='balanced')


# In[ ]:


# from sklearn.metrics import fbeta_score, make_scorer
# best_beta_ind = (Fb.Model1 - Fb.Model2).abs().idxmin()
# my_beta = Fb.Beta[best_beta_ind]
# print (my_beta)
# my_fbeta_score = \
#     make_scorer(fbeta_score, beta=my_beta, pos_label='Yes', greater_is_better=True)


# In[ ]:


my_param_grid = {'min_samples_leaf': [5, 10, 15],
                 'min_weight_fraction_leaf': [0.003, 0.005, 0.007],
                 'criterion': ['gini', 'entropy'], 
                 'min_impurity_decrease': [1e-3, 1e-4, 1e-5]}
#After running the above combinations these are the results:
# my_param_grid = {'min_samples_leaf': [10],
#                  'min_weight_fraction_leaf': [0.005],
#                  'criterion': ['gini'], 
#                  'min_impurity_decrease': [1e-4]}


# In[ ]:


dt_model_gs = GridSearchCV(estimator=dt_model_dt, 
                           param_grid=my_param_grid, 
                           cv=my_cv, 
                           scoring='roc_auc')


# In[ ]:


dt_model_gs.fit(X_train, y_train)


# In[ ]:


dt_model_3 = dt_model_gs.best_estimator_
dt_model_3


# In[ ]:


y_train_pred = pd.DataFrame(dt_model_3.predict_proba(X_train), 
                           columns=dt_model_3.classes_)
#log_loss(y_true=y_train, y_pred=y_train_pred)


# In[ ]:


scores = dt_model_3.predict_proba(X_train)[:, 1]
roc_auc_score(y_true=y_train, y_score=scores)


# In[ ]:


y_test_pred = pd.DataFrame(dt_model_3.predict_proba(X_test), 
                           columns=dt_model_3.classes_)
#log_loss(y_true=y_test, y_pred=y_test_pred)


# In[ ]:


scores = dt_model_3.predict_proba(X_test)[:, 1]
roc_auc_score(y_true=y_test, y_score=scores)


# In[ ]:


y_test_pred['predict1'] = y_test_pred.apply(lambda x: x[1] > 0.1, axis=1)
#asses the model
cm = confusion_matrix(y_true=y_test,
                      y_pred=y_test_pred['predict1'])
pd.DataFrame(cm, 
             index=dt_model_3.classes_, 
             columns=dt_model_3.classes_)


# In[ ]:


print_cm_metrics(cm)


# In[ ]:


def visualize_tree(model, md=5, width=800):
    dot_data = StringIO()  
    export_graphviz(model, out_file=dot_data, feature_names=X_train.columns, max_depth=md)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())[0]  
    return Image(graph.create_png(), width=width) 


# In[ ]:


def print_dot_text(model, md=5):
    """The output of this function can be copied to http://www.webgraphviz.com/"""
    dot_data = StringIO()
    export_graphviz(model, out_file=dot_data, feature_names=X_train.columns, max_depth=md)
    dot_text = dot_data.getvalue()
    print(dot_text)


# In[ ]:


visualize_tree(dt_model_3, md=3, width=1200)


# In[ ]:


pd.Series(dt_model_3.feature_importances_, index=X_train.columns).sort_values()    .plot.barh(figsize=(4, 10), rot=0, title='Feature importances')


# In[ ]:


pd.Series(dt_model_3.feature_importances_, index=X_train.columns).sort_values()


# # Random forest

# In[ ]:


rf_model = RandomForestClassifier(class_weight='balanced',**dt_model_gs.best_params_)


# In[ ]:


my_param_grid = {'bootstrap': [True, False], 
                 'n_estimators': [50, 100, 200], 
                 'oob_score': [True, False], 
                 'warm_start': [True, False]}
#After running the above combinations these are the best results:
# my_param_grid = {'bootstrap': [True], 
#                  'n_estimators': [200], 
#                  'oob_score': [False], 
#                  'warm_start': [True]}


# In[ ]:


rf_model_gs = GridSearchCV(estimator=rf_model, 
                        param_grid=my_param_grid, 
#                        scoring='neg_log_loss', 
                        scoring='roc_auc', 
                        cv=my_cv)
#roc_auc_score


# In[ ]:


t1 = time()
rf_model_gs.fit(X_train, y_train)
t2 = time()
print(f"It took {t2-t1:.2f} seconds")


# In[ ]:


rf_model_gs_1 = rf_model_gs.best_estimator_
rf_model_gs_1


# In[ ]:


y_train_pred = pd.DataFrame(rf_model_gs_1.predict_proba(X_train), 
                           columns=rf_model_gs_1.classes_)
#log_loss(y_true=y_train, y_pred=y_train_pred)


# In[ ]:


scores = rf_model_gs_1.predict_proba(X_train)[:, 1]
roc_auc_score(y_true=y_train, y_score=scores)


# In[ ]:


y_test_pred = pd.DataFrame(rf_model_gs_1.predict_proba(X_test), 
                           columns=rf_model_gs_1.classes_)
#log_loss(y_true=y_test, y_pred=y_test_pred)


# In[ ]:


pd.Series(rf_model_gs_1.feature_importances_, index=X_train.columns).sort_values()    .plot.barh(figsize=(4, 10), rot=0, title='Feature importances')


# In[ ]:


y_test_pred['predict1'] = y_test_pred.apply(lambda x: x[1] > 0.4, axis=1)
#y_test_pred.head(10)


# In[ ]:


#asses the model
cm = confusion_matrix(y_true=y_test,
                      y_pred=y_test_pred['predict1'])
pd.DataFrame(cm, 
             index=rf_model_gs_1.classes_, 
             columns=rf_model_gs_1.classes_)


# In[ ]:


print_cm_metrics(cm)


# In[ ]:


X_train.info()
#scores = rf_model_gs_1.predict_proba(X_train)[:, 1]
#roc_auc_score(y_true=y_train, y_score=scores)


# In[ ]:


scores = rf_model_gs_1.predict_proba(X_test)[:, 1]
roc_auc_score(y_true=y_test, y_score=scores)


# In[ ]:


fpr, tpr, thresholds = roc_curve(y_test, scores, pos_label=1)
res = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Threshold': thresholds})
#res[['TPR', 'FPR', 'Threshold']][::100]


# In[ ]:


plt.plot(fpr, tpr, '-o')
plt.title('ROC')
plt.xlabel('FPR (False Positive Rate = 1-specificity)')
plt.ylabel('TPR (True Positive Rate = sensitivity)')
plt.xlim([0, 1])
plt.ylim([0, 1])


# In[ ]:





# # [](http://)Logistic Regression

# In[ ]:


def change_age(df):
    df_new = df.copy()
    df_new['age_class'] = df_new['age'].apply(lambda x: 'young' if x<=25 else 'old' if x >= 61 else 'mid')
    df_new = pd.get_dummies(df_new, columns=['age_class'],drop_first=False)
    return df_new


# In[ ]:


# from sklearn.pipeline import Pipeline, FeatureUnion

# lr_model = LogisticRegression(class_weight='balanced', max_iter=10)
# change_a = change_age()
# # change_a.head()
# steps = [('change_age', change_a), 
#          ('lr', lr_model)]
# my_pipeline = Pipeline(steps)


# In[ ]:


#Consider use pandas.cut
X_train_all = change_age(X_train)
X_train_lr = X_train_all.drop(columns=['age'])
# X_train_lr['age'] = X_train_lr['age'].apply(lambda x: 'young' if x<=25 else 'old' if x >= 61 else 'mid')
# X_train_lr = pd.get_dummies(X_train_lr, columns=['age'],drop_first=False)
my_param_grid = {'C': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3], 'penalty': ['l1', 'l2']}
lr_model = GridSearchCV(estimator=LogisticRegression(class_weight='balanced', max_iter=10), 
                        param_grid=my_param_grid, 
                        scoring='roc_auc', 
                        cv=my_cv)
lr_model.fit(X_train_lr, y_train)


# In[ ]:


y_train_pred = pd.DataFrame(lr_model.predict_proba(X_train_lr), 
                           columns=lr_model.classes_)


# In[ ]:


scores = lr_model.predict_proba(X_train_lr)[:, 1]
roc_auc_score(y_true=y_train, y_score=scores)


# In[ ]:


X_test_all = change_age(X_test)
X_test_lr = X_test_all.drop(columns=['age'])

# X_test_lr['age'] = X_test_lr['age'].apply(lambda x: 'young' if x<=25 else 'old' if x >= 61 else 'mid')
# X_test_lr = pd.get_dummies(X_test_lr, columns=['age'],drop_first=False)
#y_test_pred = lr_1.predict(X_test)
y_test_pred = pd.DataFrame(lr_model.predict_proba(X_test_lr),
                           columns=lr_model.classes_)


# In[ ]:


scores = lr_model.predict_proba(X_test_lr)[:, 1]
roc_auc_score(y_true=y_test, y_score=scores)


# In[ ]:


y_test_pred['predict1'] = y_test_pred.apply(lambda x: x[1] > 0.4, axis=1)


# In[ ]:


#asses the model
cm = confusion_matrix(y_true=y_test, y_pred=y_test_pred['predict1'])
pd.DataFrame(cm, 
             index=lr_model.classes_, 
             columns=lr_model.classes_)


# In[ ]:


print_cm_metrics(cm)


# # Ensemble methods - voting

# In[ ]:


from sklearn.ensemble import VotingClassifier, BaggingClassifier,     AdaBoostClassifier, GradientBoostingClassifier


# In[ ]:


clf1 = rf_model_gs_1 #RandomForestClassifier 
clf2 = lr_model #LogisticRegression
classifiers = [('LR', clf1), ('DT', clf2)]


# In[ ]:


clf_voting = VotingClassifier(estimators=classifiers,
                              voting='soft')
clf_voting.fit(X_train_all, y_train)


# In[ ]:


scores = clf_voting.predict_proba(X_train_all)[:, 1]
roc_auc_score(y_true=y_train, y_score=scores)


# In[ ]:


y_test_pred = pd.DataFrame(clf_voting.predict_proba(X_test_all),
                           columns=clf_voting.classes_)
y_test_pred['predict1'] = y_test_pred.apply(lambda x: x[1] > 0.2, axis=1)


# In[ ]:


scores = clf_voting.predict_proba(X_test_all)[:, 1]
roc_auc_score(y_true=y_test, y_score=scores)


# In[ ]:


print(f"train accuracy: {clf_voting.score(X_train_all, y_train):.2f}\ntest accuracy: {clf_voting.score(X_test_all, y_test):.2f}")


# In[ ]:


#asses the model
cm = confusion_matrix(y_true=y_test,
                      y_pred=y_test_pred['predict1'])
pd.DataFrame(cm, 
             index=lr_model.classes_, 
             columns=lr_model.classes_)
print_cm_metrics(cm)


# In[ ]:




