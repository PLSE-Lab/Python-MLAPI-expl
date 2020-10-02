#!/usr/bin/env python
# coding: utf-8

# # Import Modules

# In[ ]:


import numpy as np
import scipy as sp
import pandas as pd


# # Input Data

# In[ ]:


df = pd.read_csv('../input/pre-eda-ta/pre_eda_.csv', index_col=None)
df = df.drop('Unnamed: 0',axis=1)
df.head()


# # Data Preprocessing

# ## Penghapusan Fitur
# drop policy_code,payment plan

# In[ ]:


print(list(df.columns))


# In[ ]:


df = df.drop(['policy_code','pymnt_plan'],axis=1)


# In[ ]:


df.shape


# In[ ]:


#splitting categorical and continuous features
continuous_features=[value for value in list(df._get_numeric_data().columns) if value not in ["is_charged_off","issue_d"]]
categorical_features=[value for value in df.columns if value not in [*continuous_features,"is_charged_off","issue_d"]]
target_label="is_charged_off"


# ## Modifikasi Data Khusus

# **dti**

# In[ ]:


df[(df.dti < 0)]


# In[ ]:


df = df[(df.dti >= 0)]


# ecl

# In[ ]:


df['earliest_cr_line'] = df['earliest_cr_line'].astype('datetime64[ns]')


# In[ ]:


df['ecl_year'] = pd.DatetimeIndex(df['earliest_cr_line']).year
df['ecl_month'] = pd.DatetimeIndex(df['earliest_cr_line']).month


# In[ ]:


df= df.drop('earliest_cr_line',axis=1)


# In[ ]:


df['issue_d'] = df['issue_d'].astype('datetime64[ns]')


# ## Outlier Handling
# acc_open_past_24mths, annual_inc, delinq_amnt, mo_sin_old_il_acct, mths_since_recent_bc, num_accts_ever_120_pd, num_il_tl, num_tl_90g_dpd_24m, total_bal_ex_mort,delinq_2yrs

# In[ ]:


def outlier_treatment(datacolumn):
    sorted(datacolumn)
    Q1,Q3 = np.percentile(datacolumn , [25,75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range,upper_range


# In[ ]:


def check_outlier(col):
    l,u = outlier_treatment(df[col])
    print("----",col,"----")
    print("Batas atas: ", u)
    print("Batas bawah: ", l)
    print(len(df[(df[col] < l) | (df[col] > u)]))
    print()


# In[ ]:


def remove_outlier(df,col):
    l,u = outlier_treatment(df[col])
    print("----",col,"----")
    print("Batas atas: ", u)
    print("Batas bawah: ", l)
    df_out = df[(df[col] >= l) & (df[col] <= u)]
    return df_out


# In[ ]:


outlier_features = ['acc_open_past_24mths', 'annual_inc',  'mo_sin_old_il_acct', 'mths_since_recent_bc', 'num_il_tl', 'total_bal_ex_mort']


# In[ ]:


for col in outlier_features:
    check_outlier(col)
    print(df[col].skew())


# In[ ]:


df_out = df.copy()


# In[ ]:


df_out = df.copy()
for col in outlier_features:
    check_outlier(col)
    df_out = remove_outlier(df_out,col)


# In[ ]:


df.shape


# In[ ]:


df_out.shape


# In[ ]:


df = df_out.copy()


# In[ ]:


df.shape


# ## Feature Engineering
# fitur **is_poverty_prone** yaitu kategori tingkat ekonomi yang rawan untuk jatuh miskin pada saat krisis terjadi. Seseorang dianggap rawan miskin is_industry_prone, is_newly_employed

# In[ ]:


df['emp_length'].describe


# In[ ]:


df['is_newly_employed'] = np.where((df['emp_length'] == '< 1 year') | (df['emp_length'] == '2 years') | (df['emp_length'] == '3 years'),'Yes','No')


# In[ ]:


jobs_in_risk = ['Agent','Bartender','Chef','Cook','Customer Service','Driver','driver','Flight Attendant','Instructor','Mechanic','mechanic','Pilot','Property Manager','Realtor','Receptionist','Sales','sales','Sales Representative','Server','server','Truck Driver','truck driver','Truck driver']


# In[ ]:


df['is_industry_prone'] = np.where(df['emp_title'].isin(jobs_in_risk),'Yes','No')


# In[ ]:


df.columns


# In[ ]:


df['salary'] = pd.cut(df.annual_inc,10,labels=["a","b","c","d","e","f","g","h","i","j"])
df['dti_'] = pd.cut(df.dti,10,labels=["a","b","c","d","e","f","g","h","i","j"])
df['int_rate_'] = pd.cut(df.int_rate,10,labels=["a","b","c","d","e","f","g","h","i","j"])
df['fico'] = pd.cut(df.last_fico_range_high,10,labels=["a","b","c","d","e","f","g","h","i","j"])
df['balance'] = pd.cut(df.avg_cur_bal,10,labels=["a","b","c","d","e","f","g","h","i","j"])
df['installment_'] = pd.cut(df.installment,10,labels=["a","b","c","d","e","f","g","h","i","j"])


# In[ ]:


#splitting categorical and continuous features
continuous_features=[value for value in list(df._get_numeric_data().columns) if value not in ["is_charged_off","issue_d"]]
categorical_features=[value for value in df.columns if value not in [*continuous_features,"is_charged_off","issue_d"]]
target_label="is_charged_off"


# ## Encoding Categorical Data
# using label encoding

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df[categorical_features] = df[categorical_features].apply(lambda col: le.fit_transform(col))


# ## Pembagian Dataset Latih dan Uji
# 80% latih, 20% uji

# In[ ]:


def split_data(df):
    print(df.columns)
    #test train split time
    from sklearn.model_selection import train_test_split
    y = df[target_label].values #target
    X = df.drop([target_label],axis=1).values #features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                    random_state=42, stratify=None, shuffle=False)

    print("train-set size: ", len(y_train),
      "\ntest-set size: ", len(y_test))
    print("charged-off cases in test-set: ", sum(y_test)/len(y_test)*100,"%")
    return X_train, X_test, y_train, y_test


# In[ ]:


# def split_data(df):
#     print(df.columns)
#     #test train split time
#     from sklearn.model_selection import train_test_split
#     y = df[target_label].values #target
#     X = df.drop([target_label],axis=1).values #features
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
#                                                     random_state=42, stratify=y)

#     print("train-set size: ", len(y_train),
#       "\ntest-set size: ", len(y_test))
#     print("charged-off cases in test-set: ", sum(y_test)/len(y_test)*100,"%")
#     return X_train, X_test, y_train, y_test


# In[ ]:


# droplist = ['zip_code']
# df = df.drop(droplist,axis=1)


# In[ ]:


df_cat = df.drop(continuous_features,axis=1)
df_cat.sort_values(by='issue_d')
df_cat = df_cat.drop("issue_d",axis=1)
X_c, X_test_c, y_c, y_test_c = split_data(df_cat)


# In[ ]:


df_con = df.drop(categorical_features,axis=1)
df_con.sort_values(by='issue_d')
df_con = df_con.drop("issue_d",axis=1)
X_g, X_test_g, y_g, y_test_g = split_data(df_con)


# In[ ]:





# In[ ]:


df.sort_values(by='issue_d')
df = df.drop("issue_d",axis=1)
X, X_test, y, y_test = split_data(df)


# In[ ]:


del df_cat
del df_con


# ## Handling Imbalance Data

# In[ ]:


import imblearn
from collections import Counter
from matplotlib import pyplot
from numpy import where
from imblearn.over_sampling import SMOTE


# In[ ]:


oversample = SMOTE()


# In[ ]:


counter = Counter(y)
print('before: ',counter)
X_train, y_train = oversample.fit_resample(X, y)
# summarize the new class distribution
counter = Counter(y_train)
print('after: ',counter)


# In[ ]:


counter = Counter(y_c)
print('before y categorical: ',counter)
X_train_c, y_train_c = oversample.fit_resample(X_c, y_c)
# summarize the new class distribution
counter = Counter(y_train_c)
print('after categorical: ',counter)


# In[ ]:


counter = Counter(y_g)
print('before y Gaussian: ',counter)
X_train_g, y_train_g = oversample.fit_resample(X_g, y_g)
# summarize the new class distribution
counter = Counter(y_train_g)
print('after Gaussian: ',counter)


# # Modelling

# In[ ]:


pip install mixed-naive-bayes


# In[ ]:


from mixed_naive_bayes import MixedNB
from sklearn import metrics
import scikitplot as skplt
from sklearn.metrics import confusion_matrix,auc,roc_auc_score
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB, CategoricalNB 
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import time


# # Initial Modelling

# Tanpa Tuning Parameter

# In[ ]:


def get_predictions(clf, X_train, y_train, X_test, y_test):
    # create classifier
    clf = clf
    # fit it to training data
    start = time.time()
    clf.fit(X_train,y_train)
    stop = time.time()
    print(f"Training time: {stop - start} s")
    train_time = stop - start
    # predict using test data
    start = time.time()
    y_pred = clf.predict(X_test)
    # Compute predicted probabilities: y_pred_prob
    y_pred_prob = clf.predict_proba(X_test)
    stop = time.time()
    print(f"Prediction time: {stop - start} s")
    pred_time = stop - start
    #for fun: train-set predictions
    train_pred = clf.predict(X_train)
    print('train-set confusion matrix:\n', confusion_matrix(y_train,train_pred)) 
    metrics.plot_roc_curve(clf, X_test, y_test)
    return y_pred, y_pred_prob, pred_time, train_time


# In[ ]:


def print_scores(y_test,y_pred,y_pred_prob,pred_time,train_time):
    cm1= confusion_matrix(y_test,y_pred)
    print('test-set confusion matrix:\n', confusion_matrix(y_test,y_pred)) 
    print("accuracy score: ", accuracy_score(y_test,y_pred))
    print("precision score: ", precision_score(y_test,y_pred))
    
    print("recall score: ", recall_score(y_test,y_pred))
    print("specificity score: ", cm1[0,0]/(cm1[0,0]+cm1[0,1]))
    print("ROC AUC: {}".format(roc_auc_score(y_test, y_pred_prob[:,1])))
    print("f1 score: ", f1_score(y_test,y_pred))
    print ("-------------------")
    print_metrics(y_test,y_pred,y_pred_prob,pred_time,train_time)
    print ("-------------------")
    print_cm(cm1)


# In[ ]:


def apply_model(clf, X_train, y_train, X_test, y_test):
    start = time.time()
    y_pred, y_pred_prob, pred_time, train_time = get_predictions(clf, X_train, y_train, X_test, y_test)
    stop = time.time()
    print_scores(y_test,y_pred,y_pred_prob,pred_time,train_time)


# In[ ]:


def predict(clf,X_test, y_test):
    #predict using test data
    start = time.time()
    y_pred = clf.predict(X_test)
    # Compute predicted probabilities: y_pred_prob
    y_pred_prob = clf.predict_proba(X_test)
    stop = time.time()
    print(f"Prediction time: {stop - start} s")
    metrics.plot_roc_curve(clf, X_test, y_test)
    skplt.metrics.plot_cumulative_gain(y_test, y_pred_prob)
    plt.show()
    pred_time= stop - start
    return y_pred, y_pred_prob, pred_time


# In[ ]:


def test_model(clf, X_test, y_test):
    start = time.time()
    y_pred, y_pred_prob, pred_time = predict(clf, X_test, y_test)
    stop = time.time()
    train_time = 999999
    print_scores(y_test,y_pred,y_pred_prob,pred_time, train_time)


# In[ ]:


scoring = ['accuracy','precision','recall','roc_auc','f1']


# In[ ]:


def print_metrics(y_test,y_pred,y_pred_prob,pred_time,train_time):
    cm1= confusion_matrix(y_test,y_pred)
    text = """\hline
\\textbf{{Accuracy}}  & : {0:.3f} & \\textbf{{Specificity}} & : {1:.3f} \\\\ \hline
\\textbf{{Precision}} & : {2:.3f} & \\textbf{{ROC/AUC}}     & : {3:.3f} \\\\ \hline
\\textbf{{Recall}}    & : {4:.3f} & \\textbf{{F1 Score}}    & : {5:.3f} \\\\ \hline
\\textbf{{Training Time}} & \multicolumn{{1}}{{r|}}{{: {6:.3f} s}} & \\textbf{{Prediction Time}} & \multicolumn{{1}}{{r|}} {{: {7:.3f} s}} \\\\ \hline
\end{{tabular}}
\end{{table}}""".format(accuracy_score(y_test,y_pred),cm1[0,0]/(cm1[0,0]+cm1[0,1]),precision_score(y_test,y_pred),roc_auc_score(y_test, y_pred_prob[:,1]),recall_score(y_test,y_pred),f1_score(y_test,y_pred),train_time,pred_time)
    print(text)


# In[ ]:


def print_cm(cm1):
    text = """\\hline
\\multicolumn{{2}}{{|l|}}{{\\cellcolor[HTML]{{EFEFEF}}}}                   & \\multicolumn{{2}}{{c|}}{{\\cellcolor[HTML]{{EFEFEF}}\\textbf{{Prediction}}}} \\\\ \\cline{{3-4}} 
\\multicolumn{{2}}{{|l|}}{{\\multirow{{-2}}{{*}}{{\\cellcolor[HTML]{{EFEFEF}}}}}} & \\textbf{{Fully Paid (0)}}        & \\textbf{{Charged-off (1)}}        \\\\ \\hline
\\multicolumn{{1}}{{|c|}}{{\\cellcolor[HTML]{{EFEFEF}}}} &
  \\textbf{{Fully Paid (0)}} &
  \\cellcolor[HTML]{{FFFFFF}}\\textbf{{{}}} &
  \\cellcolor[HTML]{{FFFFFF}}\\textbf{{{}}} \\\\ \\cline{{2-4}} 
\\multicolumn{{1}}{{|c|}}{{\\multirow{{-2}}{{*}}{{\\cellcolor[HTML]{{EFEFEF}}\\textbf{{Actual}}}}}} &
  \\textbf{{Charged-off (1)}} &
  \\cellcolor[HTML]{{FFFFFF}}\\textbf{{{}}} &
  \\cellcolor[HTML]{{FFFFFF}}\\textbf{{{}}} \\\\ \\hline
\\end{{tabular}}
\\end{{table}}""".format(cm1[0,0],cm1[0,1],cm1[1,0],cm1[1,1])
    print(text)


# # Gaussian

# In[ ]:


apply_model(GaussianNB(), X_train_g, y_train_g, X_test_g, y_test_g)


# ## GridSearch

# In[ ]:


grid_param_g = {
    'priors': [(0.8,0.2),(0.7,0.3),(0.75,0.25),(0.725,0.275),(None)],
    'var_smoothing': [1e-3,1e-4,1e-5,1e-6,1e-7]
}


# In[ ]:


classifier = GaussianNB()
gd_sr_g = GridSearchCV(estimator=classifier,
                     param_grid=grid_param_g,
                     scoring=scoring,refit='roc_auc',
                     cv=10,
                     n_jobs=-1)


# In[ ]:


gd_sr_g.fit(X_train_g, y_train_g)


# In[ ]:


allscores=gd_sr_g.cv_results_
score = pd.DataFrame(allscores,columns=['param_priors','param_var_smoothing','mean_fit_time','mean_test_accuracy','mean_test_precision','mean_test_recall','mean_test_roc_auc','mean_test_f1'])
score


# In[ ]:


best_parameters = gd_sr_g.best_params_
print(best_parameters)


# In[ ]:


score.to_csv('gnb_res.csv')
allscore=pd.DataFrame(allscores)
allscore.to_csv('all_gnb_res.csv')


# In[ ]:


test_model(gd_sr_g,X_test_g,y_test_g)


# # Categorical

# In[ ]:


apply_model(CategoricalNB(), X_train_c, y_train_c, X_test_c, y_test_c)


# In[ ]:


grid_param_c = {
    'alpha': [0.25,0.5,0.75,1]
    ,'fit_prior': [True,False]
    ,'class_prior': [(0.8,0.2),(0.7,0.3),(0.75,0.25),(0.725,0.275),(None)]
}


# In[ ]:


classifier = CategoricalNB()
gd_sr = GridSearchCV(estimator=classifier,
                     param_grid=grid_param_c,
                     scoring=scoring,refit='roc_auc',
                     cv=10,
                     n_jobs=-1)


# In[ ]:


gd_sr.fit(X_train_c, y_train_c)


# In[ ]:


allscores=gd_sr.cv_results_
score = pd.DataFrame(allscores,columns=['param_alpha','param_class_prior','param_fit_prior','mean_fit_time','mean_test_accuracy','mean_test_precision','mean_test_recall','mean_test_roc_auc','mean_test_f1'])
score


# In[ ]:


best_parameters = gd_sr.best_params_
print(best_parameters)


# In[ ]:


score.to_csv('cnb_res.csv')
allscore=pd.DataFrame(allscores)
allscore.to_csv('all_cnb_res.csv')


# In[ ]:


test_model(gd_sr,X_test_c,y_test_c)

# # %% [markdown]
# # # Mixed

# # %% [code]
# df=df.drop('is_charged_off',axis=1)

# # %% [code]
# def idx_cat(df):
#     q = []
#     n = []
#     for col in categorical_features:
#         print(col,' ',df.columns.get_loc(col),' ',df[col].nunique())
#         q.append(df.columns.get_loc(col)) 
#         n.append(df[col].nunique()) 
#     return q,n

# # %% [code]
# idx, nmax = idx_cat(df)

# # %% [code]
# classifier = MixedNB(categorical_features=idx,max_categories=nmax)

# # %% [code]
# apply_model(MixedNB(categorical_features=idx,max_categories=nmax), X_train, y_train, X_test, y_test)

# # %% [markdown]
# # ## Pencarian Parameter Terbaik: Grid Search

# # %% [code]
# grid_param_m = {
#     'categorical_features': idx, 
#     'max_categories': nmax, 
#     'alpha': [0.25,0.5,0.75,None],
#     'priors': [0.7,0.3],
#     'var_smoothing': [1e-9,2e-9]
# }

# # %% [code]
# classifier = MixedNB()
# gd_sr = GridSearchCV(estimator=classifier,
#                      param_grid=grid_param_m,
#                      scoring=scoring,refit='accuracy',
#                      cv=5,
#                      n_jobs=-1)

# # %% [code]
# gd_sr.fit(X_train, y_train)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




