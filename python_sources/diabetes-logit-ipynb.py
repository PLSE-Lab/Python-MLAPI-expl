#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from scipy.stats.stats import pearsonr
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import GridSearchCV
import sklearn.metrics 

import matplotlib.pyplot as plt

def roc_auc(pred, act, plot=True, label = "curve"):
    prob = pred/pred.max() #normalize
    fpr, tpr, threshold = sklearn.metrics.roc_curve(act, prob, drop_intermediate=True)    
    auc = sklearn.metrics.auc(fpr, tpr)

    if plot:
        plt.scatter(x=fpr, y=tpr, color='navy')
        rcolor = tuple(np.random.rand(3,1)[:,0])
        plt.plot(fpr, tpr, c=rcolor, lw=2, label=label + ' (AUC = %0.3f)' % auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()

    return auc

### laptop
#datain_dir = "~/Data/diabetes/"
#datain_col_dir = '~/Data/diabetes/'
#dataout_dir = '~/Data/diabetes/'

### kaggle

datain_dir = '../input/diabetes/'
datain_col_dir = '../input/diabetes-readmissions-column-annotation/'
dataout_dir = ''


# # Read in the data

# In[ ]:


df_raw_all = pd.read_csv(datain_dir + 'diabetic_data.csv') 
df_raw = df_raw_all.sample(10000)
df_raw = df_raw.replace('?', np.nan) 
df_raw.shape


# # How many values are missing in patient records?

# In[ ]:


pt_sparsity = df_raw.isnull().apply(sum, axis=1)
get_ipython().run_line_magic('matplotlib', 'inline')
myhist = pt_sparsity.hist()


# No need to drop patients due to missing values. 3 out of 50 missing values is not bad at all.
# 
# # Explore the variables, compile information about them

# In[ ]:


col_data = df_raw.apply(lambda s: set(s.unique()), axis=0).to_frame('uni_val')
col_data['nan_rat'] = df_raw.isnull().sum(axis=0)/len(df_raw)
col_data['n_uni_vals'] = col_data.uni_val.apply(len)
col_data['uni_vals_str'] = col_data[col_data.n_uni_vals<2000].uni_val.astype(str)
col_data = col_data.drop('uni_val', axis=1)
col_data['var_type'] = np.nan
col_data.to_csv(dataout_dir + "columns_raw.csv")


# # Manual annotation of columns
# I manually took columns_raw.csv and annotated columns based on whether they were ordered or categorical.  I saved my annotated file as columns.csv.  I will read this in later in the notebook.

# In[ ]:


col_data = pd.read_csv(datain_col_dir + "columns.csv", index_col=0)
col_data.sample(10)


# In[ ]:


#TODO recapture medical specialty
spec_counts = df_raw.medical_specialty.value_counts()
spec_counts.head(5).to_frame('num patients')


# In[ ]:


spec_thresh = 5
for (spec, count) in spec_counts.head(spec_thresh).iteritems():
    new_col = 'spec_' + str(spec)
    df_raw[new_col] = (df_raw.medical_specialty == spec)
    
df_raw.filter(regex='spec').sample(10)


# # Identify the most common diagnoses

# In[ ]:


diag_counts = (df_raw.diag_1.value_counts() + df_raw.diag_2.value_counts() + df_raw.diag_3.value_counts()).sort_values(ascending=False)
diag_counts.head(10).to_frame('num patients w diag')


# # Generate boolean features for top N diagnoses

# In[ ]:


diag_thresh = 10
for (icd9, count) in diag_counts.head(diag_thresh).iteritems():
    new_col = 'diag_' + str(icd9)
    df_raw[new_col] = (df_raw.diag_1 == icd9)|(df_raw.diag_2 == icd9)|(df_raw.diag_3 == icd9)
    
df_raw.filter(regex='diag_').sample(10)


# # Clean the data

# In[ ]:


df_raw.age.sample(10)


# In[ ]:


df_raw2 = pd.DataFrame(df_raw, copy=True) #preserve df_raw so I can rerun this step
df_raw2['age'] = df_raw2.age.str.extract('(\d+)-\d+')

to_drop = col_data[col_data.var_type.str.contains('drop')].index
df_raw2.drop(to_drop, axis=1, inplace=True)

#break out categorical variables into binaries
cat_cols = col_data[col_data.var_type.str.contains('cat')].index
df_raw2 = pd.get_dummies(df_raw2, columns=cat_cols)

#dropping these leaves up with one binary variable, ideal for simplicity
df_raw2.drop(['readmitted_<30','readmitted_>30'], axis=1, inplace=True)

#cleaning up outcome variable
df_raw2['is_readmitted'] = (df_raw2.readmitted_NO == 0)
df_raw2.drop('readmitted_NO', axis=1, inplace=True)

#ta daaaaaah, the data is ready to go
df = pd.DataFrame(df_raw2)
df.shape


# In[ ]:


df.age.sample(10)


# In[ ]:


df.sample(15).sample(7, axis=1)


# # Examine outcome variable

# In[ ]:


df.is_readmitted.value_counts()


# # Define this machine learning problem, impute, set aside test data

# In[ ]:


#partition training and test data, one balanced training set, all remaining for testing 
outcome_column = 'is_readmitted' 

#Imputing with outlying value since we are focusing on tree based methods
dff = df.fillna(df.mean) # using mean rather than -9999 which I use for tree methods 

#%% Split data for validation
X = dff.drop(outcome_column, axis=1) 
y = dff[outcome_column] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0) 


# # One on one correlations

# In[ ]:


##### SINGULAR LOGISTIC REGRESSION
def apply_pearsonr(col):
    isnan = col.isnull()
    xi = col[isnan==False].astype(float)
    yi = y[isnan==False].astype(float)
    (r,p) = pearsonr(xi,yi)
    return (r,p)

res = X.apply(apply_pearsonr).to_frame()
res['r'] = res[0].apply(lambda x: x[0])
res['p'] = res[0].apply(lambda x: x[1])

res.drop(0, axis=1).sort_values('p').head(15)
#TODO - apply bonferoni correction


# # Fit data to Logistic Regression model, using Lasso penalization and gridsearch

# In[ ]:


C_params = [10**x for x in np.arange(-2.5,2.5,0.1)]
param_grid = {
        'penalty':['l1'], #lasso only
        'C':C_params
}

clf = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='roc_auc', verbose=3, n_jobs=10)
clf.fit(X_train, y_train)


# In[ ]:


clf.best_params_


# In[ ]:


lr = clf.best_estimator_
feat_imp = pd.Series(index=X_train.columns, data=lr.coef_[0])
feat_imp.sort_values(ascending=False)


# This looks like noise, it doesn't have any similarity with important tree variables or with what I'd expect.

# In[ ]:


#REDO, with more stringent C value
lr2 = LogisticRegression(C=.015, penalty='l1')
lr2.fit(X_train, y_train)
feat_imp = pd.Series(index=X_train.columns, data=lr2.coef_[0])
feat_imp[feat_imp!=0].sort_values(ascending=False)


# # Assess prediction accuracy

# In[ ]:


#%% assess accuracy
pred = lr.predict_proba(X_test)[:,1]
fpr, tpr, threshold = sklearn.metrics.roc_curve(y_test, pred, drop_intermediate=True)    
df_res = pd.DataFrame(data={'fpr':fpr, 'tpr':tpr, 'threshold':threshold})
df_res = df_res[['threshold','fpr','tpr']]
sklearn.metrics.auc(fpr, tpr)
t=y.value_counts()[1]/y.value_counts().sum()
sklearn.metrics.f1_score(y_test, pred>t)
sklearn.metrics.accuracy_score(y_test, pred>t)

roc_auc(pred, y_test)


# In[ ]:


pd.options.mode.chained_assignment = None
pred = lr.predict_proba(X_test)[:,1]
fpr, tpr, threshold = sklearn.metrics.roc_curve(y_test, pred, drop_intermediate=True)    
df_res = pd.DataFrame(data={'fpr':fpr, 'tpr':tpr, 'threshold':threshold})
df_res = df_res[['threshold','fpr','tpr']]


# In[ ]:


df_res['accuracy'] = df_res.threshold.apply(lambda t: sklearn.metrics.accuracy_score(y_test, pred>t))
df_res['precision'] = df_res.threshold.apply(lambda t: sklearn.metrics.precision_score(y_test, pred>t))
df_res['recall'] = df_res.threshold.apply(lambda t: sklearn.metrics.recall_score(y_test, pred>t))
df_res['f1'] = df_res.threshold.apply(lambda t: sklearn.metrics.f1_score(y_test, pred>t))
df_res['specificity'] = df_res.fpr.apply(lambda fpr: 1-fpr)

pt_opt = df_res[df_res.f1 == df_res.f1.max()].iloc[0]
pt_opt


# In[ ]:


plt.rcParams["figure.figsize"] = (20,6)
df_res.plot(x='threshold')


# In[ ]:




