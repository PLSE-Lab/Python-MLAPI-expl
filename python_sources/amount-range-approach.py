#!/usr/bin/env python
# coding: utf-8

# ## This is my first kernel, comments and corrections are very welcome!!
# ### Please help me to improve!!! :)

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer


# In[ ]:


#plot config
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.rcParams['figure.figsize'] = 24, 12
mpl.rcParams['agg.path.chunksize'] = 500000


# In[ ]:


#name of the columns that starts with V for slicing
v_cols = ["V"+str(i) for i in range(1,29)]

#read the data
df = pd.read_csv("../input/creditcard.csv")


# In[ ]:


# amount_range
df = df.assign(amount_range = pd.qcut(df.Amount, 5, duplicates="drop").cat.codes)


# #### Duplicated transactions
# 
# - It is not a normal behaviour to make the same transaction in the same second two times or more.
# - Could be an error, POS error, human error.. etc.
# - We should protect our clients of this kind of mistakes.
# - If its a normal transaction, the client can retry one second later without problem.
# - Duplicated transactions should be marked as fraud to be blocked.
# - In a previous layer of our system, we will detect this duplicated transactions and block them except the first one.
# - https://chargeback.com/velocity-checks-fraud-prevention/
# 
# #### Amount 0 transactions
# - Could be a mistake. 
# - Could be a tentative, a fraudster making 0-risk-tries to accomodate their techniques to our system to make later a real amount fraud.
# - Could be somebody trying to increase the work load of our system at cost 0, with unknown intentions.
# - We are blocking them too in a previous layer.
# 

# In[ ]:


#remove duplicated transactions
df = df[~df.duplicated(subset=v_cols)]
#remove amount==0 transactions
df = df[df.Amount!=0]


# #### Pareto principle
# - https://en.wikipedia.org/wiki/Pareto_principle
# - The 80% of our volume will come from the 20% of the transactions.
# - As Richard Koch http://richardkoch.net/ suggest in "The 80/20 Principle", we are focusing on that 20% to "achieving more with less".

# In[ ]:


a = df.groupby("amount_range").Amount.sum()
b = df.groupby("amount_range").Amount.count()

r = pd.concat((
    a, 
    b,
    a / a.sum(),
    b / b.sum()
    
), axis=1)
r.columns = ["sum_amount", "count_transactions", "% of total_amount", "% of total_count"]
r.iloc[:,2:].plot(kind="bar")
plt.title("Total amount/total count by amount_range", fontsize=16);


# In[ ]:


#we get that 20% and separate it form the rest of the dataset
principal_80_percent = df[df.amount_range==4]


# #### Our approach will be:
# - Looping through columns combinations and calculate std an kurtosis
# - Identifying the 4 column combination that have a difference which distribution has the less kurtosis.
# - Once we have those 4 combinations, we create the columns in the dataframe.
# - Given that new 4 columns, we calculate the angle of the vector that we can create with those 4 float points for each row.
# - Given the angles of that combinations, we split them into 4 categorical codes and get 4 groups for the principal_80_percent to evaluate.

# In[ ]:



res = pd.DataFrame({
    "cols" : [],
    "std" : [],
    "kur" : []
})

for c in v_cols:
    for c1 in v_cols:
        if c!=c1:
            
            
            z = (principal_80_percent[c]-principal_80_percent[c1])
            
            std  = z.std()
            sk = z.skew()
            kur = z.kurtosis()
            
            push = {
                "cols" : c+"_"+c1,
                "std" : std,
                "kur" : np.abs(kur)
            }
            
            res = res.append([push])

ops = res.sort_values("kur", ascending=True).drop_duplicates("std").iloc[:4].cols.values

for c in ops:
    c1, c2 = c.split("_")

    kwargs = {c1+"_diff_"+c2 : df[c1]-df[c2]}
    df = df.assign(**kwargs)

sel_cols = [i for i in df.columns if i.find("_diff_")!=-1]

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

def fn(x):
    return angle_between([x[0], x[1]], [x[2], x[3]])

df["angles"] = df[sel_cols].apply(fn, axis=1)
df["angles_code"] = pd.cut(df.angles, 5).cat.codes


# In[ ]:


#as we have modified df to add it angles_code, we get again the principal_80_percent
principal_80_percent = df[df.amount_range==4]


# #### For modeling, we are creating two classes of model, one for the principal_80_percent, an SGDClassifier, and a DecisionTreeClassifier for the rest of the dataset.
# -We are instanciating the classifiers for each dataset-split.

# In[ ]:



""" 
    Train-test split, normalize
"""
def tts(df):
    X, y = df[df.columns.difference(["Class"])], df.Class
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    return X_train, X_test, y_train, y_test

def normaliz(X_train,X_test, v_cols, sel_cols):
    mms = Normalizer(norm="l1")
    mms.fit(X_train.loc[:,v_cols+sel_cols])

    X_train_n = X_train.copy()
    X_train_n.at[:,v_cols+sel_cols] = mms.transform(X_train.loc[:,v_cols+sel_cols])

    X_test_n = X_test.copy()
    X_test_n.at[:,v_cols+sel_cols] = mms.transform(X_test.loc[:,v_cols+sel_cols])

    X_train = X_train_n
    X_test = X_test_n
    return X_train, X_test




""" 
    Score metrics 
"""
def confusionm(labels,pred):
    matrix = confusion_matrix(labels,pred)
    return matrix

def classif_rep(labels,pred):
    report = classification_report(labels,pred)
    print(report)

def auc(labels,pred):
    fpr, tpr, threshold = metrics.roc_curve(labels, pred)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc


def show_total_results_from_list(result_list, ds_name="Train"):
    y,p = (result_list)
    print("{} \n".format(ds_name))
    print(confusionm(y,p), "\n")
    print("AUC", auc(y,p), "\n")
    classif_rep(y,p)


def results(npreds=1):
    """
        fit and eval classifiers and stack the results for showing confusion matrix, AUC and classification_report 
    """

    # for the principal 80%, we make different classifieres for each angles_code
    pr4tr, pr4tes = fit_eval_classifier_SGD(principal_80_percent[principal_80_percent.angles_code==4], npreds=npreds)
    pr3tr, pr3tes = fit_eval_classifier_decision_tree(principal_80_percent[principal_80_percent.angles_code==3], npreds=npreds)
    pr2tr, pr2tes = fit_eval_classifier_decision_tree(principal_80_percent[principal_80_percent.angles_code==2], npreds=npreds)
    pr1tr, pr1tes = fit_eval_classifier_decision_tree(principal_80_percent[principal_80_percent.angles_code==1], npreds=npreds)
    pr0tr, pr0tes = fit_eval_classifier_decision_tree(principal_80_percent[principal_80_percent.angles_code==0], npreds=npreds)


    # for the rest of the data, we don't split by angles_code
    rest_of_the_data = df.loc[df.index.difference(principal_80_percent.index)]
    pr_rest_train, pr_rest_test = fit_eval_classifier_decision_tree(rest_of_the_data, npreds=npreds)


    # now we concat all the predictions to get the metrics
    tot_r_train = np.hstack((
        np.stack(np.hstack(pr4tr)),
        np.stack(np.hstack(pr3tr)),
        np.stack(np.hstack(pr2tr)),
        np.stack(np.hstack(pr1tr)),
        np.stack(np.hstack(pr0tr)),
        np.hstack(pr_rest_train)
    ))

    tot_r_test = np.hstack((
        np.stack(np.hstack(pr4tes)),
        np.stack(np.hstack(pr3tes)),
        np.stack(np.hstack(pr2tes)),
        np.stack(np.hstack(pr1tes)),
        np.stack(np.hstack(pr0tes)),
        np.hstack(pr_rest_test)
    ))

    show_total_results_from_list(tot_r_train, ds_name="Train")
    show_total_results_from_list(tot_r_test, ds_name="Test")


   
    
    
"""
    Models
"""    

def fit_eval_classifier_SGD(df, npreds):
    """
        SGD classifier for the principal_80_percent with angle_code == 4
    """
    
    #model and params for GridSearchCV
    parameters = {
        "loss" : ('hinge', 'log' ),
        "penalty" : ('none', 'l2'),
        "alpha" : (0.0001, 0.5, .9, 10, 100),
        "epsilon" : ( .01, 1, 100)
    }

    sgd = SGDClassifier(max_iter=100, shuffle=True)
    
    X_train, X_test, y_train, y_test = tts(df)
    X_train, X_test = normaliz(X_train, X_test, v_cols, sel_cols)

    #GridSearchCV
    clf = GridSearchCV(sgd, parameters, cv=5, scoring=metrics.make_scorer(metrics.roc_auc_score), refit=True)
   
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        clf_fit = clf.fit(X_train[v_cols+sel_cols], y_train)
    
    #fit with best_params
    clf_best = SGDClassifier(**clf_fit.best_params_).fit(X_train[v_cols+sel_cols], y_train)
    
    
    #make and store the predictions
    pr_train = []
    pr_test = []
    for n in range(npreds):

        X_train, X_test, y_train, y_test = tts(df)
        X_train, X_test = normaliz(X_train, X_test, v_cols, sel_cols)

        prs = clf_best.predict(X_train[v_cols+sel_cols])
        pr_train.append(np.array([y_train, prs]))

        prs = clf_best.predict(X_test[v_cols+sel_cols])
        pr_test.append(np.array([y_test, prs]))    

    return pr_train, pr_test


def fit_eval_classifier_decision_tree(df, npreds):

    """
        DecisionTree for the rest of the dataset
    """
        
    X_train, X_test, y_train, y_test = tts(df)
    X_train, X_test = normaliz(X_train, X_test, v_cols, sel_cols)

    pr_train = []
    pr_test = []
    
    #fit
    clf = DecisionTreeClassifier(max_depth=1000, criterion="entropy", presort=True, min_impurity_decrease=1e-20, max_leaf_nodes=500).fit(X_train[v_cols+sel_cols], y_train)

    
    #make and store the predictions
    for n in range(npreds):

        X_train, X_test, y_train, y_test = tts(df)
        X_train, X_test = normaliz(X_train, X_test, v_cols, sel_cols)

        prs = clf.predict(X_train[v_cols+sel_cols])
        pr_train.append(np.array([y_train, prs]))

        prs = clf.predict(X_test[v_cols+sel_cols])
        pr_test.append(np.array([y_test, prs]))    

    return pr_train, pr_test


# In[ ]:


#score  with 1 prediction
results(npreds=1)


# In[ ]:


#score with 10 predictions
results(npreds=10)

