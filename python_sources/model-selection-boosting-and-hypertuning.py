#!/usr/bin/env python
# coding: utf-8

# # Importing Required Packages and Modules

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.svm import SVC # For SVM model
from sklearn.tree import DecisionTreeClassifier # For Decission Tree Classifier
from sklearn import tree
from ipywidgets import interact,interactive
from sklearn.ensemble import RandomForestClassifier # For random Forest Classifier
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression # For Logistic Regression
from xgboost import XGBClassifier,plot_importance # For XGBoost Classifier
from catboost import CatBoostClassifier # For Cat Boost Classifier
from imblearn.over_sampling import SMOTE
from keras.models import Sequential # For Neural Network Sequential Model
from keras.layers import Dense, Activation,Layer,Lambda
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


# # Data Preprocessing Before Modelling

# In[ ]:


df = pd.read_csv("../input/df-cleancsv/df_Clean.csv").iloc[:,1:]
df.head()


# In[ ]:


[f"{i} is {df[i].dtype}" for i in df.columns]


# ## Categorical Data (df_cat)

# In[ ]:


df_cat = pd.DataFrame.copy(df)
columns = ['Product_Age','Call_details','Claim_Value']
for i in df_cat.columns:
    if i not in columns:
        df_cat[i] = df_cat[i].astype('category')
[f"{i} is {df_cat[i].dtype}" for i in df_cat.columns]


# ## Label Encoded data (df_Lable)

# In[ ]:


df_Label = pd.DataFrame.copy(df_cat)
for i in df_Label.columns:
    if df_Label[i].dtype.name =='category':
        enco = preprocessing.LabelEncoder()
        enco.fit(list(set(df_Label[i])))
        df_Label[i] = enco.transform(df_Label[i])


# In[ ]:


df_Label.info()


# ## Normalised Dummy Data (df_norm_dummy)

# In[ ]:


_df_norm = pd.DataFrame.copy(df_cat)
colum = []
for i in _df_norm.columns:
    if _df_norm[i].dtype.name == 'category':
        colum.append(i)
df_dummy = pd.get_dummies(_df_norm, columns =colum[:-1])
min_max_scaler = preprocessing.MinMaxScaler()
Scaled = min_max_scaler.fit_transform(df_dummy[['Product_Age','Call_details','Claim_Value']] )


# In[ ]:


col = list(df_dummy.columns)
_norm_dummy = np.concatenate((Scaled,df_dummy.values[:,3:99]),axis=1)
df_norm_dummy = pd.DataFrame(_norm_dummy,columns=col)
df_norm_dummy


# # Balancing the data Before Modeling

# ## Balancing to Lable Data (X_lab_balance,Y_lab_balance)

# In[ ]:


sm_x_lab = pd.DataFrame.copy(df_Label)
sm_y_lab = sm_x_lab.pop('Fraud')
sm = SMOTE(random_state =101)
X_lab_balance, Y_lab_balance = sm.fit_sample(sm_x_lab, sm_y_lab)
X_lab_balance.shape, Y_lab_balance.shape


# ## Balancing the Categorical Data (X_cat_balance, Y_cat_balance)

# In[ ]:


df_majority = df_cat[df_cat.Fraud==0]
df_minority = df_cat[df_cat.Fraud==1]
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=323,    # to match majority class
                                 random_state=101) # reproducible results
df_cat_balance = pd.concat([df_majority, df_minority_upsampled])
df_cat_balance.Fraud.value_counts()
X_cat_balance = pd.DataFrame.copy(df_cat_balance)
Y_cat_balance = X_cat_balance.pop('Fraud')
X_cat_balance.shape, Y_cat_balance.shape


# ## Balancing The Normalised Dummy data

# In[ ]:


sm_x_nd = pd.DataFrame.copy(df_norm_dummy)
sm_y_nd = sm_x_nd.pop('Fraud')
sm = SMOTE(random_state =101)
X_nd_balance, Y_nd_balance = sm.fit_sample(sm_x_nd, sm_y_lab)
X_nd_balance.shape, Y_nd_balance.shape


# # Train Test Split

# ## For Categorical Data

# In[ ]:


# Balanced
X_train_cat_bal,X_test_cat_bal,Y_train_cat_bal,Y_test_cat_bal = train_test_split(X_cat_balance, Y_cat_balance,test_size=0.3,random_state=101)
X_train_cat_bal.shape,X_test_cat_bal.shape,Y_train_cat_bal.shape,Y_test_cat_bal.shape


# In[ ]:


# Original
X_c = pd.DataFrame.copy(df_cat)
Y_c = X_c.pop('Fraud')
X_train_cat,X_test_cat,Y_train_cat,Y_test_cat = train_test_split(X_c,Y_c,test_size=0.3,random_state=101)
X_train_cat.shape,X_test_cat.shape,Y_train_cat.shape,Y_test_cat.shape


# ## For Labeled Data

# In[ ]:


# Balanced
X_train_lab_bal,X_test_lab_bal,Y_train_lab_bal,Y_test_lab_bal = train_test_split(X_lab_balance, Y_lab_balance,test_size=0.3,random_state=101)
X_train_lab_bal.shape,X_test_lab_bal.shape,Y_train_lab_bal.shape,Y_test_lab_bal.shape


# In[ ]:


# Original
X_l = pd.DataFrame.copy(df_Label)
Y_l = X_l.pop('Fraud')
X_train_lab,X_test_lab,Y_train_lab,Y_test_lab = train_test_split(X_l,Y_l,test_size=0.3,random_state=101)
X_train_lab.shape,X_test_lab.shape,Y_train_lab.shape,Y_test_lab.shape


# ## For Normalised Dummy Data

# In[ ]:


# Balanced
X_train_nd_bal,X_test_nd_bal,Y_train_nd_bal,Y_test_nd_bal = train_test_split(X_nd_balance, Y_nd_balance,test_size=0.3,random_state=101)
X_train_nd_bal.shape,X_test_nd_bal.shape,Y_train_nd_bal.shape,Y_test_nd_bal.shape


# In[ ]:


# Original
X_d = pd.DataFrame.copy(df_norm_dummy)
Y_d = X_d.pop('Fraud')
X_train_nd,X_test_nd,Y_train_nd,Y_test_nd = train_test_split(X_d,Y_d,test_size=0.3,random_state=101)
X_train_nd.shape,X_test_nd.shape,Y_train_nd.shape,Y_test_nd.shape


# <hr>    
# 
# # Model Fitting And Evaluation
# 
# <hr>

# ## A Simple Decission Tree (Using Labeled Data)

# In[ ]:


def treebuild(cri='entropy',mxd=10,minsl=2,rs=28,spl='best'):
    Warrenty_Tree = DecisionTreeClassifier(criterion=cri,max_depth=mxd,min_samples_leaf=minsl,random_state=rs,splitter=spl)
    Warrenty_Tree.fit(X_train_lab_bal,Y_train_lab_bal)
    pred_bal = Warrenty_Tree.predict(X_test_lab_bal)
    pred_ = Warrenty_Tree.predict(X_test_lab)
    prr_bal = Warrenty_Tree.predict(X_train_lab_bal)
    prr_ = Warrenty_Tree.predict(X_train_lab)
    print("Test Accuracy original Data",np.mean(Y_test_lab==pred_))
    print("Train Accuracy Original Data",np.mean(Y_train_lab==prr_))
    print ("Test Accuracy Balanced Data",np.mean(Y_test_lab_bal==pred_bal))
#     print(classification_report(Y_test_lab_bal,pred_bal))
    print("Train Accuracy Balanced Data",np.mean(Y_train_lab_bal==prr_bal))
    print(classification_report(Y_test_lab,pred_))
interact(treebuild,cri=['entropy','gini'],mxd=[i for i in range(1,20)],minsl=[i for i in range(1,10)],rs=[i for i in  range(30)],spl=['best','random'])


# ### Hypertuning the decission tree

# cri=['entropy','gini']
# mxd=[i for i in range(1,20)]
# minsl=[i for i in range(1,10)]
# rs=[i for i in  range(30)]
# spl=['best','random']
# Warrenty_Tree = DecisionTreeClassifier()
# parm_grid = dict(criterion=cri,max_depth=mxd,min_samples_leaf=minsl,random_state=rs,splitter=spl)
# kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=101)
# gridsearch = GridSearchCV(Warrenty_Tree,parm_grid,scoring="balanced_accuracy", n_jobs=4, cv=kfold)
# grid_result = gridsearch.fit(X_lab_balance,Y_lab_balance);
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# __Best: 0.902477 using {'criterion': 'entropy', 'max_depth': 10, 'min_samples_leaf': 2, 'random_state': 28, 'splitter': 'best'}

# ___entropy,8,2,21,random

# ## Random Forest

# In[ ]:


def forests(n_est=20,cri='entropy',mxd=9,mslf=3,mf='auto',rs=9):
    forest = RandomForestClassifier(n_estimators=n_est,criterion=cri,max_depth=mxd,min_samples_leaf=mslf,max_features=mf,random_state=rs)
    forest.fit(X_train_lab_bal,Y_train_lab_bal)
    pred_bal = forest.predict(X_test_lab_bal)
    pred_ = forest.predict(X_test_lab)
    prr_bal = forest.predict(X_train_lab_bal)
    prr_ = forest.predict(X_train_lab)
    print("Test Accuracy original Data",np.mean(Y_test_lab==pred_))
    print(classification_report(Y_test_lab,pred_))
    print ("Test Accuracy Balanced Data",np.mean(Y_test_lab_bal==pred_bal))
    print(classification_report(Y_test_lab_bal,pred_bal))
    print("Train Accuracy Original Data",np.mean(Y_train_lab==prr_))
    print("Train Accuracy Balanced Data",np.mean(Y_train_lab_bal==prr_bal))
interact(forests,n_est=[i for i in range(10,100)],cri=['gini','entropy'],mxd=[i for i in range(1,10)],mslf=[i for i in range(1,10)],mf=['auto','sqrt','log2'],rs=[i for i in range(30)])


# ___So we are getting more than 90% Accuracy from this model (20,entropy,9,3,auto,9) combination

# n_est=[50,100,200]
# cri=['gini','entropy']
# mxd=[i for i in range(1,10)]
# mslf=[i for i in range(1,10)]
# mf=['auto']
# rs=[i for i in range(30)]
# forest = RandomForestClassifier()
# parm_grid = dict(n_estimators=n_est,criterion=cri,max_depth=mxd,min_samples_leaf=mslf,max_features=mf,random_state=rs)
# kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=101)
# gridsearch = GridSearchCV(forest,parm_grid,scoring="balanced_accuracy", n_jobs=4, cv=kfold)
# grid_result = gridsearch.fit(X_lab_balance,Y_lab_balance);
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Best: 0.938080 using {'criterion': 'gini', 'max_depth': 7, 'max_features': 'auto', 'min_samples_leaf': 4, 'n_estimators': 50, 'random_state': 13}

# ## Using the XGB model

# In[ ]:


xgm = XGBClassifier(max_depth=5,learning_rate=0.2,n_estimators=200)
xgm.fit(X_train_lab_bal,Y_train_lab_bal)
pred_bal = xgm.predict(X_test_lab_bal)
pred_ = xgm.predict(X_test_lab.values)
prr_bal = xgm.predict(X_train_lab_bal)
prr_ = xgm.predict(X_train_lab.values)

print("Train Accuracy Original Data",np.mean(Y_train_lab==prr_))
print("Test Accuracy original Data",np.mean(Y_test_lab==pred_))
print(classification_report(Y_test_lab,pred_))
print ("Test Accuracy Balanced Data",np.mean(Y_test_lab_bal==pred_bal))
# print(classification_report(Y_test_lab_bal,pred_bal))

print("Train Accuracy Balanced Data",np.mean(Y_train_lab_bal==prr_bal))


# ## Using Cat Boost

# In[ ]:


modelcat = CatBoostClassifier(learning_rate=0.1,depth=3,n_estimators=400,cat_features=[0,1,2,3,4,5,6,7,8,9,10,11,12,14,16,18])
modelcat.fit(X_train_cat_bal,Y_train_cat_bal)
pred_bal = modelcat.predict(X_test_cat_bal)
pred_ = modelcat.predict(X_test_cat)
prr_bal = modelcat.predict(X_train_cat_bal)
prr_ = modelcat.predict(X_train_cat)


# In[ ]:


print("Test Accuracy original Data",np.mean(Y_test_cat==pred_))
print(classification_report(Y_test_cat,pred_))
print ("Test Accuracy Balanced Data",np.mean(Y_test_cat_bal==pred_bal))
# print(classification_report(Y_test_cat_bal,pred_bal))
print("Train Accuracy Original Data",np.mean(Y_train_cat==prr_))
print("Train Accuracy Balanced Data",np.mean(Y_train_cat_bal==prr_bal))

parm_grid = dict(learning_rate=learning_rate,depth=max_depth,n_estimators=n_estimators)
modelcat = CatBoostClassifier(cat_features=[0,1,2,3,4,5,6,7,8,9,10,11,12,14,16,18])
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
gridsearch = GridSearchCV(modelcat,parm_grid,scoring="balanced_accuracy", n_jobs=4, cv=kfold)
grid_result = gridsearch.fit(X,Y)Best: 0.532262 using {'depth': 3, 'learning_rate': 0.1, 'n_estimators': 400}
# In[ ]:


Proba = pd.DataFrame(modelcat.predict_proba(X_test_cat))
Proba.columns = ['Fraud_no','Fraud_yes']
Proba['Actual'] = list(Y_test_cat)
Proba['Predicted'] = list(pred_)
Proba.loc[(Proba.Actual!=Proba.Predicted)]


# ## Simple logistic Regression

# In[ ]:


methods = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
def Logitt(method):
    lr1 = LogisticRegression(solver=method) 
    lr1.fit(X_train_nd_bal,Y_train_nd_bal)
    pred_bal = lr1.predict(X_test_nd_bal)
    pred_ = lr1.predict(X_test_nd)
    prr_bal = lr1.predict(X_train_nd_bal)
    prr_ = lr1.predict(X_train_nd)
    print("Test Accuracy original Data",np.mean(Y_test_nd==pred_))
    print(classification_report(list(Y_test_nd),pred_))
    print ("Test Accuracy Balanced Data",np.mean(list(Y_test_nd_bal)==pred_bal))
#     print(classification_report(list(Y_test_nd_bal),pred_bal))
    print("Train Accuracy Original Data",np.mean(list(Y_train_nd)==prr_))
    print("Train Accuracy Balanced Data",np.mean(list(Y_train_nd_bal)==prr_bal))
interact(Logitt,method = methods)


# ## Neural Network

# In[ ]:


classifier = Sequential()
classifier.add(Dense(activation="relu", input_dim=97, units=10, kernel_initializer="he_uniform"))
classifier.add(Dense(activation="relu", units=10, kernel_initializer="he_uniform"))
classifier.add(Dense(activation="relu", units=10, kernel_initializer="he_uniform"))
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="glorot_uniform"))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train_nd_bal, Y_train_nd_bal, batch_size = 10, nb_epoch = 100)
pred_bal = classifier.predict(X_test_nd_bal);pred_bal=(pred_bal > 0.5)
pred_ = classifier.predict(X_test_nd);pred_=(pred_ > 0.5)
prr_bal = classifier.predict(X_train_nd_bal);prr_bal=(prr_bal > 0.5)
prr_ = classifier.predict(X_train_nd);prr_=(prr_ > 0.5)


# In[ ]:


print("Test Accuracy original Data",np.mean(list(Y_test_nd)==pred_))
print(classification_report(list(Y_test_nd),pred_))
print ("Test Accuracy Balanced Data",np.mean(list(Y_test_nd_bal)==pred_bal))
# print(classification_report(list(Y_test_nd_bal),pred_bal))
print("Train Accuracy Original Data",np.mean(list(Y_train_nd)==prr_))
print("Train Accuracy Balanced Data",np.mean(list(Y_train_nd_bal)==prr_bal))


# ## SVM

# In[ ]:


svclassifier = SVC(kernel = 'linear')
svclassifier.fit(X_train_nd_bal, Y_train_nd_bal)
pred_bal = svclassifier.predict(X_test_nd_bal)
pred_ = svclassifier.predict(X_test_nd)
prr_bal = svclassifier.predict(X_train_nd_bal)
prr_ = svclassifier.predict(X_train_nd)
print("Test Accuracy original Data",np.mean(Y_test_nd==pred_))
print(classification_report(list(Y_test_nd),pred_))
print ("Test Accuracy Balanced Data",np.mean(list(Y_test_nd_bal)==pred_bal))
# print(classification_report(list(Y_test_nd_bal),pred_bal))
print("Train Accuracy Original Data",np.mean(list(Y_train_nd)==prr_))
print("Train Accuracy Balanced Data",np.mean(list(Y_train_nd_bal)==prr_bal))

