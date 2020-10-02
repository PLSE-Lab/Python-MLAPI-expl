#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[3]:


from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display
from sklearn import metrics


# In[4]:


PATH = '../input'

get_ipython().system('ls {PATH}')


# In[5]:


df_raw = pd.read_csv(f'{PATH}/train.csv', low_memory=False)


# In[6]:


def display_all(df):
    with pd.option_context('display.max_rows',1000):
        with pd.option_context('display.max_columns', 1000):
            display(df);
            
display_all(df_raw.tail().transpose())


# In[7]:


df_raw.info()


# In[8]:


def feature_engineering(df):
    df['HF1'] = df['Horizontal_Distance_To_Hydrology']+df['Horizontal_Distance_To_Fire_Points']
    df['HF2'] = abs(df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Fire_Points'])
    df['HR1'] = abs(df['Horizontal_Distance_To_Hydrology']+df['Horizontal_Distance_To_Roadways'])
    df['HR2'] = abs(df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Roadways'])
    df['FR1'] = abs(df['Horizontal_Distance_To_Fire_Points']+df['Horizontal_Distance_To_Roadways'])
    df['FR2'] = abs(df['Horizontal_Distance_To_Fire_Points']-df['Horizontal_Distance_To_Roadways'])
    df['ele_vert'] = df.Elevation-df.Vertical_Distance_To_Hydrology

    df['slope_hyd'] = (df['Horizontal_Distance_To_Hydrology']**2+df['Vertical_Distance_To_Hydrology']**2)**0.5
    df.slope_hyd=df.slope_hyd.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any

    #Mean distance to Amenities 
    df['Mean_Amenities']=(df.Horizontal_Distance_To_Fire_Points + df.Horizontal_Distance_To_Hydrology + df.Horizontal_Distance_To_Roadways) / 3 
    #Mean Distance to Fire and Water 
    df['Mean_Fire_Hyd']=(df.Horizontal_Distance_To_Fire_Points + df.Horizontal_Distance_To_Hydrology) / 2 
    
    df['Neg_Elevation_Vertical'] = df['Elevation']-df['Vertical_Distance_To_Hydrology']
    df['Elevation_Vertical'] = df['Elevation']+df['Vertical_Distance_To_Hydrology']

    df['mean_hillshade'] =  (df['Hillshade_9am']  + df['Hillshade_Noon'] + df['Hillshade_3pm'] ) / 3

    df['Mean_HorizontalHydrology_HorizontalFire'] = (df['Horizontal_Distance_To_Hydrology']+df['Horizontal_Distance_To_Fire_Points'])/2
    df['Mean_HorizontalHydrology_HorizontalRoadways'] = (df['Horizontal_Distance_To_Hydrology']+df['Horizontal_Distance_To_Roadways'])/2
    df['Mean_HorizontalFire_Points_HorizontalRoadways'] = (df['Horizontal_Distance_To_Fire_Points']+df['Horizontal_Distance_To_Roadways'])/2

    df['MeanNeg_Mean_HorizontalHydrology_HorizontalFire'] = (df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Fire_Points'])/2
    df['MeanNeg_HorizontalHydrology_HorizontalRoadways'] = (df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Roadways'])/2
    df['MeanNeg_HorizontalFire_Points_HorizontalRoadways'] = (df['Horizontal_Distance_To_Fire_Points']-df['Horizontal_Distance_To_Roadways'])/2

    df['Slope2'] = np.sqrt(df['Horizontal_Distance_To_Hydrology']**2+df['Vertical_Distance_To_Hydrology']**2)
    df['Mean_Fire_Hydrology_Roadways']=(df['Horizontal_Distance_To_Fire_Points'] + df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Roadways']) / 3
    df['Mean_Fire_Hyd']=(df['Horizontal_Distance_To_Fire_Points'] + df['Horizontal_Distance_To_Hydrology']) / 2 

    df["Vertical_Distance_To_Hydrology"] = abs(df['Vertical_Distance_To_Hydrology'])

    df['Neg_EHyd'] = df.Elevation-df.Horizontal_Distance_To_Hydrology*0.2
    
    return df


# In[9]:


df_raw = feature_engineering(df_raw)


# In[10]:


import matplotlib.pyplot as plt
df_raw.hist(bins=50, figsize=(24,20))
plt.show()


# In[11]:


df_raw.info()


# In[12]:


y = df_raw['Cover_Type']
df = df_raw.drop(columns='Cover_Type', axis=1)


# In[13]:


from sklearn.model_selection  import train_test_split

X_train, X_val, y_train, y_val = train_test_split(df, y, test_size=0.2, random_state=32)


# In[14]:


from sklearn.model_selection import cross_val_score


# In[15]:


from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression


# In[16]:


rand_forest_clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=1, max_features=0.5, random_state=42)
extra_tree_clf=ExtraTreesClassifier(n_estimators=100, min_samples_leaf=1, max_features=0.5, bootstrap=True, random_state=42)
svm_clf = LinearSVC(random_state=42)
log_reg_clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)


# In[17]:


estimators = [rand_forest_clf, extra_tree_clf, svm_clf, log_reg_clf]

for estimator in estimators:
    print ("Training estimator: ", estimator)
    estimator.fit(X_train, y_train)


# In[18]:


[estimator.score(X_val, y_val) for estimator in estimators ]


# In[ ]:


from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone


# def Stacking(folds, model, df_train, y, df_test):
#     n_folds = StratifiedKFold(n_splits=folds, random_state=29).split(df_train, y.values)
#     test_pred = np.empty((df_test.shape[0],),float)
#     train_pred = np.empty((df_train.shape[0],),float)
#     test_pred_skf = np.empty((folds, df_test.shape[0]))
#     score = np.empty((0,1), float)
#     for i, (x_train_id, x_val_id) in enumerate(n_folds):
#         X_train, X_val = df_train.iloc[x_train_id],df_train.iloc[x_val_id]
#         y_train, y_val = y.iloc[x_train_id],y.iloc[x_val_id]
#         
#         model.fit(X_train, y_train)
#         train_pred[x_val_id] = model.predict(X_val)
#         test_pred_skf[i,:] =  model.predict(df_test)
#     test_pred[:] = test_pred_skf.mean(axis=0)
#         
#     return np.round(test_pred).reshape(-1,1), train_pred.reshape(-1,1)
#     
#         

# def StackedModel(model, X_train, y_train, X_test, folds):
#     kf=StratifiedKFold(n_splits=folds, random_state=29)
#     train_pred = np.zeros((0,1))
#     test_pred = np.zeros((X_test.shape[0],))
#     test_pred_skf = np.empty((folds, X_test.shape[0], 7))
#         
#     for i,(X_train_index, X_val_index) in enumerate(kf.split(X_train, y_train.values)):
#         X_tr = X_train.iloc[X_train_index]
#         X_te = X_train.iloc[X_val_index]
#         y_tr = y_train.iloc[X_train_index]
#         
#         model.fit(X_tr, y_tr)
#         train_pred = np.append(train_pred,model.predict(X_te))
#         test_pred_skf[i,:] = model.predict_proba(X_test)
#     
#     test_pred[:] = test_pred_skf.mean(axis=0)
#     return test_pred.reshape(-1,1), train_pred.reshape(-1,1)

# len(estimators)

# In[19]:


df_test = pd.read_csv(f'{PATH}/test.csv')


# In[20]:


df_test = feature_engineering(df_test)


# kf=StratifiedKFold(n_splits=5, random_state=1, shuffle=True).split(df, y.values)
# train_pred = np.empty((0,1))
# test_pred = np.empty((test.shape[0],))
# test_pred_skf = np.empty((5, test.shape[0] ,7))
#         
# for i,(X_train_index, X_val_index) in enumerate(kf):
#     X_tr = df.iloc[X_train_index]
#     X_te = df.iloc[X_val_index]
#     y_tr = y.iloc[X_train_index]
#     
#     print(i)
#     print(len(X_train_index))
#     
#     rand_forest_clf.fit(X_tr, y_tr)
#     train_pred = np.append(train_pred,rand_forest_clf.predict(X_te))
#     test_pred_skf[i,:] = log_reg_clf.predict_proba(test)
#     test_pred[:] = test_pred_skf[i].mean(axis=1)
# 

# len(test_pred)

# test_pred

# train_pred = pd.DataFrame(train_pred)

# train_pred.shape

# %%time
# #test_pred1 ,train_pred1 = StackedModel(folds=5, model=rand_forest_clf, df_train=df, y=y, df_test = df_test)
# test_pred ,train_pred =  StackedModel(folds=5,models=estimators, X_train=df, y_train=y, X_test = df_test)
# train_pred = pd.DataFrame(train_pred)
# test_pred = pd.DataFrame(test_pred)

# test_pred

# %%time
# #test_pred1 ,train_pred1 = StackedModel(folds=5, model=rand_forest_clf, df_train=df, y=y, df_test = df_test)
# test_pred1 ,train_pred1 =  StackedModel(folds=5,models=rand_forest_clf, X_train=df, y_train=y, X_test = df_test)
# train_pred1 = pd.DataFrame(train_pred1)
# test_pred1 = pd.DataFrame(test_pred1)

# %%time
# #test_pred2 ,train_pred2 = Stacking(folds=5, model=extra_tree_clf, df_train=df, y=y, df_test = df_test)
# test_pred2 ,train_pred2 =  StackedModel(folds=5,model=extra_tree_clf, X_train=df, y_train=y, X_test = df_test)
# train_pred2 = pd.DataFrame(train_pred2)
# test_pred2 = pd.DataFrame(test_pred2)
# 

# test_pred2.describe()

# %%time
# #test_pred3 ,train_pred3 = Stacking(folds=5, model=svm_clf, df_train=df, y=y, df_test = df_test)
# test_pred3 ,train_pred3 =  StackedModel(folds=5,model=svm_clf, X_train=df, y_train=y, X_test = df_test)
# train_pred3 = pd.DataFrame(train_pred3)
# test_pred3 = pd.DataFrame(test_pred3)
# 

# %%time
# #test_pred4 ,train_pred4 = Stacking(folds=5, model=log_reg_clf, df_train=df, y=y, df_test = df_test)
# test_pred4 ,train_pred4 =  StackedModel(folds=5,model=log_reg_clf, X_train=df, y_train=y, X_test = df_test)
# train_pred4 = pd.DataFrame(train_pred4)
# test_pred4 = pd.DataFrame(test_pred4)
# 

# In[ ]:





# train_pred4.describe()

# df_tr = pd.concat([train_pred1,train_pred2,train_pred3,train_pred4], axis=1)
# df_te = pd.concat([test_pred1,test_pred2,test_pred3,test_pred4], axis=1)

# df_tr.shape

# df_te.head()

# rf = RandomForestClassifier(n_estimators = 700)
# rf.fit(train_pred.ravel(), y)
# 
# preds = rf.predict(test_pred)

# In[ ]:


#score1, score2, score3, score4


# In[21]:


preds1 = rand_forest_clf.predict(df);
preds2= extra_tree_clf.predict(df);
preds3 = svm_clf.predict(df);
preds4 = log_reg_clf.predict(df);


# In[22]:


X_final = pd.DataFrame({'rf':preds1,'et':preds2,'svm':preds3,'log_reg':preds4})
X_final.head(5)


# In[23]:


ID_test = df_test.Id


# In[24]:


X_final.tail(), y.tail()


# In[25]:


preds1 = rand_forest_clf.predict(df_test);
preds2= extra_tree_clf.predict(df_test);
preds3 = svm_clf.predict(df_test);
preds4= log_reg_clf.predict(df_test);


# In[26]:


X_test_final = pd.DataFrame({'rf':preds1,'et':preds2,'svm':preds3,'log_reg':preds4})
X_test_final.head(5)


# In[29]:


from sklearn.ensemble import VotingClassifier


# In[30]:


import xgboost as xgb
from xgboost import XGBClassifier


# In[33]:


rf = RandomForestClassifier(n_estimators = 700)
et=ExtraTreesClassifier(n_estimators=700)
svm = LinearSVC(random_state=42)
log_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
xgb = XGBClassifier()


# In[45]:


voting_clf = VotingClassifier(estimators=[('rf', rf),('et', et), ('lr', log_reg),('xgb', xgb)], voting='soft')

voting_clf.fit(X_final, y)


# In[46]:


preds = voting_clf.predict(X_test_final)


# rf = RandomForestClassifier(n_estimators = 700)
# rf.fit(X_final, y)
# 
# #preds = rf.predict(X_test_final)

# In[47]:


submission = pd.DataFrame({
    "ID": df_test.Id,
    "Cover_Type": preds
})
submission.to_csv('my_submission.csv', index=False)


# In[48]:


submission.head()


# In[ ]:





# In[ ]:




