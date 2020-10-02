#!/usr/bin/env python
# coding: utf-8

# # Janata Hack eCommerce Analytics
# 
# Determine the gender of shopper from ecommerce website data
# 
# https://datahack.analyticsvidhya.com/contest/janatahack-e-commerce-analytics-ml-hackathon

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


test = pd.read_csv('../input/janatahack/test_Yix80N0.csv')
train = pd.read_csv('../input/janatahack/train_8wry4cB.csv')


# In[ ]:


test.head()


# In[ ]:


train.shape


# It seems there is lot of information hidden in the ProductList variable. We need to extract as much information as possible. Here are the new features I can think which we can come up from this data :
#     - Time spent in a session and other date time features
#     - category, sub-category, sub-sub category and the product
#     - number of products browsed per session
#     - number of categories browsed per session
#     - number of products per hour

# In[ ]:


train.nunique()


# In[ ]:


sns.heatmap(train.isnull())
plt.show()


# Awesome! No missing data :)

# In[ ]:


df=train.append(test,ignore_index=True)
df['n_product']  = df['ProductList'].apply(lambda s : s.count(';')+1)


# In[ ]:


df['n_product'].unique()


# In[ ]:


sns.countplot(df['n_product'])


# In[ ]:


new = df['ProductList'].str.split(";",expand = True)
new.fillna(value=0, inplace=True)


# In[ ]:


new.head(15)


# In[ ]:


new1 = new[0].str.split("/",expand=True)


# In[ ]:


new1.drop(labels = 4, axis = 1, inplace = True)


# In[ ]:


new1= new1.rename(columns={0: "cat", 1:"scat",2:"sscat", 3:"prod"})


# In[ ]:


df1 = pd.concat([df,new1],axis=1)


# In[ ]:


df1.info()


# In[ ]:


dateparser = lambda x : pd.datetime.strptime(x,"%d/%m/%y %H:%M")
df1['startTime'] = df1['startTime'].apply(dateparser)
df1['endTime'] = df1['endTime'].apply(dateparser)


# In[ ]:


df1['time_difference'] = (pd.to_datetime(df1['endTime']) - pd.to_datetime(df1['startTime'])).dt.total_seconds()/60
df1['st_month'] = pd.to_datetime(df1['startTime']).dt.month
df1['st_day'] = pd.to_datetime(df1['startTime']).dt.day
df1['st_hour'] = pd.to_datetime(df1['startTime']).dt.hour


# In[ ]:


def prodperhour(df):

    if (df['time_difference'] == 0):
        return 0.0
    else:
        return df['n_product']/df['time_difference']

df1['prodperhour'] = df1.apply(prodperhour, axis = 1)
                         
                        # ['red' if x == 'Z' else 'green' for x in df['Set']]


# In[ ]:


df1.head(10)


# In[ ]:


df1['cat'].unique()


# In[ ]:


df2=df1
a=0
df2['n_cat']=0
cols = []

for i in df1['cat'].unique() :
    #print(i)
    #df2.head()
    
    col = 'n_'+i
    cols.append(col)
    df2[col] = df2.ProductList.str.contains(i)
    df2[col] = df2[col].apply(lambda x : 1 if x else 0)
    df2['n_cat'] = df2[col] + df2['n_cat']

#df2.drop(columns = cols,inplace=True)

print(df2.columns)


# In[ ]:


df3=df2

df3['n_scat']=0
cols = []

for i in df1['scat'].unique() :
    #print(i)
    #df2.head()
    col = 'n_'+i
    cols.append(col)
    df3[col] = df3.ProductList.str.contains(i)
    df3[col] = df3[col].apply(lambda x : 1 if x else 0)
    df3['n_scat'] = df3[col] + df3['n_scat']

#df3.drop(columns = cols,inplace=True)


print(df3.columns)
df1=df3


# In[ ]:


df4=df3

df4['n_sscat']=0
cols = []

for i in df1['sscat'].unique() :
    #print(i)
    #df2.head()
    col = 'n_'+i
    cols.append(col)
    df4[col] = df4.ProductList.str.contains(i)
    df4[col] = df4[col].apply(lambda x : 1 if x else 0)
    df4['n_sscat'] = df4[col] + df4['n_scat']

#df3.drop(columns = cols,inplace=True)


print(df4.columns)
df1=df4


# In[ ]:


# df4=df1

# cols = []

# for i in df1['prod'].unique() :
#     #print(i)
#     #df2.head()
#     col = 'n_'+i
#     cols.append(col)
#     df4[col] = df4.ProductList.str.contains(i)
#     df4[col] = df4[col].apply(lambda x : 1 if x else 0)
#     #df4['n_sscat'] = df4[col] + df4['n_scat']

# #df3.drop(columns = cols,inplace=True)


# print(df4.columns)
# df1=df4


# In[ ]:


df1.drop(columns=['endTime','startTime','ProductList'],inplace=True)


# In[ ]:


df1.head()


# In[ ]:


df1['n_cat'].value_counts()


# In[ ]:



# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
#cols = ['prod']

df1 = pd.get_dummies(data = df1, columns = ['prod'])
#df2.drop(columns='st_month',axis=1,inplace=True)

#df1 = df1.drop(col['sex','region','smoker'],axis=1)

#  for a in cols :
#     df1[a] = le.fit_transform(df1[a])
df1.head()
#df1 = pd.concat([df1,df2],axis=1)

df1.drop(columns=['cat','scat','sscat'],axis=1,inplace=True)


# In[ ]:


df1.columns[df1.columns.duplicated()]


# In[ ]:


df_train=df1[df1['gender'].isnull()==False].copy()
df_test=df1[df1['gender'].isnull()==True].copy()

print(df_train.shape,df_test.shape)


# In[ ]:


test_ids = df_test['session_id'] 


# In[ ]:


df_train.head()


# In[ ]:


df_train['gender']=df_train['gender'].apply(lambda x : 1 if x == 'male' else 0)
df_train.drop(columns='session_id',axis=1, inplace=True)
df_test.drop(columns=['gender','session_id'],axis=1, inplace=True)


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


x = df_train.drop('gender',axis=1)
y = df_train['gender']


# In[ ]:



from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.25,random_state=42)


# Modeling

# In[ ]:


from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve,roc_auc_score

def disp_confusion_matrix(model, x, y):
    ypred = model.predict(x)
    cm = confusion_matrix(y,ypred)
    ax = sns.heatmap(cm,annot=True,fmt='d')

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    plt.show()
    
    tp = cm[1,1]
    fn = cm[1,0]
    fp = cm[0,1]
    tn = cm[0,0]
    accuracy = (tp+tn)/(tp+fn+fp+tn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = (2*precision*recall)/(precision+recall)
    print('Accuracy =',accuracy)
    print('Precision =',precision)
    print('Recall =',recall)
    print('F1 Score =',f1)

def disp_roc_curve(model, xtest, ytest):
    yprob = model.predict_proba(xtest)
    fpr,tpr,threshold = roc_curve(ytest,yprob[:,1])
    roc_auc = roc_auc_score(ytest,yprob[:,1])

    print('ROC AUC =', roc_auc)
    plt.figure()
    lw = 2
    plt.plot(fpr,tpr,color='darkorange',lw=lw,label='ROC Curve (area = %0.2f)'%roc_auc)
    plt.plot([0,1],[0,1],color='navy',lw=lw,linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()


# Feature Selection

# In[ ]:


sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
sel.fit(xtrain, ytrain)


# In[ ]:


selected_feat= xtrain.columns[(sel.get_support())]
len(selected_feat)


# In[ ]:


df1[selected_feat].head()


# In[ ]:


df_train=df1[df1['gender'].isnull()==False].copy()
df_test=df1[df1['gender'].isnull()==True].copy()

print(df_train.shape,df_test.shape)


# In[ ]:


df_train['gender']=df_train['gender'].apply(lambda x : 1 if x == 'male' else 0)
df_train.drop(columns='session_id',axis=1, inplace=True)
df_test.drop(columns=['gender','session_id'],axis=1, inplace=True)


# In[ ]:


x = df_train.drop('gender',axis=1)
y = df_train['gender']
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.25,random_state=42)


# Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

rf = RandomForestClassifier()
rf.fit(xtrain, ytrain)
print('Training set score = {:.3f}'.format(rf.score(xtrain,ytrain)))
print('Test set score = {:.3f}'.format(rf.score(xtest,ytest)))


# In[ ]:




disp_confusion_matrix(rf, xtest, ytest)
disp_roc_curve(rf, xtest, ytest)


# In[ ]:



rf.fit(x,y)
test_prob = rf.predict_proba(df_test)[:,1]
df_rf = pd.DataFrame({'session_id':test_ids,'gender':test_prob})


# In[ ]:


import time
times = time.strftime("%Y%m%d-%H%M%S")

th = 0.5

df_rf['gender'] = df_rf['gender'].apply(lambda x : 'male' if x>th else 'female')


# In[ ]:


df_rf['gender'].value_counts()


# In[ ]:


import time
times = time.strftime("%Y%m%d-%H%M%S")
df_rf.to_csv('submission-rf_'+times+'.csv',index=False)


# In[ ]:


pd.DataFrame({'Features':x.columns, 'Importance':rf.feature_importances_}).sort_values(by='Importance',ascending=False)


# Gradient Boosting

# In[ ]:


from xgboost import XGBClassifier
xgb = XGBClassifier(objective='binary:logistic')
xgb.fit(xtrain,ytrain)
print('Training set score = {:.3f}'.format(xgb.score(xtrain,ytrain)))
print('Test set score = {:.3f}'.format(xgb.score(xtest,ytest)))


# In[ ]:



disp_confusion_matrix(xgb, xtest, ytest)
disp_roc_curve(xgb, xtest, ytest)


# In[ ]:


xgb.fit(x,y)
test_prob = xgb.predict_proba(df_test)[:,1]
df_xgb = pd.DataFrame({'session_id':test_ids,'gender':test_prob})
df_xgb.head()


# In[ ]:


df_xgb['gender'] = df_xgb['gender'].apply(lambda x : 'male' if x>0.5 else 'female')


# In[ ]:


df_xgb['gender'].value_counts()


# In[ ]:


import time
times = time.strftime("%Y%m%d-%H%M%S")
df_xgb.to_csv('submission-xgb'+times+'.csv',index=False)


# In[ ]:


arr=[100,150,200,250]
cv_scores = []
for a in arr:
    model = XGBClassifier(objective='binary:logistic', n_jobs=4, n_estimators=a)
    cv_score = cross_val_score(model, x, y, cv=5, scoring='roc_auc')
    print(a, ':', cv_score)
    cv_scores.append(cv_score)
    
fig, ax = plt.subplots(figsize=(14,6))
plt.boxplot(cv_scores)
ax.set_xticklabels(arr)
plt.xlabel('n')
plt.ylabel('roc_auc')
plt.show()


# In[ ]:


param_grid = {
    'max_depth':[5,6],
    'subsample':[0.8,0.9,1],
    'colsample_bytree': [0.6,0.8,1],
    'min_child_weight': [0.5,1],
    'gamma': [0,0.5,1]
}
xgb = XGBClassifier(objective='binary:logistic', n_jobs=4, n_estimators=100)
xgb_grid = GridSearchCV(xgb, param_grid, cv=5, scoring='roc_auc', verbose=1, n_jobs=4)
xgb_grid.fit(x, y)


# In[ ]:



xgb_best = xgb_grid.best_estimator_
disp_confusion_matrix(xgb_best, xtest, ytest)
disp_roc_curve(xgb_best, xtest, ytest)


# In[ ]:



xgb_best.fit(x,y)
test_prob = xgb_best.predict_proba(df_test)[:,1]
df_xgbgs = pd.DataFrame({'session_id':test_ids,'gender':test_prob})
df_xgbgs.head()


# In[ ]:


import time
times = time.strftime("%Y%m%d-%H%M%S")
df_xgbgs['gender'] = df_xgbgs['gender'].apply(lambda x : 'male' if x>0.5 else 'female')
df_xgbgs.to_csv('submission-xgbgs'+times+'.csv',index=False)


# In[ ]:



df_rfxgb=df_rf
df_rfxgb['gender']=0.8*df_rf['gender']+0.2*df_xgb['gender']


# In[ ]:


import time
times = time.strftime("%Y%m%d-%H%M%S")
df_rfxgb['gender'] = df_rfxgb['gender'].apply(lambda x : 'male' if x>0.5 else 'female')
df_rfxgb.to_csv('submission-rfxgb_'+times+'.csv',index=False)


# LGB

# In[ ]:


from lightgbm import LGBMClassifier
lgb = LGBMClassifier()
lgb.fit(xtrain,ytrain)
print('Training set score = {:.3f}'.format(lgb.score(xtrain,ytrain)))
print('Test set score = {:.3f}'.format(lgb.score(xtest,ytest)))


# In[ ]:


disp_confusion_matrix(lgb, xtest, ytest)
disp_roc_curve(lgb, xtest, ytest)


# In[ ]:


arr=[-1,10,20,30,50]
cv_scores = []
for a in arr:
    model = LGBMClassifier(objective='binary', n_jobs=4, boosting_type='gbdt', n_estimators=100, max_depth=a)
    cv_score = cross_val_score(model, x, y, cv=5, scoring='roc_auc')
    print(a, ':', cv_score)
    cv_scores.append(cv_score)
    
fig, ax = plt.subplots(figsize=(14,6))
plt.boxplot(cv_scores)
ax.set_xticklabels(arr)
plt.xlabel('n')
plt.ylabel('roc_auc')
plt.show()


# In[ ]:


param_grid = {
    'num_leaves':[40,60],
    'max_depth':[-1,10],
    'subsample':[0.8,0.9,1],
    'colsample_bytree': [0.6,0.8,1],
    'min_child_samples': [20,10,30],
    'min_split_gain':[0,0.5,1]
}
lgb = LGBMClassifier(objective='binary', n_jobs=4, boosting_type='gbdt', learning_rate=0.01, n_estimators=100, silent=False)
lgb_grid = GridSearchCV(lgb, param_grid, cv=5, scoring='roc_auc', verbose=1, n_jobs=4)
lgb_grid.fit(x, y)


# In[ ]:


lgb_best = lgb_grid.best_estimator_
disp_confusion_matrix(lgb_best, xtest, ytest)
disp_roc_curve(lgb_best, xtest, ytest)


# In[ ]:


lgb_best.fit(x,y)
test_prob = lgb_best.predict_proba(df_test)[:,1]
df_lgbgs = pd.DataFrame({'session_id':test_ids,'gender':test_prob})
df_lgbgs.head()


# In[ ]:


import time
times = time.strftime("%Y%m%d-%H%M%S")
df_lgbgs['gender'] = df_lgbgs['gender'].apply(lambda x : 'male' if x>0.5 else 'female')
df_lgbgs.to_csv('submission-lgbgs'+times+'.csv',index=False)


# CatBoost

# In[ ]:


from catboost import CatBoostClassifier


# In[ ]:


cb = CatBoostClassifier()
cb.fit(xtrain,ytrain)
print('Training set score = {:.3f}'.format(cb.score(xtrain,ytrain)))
print('Test set score = {:.3f}'.format(cb.score(xtest,ytest)))


# In[ ]:


disp_confusion_matrix(cb, xtest, ytest)
disp_roc_curve(cb, xtest, ytest)


# In[ ]:


param_grid = {
    'depth':[2, 3, 4],
    'learning_rate' : [0.01, 0.05, 0.1],
    'loss_function': ['Logloss', 'CrossEntropy'],
    'l2_leaf_reg':np.logspace(-20, -19, 3)
}

cb = CatBoostClassifier(iterations=2500,
                            eval_metric = 'Accuracy',
                            leaf_estimation_iterations = 10)

cb_grid = GridSearchCV(cb, param_grid, cv=5, scoring='roc_auc', verbose=1, n_jobs=4) 
cb_grid.fit(x, y)


# In[ ]:


cb_best = cb_grid.best_estimator_
disp_confusion_matrix(cb_best, xtest, ytest)
disp_roc_curve(cb_best, xtest, ytest)


# In[ ]:



cb_best.fit(x,y)
test_prob = cb_best.predict_proba(df_test)[:,1]
df_cbgs = pd.DataFrame({'session_id':test_ids,'gender':test_prob})
df_cbgs.head()


# In[ ]:


import time
times = time.strftime("%Y%m%d-%H%M%S")
df_cbgs['gender'] = df_cbgs['gender'].apply(lambda x : 'male' if x>0.5 else 'female')
df_Cbgs.to_csv('submission-lgbgs'+times+'.csv',index=False)

