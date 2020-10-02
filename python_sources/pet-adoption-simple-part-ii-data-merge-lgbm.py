#!/usr/bin/env python
# coding: utf-8

# # **Pet Adoption Simple (Part II)**

# This Kernel **continues my previous** "[Pet Adoption Simple](https://www.kaggle.com/ludovicoristori/pets-adoption-simple-pandas-random-forest)". Actually, when I started loading and processing the json data, that notebook became all but simple. So I took this decision: one new kernel (this) to use the data prepared and to add some modification to the basic dataframe df_all, too. 
# 
# Topics of this notebook: focus on **categorical data** and **merging different dataframes** (images, SVD of descriptions). In addition, I substituted Random Forests and XGBM with **LGBM** which I found better with columns like description or breed (categorical with many values).
# 
# Thanks to [Abishek](http://https://www.kaggle.com/abhishek/maybe-something-interesting-here) for the idea of rescaling/rounding outputs. It's the first time I have seen something like that and started using it practically anytime. Probably I abused a bit in the final part of this notebook, let's see after the submission...

# # Imports and Data Preparation

# In[58]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[59]:


import scipy as sp
from collections import Counter
from functools import partial


# In[60]:


import os
print(os.listdir("../input"))


# In[61]:


data_path="../input/petfinder-adoption-prediction"
first_kernel_path="../input/pets-adoption-simple-pandas-random-forest"
image_kernel_path="../input/pet-adoption-only-images"
svd_kernel_path="../input/pet-adoption-only-text-svd"


# In[62]:


train = pd.read_csv(data_path+"/train/train.csv")
test = pd.read_csv(data_path+"/test/test.csv")
color_labels = pd.read_csv(data_path+"/color_labels.csv")
breed_labels = pd.read_csv(data_path+"/breed_labels.csv")
state_labels = pd.read_csv(data_path+"/state_labels.csv")


# In[63]:


import warnings
warnings.filterwarnings('ignore')


# In[64]:


def print_columns_with_null(df):
    dfn=df.isnull().sum()
    return dfn[dfn>0]


# From "Pet Adoption Simple" we have:
# 
# * df_all0, basic dataframe
# * df_svd, SVD of descriptions in df_all0 and img_data 
# * txt_data, data from sentiment analysis
# * img_data, data from images
# 
# The preparation basically consists of merging the information in one dataframe, the "new" df_all
# 

# In[65]:


df_all0 = pd.read_csv(first_kernel_path+"/df_all0.csv")
df_all0.head()


# In[66]:


txt_data = pd.read_csv(first_kernel_path+"/txt_data.csv")
txt_data.columns = ['PetID','sent_magnitude','sent_score','sent_language']
txt_data.head()


# In[67]:


img_df1a = pd.read_csv(image_kernel_path+"/img_df1a_local.csv")
img_df1a.columns = ['PetID','ImageID','img_met_score','img_met_description']
img_df1a.head()


# In[68]:


img_df1c = pd.read_csv(image_kernel_path+"/img_df1c_local.csv")
img_df1c.columns = ['PetID','ImageID','img_crp_x','img_crp_y','img_crp_conf','img_crp_if']
img_df1c.head()


# In[69]:


img_df1p = pd.read_csv(image_kernel_path+"/img_df1p_local.csv")
img_df1p.columns = ['PetID','ImageID','img_par_red','img_par_green','img_par_blue','img_par_score','img_par_pf']
img_df1p.head()


# In[70]:


des_svd_df = pd.read_csv(svd_kernel_path+"/des_svd_df.csv")
des_svd_df.iloc[:,0:10].head()


# In[71]:


prev_subm = pd.read_csv("../input/pets-adoption-simple-pandas-random-forest/submission.csv")
prev_subm.head()


# # Merging dataframes

# In[72]:


df_all0.shape


# In[73]:


txt_data.shape


# In[74]:


img_df1a.shape


# In[75]:


img_df1c.shape


# In[76]:


img_df1p.shape


# In[77]:


des_svd_df.shape


# What can we do with RescuerID, as there are so many? In the previous kernel I distincted top rescuers from others. This time I try to count the occurrences of every rescuer, and then find a sort of rescuer level or rank to include in the main dataframe.

# In[78]:


rescuers=df_all0.groupby(by='RescuerID')['RescuerID'].count()
df_rescuers=pd.DataFrame(rescuers)
df_rescuers.columns=['ResLev']
df_rescuers.reset_index(inplace=True)
df_rescuers.head()


# In[79]:


dfm=df_all0.merge(df_rescuers,on='RescuerID')
numeric_cols=['Age','PhotoAmt','Quantity','Fee','DescriptionLength','ResLev']
categorical_cols=['Sterilized','FurLength','Breed1','State','AdoptionSpeed','Breed2','MaturitySize','Gender','Dewormed','Color1','Color2','Color3','Health']
cols=['PetID']+numeric_cols+categorical_cols
dfm=dfm[cols]
dfm.shape


# In[80]:


dfm=dfm.merge(txt_data,on='PetID', how='left')
categorical_cols=categorical_cols+['sent_language']
dfm.shape


# In[81]:


n_svd=32
svd=des_svd_df.iloc[:,0:n_svd+3]
svd.drop('Description',axis=1,inplace=True)
svd.drop('AdoptionSpeed',axis=1,inplace=True)
dfm=dfm.merge(svd,on='PetID', how='left')
dfm.shape


# In[82]:


img_df1ad=img_df1a.groupby(['PetID'])['img_met_description'].apply(', '.join).reset_index()


# In[83]:


img_df1ad['img_met_description1']=img_df1ad['img_met_description'].apply(lambda s:s.split(',')).apply(set).apply(','.join)


# In[84]:


img_df1ad['img_met_description1'].loc[0]


# In[85]:


img_df1ad['img_met_description'].loc[0]


# In[86]:


img_df1ad.drop('img_met_description',axis=1,inplace=True)
img_df1ad.head()


# In[87]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


# In[88]:


def find_svd(df,txt_col_name,n_comp):
    tfv = TfidfVectorizer(analyzer='word', stop_words = 'english', token_pattern=r'\b[a-zA-Z]\w+\b',
                      min_df=1,  max_features=10000, strip_accents='unicode', 
                      ngram_range=(1, 32), use_idf=1, smooth_idf=1, sublinear_tf=1,)
    corpus=list(df[txt_col_name])
    txt_trasf=tfv.fit_transform(corpus)
    svd = TruncatedSVD(n_components=n_comp, n_iter=10, tol=1.0)
    svd.fit(txt_trasf)
    txt_svd=svd.transform(txt_trasf)
    txt_svd_df=pd.DataFrame(txt_svd)
    return(txt_svd_df)


# In[89]:


nc=16
img_met_svd=find_svd(img_df1ad,'img_met_description1',nc)
img_met_svd.columns=['SVD_'+str(c) for c in range(0,nc)]
img_met_svd['PetID']=img_df1ad['PetID']


# In[90]:


img_df1ar=img_met_svd
img_df1ar['img_met_score_max']=img_df1a.groupby(by=['PetID','ImageID'],as_index=False).agg({'img_met_score': 'max'})['img_met_score'] 
img_df1ar['img_met_score_min']=img_df1a.groupby(by=['PetID','ImageID'],as_index=False).agg({'img_met_score': 'min'})['img_met_score'] 
img_df1ar.head()


# In[91]:


img_df1cr1=img_df1c[img_df1c['ImageID']==1]
img_df1cr1.drop('ImageID', axis=1, inplace=True)
img_df1cr1.columns=['PetID','img_crp_x1','img_crp_y1','img_crp_conf1','img_crp_if1']
img_df1cr2=img_df1c[img_df1c['ImageID']==2]
img_df1cr2.drop('ImageID', axis=1, inplace=True)
img_df1cr2.columns=['PetID','img_crp_x2','img_crp_y2','img_crp_conf2','img_crp_if2']
img_df1cr3=img_df1c[img_df1c['ImageID']==3]
img_df1cr3.drop('ImageID', axis=1, inplace=True)
img_df1cr3.columns=['PetID','img_crp_x3','img_crp_y3','img_crp_conf3','img_crp_if3']
img_df1cr=img_df1cr1
img_df1cr.head()


# In[92]:


img_df1cr=img_df1cr.merge(img_df1cr2,on='PetID')
img_df1cr=img_df1cr.merge(img_df1cr3,on='PetID')
img_df1cr.head()


# In[93]:


img_df1cr.isna().sum()


# In[94]:


img_df1cr.fillna(-1, inplace=True)


# In[95]:


img_df1p.head()


# In[96]:


img_df1p.shape


# In[97]:


img_df1pg=img_df1p.groupby(by=['PetID','ImageID'],as_index=False).agg({'img_par_score': 'max', 
                                                             'img_par_red':'first',
                                                             'img_par_green':'first',
                                                             'img_par_blue':'first',
                                                             'img_par_pf':'first'})
img_df1pg.head()


# In[98]:


img_df1pr1=img_df1pg[img_df1pg['ImageID']==1]
img_df1pr1.drop('ImageID', axis=1, inplace=True)
img_df1pr1.columns=['PetID','img_par_red1','img_par_green1','img_par_blue1','img_par_pf1','img_par_score1']
img_df1pr2=img_df1pg[img_df1pg['ImageID']==2]
img_df1pr2.drop('ImageID', axis=1, inplace=True)
img_df1pr2.columns=['PetID','img_par_red2','img_par_green2','img_par_blue2','img_par_pf2','img_par_score2']
img_df1pr3=img_df1pg[img_df1pg['ImageID']==3]
img_df1pr3.drop('ImageID', axis=1, inplace=True)
img_df1pr3.columns=['PetID','img_par_red3','img_par_green3','img_par_blue3','img_par_pf3','img_par_score3']
img_df1pr=img_df1pr1
img_df1pr.head()


# In[99]:


img_df1pr=img_df1pr.merge(img_df1pr2,on='PetID')
img_df1pr=img_df1pr.merge(img_df1pr3,on='PetID')
img_df1pr.head()


# Ok, it's time to merge all the images tables:

# In[100]:


img=img_df1ar
img=img.merge(img_df1cr,on=['PetID'], how='left')
img=img.merge(img_df1pr,on=['PetID'], how='left')
img.shape


# In[101]:


img.fillna(-1,inplace=True)
img.head()


# In[102]:


dfm=dfm.merge(img,on='PetID', how='left')
dfm.shape


# In[103]:


df_all=dfm
df_all.head()


# In[104]:


print_columns_with_null(df_all)


# In[105]:


df_all['sent_magnitude'].fillna(-1, inplace=True) # -1=no comments received
df_all['sent_score'].fillna(-1, inplace=True) # -1=no comments received
df_all['sent_language'].fillna('en', inplace=True) # default=english


# In[106]:


df_all.fillna(-1, inplace=True) # -1=no images


# In[107]:


print_columns_with_null(df_all)


# In[108]:


# categorical_cols=categorical_cols # no add to do
categorical_cols


# In[109]:


df_all[categorical_cols]=df_all[categorical_cols].apply(lambda c : c.astype('category'))


# # Modelling

# In[110]:


df_all.to_csv('df_all.csv')
df_all.head()


# In[111]:


df_all.dtypes[df_all.dtypes=='object']


# In[112]:


df_all.columns


# In[113]:


dftrain=df_all[np.invert(df_all['AdoptionSpeed']==-1)].copy()
dftest=df_all[df_all['AdoptionSpeed']==-1].copy()


# In[114]:


dftest_ids=dftest['PetID']
dftest_ids.head()


# In[115]:


dftrain = dftrain.drop(['PetID'],axis=1)
dftest = dftest.drop(['PetID'],axis=1)


# In[116]:


XT = dftest.drop('AdoptionSpeed',axis=1)
y  = dftrain['AdoptionSpeed']
X  = dftrain.drop('AdoptionSpeed',axis=1)


# In[117]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold


# Classifier Choice: I used **LGBM instead of XGB**. **We have to perform a Classification but we use the LGBM Regressor followed by a custom rounding** (credits: Abishek).

# In[118]:


import lightgbm as lgb


# In[119]:


cat_features=[x for x in categorical_cols if x!='AdoptionSpeed']


# In[154]:


parameters = {'application': 'regression',
              'boosting': 'gbdt',
              'metric': 'rmse',
              'max_bin' : 8,
              'num_leaves': 12,
              'max_depth': 4,
              'learning_rate': 0.01,
              'bagging_fraction': 0.8,
              'feature_fraction': 0.8,
              'min_split_gain': 0.01,
              'min_child_samples': 128,
              'min_child_weight': 0.1,
              'data_random_seed': 123,
              'verbosity': -1,
              'early_stopping_rounds': 50,
              'num_rounds': 3000}
evals_result={}


# In[155]:


def qks(a,b):
    return cohen_kappa_score(np.round(a), np.round(b), weights='quadratic')


# **Cross Fold Validation** 

# In[156]:


kf_splits=10
k_fold = KFold(n_splits=kf_splits, shuffle=True)
k=0
df_qks=pd.DataFrame(columns=['best_round','qks_train','qks_valid'])
df_y=pd.DataFrame(index=XT.index)
perf_list=[]
for train_idx, valid_idx in k_fold.split(X,y):
    k=k+1
    print('Step k={}'.format(k))
    X_train = X.iloc[train_idx, :]
    X_valid = X.iloc[valid_idx, :]
    y_train = y.iloc[train_idx]
    y_valid = y.iloc[valid_idx]
    lgb_train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features,free_raw_data=False)
    lgb_valid_set = lgb.Dataset(X_valid, label=y_valid,free_raw_data=False)
    lgbm=lgb.LGBMRegressor()
    lgbm = lgb.train(parameters,
                     train_set=lgb_train_set,
                     valid_sets=[lgb_train_set,lgb_valid_set],
                     verbose_eval=100,
                     evals_result=evals_result)
    best_round=lgbm.best_iteration
    y_train_pred = lgbm.predict(X_train,num_iteration=best_round)
    y_valid_pred = lgbm.predict(X_valid,num_iteration=best_round)
    y_test_pred = lgbm.predict(XT,num_iteration=best_round)
    qks1=qks(y_train_pred,y_train)
    qks2=qks(y_valid_pred,y_valid)
    df_qks.loc[k]=[best_round,qks1,qks2]
    perf_list=perf_list+[evals_result]
    df_y[k]=y_test_pred


# In[144]:


evals_result.keys()


# In[146]:


fig, ax = plt.subplots(1,1, figsize=(8,6))
for i in range(0,kf_splits):
    ax.plot(perf_list[i]['training']['rmse'], color='blue')
    ax.plot(perf_list[i]['valid_1']['rmse'], color='red')


# In[150]:


fig,ax = plt.subplots(1,1,figsize=(12,32))
lgb.plot_importance(lgbm, ax=ax)


# In[123]:


df_qks


# In[124]:


ym=df_y.mean(axis=1)
ym.describe()


# In[125]:


sns.distplot(ym)


# In[126]:


df_all[df_all['AdoptionSpeed'].astype(int)>=0]['AdoptionSpeed'].value_counts()


# Ok, I tried with a lot of different combinations of parameters but is very difficult to have a train error less than rmse=1%. But the most important thing is that with these predictions, it's impossible to fit the original distribution of Adoption Speed. For example, we have
# 
# 

# In[127]:


sum(ym>3.5)


# So this is the idea I think I got from the OptimizeRounder: **why simply rounding the output of LGBM**? Why not use the **additional degree of freedom** coming from a custom rounder? 

# In[128]:


def distrib_err(coef,test_proba,train_label):
    test_predictions = pd.cut(test_proba, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4])
    N_CLASS=5
    freq_train=np.zeros(N_CLASS)
    freq_test=np.zeros(N_CLASS)
    delta_freq=np.zeros(N_CLASS)
    for i in range(0,N_CLASS):
        freq_train[i]=100*Counter(train_label)[i]/len(train_label)
        freq_test[i]=100*Counter(test_predictions)[i]/len(test_predictions)
        delta_freq[i]=freq_test[i]-freq_train[i]
    return np.sum(delta_freq**2)


# In[129]:


initial_coef = [2.0, 2.5, 3.0, 3.5]
distrib_err_partial = partial(distrib_err, test_proba=ym, train_label=y)
final_coef = sp.optimize.minimize(distrib_err_partial, initial_coef, method='nelder-mead')
final_coef


# Here is a function to apply a vector of custom limits:

# In[130]:


def apply_lim(y_calc,limits):
    y_round=np.zeros(len(y_calc))
    for i,yc in enumerate(y_calc):
        if (yc<=limits[0]):
            y_round[i]=0
        if ((yc>limits[0])&(yc<=limits[1])): 
            y_round[i]=1
        if ((yc>limits[1])&(yc<=limits[2])): 
            y_round[i]=2
        if ((yc>limits[2])&(yc<=limits[3])):
            y_round[i]=3
        if (yc>limits[3]):
            y_round[i]=4
    return y_round


# Let's try to map it into the original one:

# In[131]:


y_test_pred_r = apply_lim(ym,final_coef['x'])


# # Final Submission

# In[132]:


y_pred = y_test_pred_r.astype('int')


# In[133]:


subm=pd.DataFrame({'PetID': dftest_ids,'AdoptionSpeed': y_pred})
subm.head()


# In[134]:


subm['AdoptionSpeed'].value_counts()


# In[135]:


subm.to_csv('submission.csv', index=False)

