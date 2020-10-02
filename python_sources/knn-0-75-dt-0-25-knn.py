#!/usr/bin/env python
# coding: utf-8

# ### vooraguptha@gmail.com

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
from scipy import stats
from sklearn.model_selection import KFold

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/train.csv')
test = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/test.csv')
print(train.shape,test.shape)


# In[ ]:


# print(train.workclass.value_counts())
train["workclass"] =np.where(train["workclass"].isin([' ?']),"unknown",train["workclass"])
train["occupation"] =np.where(train["occupation"].isin([' ?']),"unknown",train["occupation"])
train["native-country"] =np.where(train["native-country"].isin([' ?']),"unknown",train["native-country"])


# In[ ]:


list_cols=[]
list_cols=[pd.get_dummies(train["workclass"],prefix="workclass"),
                 pd.get_dummies(train["education"],prefix="education"),
                 pd.get_dummies(train["marital-status"],prefix="marital"),
                 pd.get_dummies(train["occupation"],prefix="occupation"),
                 pd.get_dummies(train["relationship"],prefix="relationship"),
                 pd.get_dummies(train["race"],prefix="race"),
                 pd.get_dummies(train["native-country"],prefix="country"),
                 pd.get_dummies(train["sex"],prefix="sex")]
list_cols_df=pd.concat(list_cols, axis=1)
list_cols_df.head(1)
train_dmy=pd.concat([train,list_cols_df],axis=1)
# list_cols_df.columns
train_dmy.shape


# In[ ]:


train_dmy["workclass_local_selfempnot_state"]=train_dmy["workclass_ Self-emp-not-inc"] + train_dmy["workclass_ Local-gov"] + train_dmy["workclass_ State-gov"]
train_dmy["workclass_never_without_private"]=train_dmy["workclass_ Never-worked"] + train_dmy["workclass_ Without-pay"] + train_dmy["workclass_ Private"]
train_dmy["workclass_sunk"]= train_dmy["workclass_unknown"]
list_class =[ "workclass_ Self-emp-inc" , "workclass_ Federal-gov" , "workclass_local_selfempnot_state",   "workclass_never_without_private" ]


train_dmy["education_doc_prof"] = train_dmy["education_ Doctorate"] + train_dmy["education_ Prof-school"]
train_dmy["education_voc_acdm_preschool"] = train_dmy["education_ Assoc-voc"] + train_dmy["education_ Assoc-acdm"] + train_dmy["education_ Preschool"]
train_dmy["education_coll_hsgrad"] = train_dmy["education_ Some-college"] + train_dmy["education_ HS-grad"] 
train_dmy["education_xii_iiv_x"] = train_dmy["education_ 12th"] + train_dmy["education_ 7th-8th"]  + train_dmy["education_ 10th"]
train_dmy["education_ix_xi_v_vi_i_iv"] = train_dmy["education_ 9th"] + train_dmy["education_ 11th"]  + train_dmy["education_ 5th-6th"] + train_dmy["education_ 1st-4th"]
list_edu = ["education_doc_prof", "education_ Masters", "education_ Bachelors" , "education_voc_acdm_preschool", "education_coll_hsgrad", "education_xii_iiv_x"]

train_dmy["marital_married_civ_af"] = train_dmy["marital_ Married-civ-spouse"] + train_dmy["marital_ Married-AF-spouse"]
train_dmy["marital_widow_spos_abs"] = train_dmy["marital_ Widowed"] + train_dmy["marital_ Married-spouse-absent"]
list_marrital = ["marital_married_civ_af", "marital_widow_spos_abs" , "marital_ Divorced","marital_ Separated"]

train_dmy["occupation_exec_prof"] = train_dmy["occupation_ Exec-managerial"]+train_dmy["occupation_ Prof-specialty"]
train_dmy["occupation_prot_tech"] = train_dmy["occupation_ Protective-serv"]+train_dmy["occupation_ Tech-support"]
train_dmy["occupation_crft_trans"] = train_dmy["occupation_ Craft-repair"]+train_dmy["occupation_ Transport-moving"]
train_dmy["occupation_adm_machine_farming_armed_unk"] = train_dmy["occupation_ Adm-clerical"]+train_dmy["occupation_ Machine-op-inspct"]+train_dmy["occupation_ Farming-fishing"]+train_dmy["occupation_ Armed-Forces"]+train_dmy["occupation_unknown"]
train_dmy["occupation_oth_priv"] = train_dmy["occupation_ Priv-house-serv"]+train_dmy["occupation_ Other-service"]
list_occu = ["occupation_exec_prof","occupation_prot_tech","occupation_ Sales","occupation_crft_trans","occupation_adm_machine_farming_armed_unk","occupation_ Handlers-cleaners"]

train_dmy["relationship_wife_husband"] = train_dmy["relationship_ Husband"] + train_dmy["relationship_ Wife"]
list_relat=["relationship_wife_husband","relationship_ Not-in-family","relationship_ Unmarried","relationship_ Other-relative"]

list_race = ['race_ Amer-Indian-Eskimo', 'race_ Asian-Pac-Islander',       'race_ Black', 'race_ Other', 'race_ White']

train_dmy["country_irn_fran_ind_taiw_jap_yugo_camb"] = train_dmy['country_ Iran'] + train_dmy['country_ France'] + train_dmy['country_ India']+train_dmy['country_ Taiwan']+train_dmy['country_ Japan']+train_dmy['country_ Yugoslavia']+train_dmy['country_ Cambodia']
train_dmy["country_itl_eng_can_germ_phi_hon"] = train_dmy['country_ Italy']+ train_dmy['country_ England']+ train_dmy['country_ Canada'] +train_dmy['country_ Germany']+train_dmy['country_ Philippines'] + train_dmy['country_ Hong']
train_dmy["country_gree_chi_cub_unk_scot_us"] = train_dmy['country_ Greece'] + train_dmy['country_ China'] + train_dmy['country_ Cuba']+train_dmy['country_unknown']+train_dmy['country_ Scotland'] + train_dmy['country_ United-States']
train_dmy["country_hol_out_hun_ire_sout_pol"] = train_dmy['country_ Holand-Netherlands'] + train_dmy['country_ Outlying-US(Guam-USVI-etc)'] + train_dmy['country_ Hungary']+train_dmy['country_ Ireland'] + train_dmy['country_ South'] + train_dmy['country_ Poland']
train_dmy["country_thai_ecu_jam_laos_por_tri_pue"] = train_dmy['country_ Thailand'] + train_dmy['country_ Ecuador'] + train_dmy['country_ Jamaica'] + train_dmy['country_ Laos'] + train_dmy['country_ Portugal'] + train_dmy['country_ Trinadad&Tobago'] + train_dmy['country_ Puerto-Rico']
list_coun = ["country_irn_fran_ind_taiw_jap_yugo_camb", "country_itl_eng_can_germ_phi_hon" ,"country_gree_chi_cub_unk_scot_us" ,"country_hol_out_hun_ire_sout_pol","country_thai_ecu_jam_laos_por_tri_pue"]

train_dmy["education-num_enc"]=train_dmy["education-num"].replace({1.0:0.0,2.0:-2.147591,3.0:-1.838067,4.0:-1.569754,5.0:-1.744181,6.0:-1.494261,7.0:-1.774019,8.0:-1.346711,9.0:-0.513642,10.0:-0.300241,11.0:0.108586,                                                                  12.0:0.040867,13.0:0.803894,14.0:1.375570,15.0:2.165181,16.0:2.199003}) 
print(train_dmy["education-num_enc"].unique())
train_dmy["education-num_1_3"] = np.where(train_dmy["education-num"]<=3,1,0)
train_dmy["education-num_4_8"] = np.where((train_dmy["education-num"]>3) & (train_dmy["education-num"]<=8),1,0)
train_dmy["education-num_9_10"] = np.where((train_dmy["education-num"]>8) & (train_dmy["education-num"]<=10),1,0)
train_dmy["education-num_11"] = np.where(train_dmy["education-num"]==11,1,0)
train_dmy["education-num_12"] = np.where(train_dmy["education-num"]==12,1,0)
train_dmy["education-num_13"] = np.where(train_dmy["education-num"]==13,1,0)
train_dmy["education-num_14_16"] = np.where((train_dmy["education-num"]>13) & (train_dmy["education-num"]<=16),1,0)
# train_dmy["education_"]
list_edunum = ["education-num_1_3","education-num_4_8","education-num_9_10","education-num_11","education-num_12","education-num_13","education-num_14_16"]

train_dmy['age_lt22'] = np.where(train_dmy["age"]<= 22 , 1 , 0)
train_dmy['age_bt_22_26'] = np.where(((train_dmy["age"]> 22) & (train_dmy["age"]<= 26)) , 1 , 0)
train_dmy['age_bt_26_30'] = np.where(((train_dmy["age"]> 26) & (train_dmy["age"]<= 30)) , 1 , 0)
train_dmy['age_bt_30_33'] = np.where(((train_dmy["age"]> 30) & (train_dmy["age"]<= 33)) , 1 , 0)
train_dmy['age_bt_33_37'] = np.where(((train_dmy["age"]> 33) & (train_dmy["age"]<= 37)) , 1 , 0)
train_dmy['age_bt_37_41'] = np.where(((train_dmy["age"]> 37) & (train_dmy["age"]<= 41)) , 1 , 0)
train_dmy['age_bt_41_58'] = np.where(((train_dmy["age"]> 41) & (train_dmy["age"]<= 58)) , 1 , 0)
train_dmy['age_bt_58_90'] = np.where(((train_dmy["age"]> 58) & (train_dmy["age"]<= 90)) , 1 , 0)
# train_dmy["ageenc"]= np.where
list_age= ['age_bt_22_26','age_bt_26_30','age_bt_30_33','age_bt_33_37','age_bt_37_41','age_bt_41_58','age_bt_58_90']

train_dmy["capfact_enc"]=np.where(train_dmy["capital-gain"]<=9999.9,-0.100820,                                  np.where(((train_dmy["capital-gain"]>9999)& (train_dmy["capital-gain"]<=19999.8)),5.592856,                                          np.where(( train_dmy["capital-gain"] >19999.8 ), 5.602594,0)                                          ))

train_dmy["caplsfact_enc"]=np.where(train_dmy["capital-loss"]<=1306.8,-0.073605,                                  np.where(((train_dmy["capital-loss"]>1306.8)& (train_dmy["capital-loss"]<=3049.2)),2.940006,                                          np.where(( train_dmy["capital-loss"] >3049.2 ), 5.602594,0)                                          ))


# In[ ]:


edu=train_dmy.groupby("education")["target"].agg({np.mean,np.std})
edu["education"]=edu.index
edu=edu.reset_index(drop=True)
edu
mrt=train_dmy.groupby("marital-status")["target"].agg({"mean1":np.mean,"stdd":np.std})
mrt["marital-status"]=mrt.index
mrt=mrt.reset_index(drop=True)
mrt
wrk=train_dmy.groupby("workclass")["target"].agg({"mean11":np.mean,"stdd1":np.std})
wrk["workclass"]=wrk.index
wrk=wrk.reset_index(drop=True)
wrk
occ=train_dmy.groupby("occupation")["target"].agg({"mean111":np.mean,"stdd11":np.std})
occ["occupation"]=occ.index
occ=occ.reset_index(drop=True)
occ
train_dmy=pd.merge(train_dmy,edu,on="education",how="left")
train_dmy=pd.merge(train_dmy,mrt,on="marital-status",how="left")


# In[ ]:


dummylist_cat = list_class  + list_marrital   + ['sex_ Male','uid','target'] +list_race +list_occu
numrics = ["capital-gain","age","hours-per-week","capital-loss","education-num"]
feature_list  = dummylist_cat + numrics
# feature_list


# In[ ]:


feature_list=['relationship_wife_husband', 'education-num_enc', 'capital-gain', 'age', 'occupation_exec_prof', 'hours-per-week', 'capital-loss', 'mean', 'std', 
'education_ Masters','occupation_ Farming-fishing', 'occupation_ Sales','target','uid','workclass_local_selfempnot_state','marital_married_civ_af']


# In[ ]:



from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score ,f1_score, roc_auc_score , confusion_matrix , classification_report, roc_curve, auc , log_loss
from sklearn.model_selection import GridSearchCV,LeaveOneOut,cross_val_score,cross_val_predict,KFold
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


train_model=train_dmy[feature_list]
train_model.shape


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(train_model.drop(['target','uid'],axis=1), train_model["target"],test_size=0.2,random_state=5)
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)


# In[ ]:


# #List Hyperparameters that we want to tune.
# leaf_size = list(range(30,50))
# n_neighbors = list(range(10,30))
# p=[1,2]
# #Convert to dictionary
# hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
# #Create new KNN object
# knn_2 = KNeighborsClassifier()
# #Use GridSearch
# clf = GridSearchCV(knn_2, hyperparameters, cv=3,verbose=1,n_jobs=-1)
# #Fit the model
# best_model = clf.fit(x_train,y_train)
# #Print The value of best Hyperparameters
# print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
# print('Best p:', best_model.best_estimator_.get_params()['p'])
# print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])


# In[ ]:


# neighbors = list(range(1,20))
# train_results = []
# test_results = []
# for n in neighbors:
#    model = KNeighborsClassifier(n_neighbors=71,n_jobs=-1,p=1,leaf_size=n)
#    model.fit(x_train, y_train)
#    train_pred = model.predict_proba(x_train)
#    loglos=log_loss(y_train,train_pred[:,1])
#    train_results.append(loglos)
#    y_pred = model.predict(x_test)
#    predprob = model.predict_proba(x_test)
#    loglos=log_loss(y_test,predprob[:,1])
    
#    test_results.append(loglos)
# from matplotlib.legend_handler import HandlerLine2D
# line1, = plt.plot(neighbors, train_results, 'b', label="Train AUC")
# line2, = plt.plot(neighbors, test_results, 'r', label="Test AUC")
# plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
# plt.ylabel('AUC score')
# plt.xlabel('n_neighbors')
# plt.show()


# In[ ]:


# train_results
# test_results


# In[ ]:


#Create KNN Object.
clf = KNeighborsClassifier(n_neighbors=65,leaf_size=1,n_jobs=-1,p=1,weights='uniform')

clf.fit(x_train, y_train)
print(clf)

pred= clf.predict(x_train)
predprob = clf.predict_proba(x_train)
print(accuracy_score(pred,y_train))
print(confusion_matrix(pred,y_train))
print(classification_report(pred,y_train))
print(predprob,pred)
false_positive_rate, true_positive_rate, thresholds = roc_curve(pred,y_train)
print(auc(false_positive_rate, true_positive_rate))
log_loss(y_train,predprob[:,1])


# In[ ]:


test_pred=clf.predict(x_test)
test_predprob1=clf.predict_proba(x_test)
print(accuracy_score(test_pred,y_test))
print(confusion_matrix(test_pred,y_test))
print(classification_report(test_pred,y_test))
false_positive_rate, true_positive_rate, thresholds = roc_curve(test_pred,y_test)
print(auc(false_positive_rate, true_positive_rate))
log_loss(y_test,test_predprob1[:,1])


# In[ ]:


test["workclass"] =np.where(test["workclass"].isin([' ?']),"unknown",test["workclass"])
test["occupation"] =np.where(test["occupation"].isin([' ?']),"unknown",test["occupation"])
test["native-country"] =np.where(test["native-country"].isin([' ?']),"unknown",test["native-country"])


# In[ ]:


list_cols=[]
list_cols=[pd.get_dummies(test["workclass"],prefix="workclass"),
                 pd.get_dummies(test["education"],prefix="education"),
                 pd.get_dummies(test["marital-status"],prefix="marital"),
                 pd.get_dummies(test["occupation"],prefix="occupation"),
                 pd.get_dummies(test["relationship"],prefix="relationship"),
                 pd.get_dummies(test["race"],prefix="race"),
                 pd.get_dummies(test["native-country"],prefix="country"),
                 pd.get_dummies(test["sex"],prefix="sex")]
list_cols_df=pd.concat(list_cols, axis=1)
list_cols_df.head(1)
test_dmy=pd.concat([test,list_cols_df],axis=1)
# list_cols_df.columns
test_dmy.shape


# In[ ]:


test_dmy["workclass_local_selfempnot_state"]=test_dmy["workclass_ Self-emp-not-inc"] + test_dmy["workclass_ Local-gov"] + test_dmy["workclass_ State-gov"]
test_dmy["workclass_never_without_private"]=test_dmy["workclass_ Never-worked"] + test_dmy["workclass_ Without-pay"] + test_dmy["workclass_ Private"]
test_dmy["workclass_sunk"]= test_dmy["workclass_unknown"]

test_dmy["education_doc_prof"] = test_dmy["education_ Doctorate"] + test_dmy["education_ Prof-school"]
test_dmy["education_voc_acdm_preschool"] = test_dmy["education_ Assoc-voc"] + test_dmy["education_ Assoc-acdm"] + test_dmy["education_ Preschool"]
test_dmy["education_coll_hsgrad"] = test_dmy["education_ Some-college"] + test_dmy["education_ HS-grad"] 
test_dmy["education_xii_iiv_x"] = test_dmy["education_ 12th"] + test_dmy["education_ 7th-8th"]  + test_dmy["education_ 10th"]
test_dmy["education_ix_xi_v_vi_i_iv"] = test_dmy["education_ 9th"] + test_dmy["education_ 11th"]  + test_dmy["education_ 5th-6th"] + test_dmy["education_ 1st-4th"]

test_dmy["marital_married_civ_af"] = test_dmy["marital_ Married-civ-spouse"] + test_dmy["marital_ Married-AF-spouse"]
test_dmy["marital_widow_spos_abs"] = test_dmy["marital_ Widowed"] + test_dmy["marital_ Married-spouse-absent"]

test_dmy["relationship_wife_husband"] = test_dmy["relationship_ Husband"] + test_dmy["relationship_ Wife"]

test_dmy["occupation_exec_prof"] = test_dmy["occupation_ Exec-managerial"]+test_dmy["occupation_ Prof-specialty"]
test_dmy["occupation_prot_tech"] = test_dmy["occupation_ Protective-serv"]+test_dmy["occupation_ Tech-support"]
test_dmy["occupation_crft_trans"] = test_dmy["occupation_ Craft-repair"]+test_dmy["occupation_ Transport-moving"]
test_dmy["occupation_adm_machine_farming_armed_unk"] = test_dmy["occupation_ Adm-clerical"]+test_dmy["occupation_ Machine-op-inspct"]+test_dmy["occupation_ Farming-fishing"]+test_dmy["occupation_ Armed-Forces"]+test_dmy["occupation_unknown"]
test_dmy["occupation_oth_priv"] = test_dmy["occupation_ Priv-house-serv"]+test_dmy["occupation_ Other-service"]
test_dmy["education-num_enc"]=train_dmy["education-num"].replace({1.0:0.0,2.0:-2.147591,3.0:-1.838067,4.0:-1.569754,5.0:-1.744181,6.0:-1.494261,7.0:-1.774019,8.0:-1.346711,9.0:-0.513642,10.0:-0.300241,11.0:0.108586,                                                                  12.0:0.040867,13.0:0.803894,14.0:1.375570,15.0:2.165181,16.0:2.199003}) 
test_dmy['country_ Holand-Netherlands']=0
test_dmy["country_irn_fran_ind_taiw_jap_yugo_camb"] = test_dmy['country_ Iran'] + test_dmy['country_ France'] + test_dmy['country_ India']+test_dmy['country_ Taiwan']+test_dmy['country_ Japan']+test_dmy['country_ Yugoslavia']+test_dmy['country_ Cambodia']
test_dmy["country_itl_eng_can_germ_phi_hon"] = test_dmy['country_ Italy']+ test_dmy['country_ England']+ test_dmy['country_ Canada'] +test_dmy['country_ Germany']+test_dmy['country_ Philippines'] + test_dmy['country_ Hong']
test_dmy["country_gree_chi_cub_unk_scot_us"] = test_dmy['country_ Greece'] + test_dmy['country_ China'] + test_dmy['country_ Cuba']+test_dmy['country_unknown']+test_dmy['country_ Scotland'] + test_dmy['country_ United-States']
test_dmy["country_hol_out_hun_ire_sout_pol"] = test_dmy['country_ Holand-Netherlands'] + test_dmy['country_ Outlying-US(Guam-USVI-etc)'] + test_dmy['country_ Hungary']+test_dmy['country_ Ireland'] + test_dmy['country_ South'] + test_dmy['country_ Poland']
test_dmy["country_thai_ecu_jam_laos_por_tri_pue"] = test_dmy['country_ Thailand'] + test_dmy['country_ Ecuador'] + test_dmy['country_ Jamaica'] + test_dmy['country_ Laos'] + test_dmy['country_ Portugal'] + test_dmy['country_ Trinadad&Tobago'] + test_dmy['country_ Puerto-Rico']
list_coun = ["country_irn_fran_ind_taiw_jap_yugo_camb", "country_itl_eng_can_germ_phi_hon" ,"country_gree_chi_cub_unk_scot_us" ,"country_hol_out_hun_ire_sout_pol","country_thai_ecu_jam_laos_por_tri_pue"]


# In[ ]:


dummylist_cat = list_class  + list_marrital   + ['sex_ Male'] +list_race +list_occu
numrics = ["capital-gain","age","hours-per-week","capital-loss","education-num"]
feature_list  = dummylist_cat + numrics
# feature_list
test_dmy[feature_list].shape
feature_list


# In[ ]:


edu=train_dmy.groupby("education")["target"].agg({np.mean,np.std})
edu["education"]=edu.index
edu=edu.reset_index(drop=True)
edu
mrt=train_dmy.groupby("marital-status")["target"].agg({"mean1":np.mean,"stdd":np.std})
mrt["marital-status"]=mrt.index
mrt=mrt.reset_index(drop=True)
mrt
wrk=train_dmy.groupby("workclass")["target"].agg({"mean11":np.mean,"stdd1":np.std})
wrk["workclass"]=wrk.index
wrk=wrk.reset_index(drop=True)
wrk
occ=train_dmy.groupby("occupation")["target"].agg({"mean111":np.mean,"stdd11":np.std})
occ["occupation"]=occ.index
occ=occ.reset_index(drop=True)
occ
test_dmy=pd.merge(test_dmy,edu,on="education",how="left")
test_dmy=pd.merge(test_dmy,mrt,on="marital-status",how="left")
# train_dmy=pd.merge(train_dmy,occ,on="occupation",how="left")
# train_dmy=pd.merge(train_dmy,wrk,on="workclass",how="left")


# In[ ]:


feature_list=['relationship_wife_husband', 'education-num_enc', 'capital-gain', 'age', 'occupation_exec_prof', 'hours-per-week', 'capital-loss', 'mean', 'std', 
'education_ Masters','occupation_ Farming-fishing', 'occupation_ Sales','workclass_local_selfempnot_state','marital_married_civ_af']


# In[ ]:



predoot= clf.predict(test_dmy[feature_list])
predproboot = clf.predict_proba(test_dmy[feature_list])
test_dmy["predoot"]=predoot
test_dmy["predproboot0"]=predproboot[:,0]
test_dmy["predproboot1"]=predproboot[:,1]
test_dmy.head()
predproboot[:,0]


# In[ ]:


test_dmy[["uid","predproboot1"]].to_csv("result1.csv")


# In[ ]:




