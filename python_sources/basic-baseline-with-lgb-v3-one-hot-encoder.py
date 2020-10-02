#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
import lightgbm as lgb
get_ipython().run_line_magic('matplotlib', 'inline')
# Any results you write to the current directory are saved as output.


# In[2]:


PATH = "../input"
list_of_files = os.listdir(PATH)

application_train = pd.read_csv(PATH+"/application_train.csv")
application_test = pd.read_csv(PATH+"/application_test.csv")
bureau = pd.read_csv(PATH+"/bureau.csv")
bureau_balance = pd.read_csv(PATH+"/bureau_balance.csv")
credit_card_balance = pd.read_csv(PATH+"/credit_card_balance.csv")
installments_payments = pd.read_csv(PATH+"/installments_payments.csv")
previous_application = pd.read_csv(PATH+"/previous_application.csv")
POS_CASH_balance = pd.read_csv(PATH+"/POS_CASH_balance.csv")


# Now let's look at the shape of this dataset

# In[3]:


list_of_files.sort()
list_of_files.remove("sample_submission.csv")
print(list_of_files)
shape = {"rows" : [POS_CASH_balance.shape[0],
        application_test.shape[0], 
        application_train.shape[0], 
        bureau.shape[0], 
        bureau_balance.shape[0],
        credit_card_balance.shape[0],
        installments_payments.shape[0],
        previous_application.shape[0]], 
        "cols" : [POS_CASH_balance.shape[1],
        application_test.shape[1], 
        application_train.shape[1], 
        bureau.shape[1], 
        bureau_balance.shape[1],
        credit_card_balance.shape[1],
        installments_payments.shape[1],
        previous_application.shape[1]]}

shapes = pd.DataFrame(shape, index = list_of_files)
print (shapes)


# **Let's look at the data**
# Number of rows of most files are larger than numbers of rows of test / train data. Therefore, if all if entities (IDs) from train / test dataset are included in other files, we can merge other files' features into applicaion_test / application_train dataframe. If not all, but if most of IDs from applicaion_test / application_train are included in other files, we could merge them by estimating empty values. 

# In[4]:


POS_CASH_balance.head()


# In[5]:


application_test.head()


# In[6]:


application_train.head()


# In[7]:


bureau.head()


# In[8]:


bureau_balance.head()


# In[9]:


credit_card_balance.head()


# In[10]:


installments_payments.head()


# In[11]:


previous_application.head()


# **Ideas on hot to merge / concatenate data**
# 
# ![](https://storage.googleapis.com/kaggle-media/competitions/home-credit/home_credit.png)
# According to the image on dataset explanation, there are some keys to merge / concatenate sparse data into some organized form. 
# 
# 1. First of all, with SK_ID CURR, we can find linkage between ***train / test dataset , bureau, POS_CASH_balance, credit_card_balance, installment_payment, and previous_application*** together. Here, since train / test dataset is our main target to work on, train/test dataset could be the hub of these data. 
# 
# 2. With SK_ID_PREV, we cand find linkage between ***previous_application, POS_CASH_balance,  credit_card_balance, installment_payment*** and here, previoud_application may be the hub. (Since the linkage is 'SK_ID_PREV' and the infomation is mainly about previous information.
# 
# 3. Lastly, with SK_ID_BUREAU, we can link ***bureau and bureau_balance*** thereby linking these to the first group - to train / test dataset. 
# 
# When we merge or link these separate data, we should consider** how to deal with Nan values. **If most of the files in each group shares most of the SK_IDs within there group, we can merge them on SK_ID and fill Nans by rather **1) putting estimated values, or 2) putting 0 or negative values as a sign of 'unidentified''** If 'unable to identify value' itself have significant implication (ex. if, say,  applicants with certain range of risk probability have tendency to have more missing values), putting 0 or negative value as sign of missing value would be more effective. 

# First let's see to what extent each group shares SK_IDs

# In[12]:


total_IDS = np.concatenate((application_test["SK_ID_CURR"].values, application_train["SK_ID_CURR"].values))
print(len(np.unique(np.array(total_IDS))) == len(total_IDS))#No redundent IDs within train and test data


# **Group 1 - train / test dataset , bureau, POS_CASH_balance, credit_card_balance, installment_payment, and previous_application**

# In[13]:



POS_CASH_balance_IDS = POS_CASH_balance["SK_ID_CURR"].values
bureau_IDS = bureau["SK_ID_CURR"].values
credit_card_balance_IDS = credit_card_balance["SK_ID_CURR"].values
installments_payments_IDS = installments_payments["SK_ID_CURR"].values
previous_application_IDS = previous_application["SK_ID_CURR"].values

tot = len(total_IDS)
print(tot)

print (len(np.intersect1d(POS_CASH_balance_IDS, total_IDS))/tot*100,
len(np.intersect1d(bureau_IDS, total_IDS))/tot*100,
len(np.intersect1d(credit_card_balance_IDS, total_IDS))/tot*100,
len(np.intersect1d(installments_payments_IDS, total_IDS))/tot*100,
len(np.intersect1d(previous_application_IDS, total_IDS))/tot*100)


# **credit_card_balance**shows only little amount of IDS shared with training / test dataset. Others seems like including most of ID from training / test data. Seems like only if we handle them well, we could link them and work on it together. 

# **Group 2 - previous_application, POS_CASH_balance,  credit_card_balance, installment_payment**

# In[14]:


prev = previous_application["SK_ID_PREV"].values

POS_CASH_balance_IDS_prev = POS_CASH_balance["SK_ID_PREV"].values
credit_card_balance_IDS_prev = credit_card_balance["SK_ID_PREV"].values
installments_payments_IDS_prev = installments_payments["SK_ID_PREV"].values

prev_num = len(prev)

print (prev_num)

print(len(np.intersect1d(POS_CASH_balance_IDS_prev, prev))/prev_num*100,
len(np.intersect1d(credit_card_balance_IDS_prev, prev))/prev_num*100,
len(np.intersect1d(installments_payments_IDS_prev, prev))/prev_num*100)


# **Group 3 - bureau and bureau_balance**

# In[15]:


bureau_br = np.unique(bureau["SK_ID_BUREAU"].values)
print(len(np.intersect1d(np.unique(bureau_balance["SK_ID_BUREAU"].values), bureau_br))/len(bureau_br)*100)


# However, here, I'm thinking of linking Bureau data train / test dataset. Since Bureau shares 85% of it's id with train / test data, I'll check how much of those shared IDs are also shared with bureau_balance. 

# In[16]:


breau_total = np.unique(np.intersect1d(bureau_IDS, total_IDS)) #by SK_ID_CURR
bureau_filtered = bureau.loc[bureau["SK_ID_CURR"].isin(breau_total)] #bureau & train / test - sharing SK_ID_CURR

b = np.intersect1d(np.unique(bureau_filtered["SK_ID_BUREAU"].values), np.unique(bureau_balance["SK_ID_BUREAU"].values)) #br & br balance


bureau_filtered = bureau_filtered.loc[bureau_filtered["SK_ID_BUREAU"].isin(b)]
len(bureau_filtered["SK_ID_CURR"].values)
bureau_filtered


# There are many redundant IDs. There are some value differences within the identical IDs, so later we'll think of ways to work on them - say, merge their values into the average, etc. For now, let's just check how many of independent IDs are shared. 

# In[17]:


print (len(np.unique(bureau_filtered["SK_ID_CURR"].values))/tot*100)


# **To sum up, In group 1, train / test data as hub, we can merge : **
# 
# * POS_CASH_balance : 94.66589942597297 %
# * bureau : 85.84047943186762 %
# * credit_card_balance : 29.06850430169401 %
# * installments_payments : 95.3213288234551 %
# * previous_application : 95.11641941867482%
# 
# ** In group 2, previous_application as a hub, we can merge: **
# * POS_CASH_balance : 53.81963029887188 %
# * credit_card_balance : 5.564257035326012  %
# * installments_payments : 57.41210407768106 %
# 
# ** Linking train / test data with bureau_balance, using bureau as a link: **
# * 37.7656453944506%
# 
# of entire rows (IDs) could be merged

# **Now, let's merge group 1**
# 
# 

# Processing application_train/test

# In[18]:


train = application_train.drop(["TARGET"], axis = 1)
train_target = application_train["TARGET"]
test= application_test.copy()
tr = len(application_train)
print (all(i ==True for i in train.columns==test.columns))


# In[19]:


#Dividing categorical and numerical features: 
df = pd.concat([train, test])

del train, test, application_train, application_test
gc.collect()


def categorical_features(data):
    features = [i for i in list(data.columns) if data[i].dtype == 'object']
    return features

categorical = categorical_features(df)
numerical = [i for i in df.columns if i not in categorical]
numerical.remove("SK_ID_CURR")
IDs = df["SK_ID_CURR"]


# In[20]:


#Processing categorical features
for feature in categorical:
    df[feature].fillna("unidentified")
    print(f'Transforming {feature}...')
    encoder = LabelEncoder()
    encoder.fit(df[feature].astype(str))
    df[feature] = encoder.transform(df[feature].astype(str))
    
df.head()


# In[21]:


#processing numeric features #Try log operation later
for feats in df.columns:
    df[feats] = df[feats].fillna(-1)
        
df.head()


# **Processing and merging *POS_CASH_balance***

# In[22]:


#POS_CASH_balance, bureau, credit_card_balance, installments_payments, previous_application

#POS_CASH_balance
POS_CASH_balance_G1 = POS_CASH_balance.loc[POS_CASH_balance["SK_ID_CURR"].isin(total_IDS)]
print (len(np.unique(POS_CASH_balance_G1["SK_ID_CURR"].values)))
POS_CASH_balance_G1.head()


# 

# let's look at how it is distribute.
#  I refered to the function of plotting from [this kernal](http://https://www.kaggle.com/gpreda/home-credit-default-risk-extensive-eda)

# In[23]:


def plot_distribution(dataframe,feature,color):
    plt.figure(figsize=(10,6))
    plt.title("Distribution of %s" % feature)
    sns.distplot(dataframe[feature].dropna(),color=color, kde=True,bins=100)
    plt.show()   

colors = ["blue", "red", "green", "tomato", "brown", "black", "Gray"]
for i, j in zip(POS_CASH_balance_G1.drop("NAME_CONTRACT_STATUS", axis =1).columns, colors):
    plot_distribution(POS_CASH_balance_G1, i, j)
    
dic = Counter(POS_CASH_balance_G1["NAME_CONTRACT_STATUS"])
plt.bar(range(len(dic)), list(dic.values()))
plt.xticks(range(len(dic)), list(dic.keys()), rotation = 90)
plt.show()


# Let's merge numerical values by the means within same SK_ID

# In[24]:


np.unique(POS_CASH_balance_G1["NAME_CONTRACT_STATUS"].values)
POS_CASH_balance_G1_num = (POS_CASH_balance_G1.groupby("SK_ID_CURR", as_index=False).mean())
nb = POS_CASH_balance_G1[["SK_ID_CURR", "NAME_CONTRACT_STATUS"]].groupby("SK_ID_CURR", as_index = False).count()
nb["num_in_POS_CASH"] = nb["NAME_CONTRACT_STATUS"]

df = df.merge(POS_CASH_balance_G1_num.drop("SK_ID_PREV", axis = 1), on='SK_ID_CURR', how='left').fillna(-1)
df = df.merge(nb.drop("NAME_CONTRACT_STATUS", axis = 1), on='SK_ID_CURR', how='left').fillna(-1)


del nb, POS_CASH_balance_G1_num, POS_CASH_balance_G1
gc.collect()


df.head()


# In[25]:


#from POS_CASH_balance_G1, let's merge categorical feature - NAME_CONTRACT_STATUS
#Also let's do log processing for large numbers


# **Processing and merging *bureau***

# In[26]:


#Bureau
bureau_G1 = bureau.drop(["SK_ID_BUREAU"], axis = 1).loc[bureau["SK_ID_CURR"].isin(total_IDS)]
print (len(np.unique(bureau_G1["SK_ID_CURR"].values)))
bureau_G1.head()


# In[27]:


for i in (bureau_G1.drop(["CREDIT_ACTIVE", "CREDIT_CURRENCY", "CREDIT_TYPE"], axis =1).columns): #numerical values
    plot_distribution(bureau_G1, i, "blue")


for i in ["CREDIT_ACTIVE", "CREDIT_CURRENCY", "CREDIT_TYPE"]: #categorical values
    dic = Counter(bureau_G1[i])
    plt.bar(range(len(dic)), list(dic.values()))
    plt.xticks(range(len(dic)), list(dic.keys()), rotation = 90)
    plt.title(i)
    plt.show()


# In[28]:


bureau_G1_num = (bureau_G1.groupby("SK_ID_CURR", as_index=False).mean())
nb = bureau_G1[["SK_ID_CURR", "CREDIT_ACTIVE"]].groupby("SK_ID_CURR", as_index = False).count()
nb["num_in_bureau"] = nb["CREDIT_ACTIVE"]

df = df.merge(bureau_G1_num, on='SK_ID_CURR', how='left').fillna(-1)
df = df.merge(nb.drop("CREDIT_ACTIVE", axis=1), on='SK_ID_CURR', how='left').fillna(-1)


del nb, bureau_G1_num, bureau_G1
gc.collect()


df.head()


# In[29]:


#let's work on categoricals - "CREDIT_ACTIVE", "CREDIT_CURRENCY", "CREDIT_TYPE"


# * **Processing and merging* credit_card_balance***

# In[30]:


##credit_card_balance
credit_card_balance_G1 = credit_card_balance.drop(["SK_ID_PREV"], axis = 1).loc[credit_card_balance["SK_ID_CURR"].isin(total_IDS)]
print (len(np.unique(credit_card_balance_G1["SK_ID_CURR"].values)))
credit_card_balance_G1.head()


# In[31]:


for i in (credit_card_balance_G1.drop(["NAME_CONTRACT_STATUS"], axis =1).columns): #numerical values
    plot_distribution(credit_card_balance_G1, i, "blue")


for i in ["NAME_CONTRACT_STATUS"]: #categorical values
    dic = Counter(credit_card_balance_G1[i])
    plt.bar(range(len(dic)), list(dic.values()))
    plt.xticks(range(len(dic)), list(dic.keys()), rotation = 90)
    plt.title(i)
    plt.show()


# In[32]:


credit_card_balance_G1_num = (credit_card_balance_G1.groupby("SK_ID_CURR", as_index=False).mean())
nb = credit_card_balance_G1[["SK_ID_CURR", "NAME_CONTRACT_STATUS"]].groupby("SK_ID_CURR", as_index = False).count()
nb["num_in_credit_card"] = nb["NAME_CONTRACT_STATUS"]

df = df.merge(credit_card_balance_G1_num, on='SK_ID_CURR', how='left').fillna(-1)
df = df.merge(nb.drop("NAME_CONTRACT_STATUS", axis=1), on='SK_ID_CURR', how='left').fillna(-1)


del nb, credit_card_balance_G1_num, credit_card_balance_G1
gc.collect()


df.head()


# **Processing and mergins *installments_payments***

# In[33]:


##installments_payments
installments_payments_G1 = installments_payments.drop(["SK_ID_PREV"], axis = 1).loc[installments_payments["SK_ID_CURR"].isin(total_IDS)]
print (len(np.unique(installments_payments_G1["SK_ID_CURR"].values)))
installments_payments_G1.head()


# In[34]:


for i in (installments_payments_G1.columns): #numerical values
    plot_distribution(installments_payments_G1, i, "blue")


# In[35]:


installments_payments_G1_num = (installments_payments_G1.groupby("SK_ID_CURR", as_index=False).mean())
nb = installments_payments_G1[["SK_ID_CURR", "NUM_INSTALMENT_VERSION"]].groupby("SK_ID_CURR", as_index = False).count()
nb["num_in_install_pay"] = nb["NUM_INSTALMENT_VERSION"]

df = df.merge(installments_payments_G1_num, on='SK_ID_CURR', how='left').fillna(-1)
df = df.merge(nb.drop("NUM_INSTALMENT_VERSION", axis=1), on='SK_ID_CURR', how='left').fillna(-1)


del nb, installments_payments_G1_num, installments_payments_G1
gc.collect()


df.head()


# **Processing and merging *previous_application***

# In[36]:


##previous_application
previous_application_G1 = previous_application.drop(["SK_ID_PREV"], axis = 1).loc[previous_application["SK_ID_CURR"].isin(total_IDS)]
print (len(np.unique(previous_application_G1["SK_ID_CURR"].values)))
previous_application_G1.head()


# In[37]:


categorical = categorical_features(previous_application_G1)
numerical = [i for i in previous_application_G1.columns if i not in categorical]
numerical.remove("SK_ID_CURR")

for i in numerical: #numerical values
    plot_distribution(previous_application_G1, i, "blue")


for i in categorical: #categorical values
    dic = Counter(previous_application_G1[i])
    plt.bar(range(len(dic)), list(dic.values()))
    plt.xticks(range(len(dic)), list(dic.keys()), rotation = 90)
    plt.title(i)
    plt.show()


# In[38]:


#Merge with numerical values
previous_application_G1_num = (previous_application_G1.groupby("SK_ID_CURR", as_index=False).mean())
nb = previous_application_G1[["SK_ID_CURR", "NAME_CONTRACT_TYPE"]].groupby("SK_ID_CURR", as_index = False).count()
nb["num_in_previous_app"] = nb["NAME_CONTRACT_TYPE"]

df = df.merge(previous_application_G1_num, on='SK_ID_CURR', how='left').fillna(-1)
df = df.merge(nb.drop("NAME_CONTRACT_TYPE", axis=1), on='SK_ID_CURR', how='left').fillna(-1)


del nb, previous_application_G1_num, previous_application_G1
gc.collect()


df.head()


# **Try LGBM with merged data**
# 
# Yet, I have much more things to be done (I noted them at the bottom of this notebook). However, I'll try LGBM with what I've got so far (Group 1)

# In[ ]:


from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold, KFold

train_X = df[:tr].drop("SK_ID_CURR", axis = 1)
test_X = df[tr:].drop("SK_ID_CURR", axis = 1)

train_X["TARGET"] = train_target
#label: train_target
y = train_target

folds = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(train_X.shape[0])
sub_preds = np.zeros(test_X.shape[0])
feats = [f for f in train_X.columns if f not in ['SK_ID_CURR','TARGET']]


# In[ ]:


for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train_X)):
    trn_x, trn_y = train_X[feats].iloc[trn_idx], train_X.iloc[trn_idx]['TARGET']
    val_x, val_y = train_X[feats].iloc[val_idx], train_X.iloc[val_idx]['TARGET']
    
    clf = LGBMClassifier(
        n_estimators=10000,
        learning_rate=0.01,
        num_leaves=30,
        colsample_bytree=.8,
        subsample=.9,
        max_depth=7,
        reg_alpha=.1,
        reg_lambda=.1,
        min_split_gain=.01,
        min_child_weight=100,
        #scale_pos_weight=12.5,
        silent=-1,
        verbose=-1,
    )
    
    clf.fit(trn_x, trn_y, 
            eval_set= [(trn_x, trn_y), (val_x, val_y)], 
            eval_metric='auc', verbose=100, early_stopping_rounds=100  #30
           )
    
    oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
    sub_preds += clf.predict_proba(test_X[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
    
    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
    del trn_x, trn_y, val_x, val_y
    gc.collect()


# In[ ]:


submission = pd.read_csv("../input/sample_submission.csv")
submission['TARGET'] = sub_preds
submission.to_csv("baseline2.csv", index=False)
submission.head()


# More things to do:
# - Adding categorical features when merging files (I only added numerical features for now. This was because each file have so many data on redundant SK_ID, so in case of numerical data I just merged average value within identical IDs, but in case of categorical data I'm not sure what would be the best way to merge it. For now I'm trying to get mode of categorical values** (But identifying mode of each IDs and merging them takes too much time and memory since there exists too much different IDs and categories.) I'll be glad to share any ideas on how to merge categorical values here!**
# 
# - Do log processing for numerical values (Haven't processed yet due to irregularity of numerical features, and some of files having too much features)
# 
# - More carefully handle Nan values : I just put -1 to all Nan values, but in some cases, say, if there exists value '-1' (or near -1) within the columns -1 would not appropriately identify 'Non existing value' for the column.
# 
# - Merge group 2 (previous payment as a hub, SK_ID_PREV as keys) to find out information on previous application. 
# 
# - Merge Bureau balance to df (Group 3)
