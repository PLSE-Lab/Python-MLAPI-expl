#!/usr/bin/env python
# coding: utf-8

# # **PLEASE SAVE YOUR WORK BY CREATING A VERSION**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pylab as plt
import seaborn as sns

import lightgbm as lgb

from sklearn.decomposition import PCA
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import *
from sklearn.metrics import *

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


sample_submission = pd.read_csv('sample_submission.csv')
print(sample_submission)


# In[ ]:


def reduce_size(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    print("Initial memory usage is ",start_mem," Mb")
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[ ]:


#Obtaining data from V,C,D,M and other important transaction columns
V_cols = ['V'+str(i) for i in range(1,340)]
C_cols = ['C'+str(i) for i in range(1,14)]
D_cols = ['D'+str(i) for i in range(1,15)]
M_cols = ['M'+str(i) for i in range(1,9)]
card_cols = ['card'+str(i) for i in range(1,6)]
trans_cols = ['TransactionID', 'TransactionDT', 'TransactionAmt','ProductCD','addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain']+C_cols+D_cols+M_cols+card_cols
train_Vcols = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv',usecols = trans_cols+V_cols+['isFraud'])
#train_Vcols = pd.read_csv('train_transaction.csv',usecols = trans_cols+V_cols+['isFraud'])

train_Vcols = reduce_size(train_Vcols)


# In[ ]:


nangp = {}
all_nans = train_Vcols.isna()
for i in train_Vcols:
    group = all_nans[i].sum()
    try:
        nangp[group].append(i)
    except:
        nangp[group] = [i]
for group,cols in nangp.items():
    print(group,' NaNs: ',cols)

def hmap(cols,train_Vcols):
    title = cols[0]+'-'+cols[-1]
    corr_mat = train_Vcols[cols].corr()
    plt.figure(figsize=(20,20))
    sns.heatmap(corr_mat,cmap='RdBu_r',annot=True,center=0.0)
    plt.title(title)
    plt.show()
    return corr_mat

for group,cols in nangp.items():
    if len(cols)<5:
        continue
    cmat = hmap(cols,train_Vcols)
    


# In[ ]:


def usecols(grouped_cols,train_Vcols):
    use = []
    for g in grouped_cols:
        mx = 0; vx = g[0]
        for gg in g:
            n = train_Vcols['V'+str(gg)].nunique()
            if n>mx:
                mx = n
                vx = gg
            #print(str(gg)+'-'+str(n),', ',end='')
        use.append(vx)
        #print()
    #print('Use these',use)
    return use

grouped_cols = [[[1],[2,3],[4,5],[6,7],[8,9],[10,11]],[[12,13],[14],[15,16,17,18,21,22,31,32,33,34],[19,20],[23,24],[25,26],[27,28],[29,30]],[[35,36],[37,38],[39,40,42,43,50,51,52],[41],[44,45],[46,47],[48,49]],[[53,54],[55,56],[57,58,59,60,63,64,71,72,73,74],[61,62],[65],[66,67],[68],[69,70]],[[75,76],[77,78],[79,80,81,84,85,92,93,94],[82,83],[86,87],[88],[89],[90,91]],[[95,96,97,101,102,103,105,106],[98],[99,100],[104]],               [[107],[108,109,110,114],[111,112,113],[115,116],[117,118,119],[120,122],[121],[123]],[[124,125],[126,127,128,132,133,134],[129],[130,131],[135,136,137]],[[138],[139,140],[141,142],[146,147],[148,149,153,154,156,157,158],[161,162,163]],               [[143,164,165],[144,145,150,151,152,159,160],[166]],[[167,168,177,178,179],[172,176],[173],[181,182,183]],[[186,187,190,191,192,193,196,199],[202,203,204,211,212,213],[205,206],[207],[214,215,216]],               [[169],[170,171,200,201],[174,175],[180],[184,185],[188,189],[194,195,197,198],[208,210],[209]],[[217,218,219,231,232,233,236,237],[223],[224,225],[226],[228],[229,230],[235]],               [[240,241],[242,243,244,258],[246,257],[247,248,249,253,254],[252],[260],[261,262]],[[263,265,264],[266,269],[267,268],[273,274,275],[276,277,278]],[[220],[221,222,227,245,255,256,259],[234],[238,239],[250,251],[270,271,272]],               [[279,280,293,294,295,298,299],[284],[285,287],[286],[290,291,292],[297]],[[302,303,304],[305],[306,307,308,316,317,318],[309,311],[310,312],[319,320,321]],[[281],[282,283],[288,289],[296],[300,301],[313,314,315]],               [[322,323,324,326,327,328,329,330,331,332,333],[325],[334,335,336],[337,338,339]]]
good_cols = [usecols(grouped_cols[i],train_Vcols) for i in range(len(grouped_cols))]
good_cols = [item for col in good_cols for item in col]
good_cols = ['V'+str(c) for c in good_cols]
print(good_cols)
train_trans = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv',usecols = trans_cols+good_cols+['isFraud'])
test_trans = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv',usecols = trans_cols+good_cols)#+['isFraud'])

#train_trans = pd.read_csv('train_transaction.csv',usecols = trans_cols+good_cols+['isFraud'])
#test_trans = pd.read_csv('test_transaction.csv',usecols = trans_cols+good_cols)

train_trans = reduce_size(train_trans)
test_trans = reduce_size(test_trans)
train_trans_Vpca = train_trans.copy()
test_trans_Vpca = test_trans.copy()


# In[ ]:


train_trans_gc = train_trans[good_cols]
test_trans_gc = test_trans[good_cols]
def get_pca(train_trans_gc,test_trans_gc):
    train_trans_gc.fillna(train_trans_gc.min(),inplace=True)
    test_trans_gc.fillna(train_trans_gc.min(),inplace=True)
    sc = StandardScaler()
    train_trans_gc = sc.fit_transform(train_trans_gc)
    #train_trans_gc = minmax_scale(train_trans_gc,feature_range = (0,1))
    #test_trans_gc = minmax_scale(test_trans_gc,feature_range = (0,1))
    test_trans_gc = sc.fit_transform(test_trans_gc)
        
    pca = PCA(n_components=50)
    pcomps_train = pca.fit_transform(train_trans_gc)
    pcomps_test = pca.transform(test_trans_gc)
    return pcomps_train,pcomps_test
train_trans_vred,test_trans_vred = get_pca(train_trans_gc,test_trans_gc)


# In[ ]:


pca_cols = ['Vpca_'+str(i) for i in range(50)]
train_trans_vred = pd.DataFrame(train_trans_vred,columns = pca_cols)
test_trans_vred = pd.DataFrame(test_trans_vred,columns = pca_cols)

train_trans_Vpca.drop(good_cols,axis=1,inplace=True)
train_trans_Vpca = pd.concat([train_trans_Vpca,train_trans_vred],axis=1)

test_trans_Vpca.drop(good_cols,axis=1,inplace=True)
test_trans_Vpca = pd.concat([test_trans_Vpca,test_trans_vred],axis=1)


# In[ ]:


train_id = pd.read_csv("../input/ieee-fraud-detection/train_identity.csv")
test_id = pd.read_csv("../input/ieee-fraud-detection/train_identity.csv")

#train_id = pd.read_csv("train_identity.csv")
#test_id = pd.read_csv("test_identity.csv")
train_id =  reduce_size(train_id)
test_id =  reduce_size(test_id)


# In[ ]:


train_id = reduce_size(train_id)
train_trans = reduce_size(train_trans)

 test_id = reduce_size(test_id)
test_trans = reduce_size(test_trans)

# Remove infinities
train_id.replace([np.inf, -np.inf], np.nan, inplace=True)
train_trans.replace([np.inf, -np.inf], np.nan, inplace=True)
test_id.replace([np.inf, -np.inf], np.nan, inplace=True)
test_trans.replace([np.inf, -np.inf], np.nan, inplace=True)

train_trans_Vpca.replace([np.inf, -np.inf], np.nan, inplace=True)
test_trans_Vpca.replace([np.inf, -np.inf], np.nan, inplace=True)


#Removing the columns with >90% of data missing.
print("Training Id shape before cleanup", train_id.shape)
train_id = train_id[train_id.columns[train_id.isnull().mean() < 0.9]]
# train_trans = train_trans[train_trans.columns[train_trans.isnull().mean() < 0.9]]
print("Training data Shape after", train_id.shape)
print("Test Id data Shape before", test_id.shape)
test_id = test_id[test_id.columns[test_id.isnull().mean() < 0.9]]
print("Test data Shape after", test_id.shape)


# In[ ]:


# 'DeviceInfo' column
# Reference: https://www.kaggle.com/davidcairuz/feature-engineering-lightgbm
for df in [train_id, test_id]:
        df["device_name"] = df["DeviceInfo"].str.split("/", expand=True)[0]
        df["device_version"] = df["DeviceInfo"].str.split("/", expand=True)[1]

        df.loc[
            df["device_name"].str.contains("SM", na=False), "device_name"
        ] = "Samsung"
        df.loc[
            df["device_name"].str.contains("SAMSUNG", na=False), "device_name"
        ] = "Samsung"
        df.loc[
            df["device_name"].str.contains("GT-", na=False), "device_name"
        ] = "Samsung"
        df.loc[
            df["device_name"].str.contains("Moto G", na=False), "device_name"
        ] = "Motorola"
        df.loc[
            df["device_name"].str.contains("Moto", na=False), "device_name"
        ] = "Motorola"
        df.loc[
            df["device_name"].str.contains("moto", na=False), "device_name"
        ] = "Motorola"
        df.loc[df["device_name"].str.contains("LG-", na=False), "device_name"] = "LG"
        df.loc[df["device_name"].str.contains("rv:", na=False), "device_name"] = "RV"
        df.loc[
            df["device_name"].str.contains("HUAWEI", na=False), "device_name"
        ] = "Huawei"
        df.loc[
            df["device_name"].str.contains("ALE-", na=False), "device_name"
        ] = "Huawei"
        df.loc[df["device_name"].str.contains("-L", na=False), "device_name"] = "Huawei"
        df.loc[df["device_name"].str.contains("Blade", na=False), "device_name"] = "ZTE"
        df.loc[df["device_name"].str.contains("BLADE", na=False), "device_name"] = "ZTE"
        df.loc[
            df["device_name"].str.contains("Linux", na=False), "device_name"
        ] = "Linux"
        df.loc[df["device_name"].str.contains("XT", na=False), "device_name"] = "Sony"
        df.loc[df["device_name"].str.contains("HTC", na=False), "device_name"] = "HTC"
        df.loc[df["device_name"].str.contains("ASUS", na=False), "device_name"] = "Asus"

        df.loc[
            df.device_name.isin(
                df.device_name.value_counts()[df.device_name.value_counts() < 200].index
            ),
            "device_name",
        ] = "rare"

        df["DeviceInfo"] = df["DeviceInfo"].fillna("unknown_device").str.lower()
        df["deviceInfo_device"] = df["DeviceInfo"].apply(
            lambda x: "".join([i for i in x if i.isalpha()])
        )
        df["deviceInfo_version"] = df["DeviceInfo"].apply(
            lambda x: "".join([i for i in x if i.isnumeric()])
        )


# In[ ]:


print(train_id.columns)
print(test_id.columns)
renamed_cols = ['TransactionID', 'id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06',                'id_09', 'id_10', 'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16',                'id_17', 'id_18', 'id_19', 'id_20', 'id_28', 'id_29', 'id_30', 'id_31',                'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38',                'DeviceType', 'DeviceInfo', 'device_name', 'device_version','deviceInfo_device', 'deviceInfo_version']
test_id.columns = renamed_cols


# In[ ]:


## Get Browser and version from id_31
for df in [train_id, test_id]:
    df['browser'] = df['id_31'].str.split(' ', expand=True)[0]
    df['version'] = df['id_31'].str.split(' ', expand=True)[1]
    
    # For Safari values are misplaced in train_df and test_df
    df.loc[df['version'] == 'safari', 'browser'] = 'safari'
    df.loc[df['version'] == 'safari', 'version'] = df[df['version'] == 'safari']['id_31'].str.split(' ', expand=True)[2]

    # Get screen_width and screen_height from id_33
    df['screen_width'] = df['id_33'].str.split('x', expand=True)[0]
    df['screen_height'] = df['id_33'].str.split('x', expand=True)[1]
    


# In[ ]:


## List Numerical and Categorical Columns
## categorical_columns: Replace Nan values with missing
## numerical_columns: Replace Nan values with -1
for df in [train_id, test_id, train_trans, test_trans,train_trans_Vpca,test_trans_Vpca]:
    numerical_columns = df._get_numeric_data().columns
    categorical_columns = list(set(df.columns) - set(numerical_columns))
    df[categorical_columns] = df[categorical_columns].replace({ np.nan:'missing'})
    df[numerical_columns] = df[numerical_columns].replace({ np.nan:-1})
    
## id_30 is OS name
## Restrict to os_categories, currently ignoring versions
os_categories = ['Mac', 'iOS', 'Android', 'Windows', 'Linux']
for df in [train_id, test_id]:
    for os in os_categories:
        df.loc[df['id_30'].str.contains(os, na=False), 'id_30'] = os


# In[ ]:


# Card Noise Removal
# Card 1, 3, 5 have numeric values and may contain noise values, i.e values that occur very infrequently
# Here we remove them
card_threshold_map = {
    "card1": 2,
    "card3": 200,
    "card5": 300
}

for card, threshold in card_threshold_map.items():
    noise_card_list = list(train_trans[card].value_counts()[train_trans[card].value_counts() < threshold].index)
    for df in [train_trans, test_trans]:
        df.loc[df[card].isin(noise_card_list), card] = "Others"


# In[ ]:


# Creates four new Catgorical columns
# P_emaildomain_bin (google, microsoft etc.)
# P_emaildomain_suffix (.com etc)
# R_emaildomain_bin",
# R_emaildomain_suffix",
def bin_emails_and_domains():
    # https://www.kaggle.com/c/ieee-fraud-detection/discussion/100499#latest_df-579654
    emails = {
        "gmail": "google",
        "att.net": "att",
        "twc.com": "spectrum",
        "scranton.edu": "other",
        "optonline.net": "other",
        "hotmail.co.uk": "microsoft",
        "comcast.net": "other",
        "yahoo.com.mx": "yahoo",
        "yahoo.fr": "yahoo",
        "yahoo.es": "yahoo",
        "charter.net": "spectrum",
        "live.com": "microsoft",
        "aim.com": "aol",
        "hotmail.de": "microsoft",
        "centurylink.net": "centurylink",
        "gmail.com": "google",
        "me.com": "apple",
        "earthlink.net": "other",
        "gmx.de": "other",
        "web.de": "other",
        "cfl.rr.com": "other",
        "hotmail.com": "microsoft",
        "protonmail.com": "other",
        "hotmail.fr": "microsoft",
        "windstream.net": "other",
        "outlook.es": "microsoft",
        "yahoo.co.jp": "yahoo",
        "yahoo.de": "yahoo",
        "servicios-ta.com": "other",
        "netzero.net": "other",
        "suddenlink.net": "other",
        "roadrunner.com": "other",
        "sc.rr.com": "other",
        "live.fr": "microsoft",
        "verizon.net": "yahoo",
        "msn.com": "microsoft",
        "q.com": "centurylink",
        "prodigy.net.mx": "att",
        "frontier.com": "yahoo",
        "anonymous.com": "other",
        "rocketmail.com": "yahoo",
        "sbcglobal.net": "att",
        "frontiernet.net": "yahoo",
        "ymail.com": "yahoo",
        "outlook.com": "microsoft",
        "mail.com": "other",
        "bellsouth.net": "other",
        "embarqmail.com": "centurylink",
        "cableone.net": "other",
        "hotmail.es": "microsoft",
        "mac.com": "apple",
        "yahoo.co.uk": "yahoo",
        "netzero.com": "other",
        "yahoo.com": "yahoo",
        "live.com.mx": "microsoft",
        "ptd.net": "other",
        "cox.net": "other",
        "aol.com": "aol",
        "juno.com": "other",
        "icloud.com": "apple",
    }
    us_emails = ["gmail", "net", "edu"]

    purchaser = "P_emaildomain"
    recipient = "R_emaildomain"
    unknown = "email_not_provided"

    for df in [train_trans, test_trans]:
        df["is_proton_mail"] = (df[purchaser] == "protonmail.com") | (
            df[recipient] == "protonmail.com"
        )
        df["email_check"] = np.where(
            (df[purchaser] == df[recipient]) & (df[purchaser] != unknown), 1, 0
        )

        for c in [purchaser, recipient]:
            df[purchaser] = df[purchaser].fillna(unknown)
            df[recipient] = df[recipient].fillna(unknown)

            df[c + "_bin"] = df[c].map(emails)

            df[c + "_suffix"] = df[c].map(lambda x: str(x).split(".")[-1])

            df[c + "_suffix"] = df[c + "_suffix"].map(
                lambda x: x if str(x) not in us_emails else "us"
            )
bin_emails_and_domains()


# In[ ]:


#V values without PCA
train_df = train_trans.merge(train_id, how='left', left_index=True, right_index=True, on='TransactionID')
test_df = test_trans.merge(test_id, how='left', left_index=True, right_index=True, on='TransactionID')

#2nd dataset contains V values with PCA
train_df_Vpca = train_trans_Vpca.merge(train_id, how='left', left_index=True, right_index=True, on='TransactionID')
test_df_Vpca = test_trans_Vpca.merge(test_id, how='left', left_index=True, right_index=True, on='TransactionID')

for df in [train_df, test_df, train_df_Vpca, test_df_Vpca]:
    numerical_columns = df._get_numeric_data().columns
    categorical_columns = list(set(df.columns) - set(numerical_columns))
    df[categorical_columns] = df[categorical_columns].replace({ np.nan:'missing'})
    df[numerical_columns] = df[numerical_columns].replace({ np.nan:-1})


# In[ ]:


for f in train_df.drop('isFraud', axis=1).columns:
    if train_df[f].dtype=='object' or test_df[f].dtype=='object':
        try:
            lbl = LabelEncoder()
            lbl.fit(list(train_df[f].values) + list(test_df[f].values))
            train_df[f] = lbl.transform(list(train_df[f].values))
            test_df[f] = lbl.transform(list(test_df[f].values))   
        except:
            train_df.drop(f,axis=1,inplace=True)
            test_df.drop(f,axis=1,inplace=True)
            print(f)
            
for f in train_df_Vpca.drop('isFraud', axis=1).columns:
    if train_df_Vpca[f].dtype=='object' or test_df_Vpca[f].dtype=='object':
        try:
            lbl = LabelEncoder()
            lbl.fit(list(train_df_Vpca[f].values) + list(test_df_Vpca[f].values))
            train_df_Vpca[f] = lbl.transform(list(train_df_Vpca[f].values))
            test_df_Vpca[f] = lbl.transform(list(test_df_Vpca[f].values))   
        except:
            train_df_Vpca.drop(f,axis=1,inplace=True)
            test_df_Vpca.drop(f,axis=1,inplace=True)
            print(f)


# In[ ]:


X_train = train_df.sort_values('TransactionDT').drop(['isFraud','TransactionDT'],axis=1)
y_train = train_df.sort_values('TransactionDT')['isFraud'].astype(bool)
X_test = test_df.sort_values('TransactionDT').drop(['TransactionDT'],axis=1)

X_train_Vpca = train_df_Vpca.sort_values('TransactionDT').drop(['isFraud','TransactionDT'],axis=1)
y_train_Vpca = train_df_Vpca.sort_values('TransactionDT')['isFraud'].astype(bool)
X_test_Vpca = test_df_Vpca.sort_values('TransactionDT').drop(['TransactionDT'],axis=1)

Xtrain, Xvalid, ytrain, yvalid = train_test_split(X_train,y_train,test_size=0.25, random_state=0)


# In[ ]:


#LightGBM Model for V with and without PCA -- Version 1
clf1 = lgb.LGBMClassifier(
    num_leaves= 256,
    min_child_samples= 79,
    objective="binary",
    max_depth=13,
    learning_rate=0.03,
    boosting_type="gbdt",
    subsample_freq=3,
    subsample=0.9,
    bagging_seed=11,
    metric = 'auc',
    verbosity=-1,
    reg_alpha=0.3,
    reg_lambda=0.3,
    colsample_bytree=0.9,
    n_jobs= -1
    )

clf1.fit(X_train, y_train,verbose=True)
y_predict = clf1.predict_proba(X_test)[:,1]
sample_submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv', index_col='TransactionID')
#sample_submission = pd.read_csv('sample_submission.csv', index_col='TransactionID')

sample_submission['isFraud'] = y_predict
sample_submission.to_csv('submission1.csv')

clf1.fit(X_train_Vpca, y_train_Vpca,verbose=True)
y_predict = clf1.predict_proba(X_test_Vpca)[:,1]
sample_submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv', index_col='TransactionID')
#sample_submission = pd.read_csv('sample_submission.csv', index_col='TransactionID')

sample_submission['isFraud'] = y_predict
sample_submission.to_csv('submission2.csv')


# In[ ]:


clf2 = lgb.LGBMClassifier(
    max_bin = 63,
    num_leaves = 255,
    num_iterations = 500,
    learning_rate = 0.01,
    tree_learner = 'serial',
    task = 'train',
    is_training_metric = False,
    min_data_in_leaf = 1,
    min_sum_hessian_in_leaf = 100,
    sparse_threshold=1.0,
    device = 'cpu',
    num_thread = -1,
    save_binary= True,
    seed= 42,
    feature_fraction_seed = 42,
    bagging_seed = 42,
    drop_seed = 42,
    data_random_seed = 42,
    objective = 'binary',
    boosting_type = 'gbdt',
    verbose = 1,
    metric = 'auc',
    is_unbalance = True,
    boost_from_average = False,
)
clf2.fit(X_train, y_train)
sample_submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv', index_col='TransactionID')
#sample_submission = pd.read_csv('sample_submission.csv', index_col='TransactionID')
sample_submission['isFraud'] = clf2.predict_proba(X_test)[:,1]
sample_submission.to_csv('submission3.csv')

clf2.fit(X_train_Vpca, y_train_Vpca,verbose=True)
y_predict = clf2.predict_proba(X_test_Vpca)[:,1]
sample_submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv', index_col='TransactionID')
#sample_submission = pd.read_csv('sample_submission.csv', index_col='TransactionID')
sample_submission['isFraud'] = y_predict
sample_submission.to_csv('submission4.csv')


# In[ ]:


clf3 = RandomForestClassifier(n_jobs=-1, n_estimators = 250,verbose=1,max_features=20,max_depth=15)
clf3.fit(X_train,y_train)
y_predict = clf3.predict_proba(X_test)[:,1]
sample_submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv', index_col='TransactionID')
#sample_submission = pd.read_csv('sample_submission.csv', index_col='TransactionID')
sample_submission['isFraud'] = y_predict
sample_submission.to_csv('submission5.csv')


# In[ ]:


print(test_scores)


# In[ ]:


#XGBoost Model 
import xgboost as xgb



clf4 = xgb.XGBClassifier(n_jobs=-1, 
        objective = 'binary:logistic',
        eval_metric = 'auc',  
        n_estimators=250,
        max_depth=15,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method = 'hist'
    )

clf4.fit(X_train, y_train)
y_predict = clf4.predict_proba(X_test)[:,1]
sample_submission = pd.read_csv('./input/sample_submission.csv', index_col='TransactionID')
sample_submission['isFraud'] = y_predict
sample_submission.to_csv('submission6.csv')
# 0.928964


clf4.fit(X_train_Vpca, y_train_Vpca,verbose=True)
y_predict = clf2.predict_proba(X_test_Vpca)[:,1]
sample_submission = pd.read_csv('./input/sample_submission.csv', index_col='TransactionID')
sample_submission['isFraud'] = y_predict
sample_submission.to_csv('submission7.csv')
#0.921232


# predictions = model.predict(test)

# ### Time Features

# There is a surge in the plot. It might be Christmas season. So we will look into whether Christamas season has many fraud cases. If so, we will create new variables named Christmas season, Thanksgiving and so on. And then try to implement Von Minsur DIstribution to create new variables.

# There might be outliers in this dataset.

# In[ ]:


sns.relplot(x= 'TransactionDT',y = 'TransactionAmt', data = train_df, color = 'b')
plt.show()
train_df[train_df['TransactionAmt'] > 10000]
train_df['TransactionDT']
sns.distplot(train_df['TransactionDT'], kde=False)
plt.show()


# The logic behind using Von Mises distribution. In this dataset, we have time features. For D columns, there are many missing values which might be not be helpful. So my gerneral concept about using Time features is to use Transaction Delta Time.
# 
# * First, I will fit the data into Vonmises Distribution to estimate the parameters. 
# * Second, Create a corresponding values when testing points come.
# * Third, evaluate new variables.

# In[ ]:


from scipy.stats import vonmises
import matplotlib.pyplot as plt
#kappa, loc and scale.
vonmises.fit(train_df['TransactionDT'])

train_df['isFraud'].value_counts()

Not_Fraud = train_df[train_df['isFraud'] == 0]
Is_Fraud = train_df[train_df['isFraud'] == 1]

sns.distplot(Not_Fraud['TransactionDT'], kde=False)
plt.show()

sns.distplot(Is_Fraud['TransactionDT'], kde=False)
plt.show()
train_df[D_cols].describe()
sns.pairplot(train_df[D_cols])
plt.show()

# Overview of each col in D_cols
for col in D_cols:
    print("Column Name: {}".format(col))
    print("Null Values Number: ", train_df[col].isnull().sum())
    print(train_df[col].value_counts())


# In[ ]:


from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection, neighbors)


# In[ ]:


tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
t0 = time()
X_tsne = tsne.fit_transform(d_df)
y = train_df['isFraud']


# In[ ]:


y = train_df['isFraud']
area1 = np.ma.masked_where(y != 1, y)
area2 = np.ma.masked_where(y != 0, y)


# In[ ]:


plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=area2+5, label="Label y = 0", c='g')
plt.title("Tsne (Not Fraud)")
plt.xlabel('First Component')
plt.ylabel("Second Component")
plt.show()


# In[ ]:


plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=area1+5, label="Label y = 1", c='r')
plt.title("Tsne (isFraud)")
plt.xlabel('First Component')
plt.ylabel("Second Component")
plt.show()


# In[ ]:


t = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')
y = t['isFraud']
yy = y.to_numpy()

area1 = np.ma.masked_where(yy!=0, 5*np.ones(SY))
area2 = np.ma.masked_where(yy!=1, 5*np.ones(SY))
train_trans_vred_np = train_trans_vred.to_numpy()
train_trans_gc.fillna(train_trans_gc.min(),inplace=True)
train_trans_gc_np = train_trans_gc.to_numpy()

#############   MDS embedding    ##################################
print("Computing MDS embedding")
clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
X_mds = clf.fit_transform(train_trans_vred_np)
plt.scatter(X_mds[:,0],X_mds[:,1], s=area1,label='not fraud',color='b')
plt.xlabel("x1")
plt.ylabel("x2")
ylim((-300, 2500))
plt.legend()
plt.show()
plt.scatter(np.concatenate(X_mds[:,0],X_mds[:,1], s=area2[1:60000],label='fraud',color='r')
plt.xlabel("x1")
plt.ylabel("x2")
ylim((-300, 2000))
plt.legend()
plt.show()


# In[ ]:


#############   Spectral embedding    ##################################
print("Computing Spectral embedding")
embedder = manifold.SpectralEmbedding(n_components=2, random_state=0,eigen_solver="arpack")
X_se = embedder.fit_transform(train_trans_vred_np)
plt.scatter(X_se[:,0],X_se[:,1], s=area1,label='not fraud',color='b')
plt.xlabel("x1")
plt.ylabel("x2")
ylim((-300, 2500))
plt.legend()
plt.show()
plt.scatter(np.concatenate(X_se[:,0],X_se[:,1], s=area2[1:60000],label='fraud',color='r')
plt.xlabel("x1")
plt.ylabel("x2")
ylim((-300, 2000))
plt.legend()
plt.show()


# In[ ]:


############   PCA 3D visualization   ############################
fig = plt.figure(figsize = (25, 16))
ax = fig.add_subplot(111, projection='3d')
xs = train_trans_vred_np[:,1]
ys = train_trans_vred_np[:,0]
zs = train_trans_vred_np[:,2]
ax.scatter(xs, ys, zs, s=area1,label='y==not fraud')
ax.scatter(xs, ys, zs, s=area2,label='y==fraud')
ax.set_xlim3d(-25,80)
ax.set_ylim3d(-20,30)
ax.set_zlim3d(0,30)
ax.set_xlabel("x2")
ax.set_ylabel("x1")
ax.set_zlabel("x3")
plt.legend()
plt.show()

