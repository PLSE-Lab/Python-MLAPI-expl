#!/usr/bin/env python
# coding: utf-8

# ## Genetic variant conflicting classifications

# ### Context
# 
# According to the [data description](https://www.kaggle.com/kevinarvai/clinvar-conflicting/home) the data is from a public resource called [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar), which contains annotations about genetic variants. If the variants have conflicting classifications from laboratory to laboratory, it can cause confusion when reserachers try to assess the impact of the variant on a given patient.
# 

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import operator
import warnings
from collections import Counter
from itertools import chain
from time import time
# feature selection
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV,SelectFromModel
# classifiers
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV,cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import resample,shuffle
import matplotlib.pyplot as plt


# ### Preliminary EDA
# 
# Let's begin by reading in the data and looking at the columns:

# In[ ]:


vardat = pd.read_csv('../input/clinvar_conflicting.csv',dtype={0:object,38:object,40:object})
# explicity define dtype for pandas dtype error
vardat.head()


# And now the rows and columns in detail...

# In[ ]:


print(vardat.shape)
print(vardat.columns)


# In[ ]:


vardat.describe()


# ##### Now the total number of positive and negative esamples in the class

# In[ ]:


sns.countplot(vardat['CLASS'])
plt.show()
print(Counter(vardat['CLASS'].values))


# So, it appears that the number of negative examples are ~3 times that of the positive samples. The class imbalance problem has to be considered while building a classifer for this dataset. But, that is something for the later, now explore the other columns of this dataset:
# 
# #### Class count per chromosome

# In[ ]:


vardat.groupby(['CHROM','CLASS']).size()


# Some of the chromosomes have more misclassified variants than others, as expected.
# ##### REF and ALT allele count

# In[ ]:


Counter(vardat[['REF', 'ALT']].apply(lambda x: ':'.join(x), axis=1))


# So it appears that C&rarr;T and G&rarr;A SNPs are the most prominent in this dataset, and as mentioned in the data description, SNPs are the most prominent in this dataset.
# 
# #### ORIGIN count  

# In[ ]:


print(Counter(vardat['ORIGIN'].values))


# According to data description,`0` should be the `unknown` origin, but this doesn't seem to be the case, so fill all `nan` with 0s 

# In[ ]:


vardat['ORIGIN'].fillna(0, inplace=True)
print(Counter(vardat['ORIGIN'].values))


# #### Consequence column

# In[ ]:


cons = Counter(list(chain.from_iterable([str(v).split('&') for v in vardat['Consequence'].values])))
sorted(cons.items(),key=operator.itemgetter(1),reverse=True)


# #### CLNVC (Variant Type) column

# In[ ]:


clnvc = Counter(vardat['CLNVC'].values)
sorted(clnvc.items(),key=operator.itemgetter(1),reverse=True)


# #### CLNDN (disease name count) column

# In[ ]:


clndn = Counter(list(chain.from_iterable([str(v).split('|') for v in vardat['CLNDN'].values])))
sorted(clndn.items(),key=operator.itemgetter(1),reverse=True)


# Since the only variables with numerical variables are allele frequency **(AF..)** and CADD score **(CADD..)** explore these values in detail:

# In[ ]:


var_vals = vardat[['AF_ESP','AF_EXAC','AF_TGP','CADD_PHRED','CADD_RAW','CLASS']].dropna()
print(var_vals.info())
print(var_vals.describe())
print(Counter(var_vals['CLASS']))


# In[ ]:


sns.pairplot(var_vals,hue='CLASS')
plt.show()


# In[ ]:


var_corr = vardat[['AF_ESP','AF_EXAC','AF_TGP','CADD_PHRED','CADD_RAW']].corr()
cmap = sns.diverging_palette(220, 20, n=7,as_cmap=True)
sns.heatmap(var_corr, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5,annot=True,cbar_kws={"shrink": .5})


# ### Data formatting
# 
# The column `Consequence` has values concatenated by `&`, so these need to be separated properly, and some of the columns cannot be used as such, so these have to be either one hot encoded, or metadata (for example: count of certain attributes) have to be generated. The functions below are written to handle these issues.

# In[ ]:


# Below are some functions used to encode various columsn
def consequence_encoder(consequence,consdict):
    '''
    encoder for consequence data
    '''
    outmat = np.zeros((len(consequence),len(consdict)),dtype=np.int)
    for i,cons in enumerate(consequence):
        conslist = str(cons).split('&')
        cindex = np.zeros((len(consdict)),dtype=np.int)
        for c in conslist:
            if c in consdict:
                cindex[consdict[c]]=1
            else:
                continue
        outmat[i] = cindex
    return outmat

def get_base_dict(ref_alt,mincount=25):
    '''
    return all the Reference/Alternate bases with count>=mincount
    '''
    base_count = Counter(ref_alt)
    basedict = {}
    i = 0
    for b,c in base_count.items():
    #     print(a,c)
        if c<mincount:
            continue
        basedict[b]= i
        i+=1
    return basedict
    

def base_encoder(basedat,basedict):
    '''
    encoder for Reference/Alternate bases
    '''
    basemat = np.zeros((len(basedat),len(basedict)),dtype=np.int)
    for i,b in enumerate(basedat):
        bindex = np.zeros((len(basedict)),dtype=np.int)
        if b in basedict:
            bindex[basedict[b]] = 1
    return basemat

def CLNDISDB_count(clndisdb):
    '''
    return count of evidence ids
    '''
    clncount = np.zeros(shape=(len(clndisdb)),dtype=np.int)
    for i, cln in enumerate(clndisdb):
        clncount[i]=len(re.sub(pattern=r'\.\|',repl='',string=str(cln)).split('|'))
    return clncount

def CLNDN_count(clndn):
    '''
    return clinvar disease name
    '''
    clndncount = np.zeros(shape=(len(clndn)),dtype=np.int)
    for i, cln in enumerate(clndn):
        clndncount[i]=len(re.sub(pattern=r'\.\|',repl='',string=str(cln)).split('|'))
    return clndncount

def get_clndn_dict(clndn,mincount=25):
    '''
    return clinvar disease name dictionary, where each disease name must occur mincount times
    '''
    clndn_count = Counter(list(chain.from_iterable([str(dn).split('|') for dn in clndn])))
    clndn_dict = {}
    i = 0
    for dn,cn in clndn_count.items():
        if cn < mincount:
            continue
        clndn_dict[dn]=i
        i+=1
    return clndn_dict

def clndn_encoder(clndn,clndn_dict):
    '''
    encoder for clinvar disease names
    '''
    clndnmat = np.zeros((len(clndn),len(clndn_dict)),dtype=np.int)
    for i,dns in enumerate(clndn):
        dndat = str(dns).split('|')
        dnindex = np.zeros((len(clndn_dict)),dtype=np.int)
        for dn in dndat:
            if dn in clndn_dict:
                dnindex[clndn_dict[dn]] = 1
    return clndnmat


# In[ ]:


format_dat = vardat[['AF_ESP','AF_EXAC','AF_TGP','LoFtool']]
format_dat.fillna(0, inplace=True)
format_dat.isnull().values.any()


# In[ ]:


cons_set = set(list(chain.from_iterable([str(v).split('&') for v in vardat['Consequence'].values])))
consdict = dict(zip(cons_set,range(len(cons_set))))
# # CLNDISDB
clndb_count = CLNDISDB_count(vardat['CLNDISDB'].values)
format_dat =  np.concatenate((format_dat,clndb_count.reshape(-1,1)),axis=1)
# # CLNDN
clndn_count = CLNDN_count(vardat['CLNDN'].values)
format_dat =  np.concatenate((format_dat,clndn_count.reshape(-1,1)),axis=1)
# # Reference allele length
reflen = np.array([len(r) for r in vardat['REF'].values],dtype=np.int)
format_dat =  np.concatenate((format_dat,reflen.reshape(-1, 1)),axis=1)
# # allele length
allelelen = np.array([len(r) for r in vardat['Allele'].values],dtype=np.int)
format_dat =  np.concatenate((format_dat,allelelen.reshape(-1, 1)),axis=1)
# chromosome
chr_encoder = LabelEncoder()
chr_onehot = OneHotEncoder(sparse=False)
chr_ind = chr_encoder.fit_transform(vardat['CHROM'])
format_dat =  np.concatenate((format_dat,chr_onehot.fit_transform(chr_ind.reshape(-1, 1))),axis=1)
# # origin
origin_encoder = OneHotEncoder(sparse=False)
format_dat =  np.concatenate((format_dat,origin_encoder.fit_transform(vardat[['ORIGIN']])),axis=1)
# # CLNVC
cldn_encoder = LabelEncoder()
cldn_onehot = OneHotEncoder(sparse=False)
clndn_ind = cldn_encoder.fit_transform(vardat['CLNVC'])
format_dat =  np.concatenate((format_dat,cldn_onehot.fit_transform(clndn_ind.reshape(-1, 1))),axis=1)
# # impact 
imp_encoder = LabelEncoder()
imp_onehot = OneHotEncoder(sparse=False)
imp_ind = imp_encoder.fit_transform(vardat['IMPACT'])
format_dat =  np.concatenate((format_dat,imp_onehot.fit_transform(imp_ind.reshape(-1, 1))),axis=1)
# # Exon encoder
exon_encode = np.ones((vardat.shape[0]),dtype=np.int)
exon_encode[vardat['EXON'].isna()]=0
format_dat =  np.concatenate((format_dat,exon_encode.reshape(-1, 1)),axis=1)
# # clinical disease name
clndn_dict = get_clndn_dict(vardat['CLNDN'],100)
clndn_encode = clndn_encoder(vardat['CLNDN'],clndn_dict)
format_dat =  np.concatenate((format_dat,clndn_encode),axis=1)
# # consequence 
cons_encode = consequence_encoder(vardat['Consequence'].values,consdict)
format_dat =  np.concatenate((format_dat,cons_encode),axis=1)
# # base data
base_dict = get_base_dict(vardat[['REF', 'ALT']].apply(lambda x: ':'.join(x), axis=1),50)
base_encode = base_encoder(list(vardat[['REF', 'ALT']].apply(lambda x: ':'.join(x), axis=1)),base_dict)
format_dat =  np.concatenate((format_dat,base_encode),axis=1)
print(format_dat.shape)


# In[ ]:


dat_Xtrain,tmp_x,dat_Ytrain,tmp_y = train_test_split(format_dat,vardat['CLASS'],test_size=0.3,random_state=42)
dat_Xval,dat_Xtest,dat_Yval,dat_Ytest= train_test_split(tmp_x,tmp_y,test_size=0.5,random_state=42)


# In[ ]:


print('Training data stats')
print(dat_Xtrain.shape)
print(Counter(dat_Ytrain))
print('\nValidation data stats')
print(dat_Xval.shape)
print(Counter(dat_Yval))
print('\nTest data stats')
print(dat_Xtest.shape)
print(Counter(dat_Ytest))


# ### Classification
# 
# As the first step, estimate a baseline score using random forest classifer and compare the results from other classifers  for an imporvement. Finally, we will also check whether accounting for the class imbalance problem improves final test results. 
# 
# #### Base classifier

# In[ ]:


warnings.simplefilter(action='ignore',category=FutureWarning)
rf_base = RandomForestClassifier(random_state=42,n_jobs=6,n_estimators=100)
print(cross_val_score(rf_base,dat_Xtrain,dat_Ytrain,cv=10))
rf_base.fit(dat_Xtrain,dat_Ytrain)
y_pred = rf_base.predict(dat_Xval)
print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))
print('Accuracy\n',accuracy_score(dat_Yval,y_pred))
print('Classificaton report\n',classification_report(dat_Yval,y_pred))


# From the results of the base classifier, it is clear that it performs reasonably well for class `0`, it is another case for the performance metrics for class `1`. Now before moving to other classifiers, lets explore if paramter optimization can help improve the results.

# #### Bagging classifier
# 
# Test how bagging classifier performs in this scenario

# In[ ]:


bc_base = BaggingClassifier(random_state=42,n_jobs=6,n_estimators=100)
print(cross_val_score(bc_base,dat_Xtrain,dat_Ytrain,cv=10))
bc_base.fit(dat_Xtrain,dat_Ytrain)
y_pred = bc_base.predict(dat_Xval)
print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))
print('Accuracy\n',accuracy_score(dat_Yval,y_pred))
print('Classificaton report\n',classification_report(dat_Yval,y_pred))


# So, the bagging classifier shows an improvement in `recall`, but `accuracy` remains somewhat the same, so we will keep exploring a bit further.
# 
# #### ExtraTrees classifier
# 
# Testing extratrees classifier

# In[ ]:


et_base = ExtraTreesClassifier(random_state=42,n_jobs=6,n_estimators=100)
print(cross_val_score(et_base,dat_Xtrain,dat_Ytrain,cv=10))
et_base.fit(dat_Xtrain,dat_Ytrain)
y_pred = et_base.predict(dat_Xval)
print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))
print('Accuracy\n',accuracy_score(dat_Yval,y_pred))
print('Classificaton report\n',classification_report(dat_Yval,y_pred))


# ExtraTreesClassifier do not perform well compared to other classifers in this case, so moving on to the next classifier.
# 
# #### AdaBoost classifier

# In[ ]:


abc = AdaBoostClassifier(random_state=42,n_estimators=100)
print(cross_val_score(abc,dat_Xtrain,dat_Ytrain,cv=10))
abc.fit(dat_Xtrain,dat_Ytrain)
y_pred = abc.predict(dat_Xval)
print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))
print('Accuracy\n',accuracy_score(dat_Yval,y_pred))
print('Classificaton report\n',classification_report(dat_Yval,y_pred))


# Adaboost classifier also does not perform well in this case
# 
# #### Gradient boosting classifier

# In[ ]:


xg_base = XGBClassifier(random_state=42,n_jobs=6,n_estimators=100)
print(cross_val_score(xg_base,dat_Xtrain,dat_Ytrain,cv=10))
xg_base.fit(dat_Xtrain,dat_Ytrain)
y_pred = xg_base.predict(dat_Xval)
print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))
print('Accuracy\n',accuracy_score(dat_Yval,y_pred))
print('Classificaton report\n',classification_report(dat_Yval,y_pred))


# Gradient boosting tree shows pathetic `recall` and `f1-score` on the validation data set. So, all the results above can be compressed into one table below. The results below are only for `class 1` as it is the class we are interested in.
# 
# | Classifier | Precision | Recall | F1-score|
# |------------|:---------:|:------:|:-------:|
# | Random forest| 0.57 | 0.38 | 0.46 |
# | Bagging | 0.57 | 0.42 | 0.48 |
# | Extra trees| 0.52 | 0.38 | 0.44|
# | Adaboost| 0.59 | 0.20 | 0.29 |
# | Gradient boost| 0.62 | 0.17 | 0.26 |
# 
# These results indicate that none of the classifier does a good job of classifying the variants that are potentially mislabled. So can we improve these scores by chaning the training data ? From looking at the number of `0` and `1` classes in the dataset, it is clear that this dataset suffers from class imbalance problem. Two of the common ways to deal with imbalance is to either upsample the smaller class data or downsample the larger class data. These will be handled in the next section
# 
# ### Class balancing
# 
# #### Up and down sampling
# 
# To balance the classes in the dataset, the larger class can either be upsampled or the smaller class can be downsampled. The functions below are written to handle up and down sampling. 

# In[ ]:


def upsample(x,y):
    '''
    upsample least represented class
    y should be the labels,up sample least represented class in y
    '''
    x = np.array(x)
    ycount = Counter(y)
    ymin = min(list(ycount.values()))
    ymax = max(list(ycount.values()))
    yind = {}
    rex = None
    rey = list()
    for yi,c in ycount.items():
        if c==ymax:
            ind = np.where(y==yi)[0]
            if rex is None:
                rex = x[ind]
            else:
                rex = np.concatenate((rex,x[ind]),axis=0)
            rey.extend([yi]*ymax)
        elif c==ymin:
            ind = np.where(y==yi)[0]
            tmp_dat = resample(x[ind],replace=True,n_samples=ymax,random_state=42)
            if rex is None:
                rex = tmp_dat
            else:
                rex = np.concatenate((rex,tmp_dat),axis=0)
            rey.extend([yi]*ymax)
    return shuffle(rex,np.array(rey),random_state=42,replace=False)

def downsample(x,y):
    '''
    downsample over represented class
    y should be the labels,up sample least represented class in y
    '''
    x = np.array(x)
    ycount = Counter(y)
    ymin = min(list(ycount.values()))
    ymax = max(list(ycount.values()))
    yind = {}
    rex = None
    rey = list()
    for yi,c in ycount.items():
        if c==ymin:
            ind = np.where(y==yi)[0]
            if rex is None:
                rex = x[ind]
            else:
                rex = np.concatenate((rex,x[ind]),axis=0)
            rey.extend([yi]*ymin)
        elif c==ymax:
            ind = np.where(y==yi)[0]
            tmp_dat = resample(x[ind],replace=False,n_samples=ymin,random_state=42)
            if rex is None:
                rex = tmp_dat
            else:
                rex = np.concatenate((rex,tmp_dat),axis=0)
            rey.extend([yi]*ymin)
    return shuffle(rex,np.array(rey),random_state=42,replace=False)


# Let's up and downsample the training data and check how these affect the performance scores

# In[ ]:


dat_Xup, dat_Yup = upsample(dat_Xtrain,dat_Ytrain)
dat_Xdown, dat_Ydown = downsample(dat_Xtrain,dat_Ytrain)
print('\nUpsample stats')
print(dat_Xup.shape)
print(Counter(dat_Yup))
print('\nDownsample stats')
print(dat_Xdown.shape)
print(Counter(dat_Ydown))


# So, we have equalized the classes. But how does this affect the performance ? We can go through the classifiers again...
# 
# #### Randomforest classifier

# In[ ]:


rf_up = RandomForestClassifier(random_state=42,n_jobs=6,n_estimators=100)
print(cross_val_score(rf_up,dat_Xup,dat_Yup,cv=10))
rf_up.fit(dat_Xup,dat_Yup)
y_pred = rf_up.predict(dat_Xval)
print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))
print('Accuracy\n',accuracy_score(dat_Yval,y_pred))
print('Classificaton report\n',classification_report(dat_Yval,y_pred))


# In[ ]:


rf_down = RandomForestClassifier(random_state=42,n_jobs=6,n_estimators=100)
print(cross_val_score(rf_down,dat_Xdown,dat_Ydown,cv=10))
rf_down.fit(dat_Xdown,dat_Ydown)
y_pred = rf_down.predict(dat_Xval)
print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))
print('Accuracy\n',accuracy_score(dat_Yval,y_pred))
print('Classificaton report\n',classification_report(dat_Yval,y_pred))


# #### Bagging classifier

# In[ ]:


bc_up = BaggingClassifier(random_state=42,n_jobs=6,n_estimators=100)
print(cross_val_score(bc_up,dat_Xup,dat_Yup,cv=10))
bc_up.fit(dat_Xup,dat_Yup)
y_pred = bc_up.predict(dat_Xval)
print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))
print('Accuracy\n',accuracy_score(dat_Yval,y_pred))
print('Classificaton report\n',classification_report(dat_Yval,y_pred))


# In[ ]:


bc_down = BaggingClassifier(random_state=42,n_jobs=6,n_estimators=100)
print(cross_val_score(bc_down,dat_Xdown,dat_Ydown,cv=10))
bc_down.fit(dat_Xdown,dat_Ydown)
y_pred = bc_down.predict(dat_Xval)
print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))
print('Accuracy\n',accuracy_score(dat_Yval,y_pred))
print('Classificaton report\n',classification_report(dat_Yval,y_pred))


# #### ExtraTrees classifier

# In[ ]:


et_up = ExtraTreesClassifier(random_state=42,n_jobs=6,n_estimators=100)
print(cross_val_score(et_up,dat_Xup,dat_Yup,cv=10))
et_up.fit(dat_Xup,dat_Yup)
y_pred = et_up.predict(dat_Xval)
print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))
print('Accuracy\n',accuracy_score(dat_Yval,y_pred))
print('Classificaton report\n',classification_report(dat_Yval,y_pred))


# In[ ]:


et_down = ExtraTreesClassifier(random_state=42,n_jobs=6,n_estimators=100)
print(cross_val_score(et_down,dat_Xdown,dat_Ydown,cv=10))
et_down.fit(dat_Xdown,dat_Ydown)
y_pred = et_down.predict(dat_Xval)
print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))
print('Accuracy\n',accuracy_score(dat_Yval,y_pred))
print('Classificaton report\n',classification_report(dat_Yval,y_pred))


# #### Adaboost classifier

# In[ ]:


abc_up = AdaBoostClassifier(random_state=42,n_estimators=100)
print(cross_val_score(abc_up,dat_Xup,dat_Yup,cv=10))
abc_up.fit(dat_Xup,dat_Yup)
y_pred = abc_up.predict(dat_Xval)
print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))
print('Accuracy\n',accuracy_score(dat_Yval,y_pred))
print('Classificaton report\n',classification_report(dat_Yval,y_pred))


# In[ ]:


abc_down = AdaBoostClassifier(random_state=42,n_estimators=100)
print(cross_val_score(abc_down,dat_Xdown,dat_Ydown,cv=10))
abc_down.fit(dat_Xdown,dat_Ydown)
y_pred = abc_down.predict(dat_Xval)
print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))
print('Accuracy\n',accuracy_score(dat_Yval,y_pred))
print('Classificaton report\n',classification_report(dat_Yval,y_pred))


# #### Gradient boosting classifier

# In[ ]:


xg_up = XGBClassifier(random_state=42,n_jobs=6,n_estimators=100)
print(cross_val_score(xg_up,dat_Xup,dat_Yup,cv=10))
xg_up.fit(dat_Xup,dat_Yup)
y_pred = xg_up.predict(dat_Xval)
print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))
print('Accuracy\n',accuracy_score(dat_Yval,y_pred))
print('Classificaton report\n',classification_report(dat_Yval,y_pred))


# In[ ]:


xg_down = XGBClassifier(random_state=42,n_jobs=6,n_estimators=100)
print(cross_val_score(xg_down,dat_Xdown,dat_Ydown,cv=10))
xg_down.fit(dat_Xdown,dat_Ydown)
y_pred = xg_down.predict(dat_Xval)
print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))
print('Accuracy\n',accuracy_score(dat_Yval,y_pred))
print('Classificaton report\n',classification_report(dat_Yval,y_pred))


# **Classification report for class `1`**
# 
# | Classifier | Precision | Recall | F1-score|
# |------------|:---------:|:------:|:-------:|
# | Random forest (up)| 0.49 | 0.51 | 0.50 |
# | Random forest (down)| 0.43 | 0.73 | 0.54 |
# | Bagging (up)| 0.49 | 0.54 | 0.51 |
# | Bagging (down)| 0.44 | 0.75 | 0.56 |
# | Extra trees (up)| 0.47 | 0.43 | 0.45 |
# | Extra trees (down)| 0.42 | 0.69 | 0.52 |
# | Adaboost (up)| 0.42 | 0.70 | 0.53 |
# | Adaboost (down)| 0.42 | 0.71 | 0.53 |
# | Gradient boost (up)| 0.42 | 0.71 | 0.53 |
# | Gradient boost (down)| 0.43 | 0.72 | 0.54 |
# 
# So it looks like down sampling the over represented samples performs (slightly) better than up sampling the under represented samples. Now to the next methods...
# 
# #### SMOTE: Synthetic Minority Over-sampling Techniques
# 
# SMOTE tries to deal with the class imbalance problem by over-sampling of the minority class. To oversample using SMOTE, consider k nearest neighbors of a sample (data point) taken from the dataset. To create a synthetic datapoint, take the vector between the current data point and one of the k neighbors sampled earlier, and multiply this vector by a random number between 0-1 to create the synthetic data point. Here is the [wikipedia entry](https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis) for SMOTE and here is a [blog post](http://rikunert.com/SMOTE_explained) explaining the algorithm. 

# In[ ]:


from imblearn.over_sampling import SMOTENC
# Synthetic Minority Over-sampling Technique for Nominal and Continuous (SMOTE-NC).


# In[ ]:


sm = SMOTENC(random_state=42,n_jobs=6,categorical_features=np.arange(8,dat_Xtrain.shape[1],1))
dat_Xsmote,dat_Ysmote = sm.fit_resample(dat_Xtrain,dat_Ytrain)
print('\nSMOTE stats')
print(dat_Xsmote.shape)
print(Counter(dat_Ysmote))


# #### Randomforest classifier

# In[ ]:


rf_smote = RandomForestClassifier(random_state=42,n_jobs=6,n_estimators=100)
print(cross_val_score(rf_smote,dat_Xsmote,dat_Ysmote,cv=10))
rf_smote.fit(dat_Xsmote,dat_Ysmote)
y_pred = rf_smote.predict(dat_Xval)
print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))
print('Accuracy\n',accuracy_score(dat_Yval,y_pred))
print('Classificaton report\n',classification_report(dat_Yval,y_pred))


# #### Bagging classifier

# In[ ]:


bc_smote = BaggingClassifier(random_state=42,n_jobs=6,n_estimators=100)
print(cross_val_score(bc_smote,dat_Xsmote,dat_Ysmote,cv=10))
bc_smote.fit(dat_Xsmote,dat_Ysmote)
y_pred = bc_smote.predict(dat_Xval)
print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))
print('Accuracy\n',accuracy_score(dat_Yval,y_pred))
print('Classificaton report\n',classification_report(dat_Yval,y_pred))


# #### ExtraTrees classifier

# In[ ]:


et_smote = ExtraTreesClassifier(random_state=42,n_jobs=6,n_estimators=100)
print(cross_val_score(et_smote,dat_Xsmote,dat_Ysmote,cv=10))
et_smote.fit(dat_Xsmote,dat_Ysmote)
y_pred = et_smote.predict(dat_Xval)
print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))
print('Accuracy\n',accuracy_score(dat_Yval,y_pred))
print('Classificaton report\n',classification_report(dat_Yval,y_pred))


# #### Adaboost classifier

# In[ ]:


abc_smote = AdaBoostClassifier(random_state=42,n_estimators=100)
print(cross_val_score(abc_smote,dat_Xsmote,dat_Ysmote,cv=10))
abc_smote.fit(dat_Xsmote,dat_Ysmote)
y_pred = abc_smote.predict(dat_Xval)
print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))
print('Accuracy\n',accuracy_score(dat_Yval,y_pred))
print('Classificaton report\n',classification_report(dat_Yval,y_pred))


# #### Gradient boosting classifier

# In[ ]:


xg_smote = XGBClassifier(random_state=42,n_jobs=6,n_estimators=100)
print(cross_val_score(xg_smote,dat_Xsmote,dat_Ysmote,cv=10))
xg_smote.fit(dat_Xsmote,dat_Ysmote)
y_pred = xg_smote.predict(dat_Xval)
print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))
print('Accuracy\n',accuracy_score(dat_Yval,y_pred))
print('Classificaton report\n',classification_report(dat_Yval,y_pred))


# | Classifier | Precision | Recall | F1-score|
# |------------|:---------:|:------:|:-------:|
# | Random forest| 0.49 | 0.57 | 0.53 |
# | Bagging | 0.49 | 0.50 | 0.49 |
# | Extra trees| 0.48 | 0.52 | 0.50 |
# | Adaboost| 0.43 | 0.63 | 0.51 |
# | Gradient boost| 0.45 | 0.62 | 0.52 |
# 
# 
# ### Summary part 1
# 
# If we compare all the results:
# 
# | Classifier | Precision | Recall | F1-score|
# |------------|:---------:|:------:|:-------:|
# | Random forest| 0.57 | 0.38 | 0.46 |
# | Random forest (up)| 0.49 | 0.51 | 0.50 |
# | Random forest (down)| 0.43 | 0.73 | 0.54 |
# | Random forest (SMOTE)| 0.49 | 0.57 | 0.53 |
# | Bagging | 0.57 | 0.42 | 0.48 |
# | Bagging (up)| 0.49 | 0.54 | 0.51 |
# | Bagging (down)| 0.44 | 0.75 | 0.56 |
# | Bagging (SMOTE)| 0.49 | 0.50 | 0.49 |
# | Extra trees| 0.52 | 0.38 | 0.44|
# | Extra trees (up)| 0.47 | 0.43 | 0.45 |
# | Extra trees (down)| 0.42 | 0.69 | 0.52 |
# | Extra trees (SMOTE)| 0.48 | 0.52 | 0.50 |
# | Adaboost| 0.59 | 0.20 | 0.29 |
# | Adaboost (up)| 0.42 | 0.70 | 0.53 |
# | Adaboost (down)| 0.42 | 0.71 | 0.53 |
# | Adaboost (SMOTE)| 0.43 | 0.63 | 0.51 |
# | Gradient boost| 0.62 | 0.17 | 0.26 |
# | Gradient boost (up)| 0.42 | 0.71 | 0.53 |
# | Gradient boost (down)| 0.43 | 0.72 | 0.54 |
# | Gradient boost (SMOTE)| 0.45 | 0.62 | 0.52 |
# 
# So, it looks like ```Bagging classifier``` performs the best for predicting the results in this case. But if we look at all the results in detail, we can see that this classifier underperforms in predicting `0` class label.
# 
# ### Using balanced classifiers
# 
# In this section we will use ensemble classifiers from [imblearn](https://pypi.org/project/imblearn/) python package. Most of these classifiers are similar to the implementations in `sci-kit` package with an additional step at training to balance  samples using `RandomUnderSampler`.
# 
# #### BalancedRandomForestClassifier
# 

# In[ ]:


from imblearn.ensemble import BalancedRandomForestClassifier 


# In[ ]:


brf = BalancedRandomForestClassifier(random_state = 42,n_estimators=100)
print(cross_val_score(brf,dat_Xtrain,dat_Ytrain,cv=10))
brf.fit(dat_Xtrain,dat_Ytrain)
y_pred =brf.predict(dat_Xval)
print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))
print('Accuracy\n',accuracy_score(dat_Yval,y_pred))
print('Classificaton report\n',classification_report(dat_Yval,y_pred))


# #### BalancedBaggingClassifier

# In[ ]:


from imblearn.ensemble import BalancedBaggingClassifier 


# In[ ]:


bbc = BalancedBaggingClassifier(random_state = 42,n_estimators=100)
print(cross_val_score(bbc,dat_Xtrain,dat_Ytrain,cv=10))
bbc.fit(dat_Xtrain,dat_Ytrain)
y_pred =bbc.predict(dat_Xval)
print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))
print('Accuracy\n',accuracy_score(dat_Yval,y_pred))
print('Classificaton report\n',classification_report(dat_Yval,y_pred))


# #### EasyEnsembleClassifier

# In[ ]:


from imblearn.ensemble import EasyEnsembleClassifier


# In[ ]:


eec = EasyEnsembleClassifier(random_state = 42,n_estimators=100)
#print(cross_val_score(eec,dat_Xtrain,dat_Ytrain,cv=10))
eec.fit(dat_Xtrain,dat_Ytrain)
y_pred =eec.predict(dat_Xval)
print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))
print('Accuracy\n',accuracy_score(dat_Yval,y_pred))
print('Classificaton report\n',classification_report(dat_Yval,y_pred))


# #### RUSBoostClassifier

# In[ ]:


from imblearn.ensemble import RUSBoostClassifier


# In[ ]:


rbc = RUSBoostClassifier(random_state = 42,n_estimators=100)
#print(cross_val_score(eec,dat_Xtrain,dat_Ytrain,cv=10))
rbc.fit(dat_Xtrain,dat_Ytrain)
y_pred =rbc.predict(dat_Xval)
print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))
print('Accuracy\n',accuracy_score(dat_Yval,y_pred))
print('Classificaton report\n',classification_report(dat_Yval,y_pred))


# ### Summary part 2
# 
# | Classifier | Precision | Recall | F1-score|
# |------------|:---------:|:------:|:-------:|
# | BalancedRandomForestClassifier| 0.44 | 0.77 | 0.56 |
# | BalancedBaggingClassifier | 0.48 | 0.70 | 0.57 |
# | EasyEnsembleClassifier| 0.42 | 0.70 | 0.53 |
# | RUSBoostClassifier| 0.47 | 0.32 | 0.38 |
# 
# From these results, it looks like `BalancedBaggingClassifier` performs the best among balanced classifiers. 
# 
# ### Conclusion
# 
# Using random over sampling/under sampling or SMOTE techniques can help improve the results in predicting genetic variants with conflicting classifications.
