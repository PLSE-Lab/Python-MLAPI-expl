#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython.core.display import display, HTML
display(HTML('<style>.container {width:98% !important;}</style>'))


# In[ ]:


import numpy as np
import pandas as pd
import random
random.seed(28)
np.random.seed(28)
import itertools
import scipy
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)
import os
import copy
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler
pd.options.display.precision = 15
from collections import defaultdict
import time
from collections import Counter
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit, RepeatedStratifiedKFold
from sklearn import metrics
import gc
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from IPython.display import HTML
import json

from category_encoders.ordinal import OrdinalEncoder
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from typing import List
import datetime
from functools import partial
import scipy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from sklearn import metrics
pd.set_option('max_rows', 500)
import re
from collections import Counter
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:20,.2f}'.format)
pd.set_option('display.max_colwidth', -1)


# In[ ]:


from sklearn import metrics, preprocessing
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from tensorflow.keras import utils


# In[ ]:


import numpy as np
from collections import Counter, defaultdict
from sklearn.utils import check_random_state

class RepeatedStratifiedGroupKFold():

    def __init__(self, n_splits=5, n_repeats=1, random_state=None):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        
    # Implementation based on this kaggle kernel:
    #    https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
    def split(self, X, y=None, groups=None):
        k = self.n_splits
        def eval_y_counts_per_fold(y_counts, fold):
            y_counts_per_fold[fold] += y_counts
            std_per_label = []
            for label in range(labels_num):
                label_std = np.std(
                    [y_counts_per_fold[i][label] / y_distr[label] for i in range(k)]
                )
                std_per_label.append(label_std)
            y_counts_per_fold[fold] -= y_counts
            return np.mean(std_per_label)
            
        rnd = check_random_state(self.random_state)
        for repeat in range(self.n_repeats):
            labels_num = np.max(y) + 1
            y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
            y_distr = Counter()
            for label, g in zip(y, groups):
                y_counts_per_group[g][label] += 1
                y_distr[label] += 1

            y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
            groups_per_fold = defaultdict(set)
        
            groups_and_y_counts = list(y_counts_per_group.items())
            rnd.shuffle(groups_and_y_counts)

            for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
                best_fold = None
                min_eval = None
                for i in range(k):
                    fold_eval = eval_y_counts_per_fold(y_counts, i)
                    if min_eval is None or fold_eval < min_eval:
                        min_eval = fold_eval
                        best_fold = i
                y_counts_per_fold[best_fold] += y_counts
                groups_per_fold[best_fold].add(g)

            all_groups = set(groups)
            for i in range(k):
                train_groups = all_groups - groups_per_fold[i]
                test_groups = groups_per_fold[i]

                train_indices = [i for i, g in enumerate(groups) if g in train_groups]
                test_indices = [i for i, g in enumerate(groups) if g in test_groups]

                yield train_indices, test_indices


# read the data

# In[ ]:


train = pd.read_csv("../input/widsdatathon2020/training_v2.csv")
samplesubmission = pd.read_csv("../input/widsdatathon2020/samplesubmission.csv")
test = pd.read_csv("../input/widsdatathon2020/unlabeled.csv")
dictionary = pd.read_csv("../input/widsdatathon2020/WiDS Datathon 2020 Dictionary.csv")
solution_template = pd.read_csv("../input/widsdatathon2020/solution_template.csv")

print('train ' , train.shape)
print('test ' , test.shape)
print('samplesubmission ' , samplesubmission.shape)
print('solution_template ' , solution_template.shape)
print('dictionary ' , dictionary.shape)


# # feature eng .

# In[ ]:


# some data cleaning
if 1 :
    #d1_mbp_min
    train[ 'd1_mbp_min'] = np.where(train[ 'd1_mbp_min'].isna(), train[ 'd1_mbp_noninvasive_min']  ,train[ 'd1_mbp_min'] )
    test[  'd1_mbp_min'] = np.where(test[ 'd1_mbp_min'].isna(), test[ 'd1_mbp_noninvasive_min']  ,test[ 'd1_mbp_min'] )
    
    train[ 'd1_mbp_noninvasive_min']  = np.where(train[ 'd1_mbp_noninvasive_min'].isna(), train[ 'd1_mbp_min']  ,train[ 'd1_mbp_noninvasive_min'] )
    test[  'd1_mbp_noninvasive_min']  = np.where(test[ 'd1_mbp_noninvasive_min'].isna() , test[ 'd1_mbp_min']   ,test[ 'd1_mbp_noninvasive_min'] )
    #d1_mbp_max
    train[ 'd1_mbp_max'] = np.where(train[ 'd1_mbp_max'].isna(), train[ 'd1_mbp_noninvasive_max']  ,train[ 'd1_mbp_max'] )
    test[ 'd1_mbp_max'] = np.where(test[ 'd1_mbp_max'].isna(), test[ 'd1_mbp_noninvasive_max']  ,test[ 'd1_mbp_max'] )
    
    train[ 'd1_mbp_noninvasive_max']  = np.where(train[ 'd1_mbp_noninvasive_max'].isna(), train[ 'd1_mbp_max']  ,train[ 'd1_mbp_noninvasive_max'] )
    test[  'd1_mbp_noninvasive_max']  = np.where(test[ 'd1_mbp_noninvasive_max'].isna() , test[ 'd1_mbp_max']   ,test[ 'd1_mbp_noninvasive_max'] )
    

    #d1_diasbp_min
    train[ 'd1_diasbp_min'] = np.where(train[ 'd1_diasbp_min'].isna(), train[ 'd1_diasbp_noninvasive_min']  ,train[ 'd1_diasbp_min'] )
    test[  'd1_diasbp_min'] = np.where(test[  'd1_diasbp_min'].isna(), test [ 'd1_diasbp_noninvasive_min']  ,test[  'd1_diasbp_min'] )

    train[ 'd1_diasbp_noninvasive_min']  = np.where(train[ 'd1_diasbp_noninvasive_min'].isna(), train[ 'd1_diasbp_min']  ,train[ 'd1_diasbp_noninvasive_min'] )
    test[  'd1_diasbp_noninvasive_min']  = np.where(test[  'd1_diasbp_noninvasive_min'].isna() , test[ 'd1_diasbp_min']   ,test[ 'd1_diasbp_noninvasive_min'] )

    train[ 'd1_diasbp_max'] = np.where(train[ 'd1_diasbp_max'].isna(), train[ 'd1_diasbp_noninvasive_max']  ,train[ 'd1_diasbp_max'] )
    test[  'd1_diasbp_max']  = np.where(test[ 'd1_diasbp_max'].isna(), test[  'd1_diasbp_noninvasive_max']  ,test[  'd1_diasbp_max'] )

    train[ 'd1_diasbp_noninvasive_max']  = np.where(train[ 'd1_diasbp_noninvasive_max'].isna(), train[ 'd1_diasbp_max']  ,train[ 'd1_diasbp_noninvasive_max'] )
    test[  'd1_diasbp_noninvasive_max']  = np.where(test[  'd1_diasbp_noninvasive_max'].isna() , test[ 'd1_diasbp_max']   ,test[ 'd1_diasbp_noninvasive_max'] )


# In[ ]:


# verif et correction des quelques valeures incoherentes genre max < min 
if 1 :
    min_col= [col for col in train.columns if 'd1' in col and col.endswith('_min')]
    for col in min_col :
        d1_col = col.replace('_min','_max')
        train[d1_col] = np.where(train[col] > train[d1_col], np.nan ,  train[d1_col])
        test[d1_col]  = np.where(test[col]   > test[d1_col], np.nan ,test[d1_col] )
        
    min_col= [col for col in train.columns if 'h1' in col and col.endswith('_min')]
    for col in min_col :
        d1_col = col.replace('_min','_max')
        train[d1_col ] = np.where(train[col] > train[d1_col], np.nan ,  train[d1_col])
        test[d1_col]  = np.where(test[col] > test[d1_col]   , np.nan ,test[d1_col] )


       


# In[ ]:


# split the apache 3 diag in parts
train['apache_3j_diagnosis_split0'] = np.where(train['apache_3j_diagnosis'].isna() , np.nan , train['apache_3j_diagnosis'].astype('str').str.split('.',n=1,expand=True)[0]  )
test['apache_3j_diagnosis_split0']   = np.where(test['apache_3j_diagnosis'].isna() , np.nan , test['apache_3j_diagnosis'].astype('str').str.split('.',n=1,expand=True)[0]  )


train['apache_3j'] = np.where(train['apache_3j_diagnosis_split0'].isna() , np.nan ,
                            np.where(train['apache_3j_diagnosis_split0'].fillna(9999).astype('int') < 200, 'Cardiovascular' , 
                            np.where(train['apache_3j_diagnosis_split0'].fillna(9999).astype('int') < 400, 'Respiratory' , 
                            np.where(train['apache_3j_diagnosis_split0'].fillna(9999).astype('int') < 500, 'Neurological' , 
                            np.where(train['apache_3j_diagnosis_split0'].fillna(9999).astype('int') < 600, 'Sepsis' , 
                            np.where(train['apache_3j_diagnosis_split0'].fillna(9999).astype('int') < 800, 'Trauma' ,  
                            np.where(train['apache_3j_diagnosis_split0'].fillna(9999).astype('int') < 900, 'Haematological' ,         
                            np.where(train['apache_3j_diagnosis_split0'].fillna(9999).astype('int') < 1000, 'Renal/Genitourinary' ,         
                            np.where(train['apache_3j_diagnosis_split0'].fillna(9999).astype('int') < 1200, 'Musculoskeletal/Skin disease' , 'Operative Sub-Diagnosis Codes' ))))))))
                                    )
test['apache_3j'] = np.where(test['apache_3j_diagnosis_split0'].isna() , np.nan ,
                            np.where(test['apache_3j_diagnosis_split0'].fillna(9999).astype('int') < 200, 'Cardiovascular' , 
                            np.where(test['apache_3j_diagnosis_split0'].fillna(9999).astype('int') < 400, 'Respiratory' , 
                            np.where(test['apache_3j_diagnosis_split0'].fillna(9999).astype('int') < 500, 'Neurological' , 
                            np.where(test['apache_3j_diagnosis_split0'].fillna(9999).astype('int') < 600, 'Sepsis' , 
                            np.where(test['apache_3j_diagnosis_split0'].fillna(9999).astype('int') < 800, 'Trauma' ,  
                            np.where(test['apache_3j_diagnosis_split0'].fillna(9999).astype('int') < 900, 'Haematological' ,         
                            np.where(test['apache_3j_diagnosis_split0'].fillna(9999).astype('int') < 1000, 'Renal/Genitourinary' ,         
                            np.where(test['apache_3j_diagnosis_split0'].fillna(9999).astype('int') < 1200, 'Musculoskeletal/Skin disease' , 'Operative Sub-Diagnosis Codes' ))))))))
                                    )

train['apache_3j_diagnosis_split1'] = np.where(train['apache_3j_diagnosis'].isna() , np.nan , train['apache_3j_diagnosis'].astype('str').str.split('.',n=1,expand=True)[1]  )
test['apache_3j_diagnosis_split1']  = np.where(test['apache_3j_diagnosis'].isna() , np.nan , test['apache_3j_diagnosis'].astype('str').str.split('.',n=1,expand=True)[1]  )


# In[ ]:


train['pre_icu_los_days'] = train['pre_icu_los_days'].apply(lambda x:scipy.special.expit(x) )
test['pre_icu_los_days']  = test['pre_icu_los_days'].apply(lambda x:scipy.special.expit(x) )


# In[ ]:


column='pre_icu_los_days'
fig = plt.figure(figsize=(20,4))
df1      = train.loc[train['hospital_death']==0]
df2      = train.loc[train['hospital_death']==1]
sns.distplot(df1[column].fillna(-5),  color='green', label='hospital_death 0', kde=True); 
sns.distplot(df2[column].fillna(-5),  color='red'  , label='hospital_death 1', kde=True); 
fig=plt.legend(loc='best')
plt.xlabel(column, fontsize=12);
plt.show()


# In[ ]:


if 1 :
    train['comorbidity_score'] = train['aids'].values * 23 + train['cirrhosis'] *4    + train['diabetes_mellitus'] *1  + train['hepatic_failure'] *16 + train['immunosuppression'] *10     + train['leukemia'] * 10     + train['lymphoma'] * 13     + train['solid_tumor_with_metastasis'] * 11
    test['comorbidity_score'] = test['aids'].values * 23    + test['cirrhosis'] *4    + test['diabetes_mellitus'] *1  + test['hepatic_failure'] *16 + test['immunosuppression'] *10     + test['leukemia'] * 10     + test['lymphoma'] * 13     + test['solid_tumor_with_metastasis'] * 11
    train['comorbidity_score'] = train['comorbidity_score'].fillna(0)
    test['comorbidity_score'] = test['comorbidity_score'].fillna(0)
    


# In[ ]:


# Drop columns based on threshold limit
if 1:
    print(train.shape)
    threshold = len(train) * 0.20
    train=train.dropna(axis=1, thresh=threshold)
    print(train.shape)


# In[ ]:


train.shape , test.shape


# # categorical

# In[ ]:


## frequency encoding for icu_id / hospital 
all_df = pd.concat([train,test],axis=0)
frequence_encode = ['icu_id','hospital_id']
for col in frequence_encode :
    hosp_asource = pd.DataFrame(all_df[col])
    hosp_asource = hosp_asource.dropna()
    hosp_asource.columns =[col]
    fe = 100*(hosp_asource.groupby(col).size()/len(hosp_asource))
    train.loc[:,col+'_fe'] = train[col].map(fe)
    test.loc[:,col+'_fe'] = test[col].map(fe)


# In[ ]:


# fill with missing
train[["apache_2_bodysystem","apache_3j_bodysystem"]]=train[["apache_2_bodysystem","apache_3j_bodysystem"]].fillna('missing')


# In[ ]:


for bin_col in ['elective_surgery','apache_post_operative','arf_apache','intubated_apache','ventilated_apache','aids','cirrhosis','gcs_eyes_apache','gcs_verbal_apache',   'gcs_motor_apache' , 'gcs_unable_apache','diabetes_mellitus','hepatic_failure','immunosuppression','leukemia','lymphoma','solid_tumor_with_metastasis'] :
    print(bin_col)
    train[bin_col].fillna(10)
    test[bin_col].fillna(10)


# In[ ]:


categoricals_features = set(['apache_3j_diagnosis_split0','apache_3j','apache_3j_diagnosis_split1','hospital_admit_source','icu_admit_source','icu_stay_type','icu_type','elective_surgery',
                        'apache_post_operative','arf_apache','intubated_apache','ventilated_apache','aids','cirrhosis',
                        'diabetes_mellitus','hepatic_failure','immunosuppression','leukemia','lymphoma','solid_tumor_with_metastasis','apache_3j_bodysystem','apache_2_bodysystem',
                         'apache_2_diagnosis'])       
   


# In[ ]:


# this is the list of all input feature we would like our model to use (we remove the target and the ids.)
to_remove=['hospital_id','icu_id','ethnicity','gender','patient_id','encounter_id','hospital_death','apache_4a_hospital_death_prob','apache_4a_icu_death_prob' ]

features = [col for col in train.columns if col not in to_remove]

# this is a list of features that look like to be categorical
categoricals_features = [col for col in categoricals_features if col not in to_remove]


# In[ ]:


# categorical feature need to be transform to numeric for mathematical purpose.
# different technics of categorical encoding exists here we will rely on our model API to deal with categorical
# still we need to encode each categorical value to an id , for this purpose we use LabelEncoder

print('Transform all String features to category.\n')
for usecol in categoricals_features:
    colcount = train[usecol].value_counts().index[0]
    train[usecol] = train[usecol].fillna(colcount)
    test[usecol]  = test[usecol].fillna(colcount)
    
    train[usecol] = train[usecol].astype('str')
    test[usecol] = test[usecol].astype('str')
    
    #Fit LabelEncoder
    le = LabelEncoder().fit(
            np.unique(train[usecol].unique().tolist()+
                      test[usecol].unique().tolist()))

    #At the end 0 will be used for dropped values
    train[usecol] = le.transform(train[usecol])+1
    test[usecol]  = le.transform(test[usecol])+1
    
    train[usecol] = train[usecol].replace(np.nan, -1).astype('int')
    test[usecol]  = test[usecol].replace(np.nan , -1).astype('int')


# remove col

# In[ ]:


num_feature = [col for col in features if col not in categoricals_features]
drop_columns=[]
lowvar_columns=[]
if 1 :
    threshold = 0.01
    lowvar_columns = train[num_feature].std()[train[num_feature].std() < threshold].index.values
    print(len(lowvar_columns),lowvar_columns)


# In[ ]:


num_feature = [col for col in features if col not in categoricals_features and train[col].dtype != 'object']
drop_columns=[]
corr = train[num_feature].corr()
# Drop highly correlated features 
columns = np.full((corr.shape[0],), True, dtype=bool)

for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >=0.99 :
            if columns[j]:
                columns[j] = False
                print('FEAT_A: {} FEAT_B: {} - Correlation: {}'.format(train[num_feature].columns[i] , train[num_feature].columns[j], corr.iloc[i,j]))
        elif corr.iloc[i,j] <= -0.995:
            if columns[j]:
                columns[j] = False

drop_columns = train[num_feature].columns[columns == False].values
print('drop_columns',len(drop_columns),drop_columns)


# In[ ]:


features = [col for col in features if col not in drop_columns]
features = [col for col in features if col not in lowvar_columns]


# In[ ]:


print('number of features ' , len(features))
print('shape of train / test ', train.shape , test.shape)


# > Encode Null value

# In[ ]:


non_categorical = [f for f in features if f not in categoricals_features]


# In[ ]:


# create a flag for null field
for i in non_categorical:
    train[str(i)+"_Na"]=pd.get_dummies(train[i].isnull(),prefix=i).iloc[:,0]
    test[str(i)+"_Na"]=pd.get_dummies(test[i].isnull(),prefix=i).iloc[:,0]
    
train['age']=train['age'].fillna(99)
test['age']=test['age'].fillna(99)
train=train.fillna(train.median())
test=test.fillna(test.median())


# re remove new useless col

# In[ ]:


features = [col for col in train.columns if col not in to_remove]
features = [col for col in features if col not in drop_columns]
features = [col for col in features if col not in lowvar_columns]


# In[ ]:


num_feature = [col for col in features if col not in categoricals_features]
drop_columns=[]
lowvar_columns=[]
if 1 :
    threshold = 0.01
    lowvar_columns = train[num_feature].std()[train[num_feature].std() < threshold].index.values
    print(len(lowvar_columns),lowvar_columns)


# In[ ]:


num_feature = [col for col in features if col not in categoricals_features and train[col].dtype != 'object']
drop_columns=[]
corr = train[num_feature].corr()
# Drop highly correlated features 
columns = np.full((corr.shape[0],), True, dtype=bool)

for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >=0.999 :
            if columns[j]:
                columns[j] = False
                print('FEAT_A: {} FEAT_B: {} - Correlation: {}'.format(train[num_feature].columns[i] , train[num_feature].columns[j], corr.iloc[i,j]))
        elif corr.iloc[i,j] <= -0.995:
            if columns[j]:
                columns[j] = False

drop_columns = train[num_feature].columns[columns == False].values
print('drop_columns',len(drop_columns),drop_columns)


# In[ ]:


features = [col for col in features if col not in drop_columns]
features = [col for col in features if col not in lowvar_columns]


# In[ ]:


train.head()


# In[ ]:


def auc(y_true, y_pred):
    def fallback_auc(y_true, y_pred):
        try:
            return metrics.roc_auc_score(y_true, y_pred)
        except:
            return 0.5
    return tf.py_function(fallback_auc, (y_true, y_pred), tf.double)


# In[ ]:


num_features = [f for f in features if f not in categoricals_features]


# In[ ]:


data = pd.concat([train[features], test[features]]).reset_index(drop=True)


# In[ ]:


data.shape


# In[ ]:


print(len(categoricals_features))
print(len(num_features))


# In[ ]:


list(categoricals_features)


# In[ ]:


list(num_features)


# In[ ]:


test_data = [np.absolute(test[i]) for i in categoricals_features] + [test[num_features]]


# In[ ]:


print(train.shape,test.shape)#, test_data.shape)


# In[ ]:


numeric_min = [f for f in num_features if '_min' in f]
numeric_max = [f for f in num_features if '_max' in f]
numeric_apache = [f for f in num_features if 'apache' in f]
numeric_other = [f for f in num_features if f not in numeric_min and f not in numeric_max and f not in numeric_apache]

print('_____________ numeric_min ',len(numeric_min) , numeric_min)
print('_____________ numeric_max ',len(numeric_max) , numeric_max )                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
print('_____________ numeric_apache ',len(numeric_apache ) , numeric_apache )                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
print('_____________ numeric_other ',len(numeric_other) , numeric_other)


# In[ ]:


import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from tensorflow.keras.optimizers import SGD, Adam
from sklearn import metrics
from keras import backend as K

METRICS = [
      tf.keras.metrics.AUC(name='auc'),
]

def model_wids2020():    
    inputs = []
    embeddings  = []
    for c in categoricals_features:
        num_unique_values = int(data[c].nunique())
        embed_dim = int(min(np.ceil((num_unique_values)/2), 50))               
        inp = layers.Input(shape=(1,))
        out = layers.Embedding(num_unique_values + 1, embed_dim, name=c)(inp)
        out = layers.SpatialDropout1D(0.5)(out)
        out = layers.Reshape(target_shape=(embed_dim, ))(out)
        inputs.append(inp)
        embeddings.append(out)

    input_numeric1     = layers.Input(shape=(len(numeric_min),))
    embedding_numeric1 = layers.Dense(250, activation='relu')(input_numeric1)
    inputs.append(input_numeric1)
    embeddings.append(embedding_numeric1)
    input_numeric2     = layers.Input(shape=(len(numeric_max),))
    embedding_numeric2 = layers.Dense(250, activation='relu')(input_numeric2)
    inputs.append(input_numeric2)
    embeddings.append(embedding_numeric2)
    input_numeric3     = layers.Input(shape=(len(numeric_apache),))
    embedding_numeric3 = layers.Dense(250, activation='relu')(input_numeric3)
    inputs.append(input_numeric3)
    embeddings.append(embedding_numeric3)
    input_numeric4     = layers.Input(shape=(len(numeric_other),))
    embedding_numeric4 = layers.Dense(50, activation='relu')(input_numeric4)
    inputs.append(input_numeric4)
    embeddings.append(embedding_numeric4)
    
    x = layers.Concatenate()(embeddings)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128)(x)

    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    y = layers.Dense(2, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=y)
    return model

model = model_wids2020()
tf.keras.utils.plot_model(model, show_shapes=True, to_file='embeddings.png')


# In[ ]:


oof_preds  = np.zeros((len(train)))
test_preds = np.zeros((len(test)))
y = train['hospital_death']
FOLD= 7
kf = RepeatedStratifiedGroupKFold(n_splits=FOLD, n_repeats=5, random_state=42)
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2020)

fold = 0
for tdx, vdx in kf.split(train, train['hospital_death'].values):
#for tdx, vdx in   kf.split(train, train['hospital_death'], train['hospital_id']):
    X_train, X_val, y_train, y_val = train.iloc[tdx], train.iloc[vdx], y[tdx], y[vdx]
    
    scalar = StandardScaler()

    X_train[num_features] = scalar.fit_transform(X_train[num_features])
    X_val[num_features]   = scalar.transform(X_val[num_features])
    test_data = test.copy()
    test_data[num_features]  = scalar.transform(test_data[num_features])
    
    test_data = [np.absolute(test_data[i]) for i in categoricals_features] + [test_data[numeric_min]]+ [test_data[numeric_max]]+ [test_data[numeric_apache]]+ [test_data[numeric_other]]
   
    X_train = [np.absolute(X_train[i]) for i in categoricals_features] + [X_train[numeric_min]]+ [X_train[numeric_max]]+ [X_train[numeric_apache]]+ [X_train[numeric_other]]
    X_val   = [np.absolute(X_val[i])   for i in categoricals_features] + [X_val[numeric_min]]  + [X_val[numeric_max]]  + [X_val[numeric_apache]]  + [X_val[numeric_other]]
    
    
    model = model_wids2020()
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[auc])
    es = callbacks.EarlyStopping(monitor='val_auc', min_delta=0.001, patience=30,
                                 verbose=1, mode='max', baseline=None, restore_best_weights=True)
    rlr        = callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5,
                                     patience=5, min_lr=1e-6, mode='max', verbose=1)
   
    model.fit(X_train,
              utils.to_categorical(y_train),
              validation_data=(X_val, utils.to_categorical(y_val)),
              verbose=1,
              batch_size=5024,
              callbacks=[es, rlr],
              epochs=100
             )
    valid_fold_preds = model.predict(X_val)[:, 1]
    test_fold_preds  = model.predict(test_data)[:, 1]
    oof_preds[vdx] = valid_fold_preds.ravel()
    test_preds += test_fold_preds.ravel()
    print('FOLD ',str(fold) , ' AUC ...............' ,metrics.roc_auc_score(y_val, valid_fold_preds))
    fold=fold+1
    K.clear_session()


# In[ ]:


AUC_FINAL=metrics.roc_auc_score(train['hospital_death'].values, oof_preds)
import joblib
joblib.dump(oof_preds , 'NN1_oof_preds_'+str(AUC_FINAL)+'.pkl')
joblib.dump(test_preds, 'NN1_test_preds_'+str(AUC_FINAL)+'.pkl')
print("Overall AUC={}".format(AUC_FINAL))


# In[ ]:


test_preds /= FOLD


# In[ ]:


test["hospital_death"] = test_preds
test[["encounter_id","hospital_death"]].to_csv("submission_nn_"+ str(AUC_FINAL) +".csv",index=False)

