#!/usr/bin/env python
# coding: utf-8

# ![](https://cdn.cnn.com/cnnnext/dam/assets/140820084625-peanut-butter-stock-super-tease.jpg)

# This is a helper kernel for anyone struggling to deal with all the missing values and unsmooth features. Im sure it does add quite a lot of bias, but for linear models the NaN values are a pain so I hope this helps you.
# # 
# # We also encode labels for ease of use
# # 
# # If you want to see my further analysis where I build on this data using catboost and Logistic Regression  on this Please see https://www.kaggle.com/pipboyguy/simple-models-smoothing-of-features

# In[ ]:


import gc
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse
import seaborn as sns
from scipy import stats
from sklearn.impute import SimpleImputer  # for categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("darkgrid", {"axes.facecolor": ".9"})


# In[ ]:


#Helper Functions

# I have given thanks to reduce_mem in one of my previous kernels
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
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
                c_prec = df[col].apply(lambda x: np.finfo(x).precision).max()
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max and c_prec == np.finfo(np.float32).precision:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def boxcox_Numfeats(frame: pd.DataFrame, columns_to_boxcox: list):
    """
    Smoothing the numerical-dataframe with box-cox
    """
    for feature in columns_to_boxcox:
        print(f"Processing Feature: {feature}")
        
        # boxcox only takes positive values
        
        if frame[feature].min() < 0:
            
            rc_bc, bc_params = stats.boxcox(frame[feature]+np.abs(frame[feature].min())+0.0001) 
            frame[feature] = rc_bc
            
        else:
            
            rc_bc, bc_params  = stats.boxcox(frame[feature]+0.0001) 
            frame[feature] = rc_bc
            
        gc.collect()
    
    return frame


## Thanks to https://www.amazon.com/Feature-Engineering-Machine-Learning-Principles/dp/1491953241 for help on this. I engineered it a bit more ;)

def plot_boxcoxes(df : pd.DataFrame,feature_names ,number_of_feats_to_plot = 6):
    """
    Plots a random sample of the transformed features as a histrograme
    """
    fig, axes = plt.subplots(number_of_feats_to_plot,1, figsize= (10,15))
    choices = np.random.choice(feature_names, number_of_feats_to_plot, replace= False)
    
    for i, feat in enumerate(choices):

        df[feat].hist(ax=axes[i], bins=100, color = "sandybrown")
        axes[i].set_yscale('log')
        axes[i].tick_params(labelsize=14)
        axes[i].set_title(feat+ ' Histogram after boxcox', fontsize=14)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Occurrence', fontsize=14)
        fig.tight_layout()
        fig.show()


# In[ ]:


train_df_identity = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv")
train_df_transaction = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv")
test_df_identity = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_identity.csv")
test_df_transaction = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_transaction.csv")

sub_df = pd.read_csv("/kaggle/input/ieee-fraud-detection/sample_submission.csv")

train_df = pd.merge(left=train_df_identity, right=train_df_transaction, how='right', on='TransactionID', validate ='one_to_many')
test_df = pd.merge(left=test_df_identity, right=test_df_transaction, how='right', on='TransactionID', validate ='one_to_many')

train_df.set_index("TransactionID", inplace=True)
# train_df.drop(["TransactionID"], axis=1,inplace=True) #This wont be needed in training set its already in ID

train_df.drop(['isFraud'],axis=1, inplace=True) # we don't want to leak this into our predictor features

## The following code aids us in submission later on
sub_df.set_index("TransactionID", inplace=True)
test_df.set_index("TransactionID", inplace=True)


test_df = test_df.reindex(sub_df.index) #So order is similar to submission file
assert all(test_df.index.values == sub_df.index.values) #Test if this worked

del train_df_identity,train_df_transaction,test_df_identity,test_df_transaction

gc.collect()


# In[ ]:


#according to https://www.kaggle.com/c/ieee-fraud-detection/discussion/101203#latest-607486

Catfeats = ['ProductCD'] +            ["card"+f"{i+1}" for i in range(6)] +            ["addr"+f"{i+1}" for i in range(2)] +            ["P_emaildomain", "R_emaildomain"] +            ["M"+f"{i+1}" for i in range(9)] +            ["DeviceType", "DeviceInfo"] +            ["id_"+f"{i}" for i in range(12, 39)]

Numfeats = list(set(train_df.columns)-set(Catfeats))

assert set(Catfeats+Numfeats) == set(train_df.columns.values)


# Lets transform the categorical features that present as floats and text to pandas categorical

# In[ ]:


for col in Catfeats:
    train_df[col] = train_df[col].astype('str').astype('category')
    test_df[col] = test_df[col].astype('str').astype('category')
    
train_df[Catfeats] = train_df[Catfeats].replace('nan', np.nan) # to create a sparse matrix    
test_df[Catfeats] = test_df[Catfeats].replace('nan', np.nan) # to create a sparse matrix


# We combine train and test to impute values

# In[ ]:


# test_df['isFraud'] = np.NaN 

# train and test have same columns so we append

test_indeces = test_df.index
train_indeces = train_df.index

imputed_df = train_df.append(test_df, ignore_index=False, verify_integrity = True, sort=True)

del train_df,test_df

imputed_df = reduce_mem_usage(imputed_df)

gc.collect()


# In[ ]:


#######
# I still need to figure out how to save memory. If we take all data the kernel crashes

imputed_df = imputed_df.sample(frac=0.15)
gc.collect()


# In[ ]:


imputed_df.head()


# In[ ]:


imp = SimpleImputer(strategy="most_frequent", missing_values= np.nan)

for col in Catfeats:
    imputed_df[col] = imp.fit_transform(imputed_df[col].as_matrix().reshape(-1,1))


# In[ ]:


imputed_df[Catfeats].head()


# Categorical features look good. We will now focus on numerical features  Simple imputation for all features. Iterative imputer is too computationally expensive

# In[ ]:


# missing_na_perc = (imputed_df[Numfeats].isna().sum()/len(imputed_df[Numfeats])).sort_values(ascending=False)

# print(f"Missing percentage of values per feature:\n\n{missing_na_perc}")


# In[ ]:


# sixty_perc_or_less = list(missing_na_perc.loc[missing_na_perc <= .6].index)
# sixty_perc_or_more = list(missing_na_perc.loc[missing_na_perc > .6].index)


# In[ ]:


# Simple imputation for al features. Iterative umputer is too computationally expensive
from sklearn.impute import SimpleImputer 

# to introduce as less bias as we can, we iterate between mean and median for imputer

impute_method = 'mean'

for col in Numfeats:
    if impute_method == 'mean':
        impute_method = 'median'
        imp_mean = SimpleImputer(missing_values=np.nan, strategy=impute_method)
        imputed_df[col] = imp_mean.fit_transform(imputed_df[col].ravel().reshape(-1,1))
    elif impute_method == 'median':     
        impute_method = 'mean'
        imp_mean = SimpleImputer(missing_values=np.nan, strategy=impute_method)
        imp_mean.fit_transform(imputed_df[col].ravel().reshape(-1,1))
        imputed_df[col] = imp_mean.fit_transform(imputed_df[col].ravel().reshape(-1,1))
        

gc.collect()


# In[ ]:


assert imputed_df[Numfeats].isna().sum().sum() == 0, "We aren't done"


# In[ ]:


# # Complex imputation of more than 60% of missing values
# from sklearn.experimental import enable_iterative_imputer  # noqa
# # now you can import normally from sklearn.impute
# from sklearn.impute import IterativeImputer # for our numerical outpurs

# finished_cols = []

# for col in sixty_perc_or_more:
    
#     imp_iterative = IterativeImputer(max_iter=50, min_value = train_df[col].min(), max_value = train_df[col].max())
#     train_df[col] = imp_iterative.fit_transform(pd.concat([train_df[sixty_perc_or_less + finished_cols], train_df[col]], axis=1))[:,-1] # we take the last column as that is the 
#                                                                                                                         # one we are imputing using the rest
        
#     finished_cols.append(col)
    
#     gc.collect()


# In[ ]:


# assert train_df.isna().sum().sum() == 0, "We aren't done"


# ## Feature smoothing

# #### before boxcox

# In[ ]:


print("before boxcox:")
plot_boxcoxes(imputed_df[Numfeats],feature_names = Numfeats)


# In[ ]:


imputed_df = boxcox_Numfeats(imputed_df, Numfeats)

imputed_df = reduce_mem_usage(imputed_df)

gc.collect()


# Plotting a random sample of the box coxed features:

# #### after boxcox

# In[ ]:


print("after boxcox:")
plot_boxcoxes(imputed_df[Numfeats],feature_names = Numfeats)


# ### Encoding of Categorical Features

# In[ ]:


less_than_4_levels = imputed_df[Catfeats].columns[np.where(imputed_df[Catfeats].nunique() <= 4)]
more_than_4_levels = imputed_df[Catfeats].columns[np.where(imputed_df[Catfeats].nunique() > 4)]


# In[ ]:


## Less or equal than 4 levels we do label encoder:

le = LabelEncoder()
for col in less_than_4_levels:
    imputed_df[col] = le.fit_transform(imputed_df[col].astype(str))
    gc.collect()


# In[ ]:


## More than 4 levels we do one-hot encoder:

# enc = OneHotEncoder(handle_unknown='ignore')
# OH_encoded_sparse = enc.fit_transform(imputed_df[more_than_4_levels])


# We combine the sparse matrix to the original (sparsed) imputed_df

# In[ ]:


object_feats = (imputed_df.dtypes == 'O').index

strings_feats = []

for obj_f in object_feats:
    try:
         imputed_df[obj_f] = imputed_df[obj_f].astype('float')
    except:
        strings_feats.append(obj_f)


# In[ ]:


imputed_df[strings_feats]


# Let's get rid of these pesky critters

# In[ ]:


for col in strings_feats:
    imputed_df[col] = le.fit_transform(imputed_df[col].astype(str))
    gc.collect()


# In[ ]:


# we will need our indeces to restore the sparse matrix in our next kernel
# imputed_df.reset_index(inplace=True)
# imputed_df.head()

# test_csr_indeces = np.where(imputed_df.TransactionID.isin(test_indeces))


# In[ ]:


# # imputed_df.drop(more_than_4_levels,axis=1, inplace=True)
# imputed_df = scipy.sparse.csr_matrix(imputed_df)

# #adding one hot encoded features
# imputed_df = scipy.sparse.hstack((imputed_df, OH_encoded_sparse))

# gc.collect()


# In[ ]:


# # Now we get our original training and test sets back
# test_sparse = imputed_df.tocsr()[np.where(imputed_df.TransactionID.isin(test_indeces))]
# train_sparse_no_target_var = imputed_df.tocsr()

# del imputed_df;gc.collect()


# ## Finally we are done!!
# # 
# # I save the finished product. I hope it serves you well ;) Just remember that we introduced a lot of bias by imputing in this fasion## Finally we are done!!
# # 
# # I save the finished product. I hope it serves you well ;) Just remember that we introduced a lot of bias by imputing in this fasion
# 

# In[ ]:


# train_sparse_no_target_var.to_pickle("train_sparse_no_target_var.pkl") # You can load it in your kernel using df = pd.read_pickle("Imputed_Train.pkl") 

# scipy.sparse.save_npz('/tmp/imputed_df.npz', imputed_df)

# scipy.sparse.save_npz('/tmp/train_sparse_no_target_var.npz', train_sparse_no_target_var)

# You can load it in your kernel using sparse_matrix = scipy.sparse.load_npz('/tmp/sparse_matrix.npz')

imputed_df.to_csv("Imputed_df.csv",index=True)

