#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


# ## Loading libraries

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nimport sys\n!cp ../input/rapids/rapids.0.12.0 /opt/conda/envs/rapids.tar.gz\n!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null\nsys.path = ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path\nsys.path = ["/opt/conda/envs/rapids/lib/python3.6"] + sys.path\nsys.path = ["/opt/conda/envs/rapids/lib"] + sys.path \n!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/')


# In[ ]:


import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import pytz
import feather
import math
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
from hmmlearn.hmm import GaussianHMM
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import lightgbm as lgb
from tqdm.notebook import tqdm
from scipy.stats import mode
from sklearn.metrics import accuracy_score
# from cuml.neighbors import KNeighborsClassifier, NearestNeighbors
# import cuml; cuml.__version__


# In[ ]:


from cuml.neighbors import KNeighborsClassifier, NearestNeighbors
import cuml; cuml.__version__


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nNAME='KNN'\n\nNMINUS=3\nNPLUS=3\n\nweights=[4, 1, 0.5, 0.25, 1, 0.5, 0.25] # original, negative, positive\n\nNFOLDS=5\nRS=42\n\nPATH=Path('/kaggle/input/is-eda-sine-50hz-exp')\n\nprint(f'Loading data for model {NAME}\\n')\n\ntrain=feather.read_dataframe(PATH/'train.feather')\ntest=feather.read_dataframe(PATH/'test.feather')\n\nwith open(PATH/'folds_train.pickle', 'rb') as infile:\n    folds_train = pickle.load(infile)\n    \nwith open(PATH/'folds_val.pickle', 'rb') as infile:\n    folds_val = pickle.load(infile)")


# In[ ]:


models=np.sort(train['model'].unique())
models


# In[ ]:


cols=['signal_no_drift']
target=['open_channels']

for shift in range(1, NMINUS+1):
    feature='signal_shift_-' + str(shift)
    cols.append(feature)
    
for shift in range(1, NPLUS+1):
    feature='signal_shift_+' + str(shift)
    cols.append(feature)
    
print(f"The list of features included in the {NAME} model:\n")
print(cols)


# In[ ]:


train=train[cols+target+['model', 'batch', 'time']]
test=test[cols+['model', 'batch', 'segment', 'time']]


# In[ ]:


print(train.shape)
print(test.shape)


# ## Setting things up for training

# In[ ]:


classes=np.array(['class_'+str(i) for i in range(11)])
oof=pd.DataFrame(data=np.zeros((len(train), 11)), index=train.index, columns=classes)
oof_preds=np.zeros(len(train))
preds_proba=pd.DataFrame(data=np.zeros((len(test), 11)), index=test.index, columns=classes)

f1_folds=[]


# Multiply the original and shifted signal columns by their weights. 

# In[ ]:


for c, w in zip(cols, weights):
    train[c]=w*train[c]
    test[c]=w*test[c]


# ## Training a KNN model

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nKNN=100\nbatch = 1024\n\nfor fold_num in range(1, NFOLDS+1):\n    \n    print(\'-\'*50)\n    print(f\'Fold {fold_num}:\')\n    \n    train_index=folds_train[fold_num]\n    val_index=folds_val[fold_num]\n\n    X_train, Y_train = train.iloc[train_index, :], train.loc[train_index, target]\n    X_val, Y_val = train.iloc[val_index, :], train.loc[val_index, target]\n    \n    for m in models:   \n        \n#         if fold_num !=1:\n#             continue\n\n        mask_model_train=(X_train[\'model\']==m)\n        mask_model_val=(X_val[\'model\']==m)\n        mask_model_test=(test[\'model\']==m)\n\n        X_mod=X_train.loc[mask_model_train, cols].values\n        Y_mod=Y_train[mask_model_train].values.reshape(-1,)\n        \n        X_val_mod=X_val.loc[mask_model_val, cols].values\n        Y_val_mod=Y_val[mask_model_val].values.reshape(-1,)\n        \n        X_test=test.loc[mask_model_test, cols].copy()\n        \n#         clf = KNeighborsClassifier(n_neighbors=KNN)\n        \n#         clf.fit(X_mod, Y_mod)\n            \n        #Y_val_pred=clf.predict_proba(X_val_mod)#.reshape(-1, 1))     \n        #Y_test_pred=clf.predict_proba(X_test.values)#.reshape(-1, 1))\n        \n        if m==\'M4\':\n            shift=1 # recall that we removed zero open channel from model 4\n        else:\n            shift=0\n        ##############################################\n        #KNN = 99\n        #batch = 1024\n        #print(\'Training...\')\n        clf = NearestNeighbors(n_neighbors=KNN)\n        clf.fit(X_mod)\n        distances, indices = clf.kneighbors(X_val_mod)\n        #print(\'Processing validation set...\')\n        ct = indices.shape[0]\n        pred = np.zeros((ct,KNN),dtype=np.int8)\n        Y_val_pred = np.zeros((ct,len(np.unique(Y_mod))),dtype=np.float32)\n        it = ct//batch + int(ct%batch!=0)\n        for k in range(it):\n            a = batch*k; b = batch*(k+1); b = min(ct,b)\n            pred[a:b,:] = Y_mod[ indices[a:b].astype(int) ]\n            for j in np.unique(Y_mod):\n                Y_val_pred[a:b,j-shift] = np.sum(pred[a:b,]==j,axis=1)/KNN\n        \n        ##############################################\n        #print(\'Processing test set...\')\n        \n        distances, indices = clf.kneighbors(X_test.values)\n\n        ct = indices.shape[0]\n        pred = np.zeros((ct,KNN),dtype=np.int8)\n        Y_test_pred = np.zeros((ct,len(np.unique(Y_mod))),dtype=np.float32)\n        it = ct//batch + int(ct%batch!=0)\n        for k in range(it):\n            a = batch*k; b = batch*(k+1); b = min(ct,b)\n            pred[a:b,:] = Y_mod[ indices[a:b].astype(int) ]\n            for j in np.unique(Y_mod):\n                Y_test_pred[a:b,j-shift] = np.sum(pred[a:b,]==j,axis=1)/KNN\n        \n        ##############################################\n\n        classes_mod=classes[np.unique(Y_mod)]           \n        #print(\'oofs...\')\n        oof.loc[val_index[mask_model_val], classes_mod]=Y_val_pred\n        #print(\'preds_probas...\')\n        preds_proba.loc[mask_model_test, classes_mod]+=Y_test_pred\n        \n        # Compute Macro F1 score for the model:\n        \n        #print(\'Y_val_pred...\')\n        Y_val_pred=np.argmax(Y_val_pred, axis=1).astype(int).reshape(-1, ) + int(shift)\n        #print(\'f1...\')\n        f1_model=f1_score(Y_val_mod, Y_val_pred, average=\'macro\')\n        print(f\'Model {m}: done! Macro F1 score = {f1_model:.5f}\')\n    \n    oof_preds[val_index]=np.argmax(oof.iloc[val_index, :].values, axis=1).astype(int).reshape(-1, )\n    Y_val_OC=train.loc[val_index, \'open_channels\'].values.astype(np.uint8).reshape(-1, )\n    \n    f1_fold=f1_score(Y_val_OC, oof_preds[val_index], average=\'macro\')\n    f1_folds.append(f1_fold)\n    \n    print(f\'\\nFold {fold_num} is done! Macro F1 score = {f1_fold:.5f}\')\n\npreds_proba/=NFOLDS\npreds=np.argmax(preds_proba.values, axis=1).astype(int).reshape(-1, )\n\nprint(\'-\'*50)\nprint(\'Summary:\')\n\nfor m in models:\n    print(f"\\nModel {m}:")\n    mask_model=train[\'model\']==m\n    f1_model=f1_score(train.loc[mask_model, \'open_channels\'].values.reshape(-1,), \n                      oof_preds[mask_model], average=\'macro\')\n    print(classification_report(train.loc[mask_model, \'open_channels\'].values.reshape(-1,), \n                                oof_preds[mask_model], digits=5))\n    print(f\'Macro F1 score for model {m}    = {f1_model:.5f}\')\n\nf1_av=np.array(f1_folds).mean()\nf1_std=np.std(f1_folds)\nprint(f\'Macro F1 score = {f1_av:.5f} (average across the folds); std = {f1_std:.5f}\')\n\nf1=f1_score(train[\'open_channels\'].values.reshape(-1,), oof_preds, average=\'macro\')\n\nprint(f\'Macro F1 score = {f1:.5f} (out-of-folds)\')')


# Reinstate the original values of the signals:

# In[ ]:


for c, w in zip(cols, weights):
    train[c]=train[c]/w
    test[c]=test[c]/w


# To get an idea about the accuracy of our results let's print a full classification report and also take a look at the confusion matricies for different models.

# In[ ]:


get_ipython().run_cell_magic('time', '', "print(classification_report(train['open_channels'].values.reshape(-1,), oof_preds, digits=5))")


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# hidden states vs open channels\nfig, ax = plt.subplots(5, 1, figsize=(10, 10*5))\nax = ax.flatten()\n\nfor i, m in enumerate(models): \n    mask=train[\'model\']==m\n    cm = confusion_matrix(train.loc[mask, \'open_channels\'].values, oof_preds[mask])\n    sns.heatmap(cm, annot=True, lw=1, ax=ax[i])\n    ax[i].set_xlabel("Predicted open channels")\n    ax[i].set_ylabel("Actual open channels")\n    ax[i].set_title(f"Model {m}")\nplt.tight_layout()\nplt.show()')


# ## Fixing Model 1
# 
# Model 1 data in the train set contain only two possible open_channels values: 0 and 1. Graphical analysis of the test set has shown that there might be an additional channel present in Model 1 test set data. To identify the channels that are not in train earlier we used the Gaussian Mixture Model algorithm. Now, for `open_channels` of model M1 greater than 1, we will overwrite the our predictions with those of GMM.

# In[ ]:


get_ipython().run_cell_magic('time', '', "PATH=Path('/kaggle/input/is-gmm-cv5-b7-repl-seq-folds/sub_GMM_110.csv')\npreds_GMM=pd.read_csv(PATH)\n\nmask_M1_GMM=(test['model']=='M1')&(preds_GMM['open_channels']>1)\npreds[mask_M1_GMM]=preds_GMM.loc[mask_M1_GMM, 'open_channels']")


# ## Generating a submission file and saving oof's and predicted probabilities

# In[ ]:


PATH=Path('/kaggle/input/liverpool-ion-switching/')
sub=pd.read_csv(PATH/'sample_submission.csv')


# In[ ]:


sub['open_channels']=preds
sub['open_channels'].value_counts().sort_index()


# In[ ]:


sub.shape


# In[ ]:


time_zone = pytz.timezone('America/Chicago')
current_datetime = datetime.now(time_zone)
ts=current_datetime.strftime("%m%d%H%M%S")

sub_file_name='sub_'+NAME+'_'+ts+'.csv'
oof_file_name='oof_'+NAME+'.feather'#'_'+ts+'.csv'
preds_file_name='preds_'+NAME+'.feather'#'_'+ts+'.csv'

ts, sub_file_name, oof_file_name, preds_file_name


# In[ ]:


get_ipython().run_cell_magic('time', '', "\noof.to_feather(oof_file_name)\npreds_proba.to_feather(preds_file_name)\nsub.to_csv(sub_file_name, index=False, float_format='%.4f')")


# ## Visualizing the results

# In[ ]:


get_ipython().run_cell_magic('time', '', "\npalette = sns.color_palette()\npalette=[(0, 0, 0)]+palette\nsns.palplot(palette)\nplt.xlabel('Open channels', fontsize=15)\nticks=np.arange(0, 11)\nplt.xticks(ticks, ticks, fontsize=12)\nplt.show()")


# In[ ]:


def plot_signal_vs_shifted_one(df, mod, target='open_channels', batch=None, segment=None, 
                               col1='signal_no_drift', col2='signal_shift_-1', s=0.05, mk_scale=60,
                               low=math.floor(train['signal_no_drift'].min()),
                               high=math.ceil(train['signal_no_drift'].max())):
    
    mask_model=df['model']==mod
        
    if batch is not None:
        mask_batch=df['batch']==batch
        mask_model=np.logical_and(mask_model, mask_batch)

    if segment is not None:
        if 'segment' not in df.columns:
            print("There is no 'segment' column in the data frame! Can't continue!")
            return
        else:           
            mask_segment=df['segment']==segment
            mask_model=np.logical_and(mask_model, mask_segment)               
            
    if target in df.columns:
        mod_chans=np.unique(df.loc[mask_model, target].values)
        for ch in mod_chans:
            mask_channel=df[target]==ch
            mask=np.logical_and(mask_model, mask_channel)
            x=df.loc[mask, col1].values
            y=df.loc[mask, col2].values
            plt.plot(x, y, 'o', markersize=s, label=ch, c=palette[ch])
            plt.legend(markerscale=mk_scale)
    else:
        x=df.loc[mask_model, col1].values
        y=df.loc[mask_model, col2].values
        plt.plot(x, y, 'o', markersize=s)        
    
    plt.xlim((low, high))
    plt.ylim((low, high))
    
    plt.xlabel('Current now, pA')
    plt.ylabel('Current next, pA')
    
    plot_title=f'Model {mod}'
    if batch is not None:
        plot_title+=f', batch {batch}'
    if segment is not None:
        plot_title+=f', {segment}'
    plt.title(plot_title)


# In[ ]:


def plot_signal_vs_shifted_all(df, mod, target='open_channels', hsize_one=5, 
                               s=0.05, mk_scale=60, n_cols=2, style='seaborn-whitegrid',
                               col1='signal_no_drift', col2='signal_shift_-1',
                               low=math.floor(train['signal_no_drift'].min()),
                               high=math.ceil(train['signal_no_drift'].max()),):
    
    mask=df['model']==mod
    
    if 'segment' in df.columns:
        segments=np.sort(df.loc[mask, 'segment'].unique())
        batches=[None for i in range(len(segments))]
    else:
        batches=np.sort(df.loc[mask, 'batch'].unique())
        segments=[None for i in range(len(batches))]
    
    hsize=n_cols*hsize_one
    n_rows=math.ceil(len(batches) / n_cols)
    vsize= n_rows*hsize_one
    
    plt.figure(figsize=(hsize, vsize))
    plt.style.use(style)
    
    for i , (batch, segment) in enumerate(zip(batches, segments), 1):
        plt.subplot(n_rows, n_cols, i)
        plot_signal_vs_shifted_one(df, target=target, batch=batch, 
                                   segment=segment, mod=mod, s=s, 
                                   mk_scale=mk_scale,
                                   low=low, high=high)
        
    plt.tight_layout()


# In[ ]:


low={'M1':-4, 'M2':-4, 'M3':-5, 'M4':-5, 'M5':-5}
high={'M1':2, 'M2':0, 'M3':6, 'M4':10, 'M5':5}


# In[ ]:


def show_results(preds, mod='M1', lag=-1, df=train, s=0.3, mk_scale=10):
    df_new=df[['model', 'batch', 'signal_no_drift', 'signal_shift_'+str(lag)]].copy()
    df_new['open_channels']=preds.astype(np.uint8)
    if 'segment' in df.columns:
        df_new['segment']=df['segment'].copy()
    plot_signal_vs_shifted_all(df_new, mod, low=low[mod], high=high[mod], s=s, mk_scale=mk_scale)


# In[ ]:


for m in models:
    show_results(oof_preds, mod=m, lag=-1, df=train, s=0.7, mk_scale=5)


# In[ ]:


y_true=train['open_channels'].values

mask=np.equal(y_true, oof_preds)
for m in models:
    show_results(y_true[~mask], mod=m, lag=-1, df=train[~mask], s=0.7, mk_scale=5)


# In[ ]:


high['M1']=4


# In[ ]:


for m in models:
    show_results(preds, mod=m, lag=-1, df=test, s=0.7, mk_scale=5)


# In[ ]:


batches_order=np.array([0, 1, 2, 6, 3, 7, 4, 9, 5, 8])


# In[ ]:


def signal_scatter_plots(df, col='signal', order=batches_order):
    
    n_batches=df['batch'].nunique()
    
    if n_batches==4:  # if test
        vsize = 2
        hsize = 2
        fig_vsize=20
        fig_hsize=40
        name='test'
        order=np.arange(4)
    else:             # if train
        vsize = 5
        hsize = 2
        fig_vsize=60
        fig_hsize=40
        name='train'
    
    plt.figure(figsize=(fig_hsize, fig_vsize), facecolor='white')
    sns.set(font_scale=3.5)
    
    for i, b in enumerate(order):

        ax = plt.subplot(vsize, hsize, i+1)
        mask_batch=(df['batch'] == b)
        
        if np.isin(df.columns, 'open_channels').any():   
            channels=np.unique(df.loc[mask_batch, 'open_channels'].values)
            for ch in channels:
                mask_channel=(df['open_channels']==ch)
                mask=np.logical_and(mask_batch, mask_channel)
                plt.plot(df.loc[mask, 'time'].values, df.loc[mask, col].values, 
                         'o', color=palette[ch], ms=0.6, label=ch)      
            title_string='Signal vs time per batch in '
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., 
                       fontsize='36', markerscale=30)
        else:
            title_string='Signal vs time per batch in '
            plt.plot(df.loc[mask_batch, 'time'].values, df.loc[mask_batch, col].values, 
                     'o', ms=0.1)

        ax.set(xlabel='Time, s', ylabel='Current, pA', title= f'Batch {b}')

    plt.suptitle(title_string + f'{name}', y=1.02)
    plt.tight_layout()
    plt.show()


# In[ ]:


get_ipython().run_cell_magic('time', '', "train['open_channels']=oof_preds\ntrain['open_channels']=train['open_channels'].astype(np.uint8)\nsignal_scatter_plots(train, col='signal_no_drift')")


# In[ ]:


get_ipython().run_cell_magic('time', '', "test['open_channels']=preds\ntest['open_channels']=test['open_channels'].astype(np.uint8)\nsignal_scatter_plots(test, col='signal_no_drift')")


# In[ ]:




