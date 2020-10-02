#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('ls ../input/1-mpn')


# In[ ]:


get_ipython().system('ls ../input/')
get_ipython().system('ls ../input/nnet-b-seed-10')
get_ipython().system('ls ../input/nnet-b-seed-11')
get_ipython().system('ls ../input/nnet-b-seed-12')


# In[ ]:


get_ipython().system('ls ../input/champ-preds')


# In[ ]:


import pandas as pd

def get_median_from_files(files):
    print(len(files))
    outs = [pd.read_csv(f, index_col=0) for f in files]
    concat_sub = pd.concat(outs, axis=1, sort=True)
    champ_median = concat_sub.median(axis=1).values
    return champ_median


test = pd.read_csv(f"../input/champs-scalar-coupling/test.csv")
TARGET = 'scalar_coupling_constant'

test['conservative']  =  pd.read_csv('../input/champs-ensemble-conservative/ensemble_sub_conservative.csv')[TARGET]
test['n1'] =  get_median_from_files(
    [
        '../input/champ-preds/gnn_median_2302.csv',
        '../input/champ-preds/gnn_median_2301.csv',
        '../input/champ-preds/gnn_median_adjusted_1JHC_2296.csv', 
    ]
)
gnn_2312 = pd.read_csv('../input/champ-preds/gnn_median_65_68_2312.csv')[TARGET]
lastgnn = pd.read_csv('../input/champ-preds/gnn_median_69_73.csv')[TARGET]  # 2318
test['n1'] = lastgnn * 0.5 + gnn_2312 * 0.3 + test['n1'] *0.2

test['n2'] =  pd.read_csv('../input/champ-preds/gnn0_median_2068.csv')[TARGET]
test['lgb_a'] = get_median_from_files(
    [
        '../input/champ-preds/submission_type_2100.csv',
        '../input/champ-preds/submission_type_2085.csv', 
    ]
)
test['lgb_m'] = get_median_from_files(
    [
        '../input/champ-preds/lgb_type_full_f286_10.csv', # -2.027
        '../input/champ-preds/lgb_type_full_f262_10.csv', # -2.016
    ]
)
test['nnet'] = get_median_from_files(
    [
        '../input/nnpvals-seed-20/nnet_sub_s11.csv',
        '../input/champ-preds/nnet_sub.csv',  # version 6
        '../input/nn-seed-10/nnet_sub.csv',
        '../input/nn-seed-11/nnet_sub.csv',
        '../input/nnet-c-seed-10/lgb_type_cv-1.7126_mae0.23572_fd5_10.csv',
        '../input/nnet-c-seed-11/lgb_type_cv-1.70994_mae0.23497_fd5_11.csv',
        '../input/nnet-c-seed-12/lgb_type_cv-1.71029_mae0.23523_fd5_12.csv',
        '../input/nnet-b-seed-10/lgb_type_cv-1.72296_mae0.24019_bags-1_f120_fd5_10.csv',
        '../input/nnet-b-seed-11/lgb_type_cv-1.70647_mae0.2376_bags-1_f120_fd5_11.csv',
        '../input/nnet-b-seed-12/lgb_type_cv-1.69833_mae0.24106_bags-1_f120_fd5_12.csv',
#         '../input/nnet-try/lgb_type_cv-2.108373877033157_mae0.12143527465528912_bags-1_f120_fd5_10.csv',
        '../input/nnet-try-seed-11/lgb_type_cv-1.64944_mae0.23965_bags-1_f120_fd5_11.csv',
#        '../input/nnet-try-seed-12/lgb_type_cv-1.64875_mae0.2412_bags-1_f120_fd5_12.csv',
    ]    
)
test['nnet_cont'] = get_median_from_files(
    [
        '../input/nncont-seed-23-p/nnetCont_sub_only_predict.csv', 
        '../input/nncont-seed-22/nnetCont_sub-1.8213.csv', # cv
    ]   
)
nn_contof =  pd.read_csv('../input/nncont-seed-26/nnetCont_sub_-2.6677.csv')[TARGET]
test['nnet_cont'] = nn_contof * 0.6 + test['nnet_cont'] * 0.4
test['final_mpnn']  =  pd.read_csv('../input/champ-preds/final_mpnn.csv')[TARGET]
test['mpnn']  =  pd.read_csv('../input/champ-preds/mpnn_5_fold_pseudo_1449.csv')[TARGET]

test['lb'] = pd.read_csv('../input/chemistry-of-best-models-1-895/stack_median.csv')[TARGET]  # -1.895 or a bit more
test['stack15'] = pd.read_csv('../input/15th-stacking/test_stacking.csv')[TARGET]  # -2.8 :D
# test['mpnn'] = pd.read_csv('../input/champ-preds/mpnn_5_fold_pseudo_1449.csv')[TARGET]
test.head(10)


# In[ ]:


# test_with_preds = test_with_preds.sort_values('id')
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

test['nnet_ens'] = test['nnet_cont'] * 0.6 + test['nnet'] * 0.4
test['lgb_ens'] = test['lgb_a'] * 0.8 + test['lgb_m'] * 0.2

test['final_preds'] =(
    test['n1']*0.5 +
    test['n2']*0.1 +
    test['lgb_ens']*0.15 +
    test['nnet_ens']*0.1 +
    test['lb']*0.1 +
    test['final_mpnn'] * 0.03 + test['mpnn'] *0.02
)
test.head(20)

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
cols = [col for col in test.columns if col not in ['id', 'molecule_name', 'atom_index_0', 'atom_index_1']]
fig = sns.heatmap(test[cols].corr(), cmap='viridis', square=True)
test[cols].corr()


# In[ ]:


# seeing only nnet
submission = pd.DataFrame()
submission['id'] = test.id
# submission['scalar_coupling_constant'] = test['final_preds']
submission['scalar_coupling_constant'] = test['final_preds'] * 0.1 + test['stack15'] * 0.9
submission.to_csv('ensemble_sub.csv', index=False)


# In[ ]:


submission.head(20)


# In[ ]:




