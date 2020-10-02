#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# General imports
import numpy as np
import pandas as pd
import os, sys, gc, warnings, random, datetime, math, psutil

import seaborn as sns
import matplotlib.pyplot as plt

from multiprocessing import Pool

warnings.filterwarnings('ignore')


# In[ ]:


########################### Helpers
#################################################################################
## Multiprocessing Run.
# :df - DataFrame to split                      # type: pandas DataFrame
# :func - Function to apply on each split       # type: python function
# This function is NOT 'bulletproof', be carefull and pass only correct types of variables.
def df_parallelize_run(df, func):
    num_partitions, num_cores = psutil.cpu_count(), psutil.cpu_count()  # number of partitions and cores
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def check_state():
    if LOCAL_TEST:
        bad_uids = full_df.groupby(['uid'])['isFraud'].agg(['nunique', 'count'])
        bad_uids = bad_uids[(bad_uids['nunique']==2)]
        print('Inconsistent groups',len(bad_uids))

    print('Cleaning done...')
    print('Total groups:', len(full_df['uid'].unique()), 
          '| Total items:', len(full_df),
          '| Total fraud', full_df['isFraud'].sum())
    
    
########################### Sainity check 
def sanity_check_run(temp_df, verbose=False):
    temp_df = temp_df.copy()
    temp_df = temp_df.sort_values(by='TransactionID').reset_index(drop=True)
    bad_uids_groups = pd.DataFrame()

    """
    for col in ['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14']:
        temp_df['sanity_check'] = temp_df.groupby(['uid'])[col].shift()
        temp_df['sanity_check'] = (temp_df[col]-temp_df['sanity_check']).fillna(0).clip(None,0)

        bad_uids = temp_df.groupby(['uid'])['sanity_check'].agg(['sum']).reset_index()
        bad_uids = bad_uids[bad_uids['sum']<0]
        bad_uids_groups = pd.concat([bad_uids_groups,bad_uids])
        if verbose: print(col, len(bad_uids), bad_uids['uid'].values[:2])
    """
    
    bad_uids = temp_df.groupby(['uid'])['V313'].agg(['nunique']).reset_index()
    bad_uids = bad_uids[(bad_uids['nunique']>2)]
    bad_uids_groups = pd.concat([bad_uids_groups,bad_uids])
    if verbose: print('V313:', len(bad_uids), bad_uids['uid'].values[:2])

    bad_uids_groups = bad_uids_groups[['uid']].drop_duplicates()
    if verbose: print('Total bad groups:', len(bad_uids_groups))
    return bad_uids_groups



def parallel_check(bad_uids_groups):
    bad_uids_items = []
    if True:
        for cur_uid in list(bad_uids_groups['uid'].unique()):
            temp_df = full_df[full_df['uid']==cur_uid].reset_index(drop=True)
            v313_values = temp_df['V313'].value_counts()
            if len(v313_values)>1: 
                v313_values = [[col for col in list(v313_values.index)[:2] if col!=0][0]] + [0]
            else: 
                v313_values = [list(v313_values.index)[0]]+[0]
                
            for i in range(1,len(temp_df)):
                item_1 = temp_df.iloc[i]
                item_2 = temp_df.iloc[i-1]

                check_if_match = temp_df.drop([i])
                check_if_match = sanity_check_run(check_if_match)
                if len(check_if_match) == 0:
                    bad_uids_items.append(item_1['TransactionID'])
                    break

                if i!=1:
                    check_if_match = temp_df.drop([i-1])
                    check_if_match = sanity_check_run(check_if_match)
                    if len(check_if_match) == 0:
                        bad_uids_items.append(item_2['TransactionID'])
                        break
                
                """
                for col in ['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14']:
                    check_sanity = item_1[col]<item_2[col]
                    if check_sanity:
                        bad_uids_items.append(item_1['TransactionID'])
                        break 
                        
                if check_sanity:
                    break
                """    
                check_sanity = item_1['V313'] not in v313_values
                if check_sanity:
                    bad_uids_items.append(item_1['TransactionID'])
                    break 
                    
                if temp_df['TransactionID'].isin(problem_items['TransactionID']).sum()==0:
                    if cur_uid>=0:
                        check_sanity = ((item_2['DT_day']-item_1['uid_td_D3'])**2)**0.5>1

                        if check_sanity:
                            bad_uids_items.append(item_1['TransactionID'])
                            break

    bad_uids_items = pd.DataFrame(bad_uids_items, columns=['uid'])
    return bad_uids_items
 


# In[ ]:


LOCAL_TEST = True
CHECK_ORDER = True
TRUST_D1 = True
MULTI_UID_CHECK = False
FULL_GROUP_CHECK = False


# In[ ]:


########################### DATA LOAD
#################################################################################
print('Load Data')
train_df = pd.read_pickle('../input/ieee-data-minification-private/train_transaction.pkl')
test_df = pd.read_pickle('../input/ieee-data-minification-private/test_transaction.pkl')

# Full Data set (careful with target encoding)
full_df = pd.concat([train_df, test_df]).reset_index(drop=True)

if LOCAL_TEST:
    full_df = full_df.iloc[:10000] #full_df[(full_df['DT_M']==12)]


# In[ ]:


########################### Base prepartion

full_df['full_addr'] = full_df['addr1'].astype(str)+'_'+full_df['addr2'].astype(str)

for col in ['D'+str(i) for i in [1,2,3,5,10,11,15]]: 
    new_col = 'uid_td_'+str(col)
    
    full_df[new_col] = full_df['TransactionDT'] / (24*60*60)
    full_df[new_col] = np.floor(full_df[new_col] - full_df[col]) + 1000

full_df['DT_day'] = np.floor(full_df['TransactionDT']/(24*60*60)) + 1000

full_df['TransactionAmt_fix'] = np.round(full_df['TransactionAmt'],2)
full_df['V313_fix'] = np.round(full_df['V313'],2)
full_df['uid'] = np.nan

v_cols = []
v_fix_cols = []
for col in ['V'+str(i) for i in range(1,340)]:
    if (full_df[col].fillna(0) - full_df[col].fillna(0).astype(int)).sum()!=0:
        if col not in ['V313']:
            v_cols.append(col)
            v_fix_cols.append(col+'_fix')
            full_df[col+'_fix_ground'] = np.round(full_df[col],2)
            full_df[col+'_fix'] = full_df[col+'_fix_ground'] + full_df['TransactionAmt_fix']

global_bad_items = full_df[full_df['D1'].isna()]
full_df = full_df[~full_df['TransactionDT'].isin(global_bad_items['TransactionDT'])]
all_items = full_df.copy()
bkp_items = full_df.copy()

print('Total number of transactions:', len(full_df))


# In[ ]:


########################### Single Transaction
# Let's filter single card apearence card1/D1 -> single transaction per card
full_df['count'] = full_df.groupby(['card1','uid_td_D1'])['TransactionID'].transform('count')
single_items = full_df[full_df['count']==1]
single_items['uid'] = single_items['TransactionID']
del full_df, single_items['count']

all_items = all_items[~all_items['TransactionID'].isin(single_items['TransactionID'])]
print('Single transaction',len(single_items))


# In[ ]:


### Clean full_df
full_df = pd.DataFrame()
###


# In[ ]:


# First appearance of card1

first_df = all_items.copy()
first_df['counts'] = first_df.groupby(['card1','uid_td_D1']).cumcount()
first_df = first_df[first_df['counts']==0]
del first_df['counts']

first_df['uid'] = first_df['TransactionID']
print('First time in dataset', len(first_df))

full_df = pd.concat([full_df,first_df])
full_df = full_df.sort_values(by='TransactionID').reset_index(drop=True)
del first_df


# In[ ]:


check_state()


# In[ ]:


# Lets Check unassigned items again
# Let's find itmes with roots out of our dataset
nan_df_check = all_items[~all_items['TransactionID'].isin(full_df['TransactionID'])]

# if 'uid_td_D3'>1000 it means that root item is in our dataset
# >1000 will also filter NaNs values
nan_df_check['uid'] = np.where(nan_df_check['uid_td_D3']>=1001, 
                               np.nan, nan_df_check['TransactionID'])
nan_df_check = nan_df_check[~nan_df_check['uid'].isna()]

full_df = pd.concat([full_df,nan_df_check])
full_df = full_df.sort_values(by='TransactionID').reset_index(drop=True)

print('Roots out of dataset', len(nan_df_check))
#del nan_df_check
out_of_bonds = nan_df_check[['TransactionID']]


# In[ ]:


check_state()


# In[ ]:


########################### VERY IMPORTANT
# Do not do sanity D3 check for gap items
problem_items = full_df[(full_df['uid_td_D3']>1182)&(full_df['uid_td_D3']<1213)]

out_of_bonds = pd.concat([out_of_bonds, problem_items[['TransactionID']]])


# In[ ]:


########################### Sort
all_items = all_items.sort_values(by='TransactionID').reset_index(drop=True)
single_items = single_items.sort_values(by='TransactionID').reset_index(drop=True)
full_df = full_df.sort_values(by='TransactionID').reset_index(drop=True)
out_of_bonds = out_of_bonds.sort_values(by='TransactionID').reset_index(drop=True)


# We found all root items. There is just very rare cases that a new root can appear 
# _____

# In[ ]:


def find_and_append_root(df):
    new_uids_items = {'TransactionID': [],
                      'uid': [],
                      }

    for i in range(len(df)):
        item = df.iloc[i]
        if item['TransactionID'] not in list(problem_items['TransactionID']):
            mask_1 = bkp_items['card1'] == item['card1']
            mask_2 = bkp_items['uid_td_D1'] == item['uid_td_D1']
            mask_3 = bkp_items['TransactionID'] < item['TransactionID']
            mask_4 = ((bkp_items['DT_day'] == item['uid_td_D3'] + 1)|
                      (bkp_items['DT_day'] == item['uid_td_D3'] - 1)|
                      (bkp_items['DT_day'] == item['uid_td_D3']))

            df_masked = bkp_items[mask_1 & mask_2 & mask_3 & mask_4]
            no_match = len(df_masked) == 0

            if no_match:
                new_uids_items['TransactionID'].append(item['TransactionID'])
                new_uids_items['uid'].append(item['TransactionID'])
 
    return_df = pd.DataFrame.from_dict(new_uids_items)
    return return_df


# In[ ]:


########################### PART X - > 100% Root

nan_df = all_items[~all_items['TransactionID'].isin(full_df['TransactionID'])]
nan_df = nan_df[nan_df['DT_M']<18]
print('Items to check:', len(nan_df))

df_cleaned = df_parallelize_run(nan_df, find_and_append_root)
df_cleaned = df_cleaned[~df_cleaned['uid'].isna()]

df_cleaned.index = df_cleaned['TransactionID']
temp_dict = df_cleaned['uid'].to_dict()
nan_df['uid'] = nan_df['TransactionID'].map(temp_dict)
nan_df = nan_df[~nan_df['uid'].isna()]
print('Assigned root items:', len(nan_df))

# append found items
full_df = pd.concat([full_df,nan_df]).sort_values(by='TransactionID').reset_index(drop=True)
check_state()


# In[ ]:





# In[ ]:


def append_item_to_uid(df):
    new_uids_items = {'TransactionID': [],
                      'uid': [],
                      }

    for i in range(len(df)):
        item = df.iloc[i]

        mask_1 = full_df['card1'] == item['card1']
        mask_2 = full_df['uid_td_D1'] == item['uid_td_D1']
        mask_3 = full_df['TransactionID'] < item['TransactionID']
        mask_4 = full_df['DT_day'] <= item['uid_td_D3'] + 1  # +1 just to ensure that there is no
        df_masked = full_df[mask_1 & mask_2 & mask_3 & mask_4]

        has_match = len(df_masked) > 0
        can_be_root = True

        # New check
        # addr2 should be nan or same as group addr2
        for col in ['addr2','addr1']:
            if has_match:
                if not np.isnan(item[col]):
                    mask = ((df_masked[col] == item[col]) | (df_masked[col].isna()))
                    df_masked = df_masked[mask]
                else:
                    df_masked = df_masked

                if len(df_masked) == 0:
                    has_match = False
        """
        # Order                   
        for col in ['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14']:
            if has_match:
                mask = df_masked[col] <= item[col]
                df_masked = df_masked[mask]

                if len(df_masked) == 0:
                    has_match = False
        """
        
        if has_match:
            mask = (df_masked['TransactionID'] > item['TransactionID']).astype(int)
            for col in v_cols:
                mask += (df_masked[col+'_fix'] == item[col+'_fix_ground']).astype(int)
            mask = mask>0
            df_masked = df_masked[mask]
            
            if len(df_masked) == 0:
                has_match = False
                    
        # Assign  
        if has_match and len(df_masked['uid'].unique()) == 1:
            check_if_match = df_masked.append(item)
            check_if_match = sanity_check_run(check_if_match)
            if len(check_if_match) == 0:
                new_uids_items['TransactionID'].append(item['TransactionID'])
                new_uids_items['uid'].append(df_masked['uid'].unique()[0])


    return_df = pd.DataFrame.from_dict(new_uids_items)
    return return_df


# In[ ]:


########################### PART X - > 100% single match
for i in range(5):
    print('Check round:', i)
        
    nan_df = all_items[~all_items['TransactionID'].isin(full_df['TransactionID'])]
    print('Items to check:', len(nan_df))

    df_cleaned = df_parallelize_run(nan_df, append_item_to_uid)
    df_cleaned = df_cleaned[~df_cleaned['uid'].isna()]

    df_cleaned.index = df_cleaned['TransactionID']
    temp_dict = df_cleaned['uid'].to_dict()
    nan_df['uid'] = nan_df['TransactionID'].map(temp_dict)
    nan_df = nan_df[~nan_df['uid'].isna()]
    print('Assigned items:', len(nan_df))

    # append found items
    full_df = pd.concat([full_df,nan_df]).sort_values(by='TransactionID').reset_index(drop=True)
    
    for i in range(100):
        bad_uids_groups = sanity_check_run(full_df, False)
        if len(bad_uids_groups)==0:
            break
        elif len(bad_uids_groups)>64:
            bad_uids_items = df_parallelize_run(bad_uids_groups, parallel_check)
        else:
            bad_uids_items = parallel_check(bad_uids_groups)

        print('Found bad items', len(bad_uids_items))
        full_df['uid'] = np.where(full_df['TransactionID'].isin(bad_uids_items['uid']), np.nan, full_df['uid'])
        full_df = full_df[~full_df['uid'].isna()].sort_values(by='TransactionID').reset_index(drop=True)
        if len(bad_uids_items)<2:
            break
    check_state()


# In[ ]:





# In[ ]:


def find_and_append_root_test(df):
    new_uids_items = {'TransactionID': [],
                      'uid': [],
                      }

    for i in range(len(df)):
        item = df.iloc[i]
        if item['TransactionID'] not in list(problem_items['TransactionID']):
            mask_1 = bkp_items['card1'] == item['card1']
            mask_2 = bkp_items['uid_td_D1'] == item['uid_td_D1']
            mask_3 = bkp_items['TransactionID'] < item['TransactionID']
            mask_4 = ((bkp_items['DT_day'] == item['uid_td_D3'] + 1)|
                      (bkp_items['DT_day'] == item['uid_td_D3'] - 1)|
                      (bkp_items['DT_day'] == item['uid_td_D3']))

            df_masked = bkp_items[mask_1 & mask_2 & mask_3 & mask_4]
            no_match = len(df_masked) == 0

            if no_match:
                new_uids_items['TransactionID'].append(item['TransactionID'])
                new_uids_items['uid'].append(item['TransactionID'])
 
    return_df = pd.DataFrame.from_dict(new_uids_items)
    return return_df


# In[ ]:


########################### PART X - > 100% Root

nan_df = all_items[~all_items['TransactionID'].isin(full_df['TransactionID'])]
nan_df = nan_df[nan_df['DT_M']>18]
print('Items to check:', len(nan_df))

df_cleaned = df_parallelize_run(nan_df, find_and_append_root_test)
df_cleaned = df_cleaned[~df_cleaned['uid'].isna()]
df_cleaned = df_cleaned[~df_cleaned['TransactionID'].isin(full_df['TransactionID'])]

df_cleaned.index = df_cleaned['TransactionID']
temp_dict = df_cleaned['uid'].to_dict()
nan_df['uid'] = nan_df['TransactionID'].map(temp_dict)
nan_df = nan_df[~nan_df['uid'].isna()]
print('Assigned root items:', len(nan_df))

# append found items
full_df = pd.concat([full_df,nan_df]).sort_values(by='TransactionID').reset_index(drop=True)
check_state()


# In[ ]:





# In[ ]:


########################### PART X - > 100% single match
for i in range(3):
    print('Check round:', i)
        
    nan_df = all_items[~all_items['TransactionID'].isin(full_df['TransactionID'])]
    print('Items to check:', len(nan_df))

    df_cleaned = df_parallelize_run(nan_df, append_item_to_uid)
    df_cleaned = df_cleaned[~df_cleaned['uid'].isna()]

    df_cleaned.index = df_cleaned['TransactionID']
    temp_dict = df_cleaned['uid'].to_dict()
    nan_df['uid'] = nan_df['TransactionID'].map(temp_dict)
    nan_df = nan_df[~nan_df['uid'].isna()]
    print('Assigned items:', len(nan_df))

    # append found items
    full_df = pd.concat([full_df,nan_df]).sort_values(by='TransactionID').reset_index(drop=True)
    
    for i in range(100):
        bad_uids_groups = sanity_check_run(full_df, False)
        if len(bad_uids_groups)==0:
            break
        elif len(bad_uids_groups)>64:
            bad_uids_items = df_parallelize_run(bad_uids_groups, parallel_check)
        else:
            bad_uids_items = parallel_check(bad_uids_groups)

        print('Found bad items', len(bad_uids_items))
        full_df['uid'] = np.where(full_df['TransactionID'].isin(bad_uids_items['uid']), np.nan, full_df['uid'])
        full_df = full_df[~full_df['uid'].isna()].sort_values(by='TransactionID').reset_index(drop=True)
        if len(bad_uids_items)<2:
            break
    check_state()


# In[ ]:


full_df[['TransactionID','uid']].to_csv('uids_part_1_v6.csv')


# In[ ]:





# In[ ]:


def find_multigroup(df):
    new_uids_items = {'TransactionID': [],
                      'multi_uid': [],
                      }

    for i in range(len(df)):
        item = df.iloc[i]
        if item['TransactionID'] not in problem_items['TransactionID']:
            mask_1 = full_df['card1'] == item['card1']
            mask_2 = full_df['uid_td_D1'] == item['uid_td_D1']
            mask_3 = full_df['TransactionID'] < item['TransactionID']
            mask_4 = ((full_df['DT_day'] == item['uid_td_D3'] + 1)|
                      (full_df['DT_day'] == item['uid_td_D3'] - 1)|
                      (full_df['DT_day'] == item['uid_td_D3']))

            df_masked = full_df[mask_1 & mask_2 & mask_3 & mask_4]
            has_match = len(df_masked) > 0

            if has_match:
                new_uids_items['TransactionID'].append(item['TransactionID'])
                new_uids_items['multi_uid'].append(list(df_masked['uid'].unique()))
 
    return_df = pd.DataFrame.from_dict(new_uids_items)
    return return_df

def find_and_filter_groups(df):
    filtered_groups = []

    for i in range(len(df)):
        test_id = df.iloc[i]['TransactionID']
        test_item = all_items[all_items['TransactionID']==test_id].iloc[0]
        possible_groups = df.iloc[i]['multi_uid']
        clean_group = find_right_uid(possible_groups, test_item)
        filtered_groups.append([test_id, clean_group])
    filtered_groups = pd.DataFrame(filtered_groups, columns=['TransactionID','uid'])    
    return filtered_groups

import operator

def find_right_uid(possible_groups, test_item):
    separated_uids = {}

    test_features_set1 = {
        'TransactionAmt':2,
        'card2':1,
        'card3':1,
        'card4':1,
        'card5':1,
        'card6':1,
        'uid_td_D2':2,
        'uid_td_D10':2,
        'uid_td_D11':2,
        'uid_td_D15':2,
        'C14':1,
        'addr1':1,
        'addr2':1,
        'P_emaildomain':1,
        'V313_fix':1,
        }

    groups_score = {}

    for possible_group in possible_groups:
        masked_df = full_df[full_df['uid']==possible_group]
        cur_score = 0
        for col in test_features_set1:
            if test_item[col] in list(masked_df[col]):
                cur_score += test_features_set1[col]
        
        for col in v_cols:
            if test_item[col]!=0:
                if test_item[col+'_fix_ground'] in list(masked_df[col+'_fix']):
                    cur_score += 1
                    
        check_if_match = masked_df.append(test_item)
        check_if_match = sanity_check_run(check_if_match)
        if len(check_if_match)==0:
            groups_score[possible_group] = cur_score
        
    new_uid = np.nan
    try:
        new_uid = max(groups_score.items(), key=operator.itemgetter(1))[0]
    except:
        pass
    return new_uid


# In[ ]:


########################### PART X - > With multigroup check
nan_df = all_items[~all_items['TransactionID'].isin(full_df['TransactionID'])]
print('Items to check:', len(nan_df))

df_cleaned = df_parallelize_run(nan_df, find_multigroup)
df_cleaned = df_cleaned[~df_cleaned['multi_uid'].isna()]

filtered_groups = df_parallelize_run(df_cleaned, find_and_filter_groups)
filtered_groups.index = filtered_groups['TransactionID']
temp_dict = filtered_groups['uid'].to_dict()
nan_df['uid'] = nan_df['TransactionID'].map(temp_dict)
nan_df = nan_df[~nan_df['uid'].isna()]
print('Assigned items:', len(nan_df))

# append found items
full_df = pd.concat([full_df,nan_df]).sort_values(by='TransactionID').reset_index(drop=True)
check_state()


# In[ ]:


for i in range(100):
    bad_uids_groups = sanity_check_run(full_df, False)
    if len(bad_uids_groups)==0:
        break
    elif len(bad_uids_groups)>64:
        bad_uids_items = df_parallelize_run(bad_uids_groups, parallel_check)
    else:
        bad_uids_items = parallel_check(bad_uids_groups)

    print('Found bad items', len(bad_uids_items))
    full_df['uid'] = np.where(full_df['TransactionID'].isin(bad_uids_items['uid']), np.nan, full_df['uid'])
    full_df = full_df[~full_df['uid'].isna()].sort_values(by='TransactionID').reset_index(drop=True)
    if len(bad_uids_items)<2:
        break
check_state()


# In[ ]:





# In[ ]:


print('Start items:', len(bkp_items))
print('Start items Frauds:', bkp_items['isFraud'].sum())


# In[ ]:


print('Uids Items:', len(single_items)+len(full_df))
print('Uids Frauds:', full_df['isFraud'].sum() + single_items['isFraud'].sum())


# In[ ]:


full_df_final = pd.concat([full_df, 
                     single_items, 
                     global_bad_items,
                     all_items[~all_items['TransactionID'].isin(full_df['TransactionID'])]
                    ])


# In[ ]:


print('Combined Uids:', len(full_df_final))
print('CombinedUids Frauds:', full_df_final['isFraud'].sum())


# In[ ]:


full_df['count'] = full_df.groupby(['uid'])['TransactionID'].transform('count')
full_df['count'].mean()


# In[ ]:


len(full_df_final['uid'].unique())


# In[ ]:


check_state()


# In[ ]:





# ----
# ### Export

# In[ ]:


########################### Export
full_df_final[['TransactionID','uid']].to_csv('uids_full_v6.csv')

