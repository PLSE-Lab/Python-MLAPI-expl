#!/usr/bin/env python
# coding: utf-8

# This step actually 'inspired' by the fashion data leakage scenario. It's basically k-nearest neighbour with distance zero (ignoring image data). The idea is simple. Given a test data, if the **exact same title** exist in the train data, should the predictions be the same? 
# 
# The simple answer is yes and no.
# 
# There are possibilities that train data with the exact same titles can have different attributes (e.g. one '**iphone 6s for sale**' color=black, the other '**iphone 6s for sale**', color=white). So I am only considering the case where the train data have the exact same title AND same attributes (non-NaN) value.
# 
# Of course this is not a foolproof method, because the train data can contain only all the black color 'iphone 6s for sale' and the test data 'iphone 6s for sale' is in fact a **white** one with a **white** iphone image that was ignored.
# 
# Anyway, I used this discovery to overwrite [previous result](https://www.kaggle.com/szelee/aoeul-solution-step-3-linearsvc-dl-model) and gain a few extra points:
#     - 0.46814 -> 0.46823 (Public Leaderboard)
#     - 0.46673 -> 0.46681 (Private Leaderboard)
#     
# Disclaimer, the codes are not very pretty as I've only managed to mash it up towards the end of the competition.

# In[ ]:


from pathlib import Path
import json
import re
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook


# In[ ]:


DATA_DIR = Path('../input/ndsc-advanced')

BEAUTY_TRAIN_CSV = DATA_DIR / 'beauty_data_info_train_competition.csv'
FASHION_TRAIN_CSV = DATA_DIR / 'fashion_data_info_train_competition.csv'
MOBILE_TRAIN_CSV = DATA_DIR / 'mobile_data_info_train_competition.csv'

BEAUTY_TEST_CSV = DATA_DIR / 'beauty_data_info_val_competition.csv'
FASHION_TEST_CSV = DATA_DIR / 'fashion_data_info_val_competition.csv'
MOBILE_TEST_CSV = DATA_DIR / 'mobile_data_info_val_competition.csv'

# put your last best submission file here
LAST_SUBMITTED_CSV = Path('../input/aoeul-solution-step-3-linearsvc-dl-model/Ensembled_SVC_DL_predictions.csv')


# In[ ]:


beauty_train_df = pd.read_csv(BEAUTY_TRAIN_CSV)
fashion_train_df = pd.read_csv(FASHION_TRAIN_CSV)
mobile_train_df = pd.read_csv(MOBILE_TRAIN_CSV)

beauty_test_df = pd.read_csv(BEAUTY_TEST_CSV)
fashion_test_df = pd.read_csv(FASHION_TEST_CSV)
mobile_test_df = pd.read_csv(MOBILE_TEST_CSV)

prev_subm_df = pd.read_csv(LAST_SUBMITTED_CSV)


# In[ ]:


categories = ['beauty', 'fashion', 'mobile']
train_dfs = [beauty_train_df, fashion_train_df, mobile_train_df]
test_dfs = [beauty_test_df, fashion_test_df, mobile_test_df]


# In[ ]:


matched_label_df = pd.DataFrame()

for cat, train_df, test_df in zip(categories, train_dfs, test_dfs):
    print(f'Matching {cat} training and test data...')
    df_train = train_df.rename(columns={'title': 'title_train', 'itemid': 'itemid_train'})
    df_test = test_df[['itemid', 'title']].rename(columns={'title': 'title_test', 'itemid': 'itemid_test'})

    # merge train and test data on the title attrib
    # there will be 1-to-1 and many-to-1 pairings, so we drop some columns and the duplicates
    df_combined = pd.merge(df_train, df_test, how='inner', left_on=['title_train'], right_on=['title_test'])
    df_combined.drop(['image_path','itemid_train', 'title_train'], axis=1, inplace=True)
    df_clean = df_combined.drop_duplicates()

    print('Finding test data with exact same title as train data with unique labels')
    single_instance=[]
    for title in tqdm_notebook(df_clean.title_test.unique()):
        if (len(df_clean[df_clean.title_test==title]) == 1): # 
            single_instance.append(title)
    print(f'{len(single_instance)} test-train title pair matches found in {cat} category')

    single_inst_df = df_clean[df_clean.title_test.isin(single_instance)]
    feat_cols = single_inst_df.columns.drop(['itemid_test', 'title_test'])

    # filter off the NaN values and generate a list of id, labels
    id_label_list=[]
    for _, row in single_inst_df.iterrows():
        for feat in feat_cols:
            if not np.isnan(row[feat]):
                itemid = '_'.join([str(row['itemid_test']), str(feat)])
                answer = str(int(row[feat]))
                id_label_list.append((itemid, answer))

    # get the first prediction of previous submission and compare to the train data label
    # since they have both the exact same title
    prev_subm_df['first_pred'] = prev_subm_df['tagging'].apply(lambda x: x.split(' ')[0])

    list_new_id = []
    list_new_ans = []

    for id_ans in tqdm_notebook(id_label_list):
        subm_first_label = int(prev_subm_df.loc[prev_subm_df.id==id_ans[0]]['first_pred'])
        if subm_first_label != int(id_ans[1]): 
            print(f'Test id: {id_ans[0]:28} Submitted label: {subm_first_label:<5} Train data label: {id_ans[1]}')
            list_new_id.append(id_ans[0])
            list_new_ans.append(id_ans[1])

    label_df = pd.DataFrame(
        {'id': list_new_id, 'tagging': list_new_ans},
        columns = ['id', 'tagging'])
    # concat all the DataFrame into one
    matched_label_df = pd.concat([matched_label_df, label_df], axis=0)
    print()


# In[ ]:


# Update last submission file
for _, row in tqdm_notebook(matched_label_df.iterrows()):
    (old_value,) = prev_subm_df.loc[prev_subm_df.id == row.id, 'tagging']
    # swap old first prediction with the one we got earlier, and shift it to be second prediction
    prev_subm_df.loc[prev_subm_df.id == row.id, 'tagging'] = str(row.tagging) + ' ' + str(old_value.split(' ')[0])
    
prev_subm_df.drop(['first_pred'], axis=1, inplace=True)
prev_subm_df.to_csv('Test_Matched_Train_predictions.csv', index=False)

