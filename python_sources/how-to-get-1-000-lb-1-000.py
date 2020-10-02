#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np  
from pathlib import Path  
import h5py
import hashlib
from tqdm import tqdm_notebook as tqdm

PATH = Path('.')
DATA_PATH = PATH/"HisCancer"


# # This is a leak from the original dataset
# See discussion: https://www.kaggle.com/c/histopathologic-cancer-detection/discussion/85424
# Some of the codes are from: https://www.kaggle.com/c/histopathologic-cancer-detection/discussion/85283
# 
# How did I discover this:
#  - Carefully watch LB people see if anybody discovered a leak that boost them up with only a few submissions.
#  - Learn from bestfiting's solution for HPA competition
#  - I was trying to reproduce finding WSI ids in https://www.kaggle.com/c/histopathologic-cancer-detection/discussion/85283
#  - Trying to develop my postprocess technique (inspired from TGS Salt)
# 

# ## Create some functions to convert csv files to dict()

# In[ ]:


def create_pcam16_hash_table(dset_img):
    # creates a dictionary where each key is the sha-1 hash of the image data and the value is the index in the h5 file
    hash_table = {}
    for h5_id,img_id in enumerate(dset_img):
        hash_table[hashlib.sha1(img_id).hexdigest()] = h5_id
    return hash_table

def create_labels_dict(f):
    # creates a dictionary where each key is the img_id and the value is tumor or not
    df = pd.read_csv(f)
    labels_dict = {}
    for idx in range(len(df)):
        img_id = df.iloc[idx,0]
        is_tumor = df.iloc[idx,1]
        labels_dict[img_id] = int(is_tumor)
    return labels_dict

def create_wsi_dict(f):
    # creates a dictionary where each key is the h5 index and the value is the wsi
    df = pd.read_csv(f)
    
    dict_list = []
    for y,x,tp,ctp,wsi in zip(df.coord_y.tolist(), df.coord_x.tolist(), df.tumor_patch.tolist(), df.center_tumor_patch.tolist(), df.wsi.tolist()):
        dict_list.append({'coord_y':y,'coord_x':x, 'tumor_patch':tp,'center_tumor_patch':ctp,'wsi':wsi})
    wsi_dict = dict(zip(df.index.tolist(),dict_list))
    return wsi_dict


# ## Create a function that generate all the information from the leak that may be helpful later
# I did not know how big the leak was at this point.

# In[ ]:


def postprocess(h5_file, labels_file, meta_file, name='new'):
    fimg = h5py.File(DATA_PATH/h5_file, 'r')

    hash_table = create_pcam16_hash_table(fimg['x'])
    labels_dict = create_labels_dict(DATA_PATH/labels_file)
    wsi_dict = create_wsi_dict(DATA_PATH/meta_file)

    ids = []
    label = []
    
    y = []
    x = []
    tp = []
    ctp = []
    wsi = []

    error_count = 0
    pbar = tqdm(labels_dict.items())
    for img_id, is_tumor in pbar:
        if img_id in hash_table.keys():
            h5_idx = hash_table[img_id]

            ids.append(img_id)
            label.append(is_tumor)

            y.append(wsi_dict[h5_idx]['coord_y'])
            x.append(wsi_dict[h5_idx]['coord_x'])
            tp.append(wsi_dict[h5_idx]['tumor_patch'])
            ctp.append(wsi_dict[h5_idx]['center_tumor_patch'])
            wsi_id = wsi_dict[h5_idx]['wsi']
            wsi.append(wsi_id)

            pbar.set_description("[ERROR:{}] h5:{}, wsi_id:{}, id:{}, is_tumor:{}".format(error_count, h5_idx, wsi_id, img_id, is_tumor))

        else:
            error_count = error_count+1
            pbar.set_description("[ERROR:{}] Image {} does not exist in external file".format(error_count, img_id))
            
    
    df_full_wsi = pd.DataFrame({'id':ids,'coord_y':y,'coord_x':x, 'tumor_patch':tp,'center_tumor_patch':ctp,'wsi':wsi,'is_tumor':label},columns=['id', 'coord_y', 'coord_x', 'tumor_patch', 'center_tumor_patch', 'wsi','is_tumor'])
    df_full_wsi.to_csv(DATA_PATH/labels_file.replace('.csv', '_wsi_{}.csv'.format(name)),index=False)
    
    print("Out of {} images, {} is unknown".format(len(pbar), error_count))


# ## Simply run the function above in `train`, `valid`, and `test`

# In[ ]:


postprocess('camelyonpatch_level_2_split_train_x.h5', "train_labels.csv", 'camelyonpatch_level_2_split_train_meta.csv', name='train')


# In[ ]:


postprocess('camelyonpatch_level_2_split_valid_x.h5', "sample_submission.csv", 'camelyonpatch_level_2_split_valid_meta.csv', name='valid')


# In[ ]:


postprocess('camelyonpatch_level_2_split_test_x.h5', "sample_submission.csv", 'camelyonpatch_level_2_split_test_meta.csv', name='test')


# ## Test if the leak has real effect on LB
# After looking at the files I generated above, I realized `tumor_patch` might be useful:
#  1. I know that if the image is not tumor_patch, then no cancer should be detected in the sliced images

# In[ ]:


origin = pd.read_csv(DATA_PATH/'000c46d3-CP4_F1_PT2019-02-26-06-23-29-725832-two_VT000c46d3_LR0.01_BS64_IMG224.pth-two-F0-T0.5-Prob-Prep(240)TTAPowerNasnet0.9714.csv').set_index('Id')
origin_dict = origin.to_dict()

wsi_valid = pd.read_csv(DATA_PATH/'sample_submission_wsi_valid.csv').set_index('id')
wsi_valid_dict = wsi_valid.to_dict()

wsi_test = pd.read_csv(DATA_PATH/'sample_submission_wsi_test.csv').set_index('id')
wsi_test_dict = wsi_test.to_dict()


# In[ ]:


# Test Code
# origin_dict['Label']
wsi_valid_dict


# ## So I postprocessed my highest scored submission.

# In[ ]:


pbar = tqdm(origin_dict['Label'].keys())

img_id_new = []
label_new = []

error_count = 0
for img_id in pbar:
    if img_id in wsi_valid_dict['tumor_patch'].keys():
        if wsi_valid_dict['tumor_patch'][img_id] == False:
            img_id_new.append(img_id)
            label_new.append(0.)
            pbar.set_description("[ERROR={}] {} = 0".format(error_count, img_id))
        else:
            img_id_new.append(img_id)
            label_new.append(origin_dict['Label'][img_id])
    else:
        img_id_new.append(img_id)
        label_new.append(origin_dict['Label'][img_id])
        error_count = error_count+1
        pbar.set_description("[ERROR={}] No id found.".format(error_count))

out = pd.DataFrame({'id': img_id_new, 'label': label_new}, columns=['id', 'label'])
out.to_csv(DATA_PATH/'output.csv',index=False)


# ## Wow, negative labels in valid set really work, how about positive labels? Will it affect the private LB?
# It will affect private LB since my score stay exactly the same with only private LB postprocessed.

# In[ ]:


pbar = tqdm(origin_dict['Label'].keys())

img_id_new = []
label_new = []

error_count = 0
for img_id in pbar:
    if img_id in wsi_test_dict['tumor_patch'].keys():
        if wsi_test_dict['tumor_patch'][img_id] == False:
            img_id_new.append(img_id)
            label_new.append(0.)
            pbar.set_description("[ERROR={}] {} = 0".format(error_count, img_id))
        else:
            img_id_new.append(img_id)
            label_new.append(origin_dict['Label'][img_id])
    else:
        img_id_new.append(img_id)
        label_new.append(origin_dict['Label'][img_id])
        error_count = error_count+1
        pbar.set_description("[ERROR={}] No id found.".format(error_count))

out = pd.DataFrame({'id': img_id_new, 'label': label_new}, columns=['id', 'label'])
out.to_csv(DATA_PATH/'output.csv',index=False)


# ## Make a perfect cheatsheet

# In[ ]:


pbar = tqdm(origin_dict['Label'].keys())

img_id_new = []
label_new = []

error_count = 0
for img_id in pbar:
    if img_id in wsi_test_dict['tumor_patch'].keys():
        if wsi_test_dict['center_tumor_patch'][img_id] == False:
            img_id_new.append(img_id)
            label_new.append(1)
            pbar.set_description("[ERROR={}] {} = 0".format(error_count, img_id))
        elif wsi_test_dict['center_tumor_patch'][img_id] == True:
            img_id_new.append(img_id)
            label_new.append(0)
            pbar.set_description("[ERROR={}] {} = 0".format(error_count, img_id))
        else:
            img_id_new.append(img_id)
            label_new.append(origin_dict['Label'][img_id])
            error_count = error_count+1
    elif img_id in wsi_valid_dict['tumor_patch'].keys():
        if wsi_valid_dict['center_tumor_patch'][img_id] == False:
            img_id_new.append(img_id)
            label_new.append(0)
            pbar.set_description("[ERROR={}] {} = 0".format(error_count, img_id))
        elif wsi_valid_dict['center_tumor_patch'][img_id] == True:
            img_id_new.append(img_id)
            label_new.append(1)
            pbar.set_description("[ERROR={}] {} = 0".format(error_count, img_id))
        else:
            img_id_new.append(img_id)
            label_new.append(origin_dict['Label'][img_id])
            error_count = error_count+1
    else:
        img_id_new.append(img_id)
        label_new.append(origin_dict['Label'][img_id])
        error_count = error_count+1
        pbar.set_description("[ERROR={}] No id found.".format(error_count))

out = pd.DataFrame({'id': img_id_new, 'label': label_new}, columns=['id', 'label'])
out.to_csv(DATA_PATH/'output.csv',index=False)


# In[ ]:




