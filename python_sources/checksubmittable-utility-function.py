#!/usr/bin/env python
# coding: utf-8

# # Simple function to test if Dataframe is ready for submission
# 
# I trimmed down a full run to just be an outline of a run, but mostly to show the function that test submissions, there are some other helpful items here too.
# 
# ## After a failed Leader Board submission due to a simple error, I decided I needed a function to let me know if I was about to waste a submission with a dataframe with an easy error.
# 
# ## So I wrote the function below for myself, but hopefully it will help you too.  Give an upvote if you like it.
# 
# - Mark

# In[ ]:


import pandas as pd
import glob
import os
import tqdm.notebook as tqdm
import json
import numpy as np


# # Function that should catch most submit errors and let you know.
# - if bChange is True for a Submit for Leader Board, it will show a fixed score if the database is not ready for submission (it will waste a submission, but may identify the issue for you.)  The score should be over 8, something like 9 for 0.3,10 for 0.2,12 for 0.1 for each error condition, I haven't done the math, so may be higher.  This should flag you for which type of error you have even if you don't get a print.  

# In[ ]:


# Need to have 2 col, filename and label, eg: "fsjdfal.mp4", "0.5678"
# Checks total length, matching filenames, in sorted order and predictions in 0-1 range.
def checkSubmittable(data, bChange = True):
    if bChange == False:
        data_orig = data.copy()
    nReturn = 0
    test_dir = "/kaggle/input/deepfake-detection-challenge/test_videos/"
    test_videos = sorted([x for x in os.listdir(test_dir) if x[-4:] == ".mp4"])
    nVids = len(test_videos)
    nPredVids = len(data['filename'])
    if (nVids != nPredVids):
        print("ERROR - Dataframe Size mismatch")
        data = data.iloc[0:0] # erase df
        data['filename'] = test_videos
        data['label'] = 0.3 # If in LB score will be obvious
        nReturn = -1
    for i in range(nVids):
        strFn = data.at[i,'filename']
        fPred = data.at[i,'label']
        if (strFn != test_videos[i] ):
            print("Filename mismatch, either not ordered or not matching expected!")
            data['filename'] = test_videos
            data['label'] = 0.1 # If in LB score will be obvious
            nReturn = -1
        if (fPred < 0.0 ) or (fPred > 1.0):
            print("Prediction value out of range!")
            data['filename'] = test_videos
            data['label'] = 0.2 # If in LB score will be obvious
            nReturn = -1
    if nReturn < 0:
        print("************************************************************")
        print("********** SUMBIT ERROR - Pred = .1 .2 .3 ******************")
        print("************************************************************")
    else:
        print("Pred. Data looks OK, Files = ", nVids)
        
    if bChange == False:
        return data_orig   
    
    return data.reset_index(drop=True)


# In[ ]:


filenames_train = glob.glob('/kaggle/input/deepfake-detection-challenge/train_sample_videos/*.mp4')


# In[ ]:


labels = json.load(open('/kaggle/input/deepfake-detection-challenge/train_sample_videos/metadata.json', encoding="utf8"))

labels = pd.DataFrame(labels).transpose()
labels = labels.reset_index()


# In[ ]:


dictFake = { 'REAL':0, 'FAKE':1}


# In[ ]:



labels['bFake'] = 0
nCnt = 0
for sub in labels['label']:
    labels.at[nCnt,'bFake'] = int(dictFake[sub])
    nCnt += 1

labels.head()


# In[ ]:


labels.label.value_counts()


# # Make predictions

# In[ ]:



test_dir = "/kaggle/input/deepfake-detection-challenge/test_videos/"
filenames = sorted([x for x in os.listdir(test_dir) if x[-4:] == ".mp4"]) # np.zeros((5,), dtype=int)
filenames_full = filenames.copy()
for i in range(len(filenames_full)):
    filenames_full[i] = os.path.join(test_dir, filenames_full[i] )
predictions = np.zeros(len(filenames,),dtype=float)


# In[ ]:



submission_df = pd.DataFrame({"filename": filenames, "label": predictions})

submission_df.label = 0.48

submission_df.shape


# # TEST CODE

# In[ ]:


def isVideoFake(filename):
    return False


# In[ ]:


# filenames_train
# labels is the training dataframe data
nCount1=0
#labels['bFake']
for filename in tqdm.tqdm(filenames_train): ## train data
    
    fn = filename.split('/')[-1]
    if (nCount1 < 999999):
        bFakeVideo = isVideoFake(filename)
    else:
        bFakeVideo = False
    
    if bFakeVideo == True:
        labels.loc[labels['index']==fn, 'label'] = 0.80
    else:
        labels.loc[labels['index']==fn, 'label'] = 0.48

    nCount1 += 1


# In[ ]:


#labels.head(50)
print(labels[0:10])


# In[ ]:


nCount=0
## run Test Here
for filename in tqdm.tqdm(filenames_full):
    
    fn2 = filename.split('/')[-1]
    if (nCount < 999999):
        bFakeVideo = isVideoFake(filename)
    else:
        bFakeVideo = False
    
    if bFakeVideo == True:
        submission_df.at[nCount, 'label'] = 0.80
        print("file Fake = ", fn)
    else:
        submission_df.at[nCount, 'label'] = 0.48       

    nCount += 1


# In[ ]:


submission_df.label.value_counts()


# # Check if submittable before exporting to file

# In[ ]:


sub = checkSubmittable(submission_df, bChange=True)


# # Format for Submission

# In[ ]:


sub.to_csv('submission.csv', index=False)


# # Print to verify it looks good

# In[ ]:


print(sub)


# In[ ]:


print("DONE")

