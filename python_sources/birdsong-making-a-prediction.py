#!/usr/bin/env python
# coding: utf-8

# In this notebook we will import the test data and make a simple prediction.

# In[ ]:


import numpy as np
import pandas as pd
import librosa

np.random.seed(0)


# There are several pieces of information that we are given about the competition and the test set. Two of the important pieces are quoted below.
# 
# The following can be found on the [evaluation page](https://www.kaggle.com/c/birdsong-recognition/overview/evaluation):
# > Submissions will be evaluated based on their row-wise micro averaged F1 score.
# > 
# > For each row_id/time window, you need to provide a space separated list of the set of birds that made a call beginning or ending in that time window. If there are no bird calls in a time window, use the code nocall.
# > 
# > There are three sites in the test set. Sites 1 and 2 are labeled in 5 second increments, while site 3 was labeled per audio file due to the time consuming nature of the labeling process.
# 
# This explains how we need to structure our submission file. There will be several submissions for each test audio file, split by time windows (5 seconds) - unless they are from site 3.
# 
# The following can be found on the [data page](https://www.kaggle.com/c/birdsong-recognition/data):
# >The hidden test set audio consists of approximately 150 recordings in mp3 format, each roughly 10 minutes long. The recordings were taken at three separate remote locations. Sites 1 and 2 were labeled in 5 second increments and need matching predictions, but due to the time consuming nature of the labeling process the site 3 files are only labeled at the file level. Accordingly, site 3 has relatively few rows in the test set and needs lower time resolution predictions.
# >
# >Two example soundscapes from another data source are also provided to illustrate how the soundscapes are labeled and the hidden dataset folder structure. The two example audio files are BLKFR-10-CPL_20190611_093000.pt540.mp3 and ORANGE-7-CAP_20190606_093000.pt623.mp3. These soundscapes were kindly provided by Jack Dumbacher of the California Academy of Science's Department of Ornithology and Mammology.
# 
# This is just further information on the test data.

# # Loading Audio
# 
# Firstly, we need to be able to read in five second windows of the test audio. We can do this using librosa. If the audio is from site 3 then we need to whole audio clip, and we can do this by setting duration to None.

# In[ ]:


def load_test_clip(path, start_time, duration=5):
    return librosa.load(path, offset=start_time, duration=duration)[0]


# # Getting Test Data
# 
# The information on the test audio is given in test.csv. We have outputted that below. The test audio is also contained in the test_audio folder.

# In[ ]:


TEST_FOLDER = '../input/birdsong-recognition/test_audio/'
test_info = pd.read_csv('../input/birdsong-recognition/test.csv')
test_info.head()


# # Possible Birds
# 
# The possible birds can be found in the training set, with the ebird_code feature. Almost all are six letter codes. The first twenty have been outputted below.

# In[ ]:


train = pd.read_csv('../input/birdsong-recognition/train.csv')
birds = train['ebird_code'].unique()
birds[0:20]


# # Making Predictions
# 
# With all of the information we have found so far, it is now possible for us to make predictions. This can be done in different ways depending on your model - for this example we will just be selecting random birds. For each row:
# 
# 1. Extract the information from test.csv  
# 2. Load in the correct clip (using librosa)  
# 3. Make a prediction  
# 4. Store the prediction  

# In[ ]:


def make_prediction(sound_clip, birds):
    return np.random.choice(birds)


# In[ ]:


try:
    preds = []
    for index, row in test_info.iterrows():
        # Get test row information
        site = row['site']
        start_time = row['seconds'] - 5
        row_id = row['row_id']
        audio_id = row['audio_id']

        # Get the test sound clip
        if site == 'site_1' or site == 'site_2':
            sound_clip = load_test_clip(TEST_FOLDER + audio_id + '.mp3', start_time)
        else:
            sound_clip = load_test_clip(TEST_FOLDER + audio_id + '.mp3', 0, duration=None)

        # Make the prediction
        pred = make_prediction(sound_clip, birds)

        # Store prediction
        preds.append([row_id, pred])

    preds = pd.DataFrame(preds, columns=['row_id', 'birds'])
except:
    preds = pd.read_csv('../input/birdsong-recognition/sample_submission.csv')


# # Outputting Submission

# In[ ]:


preds.to_csv('submission.csv', index=False)

