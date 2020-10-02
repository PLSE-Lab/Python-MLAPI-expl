#!/usr/bin/env python
# coding: utf-8

# # Analysis of the Submission.csv 
# By running this script you can visualize how your model performed in the ```test_videos``` folder.
# 
# The resources needed:
# 
# - A dataset that contains ```submission.csv```. I have included a sample submission here.
# 
# - A metadata file to get the ground truth of the videos in the ```test_videos``` folder. It is possible because the videos inside this folder is a subset of the training folders provided in the competition. I am using metadata dataset provided in https://www.kaggle.com/calebeverett/metadata-dataframe.
# 
# Please upvote if you like it! Thanks :)
# 
# ### <span style="color:red">PLEASE NOTE:</span> 
# Of course, the model is overfitting in the test_videos folder, as it is a subset of full training data. But, this can help in comparing performance of two models (in subtle ways), not for a single model. This script is mere a tool for having more insights, not a sure-shot "metric" to determine a good model :)
# 
# 
# 

# ## Import Packages

# In[ ]:


import os, sys, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm


# ## Read Data

# ### Metadata

# In[ ]:


df_MetaData = pd.read_csv('../input/train-set-metadata-for-dfdc/metadata')
df_MetaData.head()


# ### Sample submission

# In[ ]:


sample_submission = pd.read_csv("../input/sampledeepfakesubmissioncsv/sample_submission.csv")
sample_submission.head()


# ## Generate Dataframe with Predictions and Ground Truth Labels

# In[ ]:


def label_convert(label):
    if label=="REAL":
        return 0
    else:
        return 1

test_dir = "/kaggle/input/deepfake-detection-challenge/test_videos/"
test_videos = sorted([x for x in os.listdir(test_dir) if x[-4:] == ".mp4"])

df_TestData = pd.DataFrame(columns=['label', 'prediction'])
for file_id in tqdm(test_videos):
    df_TestData.loc[file_id] = [label_convert(list(df_MetaData[df_MetaData.filename==file_id].label)[0]),list(sample_submission[sample_submission.filename==file_id].label) [0]
]
    
df_TestData.head()


# ## Visualize How Well Your Model Performed

# Now, you can visualize your predictions in a histogram. 
# 
# The blue bars show your REAL predictions, and orange bars show FAKE predictions. For a perfect prediction, all blue bars should be in '0' and all orange bars should be in '1'.

# In[ ]:


df_Real =df_TestData[df_TestData.label==0]
df_Fake =df_TestData[df_TestData.label==1]

data = list(df_Real.prediction)
count = np.histogram(data)[0]
plt.hist(data,50)

data = list(df_Fake.prediction)
count = np.histogram(data)[0]
plt.hist(data,50)

plt.legend(['REAL', 'FAKE'])

plt.axis([0,1,0,140])
plt.show()


# ## Estimate log_loss

# In[ ]:


from sklearn.metrics import log_loss
LOG_LOSS = log_loss(list(df_TestData.label),list(df_TestData.prediction))
print("Log loss in the test folder is: " + str(LOG_LOSS))


# ## Estimate Confusion Matrix

# In[ ]:


from sklearn.metrics import confusion_matrix
df_CM = df_TestData.copy()
df_CM.loc[df_CM.prediction>0.5,'prediction']=1
df_CM.loc[df_CM.prediction<=0.5,'prediction']=0

CONFUSION_MATRIX = confusion_matrix(list(df_TestData.label),list(df_CM.prediction))
print("Confusion Matrix is:\n" + str(CONFUSION_MATRIX))

