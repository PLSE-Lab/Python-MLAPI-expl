#!/usr/bin/env python
# coding: utf-8

# As stated in the data description we cannot acces the full test set in this format of the competiton
# > This is a synchronous rerun code competition, you can assume that the complete test set will contain essentially the same size and number of images as the training set. Consider performing inference on just one batch at a time to avoid memory errors. Only the first few rows/images in the test set and sample submission files can be downloaded. These samples provided so you can review the basic structure of the files and to ensure consistency between the publicly available set of file names and those your code will have access to while it is being rerun for scoring.

# You should structure your code so that it returns predictions for the partial test set images (12 images available in .parquet format) in the format specified by the public sample_submission.csv (which has id for the 12 images), but does not hard code aspects like the id or number of rows. When Kaggle runs your Kernel privately, it substitutes the partial test set with the full version and evaluates the model on that. This is done to prevent test set manipulation.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


for i in range(4):
    df_test_img = pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_{}.parquet'.format(i))
    print(df_test_img.shape)


# As you can see the publicly available test set has 3 images (the image pixels are in the rows) in each of the 4 parquet files. Total 3x4=12

# In[ ]:


df_test = pd.read_csv('/kaggle/input/bengaliai-cv19/test.csv')
df_test.shape


# And the test.csv have 3x12=36 labels

# Let's also look at the first few entries in the image dataframe and label dataframe to get acquainted with the data structure

# In[ ]:


print('Data')
display(df_test_img.head())
print('label')
display(df_test.head())


# So, to write the inference function, you have to make sure the .csv structure is similar to the given test.csv but it has to account for the fact that there are more then 12 images.

# Here's one way to do that

# In[ ]:


components = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']
target=[] # model predictions placeholder
row_id=[] # row_id place holder
n_cls = [7,168,11] # number of classes in each of the 3 targets
for i in range(4):
    df_test_img = pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_{}.parquet'.format(i)) # read image data
    df_test_img.set_index('image_id', inplace=True) # set image_id as index value
    for id in df_test_img.index.values: # df_test_img.index.values has the test image_ids 
        for i,comp in enumerate(components):
            id_sample=id+'_'+comp
            row_id.append(id_sample)
            target.append(np.random.randint(0,n_cls[i])) # our model is a random integer generator between 0 and n_cls


# In[ ]:


# create a dataframe with the solutions 
df_sample = pd.DataFrame(
    {'row_id': row_id,
    'target':target
    },
    columns =['row_id','target'] 
)
df_sample.head()


# In[ ]:


# create submission file
df_sample.to_csv('submission.csv',index=False)


# After creating the submit file, click the `commit` button on the upper right corner and let it complete. It will run the whole code and save stable version of it.
# <img src="https://i.imgur.com/aACt0QZ.png" alt="Smiley face" align="center" width="400" height="500">
# 

# Click the `Open Version` button.
# <img src="https://i.imgur.com/nvEOQVR.png" alt="Smiley face" align="center" width="700" height="900">

#  and navigate back to your commited notebook. 
# <img src="https://i.imgur.com/P71iGHj.png" alt="Smiley face" align="center" width="700" height="900">

# Scroll down to the bottom and you will see the option to `submit to competition` button.
# <img src="https://i.imgur.com/j56xJLW.png" alt="Smiley face" align="center" width="700" height="500">

#  Clicking that button will do multiple things. It will replace the partial test set dataset with the full version, rerun the kernel and evaluate the metric.
#  <img src="https://i.imgur.com/5HzYeiC.png" alt="Smiley face" align="center" width="400" height="700">

# This is might take time as there are around 200k test images. After the computation is finished the result will be shown. It takes around 10-15 minutes for me. Sometimes the computation is finished but the running status is still there which is a bit annoying. Refresh the page regularly.
