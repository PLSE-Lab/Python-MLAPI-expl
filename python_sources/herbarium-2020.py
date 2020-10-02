#!/usr/bin/env python
# coding: utf-8

# # Peek
# 
# This notebook is here to just unify the dataset into one. I will perform further analysis and the Deep Learning algorithm in a future kernel. If you like this kernel, or forked this version, please upvote.
# 
# First step, we peek at the data paths:

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename.endswith('.jpg'):
            break
        print(os.path.join(dirname, filename))


# Since there's a lot of images included there, we only checked non-image files and got the three above. Next, we will load the sample submission and check.

# In[ ]:


sample_sub = pd.read_csv('../input/herbarium-2020-fgvc7/sample_submission.csv')
display(sample_sub)


# For the `*.json` files, we cannot load them to a DataFrame as there's two items that prevents this: `license` and `info`. So, I manually read the `*.json` files as follows:

# In[ ]:


import json, codecs
with codecs.open("../input/herbarium-2020-fgvc7/nybg2020/train/metadata.json", 'r',
                 encoding='utf-8', errors='ignore') as f:
    train_meta = json.load(f)
    
with codecs.open("../input/herbarium-2020-fgvc7/nybg2020/test/metadata.json", 'r',
                 encoding='utf-8', errors='ignore') as f:
    test_meta = json.load(f)


# In[ ]:


display(train_meta.keys())


# Now, we will be unifying the metadata from the `*.json` files. We will first work with the `train` data.

# First, we access the `annotations` list and convert it to a df.

# In[ ]:


train_df = pd.DataFrame(train_meta['annotations'])
display(train_df)


# Next is for `plant categories`:

# In[ ]:


train_cat = pd.DataFrame(train_meta['categories'])
train_cat.columns = ['family', 'genus', 'category_id', 'categort_name']
display(train_cat)


# Followed by the `image properties`:

# In[ ]:


train_img = pd.DataFrame(train_meta['images'])
train_img.columns = ['file_name', 'height', 'image_id', 'license', 'width']
display(train_img)


# And lastly, the `region`:

# In[ ]:


train_reg = pd.DataFrame(train_meta['regions'])
train_reg.columns = ['region_id', 'region_name']
display(train_reg)


# Then, we will merge all the DataFrames and see what we got:

# In[ ]:


train_df = train_df.merge(train_cat, on='category_id', how='outer')
train_df = train_df.merge(train_img, on='image_id', how='outer')
train_df = train_df.merge(train_reg, on='region_id', how='outer')


# In[ ]:


print(train_df.info())

display(train_df)


# Looking closer, there's a line with `NaN` values there. We need to remove rows with `NaN`s so we proceed to the next line:

# In[ ]:


na = train_df.file_name.isna()
keep = [x for x in range(train_df.shape[0]) if not na[x]]
train_df = train_df.iloc[keep]


# After selecting the `non-NaN` items, we now reiterate on their file types. We need to save on memory, as we reached `102+ MB` for this DataFrame Only.

# In[ ]:


dtypes = ['int32', 'int32', 'int32', 'int32', 'object', 'object', 'object', 'object', 'int32', 'int32', 'int32', 'object']
for n, col in enumerate(train_df.columns):
    train_df[col] = train_df[col].astype(dtypes[n])
print(train_df.info())
display(train_df)


# Finally, for our `test` dataset. Since it only contains one key, `images`:

# In[ ]:


test_df = pd.DataFrame(test_meta['images'])
test_df.columns = ['file_name', 'height', 'image_id', 'license', 'width']
print(test_df.info())
display(test_df)


# Perfect!
# 
# Now, we can go ahead and save this dataframe as a `*.csv` file for future use!

# In[ ]:


#train_df.to_csv('full_train_data.csv', index=False)
#test_df.to_csv('full_test_data.csv', index=False)


# # Last Steps
# 
# Before we end this kernel, let's check the total number of targets for this dataset:

# In[ ]:


print(len(train_df.category_id.unique()))


# It's a shocking `32,093` unique targets! I can't think of how to approach this to simplify the data so for now, let's end it here!

# # Submission
# 
# Let's create a submission file to check the format! I'll be using a random number generator for the targets!

# In[ ]:


sub = pd.DataFrame()
sub['Id'] = test_df.image_id
sub['Predicted'] = list(map(int, np.random.randint(1, 32093, (test_df.shape[0]))))
display(sub)
sub.to_csv('submission.csv', index=False)

