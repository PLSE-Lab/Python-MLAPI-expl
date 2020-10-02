#!/usr/bin/env python
# coding: utf-8

# # Overview
# According to this post from [Max Diebold](https://www.kaggle.com/c/airbus-ship-detection/discussion/62376), an empty submission can get you  a score of 0.847.
# 
# So, this baseline model will always predict that there is no ship.

# In[ ]:


import os
import pandas as pd 
ship_dir = '../input'
train_image_dir = os.path.join(ship_dir, 'train')
test_image_dir = os.path.join(ship_dir, 'test')


# # Submission

# In[ ]:


test_files =[f for f in os.listdir(test_image_dir) ]
submission_df = pd.DataFrame({'ImageId':test_files})
submission_df['EncodedPixels']=None
print (f"There are {len(test_files)} images in the test dataset")
submission_df.head()


# In[ ]:


submission_df.to_csv('submission.csv',index=False)


# In[ ]:




