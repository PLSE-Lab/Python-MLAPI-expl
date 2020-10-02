#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reset', '-sf')
get_ipython().system('ls ../input/shopee-product-detection-student/')


# In[ ]:


import pandas as pd
import numpy as np
submission_df = pd.read_csv("../input/shopee-product-detection-student/test.csv")


# # Padding zeroes
# 
# For prediction of single-digit classes, please pad a zero. Otherwise, you will receive zero for those predictions.
# 
# You can use the following one line code `.apply(lambda x: "{:02}".format(x))` to pad zeroes.
# 
# Do also refer to the following disucssion https://www.kaggle.com/c/shopee-product-detection-student/discussion/161311

# In[ ]:


submission_df["category"] = submission_df["category"].apply(lambda x: np.random.randint(10))  # 00 to 41 inclusive
submission_df["category"] = submission_df["category"].apply(lambda x: "{:02}".format(x))  # pad zeroes
submission_df.to_csv("submission.csv", index=False)


# In[ ]:


get_ipython().system('head -10 submission.csv')


# # Number of test cases
# 
# Please predict for the filenames listed in test.csv, for all the files in the test folder.
# 
# Otherwise, you will receive a error during submission. If that happens, a submission count will not be wasted.
# 
# Do also refer to the following discussion https://www.kaggle.com/c/shopee-product-detection-student/discussion/161369

# In[ ]:


# count number of files in directory
get_ipython().system('find "../input/shopee-product-detection-student/test/test/test/" -type f | wc -l')


# In[ ]:


# count the number of lines in a submission
get_ipython().system('wc -l submission.csv')


# In[ ]:


import glob
dir_files = [file.split("/")[-1][:-4] for file in glob.glob("../input/shopee-product-detection-student/test/test/test/*.jpg")]
test_files = [file[:-4] for file in submission_df["filename"]]


# In[ ]:


set(dir_files) - set(test_files)


# In[ ]:




