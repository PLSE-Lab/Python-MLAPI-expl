#!/usr/bin/env python
# coding: utf-8

# ## Note
# 
# * Process train images (part 1 of 4) : https://www.kaggle.com/ilosvigil/shopee-competition-2-ocr?scriptVersionId=37547674
# * Process train images (part 2 of 4) : https://www.kaggle.com/ilosvigil/shopee-competition-2-ocr?scriptVersionId=37573844
# * Process train images (part 3 of 4) : https://www.kaggle.com/ilosvigil/shopee-competition-2-ocr?scriptVersionId=37601445
# * Process train images (part 4 of 4) : https://www.kaggle.com/williammulianto/shopee-2-ocr-train-41b3f4?scriptVersionId=37614455
# * Process test images : https://www.kaggle.com/ilosvigil/shopee-competition-2-ocr?scriptVersionId=37631409

# # Library

# In[ ]:


get_ipython().system('pip install keras_ocr')


# In[ ]:


get_ipython().system('pip freeze > requirements.txt')


# In[ ]:


import matplotlib.pyplot as plt
import keras_ocr

import numpy as np
import pandas as pd

from time import time
import re
import itertools
import multiprocessing
import gc


# In[ ]:


print('Numpy version:', np.__version__)
print('Pandas version:', pd.__version__)


# # Dataset

# In[ ]:


get_ipython().system('ls /kaggle/input')


# In[ ]:


# df = pd.read_csv('/kaggle/input/shopee-product-detection-student/train.csv', dtype='object')
df = pd.read_csv('/kaggle/input/shopee-product-detection-student/test.csv', dtype='object')

df['category'] = df['category'].apply(lambda c: str(c).zfill(2))
df.head()


# In[ ]:


paths = []
for i in df.index:
#     paths.append(f'/kaggle/input/shopee-product-detection-student/train/train/train/{df.loc[i, "category"]}/{df.loc[i, "filename"]}')
    paths.append(f'/kaggle/input/shopee-product-detection-student/test/test/test/{df.loc[i, "filename"]}')


# # Config OCR

# In[ ]:


pipeline = keras_ocr.pipeline.Pipeline()


# In[ ]:


# part 1 - 4 : train image
# part 1 = 0:26347
# part 2 = 26347:52695
# part 3 = 52695:79042
# part 4 = 79042:TOTAL_IMAGES
# part 5 : test image
# part 5 = 0:TOTAL_IMAGES (CURRENT)
TOTAL_IMAGES = len(paths)
BATCH_PREDICT = 5

START_INDEX = 0
END_INDEX = TOTAL_IMAGES

list_texts = []


# # Predict

# In[ ]:


for i in range(START_INDEX, END_INDEX, BATCH_PREDICT):
    try:
        if i + BATCH_PREDICT < END_INDEX:
            images = [keras_ocr.tools.read(p) for p in paths[i:i + BATCH_PREDICT]]
        else:
            images = [keras_ocr.tools.read(p) for p in paths[i:END_INDEX]]

        prediction_groups = pipeline.recognize(images)

        for x in range(len(prediction_groups)):
            texts = []

            for y in range(len(prediction_groups[x])):
                texts.append(prediction_groups[x][y][0])

            list_texts.append(texts)
        gc.collect()
    except:
        if i + BATCH_PREDICT < END_INDEX:
            for j in range(BATCH_PREDICT):
                list_texts.append([])
        else:
            for j in range(END_INDEX - i):
                list_texts.append([])


# In[ ]:


sr_text = pd.Series(list_texts)
sr_text


# In[ ]:


sr_text.to_csv('test2.csv', index=False, header=False)

