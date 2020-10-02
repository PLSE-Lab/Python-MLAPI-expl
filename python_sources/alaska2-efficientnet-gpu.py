#!/usr/bin/env python
# coding: utf-8

# Ref: https://www.kaggle.com/xhlulu/alaska2-efficientnet-on-tpus?select=sample_submission.csv
# 
# https://www.youtube.com/watch?v=u9pIhOay8Fw
# 
# https://www.kaggle.com/sgladysh/alaska2-steganalysis-tpu-efficientnet-b7-b6-b5-b4
# 
# https://www.kaggle.com/tasnimnishatislam/train-inference-gpu-baseline-tta

# In[ ]:


get_ipython().system('pip install -q efficientnet')


# In[ ]:


import math, re, os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from kaggle_datasets import KaggleDatasets
import tensorflow as tf
import tensorflow.keras.layers as L
import efficientnet.tfkeras as efn
from sklearn import metrics
from sklearn.model_selection import train_test_split


# In[ ]:


BASE_PATH = "../input/alaska2-image-steganalysis"
EPOCH = 7
BATCH_SIZE = 16

