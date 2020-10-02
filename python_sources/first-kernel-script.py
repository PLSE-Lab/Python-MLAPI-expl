# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import datetime
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import random
import time
from sklearn.metrics import log_loss
    
print('Reading Data')
app = pd.read_csv("../input/app_events.csv")
test = pd.read_csv("../input/gender_age_test.csv")
app_labels = pd.read_csv("../input/app_labels.csv")
label_categories = pd.read_csv("../input/label_categories.csv")
phone_brand_device_model = pd.read_csv("../input/phone_brand_device_model.csv")
events = pd.read_csv("../input/events.csv")
train = pd.read_csv("../input/gender_age_train.csv")
    
app.head(n=5)

    

    
    
    
    
