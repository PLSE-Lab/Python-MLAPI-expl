# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import lightgbm as lgb
import gc

sub = pd.read_csv('../input/sample_submission.csv')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = 0.01

sub.to_csv('lgb_starter.csv', index=False, float_format='%.4f')