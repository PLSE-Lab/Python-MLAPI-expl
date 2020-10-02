# Simple script to get on the board

import numpy as np 
import pandas as pd 

train = pd.read_csv('../input/train.csv')
samp = pd.read_csv('../input/sample_submission.csv')
samp['target'] = np.median(train.target.values)
samp.to_csv('./sub_median.csv', index=False)