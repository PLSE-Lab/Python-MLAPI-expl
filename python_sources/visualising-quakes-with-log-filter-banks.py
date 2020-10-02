#!/usr/bin/env python
# coding: utf-8

# # Background
# 
# In Kaggle competitions visualising and understanding data is often as important as machine learning techniques applied to it. Having deep insights into the data can help to design better features or even spot leaks in the underlying dataset. In this competition we are dealing with a 1-dimensial highly granular signal. Total amout of datapoints is almost a billion, it's easy to miss the forest for the trees.
# 
# One of the most effective techniques in signal processing is DFT (Discrete Fourier Transform). It allows to reduce granularity of the signal across time axis and shift it into frequency dimension. To compact representation further one can apply log filter banks, that combine adjucent frequencies into sub-bands.

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().system('pip install python_speech_features')
from python_speech_features import logfbank


# In[ ]:


train_df = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float16})
train_df['quake_id'] = (train_df.time_to_failure.diff() > 0.0).cumsum().astype('int16')
train_df.head()


# # Parsing the signal
# From my analysis [here](https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/85350) the sensor works in the following way:
# * It makes measurements in bursts.
# * Each burst has 4096 (in rare cases 4095) measurements (separated by around 1 nanosecond).
# * Measurement bursts happen at a frequency around 1KHz (1000 bursts per second).
# 
# Based on the information above it is natural to split signal around the burst boundaries and transform each burst into the frequency domain.
# 
# **Note**: I skip the first and last quakes as they are not complete.

# In[ ]:


lfbanks = []
quake_cnt = 15
for quake_id in range(1, quake_cnt + 1):
    quake_df = train_df.loc[train_df.quake_id == quake_id]
    burst_id = (quake_df.time_to_failure.diff() < -1e-4).cumsum().astype('int16')
    lfbank = quake_df.groupby(burst_id).apply(lambda x: logfbank(
        x.acoustic_data.values, samplerate=4096, winlen=1.0, winstep=1.0, nfilt=32, nfft=4096, highfreq=512
    ))
    lfbank = np.vstack(lfbank.values)
    lfbanks.append(lfbank)


# In[ ]:


fig, axes = plt.subplots(quake_cnt, 1, figsize=(20, 4*quake_cnt))
fig.subplots_adjust(hspace=0.3)
for idx in range(1, quake_cnt + 1):
    sns.heatmap(lfbanks[idx-1].T, ax=axes[idx-1]).set_title('Quake #{}'.format(idx))


# # Summary
# 
# * The amplitude of the frequencies increases the closer we are to the earthquake event.
# * The amplitude changes are noisy over short time periods, making it hard to predict time to quake based on samples in the test data.
# * Log filter banks can be used as additional input features to your machine learning algorithm. They don't require any tuning and have compact representation.
# 
# Happy Kaggling!

# In[ ]:




