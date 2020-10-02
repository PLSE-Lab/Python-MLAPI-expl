#!/usr/bin/env python
# coding: utf-8

# # Anchor Point Mystery
# 
# Help me solve this mystery.
# 
# During the ion challenge I found something interesting in the data that I still can't explain. Hopefully someone has a good explaination for it.
# 
# My attempt during the competition was to shift the "drift" sections of the data to maximize these points. I was successful but it didn't end up providing any fruitful to my model.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import minimize


# In[ ]:


train = pd.read_csv('../input/data-without-drift/train_clean.csv')
test = pd.read_csv('../input/data-without-drift/test_clean.csv')
tt = pd.concat([train, test], sort=False)


# In[ ]:


def add_model_groups(tt):
    tt.loc[(tt['time'] > 0) & (tt['time'] <= 10), 'sbatch'] = 0
    tt.loc[(tt['time'] > 10) & (tt['time'] <= 50), 'sbatch'] = 1
    tt.loc[(tt['time'] > 50) & (tt['time'] <= 60), 'sbatch'] = 2
    tt.loc[(tt['time'] > 60) & (tt['time'] <= 100), 'sbatch'] = 3
    tt.loc[(tt['time'] > 100) & (tt['time'] <= 150), 'sbatch'] = 4
    tt.loc[(tt['time'] > 150) & (tt['time'] <= 200), 'sbatch'] = 5
    tt.loc[(tt['time'] > 200) & (tt['time'] <= 250), 'sbatch'] = 6
    tt.loc[(tt['time'] > 250) & (tt['time'] <= 300), 'sbatch'] = 7
    tt.loc[(tt['time'] > 300) & (tt['time'] <= 350), 'sbatch'] = 8
    tt.loc[(tt['time'] > 350) & (tt['time'] <= 400), 'sbatch'] = 9
    tt.loc[(tt['time'] > 400) & (tt['time'] <= 450), 'sbatch'] = 10
    tt.loc[(tt['time'] > 450) & (tt['time'] <= 500), 'sbatch'] = 11
    # Test
    tt.loc[(tt['time'] > 500) & (tt['time'] <= 510), 'sbatch'] = 12
    tt.loc[(tt['time'] > 510) & (tt['time'] <= 520), 'sbatch'] = 13
    tt.loc[(tt['time'] > 520) & (tt['time'] <= 530), 'sbatch'] = 14
    tt.loc[(tt['time'] > 530) & (tt['time'] <= 540), 'sbatch'] = 15
    tt.loc[(tt['time'] > 540) & (tt['time'] <= 550), 'sbatch'] = 16
    tt.loc[(tt['time'] > 550) & (tt['time'] <= 560), 'sbatch'] = 17
    tt.loc[(tt['time'] > 560) & (tt['time'] <= 570), 'sbatch'] = 18
    tt.loc[(tt['time'] > 570) & (tt['time'] <= 580), 'sbatch'] = 19
    tt.loc[(tt['time'] > 580) & (tt['time'] <= 590), 'sbatch'] = 20
    tt.loc[(tt['time'] > 590) & (tt['time'] <= 600), 'sbatch'] = 21
    tt.loc[(tt['time'] > 600) & (tt['time'] <= 610), 'sbatch'] = 22
    tt.loc[(tt['time'] > 610) & (tt['time'] <= 630), 'sbatch'] = 23
    tt.loc[(tt['time'] > 630) & (tt['time'] <= 650), 'sbatch'] = 24
    tt.loc[(tt['time'] > 650) & (tt['time'] <= 670), 'sbatch'] = 25
    tt.loc[(tt['time'] > 670) & (tt['time'] <= 700), 'sbatch'] = 26
    return tt

def had_drift(tt):
    """
    I dentify if section had drift in the original dataset
    """
    tt.loc[(tt['time'] > 0) & (tt['time'] <= 10), 'drift'] = False
    tt.loc[(tt['time'] > 10) & (tt['time'] <= 50), 'drift'] = False
    tt.loc[(tt['time'] > 50) & (tt['time'] <= 60), 'drift'] = True
    tt.loc[(tt['time'] > 60) & (tt['time'] <= 100), 'drift'] = False
    tt.loc[(tt['time'] > 100) & (tt['time'] <= 150), 'drift'] = False
    tt.loc[(tt['time'] > 150) & (tt['time'] <= 200), 'drift'] = False
    tt.loc[(tt['time'] > 200) & (tt['time'] <= 250), 'drift'] = False
    tt.loc[(tt['time'] > 250) & (tt['time'] <= 300), 'drift'] = False
    tt.loc[(tt['time'] > 300) & (tt['time'] <= 350), 'drift'] = True
    tt.loc[(tt['time'] > 350) & (tt['time'] <= 400), 'drift'] = True
    tt.loc[(tt['time'] > 400) & (tt['time'] <= 450), 'drift'] = True
    tt.loc[(tt['time'] > 450) & (tt['time'] <= 500), 'drift'] = True
    # Test
    tt.loc[(tt['time'] > 500) & (tt['time'] <= 510), 'drift'] = True
    tt.loc[(tt['time'] > 510) & (tt['time'] <= 520), 'drift'] = True
    tt.loc[(tt['time'] > 520) & (tt['time'] <= 530), 'drift'] = False
    tt.loc[(tt['time'] > 530) & (tt['time'] <= 540), 'drift'] = False
    tt.loc[(tt['time'] > 540) & (tt['time'] <= 550), 'drift'] = True
    tt.loc[(tt['time'] > 550) & (tt['time'] <= 560), 'drift'] = False
    tt.loc[(tt['time'] > 560) & (tt['time'] <= 570), 'drift'] = True
    tt.loc[(tt['time'] > 570) & (tt['time'] <= 580), 'drift'] = True
    tt.loc[(tt['time'] > 580) & (tt['time'] <= 590), 'drift'] = True
    tt.loc[(tt['time'] > 590) & (tt['time'] <= 600), 'drift'] = False
    tt.loc[(tt['time'] > 600) & (tt['time'] <= 610), 'drift'] = True
    tt.loc[(tt['time'] > 610) & (tt['time'] <= 630), 'drift'] = True
    tt.loc[(tt['time'] > 630) & (tt['time'] <= 650), 'drift'] = True
    tt.loc[(tt['time'] > 650) & (tt['time'] <= 670), 'drift'] = False
    tt.loc[(tt['time'] > 670) & (tt['time'] <= 700), 'drift'] = False
    return tt


# In[ ]:


tt = had_drift(tt)
tt = add_model_groups(tt)
FILTER_TRAIN = '(time <= 47.6 or time > 48) and (time <= 364 or time > 382.4)'
tt = tt.query(FILTER_TRAIN)
tt['drift'] = tt['drift'].astype('bool')


# In[ ]:


for i, d in tt.groupby('open_channels'):
    d.query('not drift')['signal'].value_counts()         .plot(figsize=(15, 5), style='.', label=i,
              title='Value Counts by Signal (Excluding Drift Sections)')
plt.legend()
plt.show()


# # We wee a high number of values with these signal values:
# - Each corresponding to open channels:
# ```
#     open_channels -> signal
#     0 -> -2.5002
#     1 -> -1.2502
#     2 -> -0.0002
#     3 -> 1.2498
#     4 -> 2.4998
#     5 -> 3.7498
# ```

# Lets look at the value count of these in the data:

# In[ ]:


anchors = [-2.5002, -1.2502, -0.0002, 1.2498, 2.4998, 3.7498]
tt.query('drift == False and signal in @anchors').groupby(['signal','open_channels'])[['time']].count()


# # Plotting only the anchor points

# In[ ]:


fig, ax = plt.subplots()
tt.query('drift == False and signal in @anchors')     .groupby('open_channels')     .plot(x='time', y='signal', style='.', figsize=(15, 5), ax=ax)
plt.show()


# # I attempted to "shift" the drift sections to maximize the number of these values

# In[ ]:


tt['signal_shift'] = np.nan
tt.loc[~tt['drift'], 'signal_shift'] = tt.loc[~tt['drift']]['signal']


# In[ ]:


# Identify the batches with drift
drift_batches = tt.query('drift')['sbatch'].unique()
print(drift_batches)


# ## First attempt: maximize anchor count.

# In[ ]:


for db in drift_batches:
    d = tt.query('sbatch == @db')
    def shift_and_anchor_count(shift):
        anchor_count = 0
        shifted_signal = (d['signal'] + shift).round(4)
        for a in anchors:
    #         print(a)
            n_anchors = (shifted_signal == a).sum()
    #         print(n_anchors)
            anchor_count += n_anchors
    #     print(f'Shift {shift} ---> anchor count: {anchor_count}')
    

        return -anchor_count
    res = minimize(shift_and_anchor_count, [0], method='Powell', tol=1e-6)
    opt_shift = res['x']
    print(f'Drift batch {db} - optimal shift {opt_shift}')
    tt.loc[tt['sbatch'] == db, 'signal_shift'] = (tt.loc[tt['sbatch'] == db]['signal'] - opt_shift).round(4)


# ## Second attempt + max and minimize surrounding values.
# - This was a better way is to also minize if surrounding values of "anchors" have high value counts:

# In[ ]:


for db in drift_batches:
    d = tt.query('sbatch == @db')
    def shift_and_anchor_count(shift):
        shift = shift/1000
        anchor_count = 0
        shifted_signal = (d['signal'] + shift).round(4)
        for a in anchors:
            n_anchors = (shifted_signal == a).sum()
            anchor_count += n_anchors
            # Penalize for high counts neighbors numbers being high
            pprior = round(a - 0.0002, 4)
            prior = round(a - 0.0001, 4)
            post = round(a + 0.0001, 4)
            ppost = round(a + 0.0002, 4)
            n_anchor_prior = (shifted_signal == prior).sum()
            n_anchor_pprior = (shifted_signal == pprior).sum()
#             print(n_anchor_prior)
            anchor_count -= (prior - pprior)
            n_anchor_post = (shifted_signal == post).sum()
            n_anchor_ppost = (shifted_signal == ppost).sum()
            anchor_count -= (post - ppost)
#         print()
        return -anchor_count
    res = minimize(shift_and_anchor_count, [0], method='Powell') #, bounds=(-0.0001, 0.0001))
    opt_shift = res['x']
    print(f'Drift batch {db} - optimal shift {opt_shift}')
    tt.loc[tt['sbatch'] == db, 'signal_shift'] = (tt.loc[tt['sbatch'] == db]['signal'] + (opt_shift/1000)).round(4)


# In[ ]:


tt['signal_round4'] = tt['signal_shift'].round(4)


# ## We now have a high value count of our anchor points in non-drift areas

# In[ ]:


anchors = [-2.5002, -1.2502, -0.0002, 1.2498, 2.4998, 3.7498]
tt.query('drift and signal_shift in @anchors').groupby(['signal_shift','open_channels'])[['time']].count()


# # Lets plot the drift sections before and after this shift:

# In[ ]:


for i, d in tt.groupby('open_channels'):
    d.query('drift')['signal'].round(4)         .value_counts()         .plot(figsize=(15, 5), style='.', label=i,
              title='Unique Value Counts in Drift Segments before shifting')
plt.show()


# This looks much cleaner, right?

# In[ ]:


for i, d in tt.groupby('open_channels'):
    d.query('drift')['signal_round4']         .round(4).value_counts()         .plot(figsize=(15, 5), style='.', label=i,
             title='Unique Value Counts in Drift Data after Shifting to Optimize Anchors')
plt.legend()
plt.show()


# ## In the end this "shifted" data did not improve my CV/LB Score and I'm still confused by why it exists.
# - Can you solve this mystery for me?

# In[ ]:




