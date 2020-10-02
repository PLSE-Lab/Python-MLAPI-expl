#!/usr/bin/env python
# coding: utf-8

# # Box Guess
# Here we take the predictions made in the [transfer learning](https://www.kaggle.com/kmader/lung-opacity-classification-transfer-learning/notebook) kernel and refine them into reasonable bounding boxes in order to make a submission.
# - We use the original input bounding boxes as a starting point
# - We divide into left and right lung since they appear to be broken up that way
# - We find the 95th percentile (`PERCENTILE_TO_KEEP`) for each of the parameters $x, y, width, height$
# - We submit two boxes (left and right) for each case above a certain threshold (`THRESHOLD_FOR_PREDICTION`)
# - We make define the two parameters below to make hyperparameter optimization easier

# In[ ]:


THRESHOLD_FOR_PREDICTION = 0.6
PERCENTILE_TO_KEEP = 95


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sub_df = pd.read_csv('../input/lung-opacity-classification-transfer-learning/submission.csv')
sub_df['score'] = sub_df['PredictionString'].map(lambda x: float(x[:4]) if isinstance(x, str) else 0)
sub_df.drop(['PredictionString'], axis=1, inplace=True)
sub_df['score'].plot.hist()
sub_df.sample(3)


# In[ ]:


all_bbox_df = pd.read_csv('../input/lung-opacity-overview/image_bbox_full.csv')
all_bbox_df.sample(3)


# In[ ]:


mini_df = all_bbox_df.             query('Target==1')[['x', 'y', 'width', 'height', 'boxes']]
sns.pairplot(mini_df,
            hue='boxes', 
             plot_kws={'alpha': 0.1})


# In[ ]:


all_bbox_df['x'].plot.hist()
right_box = all_bbox_df.query('x>450')
left_box = all_bbox_df.query('y<450')


# In[ ]:


def percentile_box(in_df, pct=95):
    return (
        np.percentile(in_df['x'], 100-pct),
        np.percentile(in_df['y'], 100-pct),
        np.percentile(in_df['width'], pct),
        np.percentile(in_df['height'], pct)
    )


# ## Show the boxes
# Here we show the boxes that we predict

# In[ ]:


right_bbox = percentile_box(right_box, PERCENTILE_TO_KEEP)
left_bbox = percentile_box(left_box, PERCENTILE_TO_KEEP)
print(right_bbox)
print(left_bbox)
fig, c_ax = plt.subplots(1, 1, figsize = (10, 10))
c_ax.set_xlim(0, 1024)
c_ax.set_ylim(0, 1024)
for i, (x, y, width, height) in enumerate([right_bbox, left_bbox]):
    c_ax.add_patch(Rectangle(xy=(x, y),
                                    width=width,
                                    height=height, 
                                     alpha = 0.5+0.25*i))


# In[ ]:


def proc_score(in_score):
    out_str = []
    if in_score>THRESHOLD_FOR_PREDICTION:
        for n_box in [left_bbox, right_bbox]:
            out_str+=['%2.2f %f %f %f %f' % (in_score, *n_box)]
    if len(out_str)==0:
        return ''
    else:
        return ' '.join(out_str)


# In[ ]:


sub_df['PredictionString'] = sub_df['score'].map(proc_score)
sub_df.sample(5)


# In[ ]:


sub_df[['patientId','PredictionString']].to_csv('submission.csv', index=False)


# In[ ]:




