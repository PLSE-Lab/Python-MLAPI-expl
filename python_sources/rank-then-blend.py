#!/usr/bin/env python
# coding: utf-8

# # Objective
# 
# This notebook is not intended to give a high leaderboard or show an awesome blend. 
# 
# In this notebook i will use 2 public submissions that score 0.940 and 0.936 and 3 personal submissions. Blending and don't knowing the cv and the cv strategy of this models is a horrible idea because they are probably overtiffing. 
# 
# The main idea of this notebook is to talk about the predictions distribution. The predictions have different distribution so combining them in the form x1*w1 + x2*w2 + .... + xn*wn is not recommended at all. Receiver Operating Characteristic area under the curve is sensible to this distribution. For this we should first rank each of this vector and then blend the predictions in the form x1*w1 + x2*w2 + .... + xn*wn. This is just an example of how you should do that.

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


sub1 = pd.read_csv('../input/melanoma-dif-sub/pl_0.936.csv')
sub2 = pd.read_csv('../input/melanoma-dif-sub/pl_0.940.csv')
sub3 = pd.read_csv('../input/melanoma-dif-sub/sub_EfficientNetB2_384.csv')
sub4 = pd.read_csv('../input/melanoma-dif-sub/sub_EfficientNetB3_384.csv')
sub5 = pd.read_csv('../input/melanoma-dif-sub/sub_EfficientNetB3_384_v2.csv')

# lets rank each prediction and then divide it by its max value to we have our predictions between 0 and 1
def rank_data(sub):
    sub['target'] = sub['target'].rank() / sub['target'].rank().max()
    return sub

sub1 = rank_data(sub1)
sub2 = rank_data(sub2)
sub3 = rank_data(sub3)
sub4 = rank_data(sub4)
sub5 = rank_data(sub5)
sub1.columns = ['image_name', 'target1']
sub2.columns = ['image_name', 'target2']
sub3.columns = ['image_name', 'target3']
sub4.columns = ['image_name', 'target4']
sub5.columns = ['image_name', 'target5']

f_sub = sub1.merge(sub2, on = 'image_name').merge(sub3, on = 'image_name').merge(sub4, on = 'image_name').merge(sub5, on = 'image_name')
f_sub['target'] = f_sub['target1'] * 0.3 + f_sub['target2'] * 0.3 + f_sub['target3'] * 0.05 + f_sub['target4'] * 0.3 + f_sub['target5'] * 0.05
f_sub = f_sub[['image_name', 'target']]
f_sub.to_csv('blend_sub.csv', index = False)

