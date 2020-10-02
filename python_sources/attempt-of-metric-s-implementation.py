#!/usr/bin/env python
# coding: utf-8

# # Attempt of metric's implementation

# This is an attempt to step-by-step implement the code of the metric described by the organizers of the competition. 
# 
# Result is very similar on LB score!
# 
# __Request for help:__ how can I rewrite the code in terms of tensors? I need to realize a metric function such as:
# 
# `def mean_iou(y_true, y_pred):
#     y_pred = tf.round(y_pred)
#     intersect = tf.reduce_sum(y_true * y_pred, axis = [1, 2, 3])
#     union = tf.reduce_sum(y_true, axis = [1, 2, 3]) + tf.reduce_sum(y_pred, axis = [1, 2, 3])
#     smooth = tf.ones(tf.shape(intersect))
#     return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))`

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


prediction_strings = pd.read_csv('../input/rsna-pred-train/model_1_train.csv')
prediction_strings.head()


# In[ ]:


ground_truth = pd.read_csv('../input/rsna-pneumonia-detection-challenge/stage_1_train_labels.csv')
ground_truth.head()


# In[ ]:


prediction = pd.DataFrame(index = ['patientId', 'confidence', 'x', 'y', 'width', 'height'])

for i, row in prediction_strings.iterrows():
    if len(str(row[1])) > 3:
        row_array = row[1].split(' ')
        for i in range(int(len(row_array) / 5)):
            prediction[prediction.shape[1]] = [row[0], float(row_array[i * 5])] +                                               [int(b) for b in row_array[i * 5 + 1 : i * 5 + 5]]
            
prediction = prediction.T
prediction.head()


# In[ ]:


def iou(x1, y1, width1, height1, x2, y2, width2, height2):
    x1, y1, width1, height1, x2, y2, width2, height2 = [int(v) for v in [x1, y1, width1, height1, x2, y2, width2, height2]]

    mask_1 = np.zeros((max(y1 + height1, y2 + height2), max(x1 + width1, x2 + width2)))
    mask_2 = np.zeros((max(y1 + height1, y2 + height2), max(x1 + width1, x2 + width2)))
    
    mask_1[y1 : y1 + height1, x1 : x1 + width1] = 1
    mask_2[y2 : y2 + height2, x2 : x2 + width2] = 1

    mask_i = mask_1 * mask_2
    mask_u = mask_1 + mask_2 - mask_i
    
    return float(sum(sum(mask_i))) / sum(sum(mask_u))


# In[ ]:


threshold = .4


# In[ ]:


tp_candidates = pd.DataFrame(index = ['patientId', 'index_prediction', 'index_gtruth', 'iou'])

for patient in list(prediction['patientId'].unique()):
    pred = prediction[prediction['patientId'] == patient].sort_values('confidence', ascending = False)
    g_tr = ground_truth[ground_truth['patientId'] == patient]
    
    if g_tr['Target'].sum() > 0:
        for ig, g in g_tr.iterrows():
            for ip, p in pred.iterrows():
                iou_score = iou(p['x'], p['y'], p['width'], p['height'], 
                                g['x'], g['y'], g['width'], g['height'])
                if iou_score > threshold:
                    tp_candidates[tp_candidates.shape[1]] = [patient, ip, ig, iou_score]
                    break
                    
tp_candidates = tp_candidates.T
tp_candidates.head(10)


# In[ ]:


rates = pd.DataFrame(index = ['TP', 'FP', 'FN', 'score'])

for threshold in [.4, .45, .5, .55, .6, .65, .7, .75]:
    tp = sum(tp_candidates['iou'] > threshold)
    fp = prediction.shape[0] - tp
    fn = ground_truth.dropna().shape[0] -          ground_truth.loc[tp_candidates[tp_candidates['iou'] > threshold]['index_gtruth']].dropna().shape[0]
    rates[threshold] = [tp, fp, fn, float(tp) / (tp + fp + fn)]
    
rates.T


# In[ ]:


rates.T['score'].mean()


# __LB score: 0.014__

# In[ ]:




