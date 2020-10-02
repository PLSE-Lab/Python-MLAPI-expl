#!/usr/bin/env python
# coding: utf-8

# # Objective: Obtain out of fold predictions on the entire training set using cross validation and then using a mean average precision IoU metric, that closely resembles the competition metric, to improve validation

# In[ ]:


import numpy as np
import pandas as pd


# # Prepare out of fold training predictions for implementation of MAP IoU matching competition evaluation description

# Load oof predictions from CNN segmentation CV kernel https://www.kaggle.com/cchadha/cnn-segmentation-cv-with-oof-preds-on-train-set/notebook

# In[ ]:


oof_preds0 = pd.read_csv('../input/cnn-segmentation-cv-with-oof-preds-on-train-set/oof_preds0.csv')
oof_preds1 = pd.read_csv('../input/cnn-segmentation-cv-with-oof-preds-on-train-set/oof_preds1.csv')
oof_preds2 = pd.read_csv('../input/cnn-segmentation-cv-with-oof-preds-on-train-set/oof_preds2.csv')


# In[ ]:


oof_preds0.head()


# In[ ]:


oof_preds1.head()


# In[ ]:


oof_preds2.head()


# Read in training labels

# In[ ]:


df = pd.read_csv('../input/rsna-pneumonia-detection-challenge/stage_1_train_labels.csv')


# In[ ]:


df.head(20)


# Parse bounding box labels into correct format for Mean Average Precision IoU metric

# In[ ]:


df['bbox_target'] = (df['x'].astype(str) +
                    ' ' + 
                    df['y'].astype(str) +
                    ' ' +
                    df['width'].astype(str) +
                    ' ' +
                    df['height'].astype(str))


# In[ ]:


df.loc[:, 'bbox_target'] = df.loc[:, 'bbox_target'].map(lambda x: x.split(' '))


# In[ ]:


df.head(20)


# In[ ]:


df = df.groupby(['patientId'], as_index = False)['bbox_target'].agg('sum')    


# In[ ]:


df.head()


# Merge labels and oof preds

# In[ ]:


df = df.merge(oof_preds0, on = 'patientId', how = 'left')
df = df.merge(oof_preds1, on = 'patientId', how = 'left')
df = df.merge(oof_preds2, on = 'patientId', how = 'left')


# In[ ]:


df = df.fillna('')


# Parse oof preds for MAP IoU

# In[ ]:


df.loc[:, 'bbox_pred'] = (df.loc[:, 'PredictionString'] +
                         ' ' +
                         df.loc[:, 'PredictionString_x'] +
                         ' ' +
                         df.loc[:, 'PredictionString_y'])


# In[ ]:


df = df.drop(['PredictionString','PredictionString_x', 'PredictionString_y'], axis=1)


# In[ ]:


df.head(20)


# Stripping whitespace from PredictionString column

# In[ ]:


df.loc[:, 'bbox_pred'] = df.loc[:, 'bbox_pred'].str.strip()


# In[ ]:


df.loc[:, 'bbox_pred'] = df.loc[:, 'bbox_pred'].map(lambda x: x.split(' '))


# In[ ]:


df.head(10)


# In[ ]:


def parse_scores(x):
    if len(x)!=1:
        scores = [x[k] for k in range(0,len(x),5)]
        for score in range(len(scores)):
            scores[score] = float(scores[score])
        return np.asarray(scores)


# In[ ]:


df.loc[:, 'bbox_scores'] = df.loc[:, 'bbox_pred'].map(parse_scores)


# In[ ]:


df.head()


# In[ ]:


def parse_bbox(x):
    if len(x)!=1:
        bbox = [int(x[k]) for k in range(0,len(x)) if k%5 != 0]
        return np.asarray(bbox).reshape(int(len(bbox)/4),4)


# In[ ]:


df.loc[:, 'bbox_preds'] = df.loc[:, 'bbox_pred'].map(parse_bbox)


# In[ ]:


df.head()


# In[ ]:


df = df.drop(['bbox_pred'], axis=1)


# In[ ]:


df.head(20)


# Edit NaN or None values to empty numpy arrays to fit MAP IoU metric implementation

# In[ ]:


df.loc[df['bbox_scores'].isnull(),['bbox_scores']] = df.loc[df['bbox_scores'].isnull(),'bbox_scores'].apply(lambda x: np.asarray([]))


# In[ ]:


df.head()


# In[ ]:


df.loc[df['bbox_preds'].isnull(),['bbox_preds']] = df.loc[df['bbox_preds'].isnull(),'bbox_preds'].apply(lambda x: np.asarray([]))


# In[ ]:


df.head()


# In[ ]:


def parse_target_str(x):
    if x[0] != 'nan':
        bbox = np.asarray([int(float(x[k])) for k in range(0,len(x))])
        return bbox.reshape(int(len(bbox)/4),4)


# In[ ]:


df.loc[:,'bbox_target'] = df.loc[:,'bbox_target'].map(parse_target_str)


# In[ ]:


df.loc[df['bbox_target'].isnull(),['bbox_target']] = df.loc[df['bbox_target'].isnull(),'bbox_target'].apply(lambda x: np.asarray([]))


# In[ ]:


df.head()


# # Find mean average precision IoU using implementation by chenyc15 https://www.kaggle.com/chenyc15/mean-average-precision-metric and edited herein

# In[ ]:


# helper function to calculate IoU
def iou(box1, box2):
    x11, y11, w1, h1 = box1
    x21, y21, w2, h2 = box2
    assert w1 * h1 > 0
    assert w2 * h2 > 0
    x12, y12 = x11 + w1, y11 + h1
    x22, y22 = x21 + w2, y21 + h2

    area1, area2 = w1 * h1, w2 * h2
    xi1, yi1, xi2, yi2 = max([x11, x21]), max([y11, y21]), min([x12, x22]), min([y12, y22])
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0
    else:
        intersect = (xi2-xi1) * (yi2-yi1)
        union = area1 + area2 - intersect
        return intersect / union


# In[ ]:


def map_iou(boxes_true, boxes_pred, scores, thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]):
    """
    Mean average precision at differnet intersection over union (IoU) threshold
    
    input:
        boxes_true: Mx4 numpy array of ground true bounding boxes of one image. 
                    bbox format: (x1, y1, w, h)
        boxes_pred: Nx4 numpy array of predicted bounding boxes of one image. 
                    bbox format: (x1, y1, w, h)
        scores:     length N numpy array of scores associated with predicted bboxes
        thresholds: IoU shresholds to evaluate mean average precision on
    output: 
        map: mean average precision of the image
    """
    
    # According to the introduction, images with no ground truth bboxes will not be 
    # included in the map score unless there is a false positive detection (?)
        
    # return None if both are empty, don't count the image in final evaluation (?)
    if len(boxes_true) == 0 and len(boxes_pred) == 0:
        return None
    elif len(boxes_true) == 0 and len(boxes_pred) > 0:
        return 0
    elif len(boxes_true) > 0 and len(boxes_pred) == 0:
        return 0
    elif len(boxes_true) > 0 and len(boxes_pred) > 0:
        assert boxes_true.shape[1] == 4 or boxes_pred.shape[1] == 4, "boxes should be 2D arrays with shape[1]=4"
        if len(boxes_pred):
            assert len(scores) == len(boxes_pred), "boxes_pred and scores should be same length"
            # sort boxes_pred by scores in decreasing order
            boxes_pred = boxes_pred[np.argsort(scores)[::-1], :]

        map_total = 0

        # loop over thresholds
        for t in thresholds:
            matched_bt = set()
            tp, fn = 0, 0
            for i, bt in enumerate(boxes_true):
                matched = False
                for j, bp in enumerate(boxes_pred):
                    miou = iou(bt, bp)
                    if miou >= t and not matched and j not in matched_bt:
                        matched = True
                        tp += 1 # bt is matched for the first time, count as TP
                        matched_bt.add(j)
                if not matched:
                    fn += 1 # bt has no match, count as FN

            fp = len(boxes_pred) - len(matched_bt) # FP is the bp that not matched to any bt
            m = tp / (tp + fn + fp)
            map_total += m
    
    return map_total / len(thresholds)


# In[ ]:


df.head(20)


# In[ ]:


for row in range(10):
    print(map_iou(df['bbox_target'][row], df['bbox_preds'][row], df['bbox_scores'][row]))


# In[ ]:


map_scores = [x for x in [map_iou(df['bbox_target'][row], df['bbox_preds'][row], df['bbox_scores'][row]) for row in range(len(df))] if x is not None]


# In[ ]:


np.mean(map_scores)


# In[ ]:




