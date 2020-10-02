#!/usr/bin/env python
# coding: utf-8

# ## YoloV5 Pseudo Labeling + OOF Evaluation
# 
# This notebook is a clean up of [Yolov5 Pseudo Labeling](https://www.kaggle.com/nvnnghia/yolov5-pseudo-labeling), with OOF-evaluation to search the best `score_threshold` for final prediction.  
# My pretrained checkpoint gives worse results compared to the original notebook, but to my surprise adding OOF-evaluation gives me a large boost about 1% LB.   
# You can try your pretrained model on this notebook.
# 
# update: fixed a bug in `calculate_final_score`, boxes should be sorted with discending conf score.
# 
# References:  
# Awesome original Pseudo Labeling notebook: https://www.kaggle.com/nvnnghia/yolov5-pseudo-labeling  
# Evaluation Script: https://www.kaggle.com/pestipeti/competition-metric-details-script  
# OOF-Evaluation: https://www.kaggle.com/shonenkov/oof-evaluation-mixup-efficientdet  
# Bayesian Optimization (though failed to improve my results): https://www.kaggle.com/shonenkov/bayesian-optimization-wbf-efficientdet  
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from tqdm.auto import tqdm
import shutil as sh
import torch
import sys
import glob
import random

get_ipython().system('cp -r ../input/yolov5train/* .')
#sys.path.insert(0, "../input/yolov5tta/")
sys.path.insert(0, "../input/weightedboxesfusion")


# In[ ]:


NMS_IOU_THR = 0.6
NMS_CONF_THR = 0.25

# WBF
best_iou_thr = 0.6
best_skip_box_thr = 0.43

# Box conf threshold
best_final_score = 0
best_score_threshold = 0

SEED = 42

EPO = 15

WEIGHTS = '../input/wheatyolov5/best_wheat0.pt'

CONFIG = '../input/wheatyolov5/yolov5x.yaml'

DATA = '../input/configyolo5/wheat0.yaml'

is_TEST = len(os.listdir('../input/global-wheat-detection/test/'))>11

is_AUG = True
is_ROT = True

VALIDATE = True

PSEUDO = True


# In[ ]:


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
    
set_seed(SEED)


# In[ ]:


# For OOF evaluation
marking = pd.read_csv('../input/global-wheat-detection/train.csv')

bboxs = np.stack(marking['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
for i, column in enumerate(['x', 'y', 'w', 'h']):
    marking[column] = bboxs[:,i]
marking.drop(columns=['bbox'], inplace=True)


# In[ ]:


def convertTrainLabel():
    df = pd.read_csv('../input/global-wheat-detection/train.csv')
    bboxs = np.stack(df['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
    for i, column in enumerate(['x', 'y', 'w', 'h']):
        df[column] = bboxs[:,i]
    df.drop(columns=['bbox'], inplace=True)
    df['x_center'] = df['x'] + df['w']/2
    df['y_center'] = df['y'] + df['h']/2
    df['classes'] = 0
    from tqdm.auto import tqdm
    import shutil as sh
    df = df[['image_id','x', 'y', 'w', 'h','x_center','y_center','classes']]
    
    index = list(set(df.image_id))
    
    source = 'train'
    if True:
        for fold in [0]:
            val_index = index[len(index)*fold//5:len(index)*(fold+1)//5]
            for name, mini in tqdm(df.groupby('image_id')):
                path2save = 'val2017/' if name in val_index else 'train2017/'
                os.makedirs('convertor/fold{}/labels/'.format(fold)+path2save, exist_ok=True)
                with open('convertor/fold{}/labels/'.format(fold)+path2save+name+".txt", 'w+') as f:
                    row = mini[['classes','x_center','y_center','w','h']].astype(float).values
                    row = row/1024
                    row = row.astype(str)
                    for j in range(len(row)):
                        text = ' '.join(row[j])
                        f.write(text)
                        f.write("\n")
                os.makedirs('convertor/fold{}/images/{}'.format(fold,path2save), exist_ok=True)
                sh.copy("../input/global-wheat-detection/{}/{}.jpg".format(source,name),'convertor/fold{}/images/{}/{}.jpg'.format(fold,path2save,name))


# In[ ]:


from ensemble_boxes import *

def run_wbf(boxes, scores, image_size=1024, iou_thr=0.5, skip_box_thr=0.7, weights=None):
    labels = [np.zeros(score.shape[0]) for score in scores]
    boxes = [box/(image_size) for box in boxes]
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes = boxes*(image_size)
    return boxes, scores, labels

def TTAImage(image, index):
    image1 = image.copy()
    if index==0: 
        rotated_image = cv2.rotate(image1, cv2.ROTATE_90_CLOCKWISE)
        return rotated_image
    elif index==1:
        rotated_image2 = cv2.rotate(image1, cv2.ROTATE_90_CLOCKWISE)
        rotated_image2 = cv2.rotate(rotated_image2, cv2.ROTATE_90_CLOCKWISE)
        return rotated_image2
    elif index==2:
        rotated_image3 = cv2.rotate(image1, cv2.ROTATE_90_CLOCKWISE)
        rotated_image3 = cv2.rotate(rotated_image3, cv2.ROTATE_90_CLOCKWISE)
        rotated_image3 = cv2.rotate(rotated_image3, cv2.ROTATE_90_CLOCKWISE)
        return rotated_image3
    elif index == 3:
        return image1
    
def rotBoxes90(boxes, im_w, im_h):
    ret_boxes =[]
    for box in boxes:
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = x1-im_w//2, im_h//2 - y1, x2-im_w//2, im_h//2 - y2
        x1, y1, x2, y2 = y1, -x1, y2, -x2
        x1, y1, x2, y2 = int(x1+im_w//2), int(im_h//2 - y1), int(x2+im_w//2), int(im_h//2 - y2)
        x1a, y1a, x2a, y2a = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
        ret_boxes.append([x1a, y1a, x2a, y2a])
    return np.array(ret_boxes)

def detect1Image(img, img0, model, device, aug):
    img = img.transpose(2,0,1)
    img = torch.from_numpy(img).to(device)
    img =  img.float()  # uint8 to fp16/32
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    # Inference
    pred = model(img, augment=aug)[0]
    
    # Apply NMS
    pred = non_max_suppression(pred, NMS_CONF_THR, NMS_IOU_THR, merge=True, classes=None, agnostic=False)
    
    boxes = []
    scores = []
    for i, det in enumerate(pred):  # detections per image
        # save_path = 'draw/' + image_id + '.jpg'
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Write results
            for *xyxy, conf, cls in det:
                boxes.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                scores.append(conf)

    return np.array(boxes), np.array(scores) 


# In[ ]:


# validate

import pandas as pd
import numpy as np
import numba
import re
import cv2
import ast
import matplotlib.pyplot as plt

from numba import jit
from typing import List, Union, Tuple


@jit(nopython=True)
def calculate_iou(gt, pr, form='pascal_voc') -> float:
    """Calculates the Intersection over Union.

    Args:
        gt: (np.ndarray[Union[int, float]]) coordinates of the ground-truth box
        pr: (np.ndarray[Union[int, float]]) coordinates of the prdected box
        form: (str) gt/pred coordinates format
            - pascal_voc: [xmin, ymin, xmax, ymax]
            - coco: [xmin, ymin, w, h]
    Returns:
        (float) Intersection over union (0.0 <= iou <= 1.0)
    """
    if form == 'coco':
        gt = gt.copy()
        pr = pr.copy()

        gt[2] = gt[0] + gt[2]
        gt[3] = gt[1] + gt[3]
        pr[2] = pr[0] + pr[2]
        pr[3] = pr[1] + pr[3]

    # Calculate overlap area
    dx = min(gt[2], pr[2]) - max(gt[0], pr[0]) + 1
    
    if dx < 0:
        return 0.0
    
    dy = min(gt[3], pr[3]) - max(gt[1], pr[1]) + 1

    if dy < 0:
        return 0.0

    overlap_area = dx * dy

    # Calculate union area
    union_area = (
            (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1) +
            (pr[2] - pr[0] + 1) * (pr[3] - pr[1] + 1) -
            overlap_area
    )

    return overlap_area / union_area


@jit(nopython=True)
def find_best_match(gts, pred, pred_idx, threshold = 0.5, form = 'pascal_voc', ious=None) -> int:
    """Returns the index of the 'best match' between the
    ground-truth boxes and the prediction. The 'best match'
    is the highest IoU. (0.0 IoUs are ignored).

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        pred: (List[Union[int, float]]) Coordinates of the predicted box
        pred_idx: (int) Index of the current predicted box
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (int) Index of the best match GT box (-1 if no match above threshold)
    """
    best_match_iou = -np.inf
    best_match_idx = -1

    for gt_idx in range(len(gts)):
        
        if gts[gt_idx][0] < 0:
            # Already matched GT-box
            continue
        
        iou = -1 if ious is None else ious[gt_idx][pred_idx]

        if iou < 0:
            iou = calculate_iou(gts[gt_idx], pred, form=form)
            
            if ious is not None:
                ious[gt_idx][pred_idx] = iou

        if iou < threshold:
            continue

        if iou > best_match_iou:
            best_match_iou = iou
            best_match_idx = gt_idx

    return best_match_idx

@jit(nopython=True)
def calculate_precision(gts, preds, threshold = 0.5, form = 'coco', ious=None) -> float:
    """Calculates precision for GT - prediction pairs at one threshold.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (float) Precision
    """
    n = len(preds)
    tp = 0
    fp = 0
    
    # for pred_idx, pred in enumerate(preds_sorted):
    for pred_idx in range(n):

        best_match_gt_idx = find_best_match(gts, preds[pred_idx], pred_idx,
                                            threshold=threshold, form=form, ious=ious)

        if best_match_gt_idx >= 0:
            # True positive: The predicted box matches a gt box with an IoU above the threshold.
            tp += 1
            # Remove the matched GT box
            gts[best_match_gt_idx] = -1

        else:
            # No match
            # False positive: indicates a predicted box had no associated gt box.
            fp += 1

    # False negative: indicates a gt box had no associated predicted box.
    fn = (gts.sum(axis=1) > 0).sum()

    return tp / (tp + fp + fn)


@jit(nopython=True)
def calculate_image_precision(gts, preds, thresholds = (0.5, ), form = 'coco') -> float:
    """Calculates image precision.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        thresholds: (float) Different thresholds
        form: (str) Format of the coordinates

    Return:
        (float) Precision
    """
    n_threshold = len(thresholds)
    image_precision = 0.0
    
    ious = np.ones((len(gts), len(preds))) * -1
    # ious = None

    for threshold in thresholds:
        precision_at_threshold = calculate_precision(gts.copy(), preds, threshold=threshold,
                                                     form=form, ious=ious)
        image_precision += precision_at_threshold / n_threshold

    return image_precision

    
# Numba typed list!
iou_thresholds = numba.typed.List()

for x in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
    iou_thresholds.append(x)
    
def validate():
    source = 'convertor/fold0/images/val2017'
    
    weights = 'weights/best.pt'
    if not os.path.exists(weights):
        weights = WEIGHTS
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load model
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    model.to(device).eval()
    
    dataset = LoadImages(source, img_size=1024)

    results = []
    
    for path, img, img0, vid_cap in dataset:
            
        image_id = os.path.basename(path).split('.')[0]
        img = img.transpose(1,2,0) # [H, W, 3]
        
        enboxes = []
        enscores = []
        
        # only rot, no flip
        if is_ROT:    
            for i in range(4):
                img1 = TTAImage(img, i)
                boxes, scores = detect1Image(img1, img0, model, device, aug=False)
                for _ in range(3-i):
                    boxes = rotBoxes90(boxes, *img.shape[:2])            
                enboxes.append(boxes)
                enscores.append(scores) 
        
        # flip
        boxes, scores = detect1Image(img, img0, model, device, aug=is_AUG)
        enboxes.append(boxes)
        enscores.append(scores) 
            
        #boxes, scores, labels = run_wbf(enboxes, enscores, image_size=1024, iou_thr=WBF_IOU_THR, skip_box_thr=WBF_CONF_THR)    
        #boxes = boxes.astype(np.int32).clip(min=0, max=1024)
        #boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        #boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        
        #boxes = boxes[scores >= 0.05].astype(np.int32)
        #scores = scores[scores >= float(0.05)]
        
        records = marking[marking['image_id'] == image_id]
        gtboxes = records[['x', 'y', 'w', 'h']].values
        gtboxes = gtboxes.astype(np.int32).clip(min=0, max=1024)
        gtboxes[:, 2] = gtboxes[:, 0] + gtboxes[:, 2]
        gtboxes[:, 3] = gtboxes[:, 1] + gtboxes[:, 3]
        
            
        result = {
            'image_id': image_id,
            'pred_enboxes': enboxes, # xyhw
            'pred_enscores': enscores,
            'gt_boxes': gtboxes, # xyhw
        }

        results.append(result)
        
    return results

def calculate_final_score(all_predictions, iou_thr, skip_box_thr, score_threshold):
    final_scores = []
    for i in range(len(all_predictions)):
        gt_boxes = all_predictions[i]['gt_boxes'].copy()
        enboxes = all_predictions[i]['pred_enboxes'].copy()
        enscores = all_predictions[i]['pred_enscores'].copy()
        image_id = all_predictions[i]['image_id']
        
        pred_boxes, scores, labels = run_wbf(enboxes, enscores, image_size=1024, iou_thr=iou_thr, skip_box_thr=skip_box_thr)    
        pred_boxes = pred_boxes.astype(np.int32).clip(min=0, max=1024)

        indexes = np.where(scores>score_threshold)
        pred_boxes = pred_boxes[indexes]
        scores = scores[indexes]
        
        # descending conf
        rank = np.argsort(scores)[::-1]
        pred_boxes = pred_boxes[rank]
        scores = scores[rank]

        image_precision = calculate_image_precision(gt_boxes, pred_boxes,thresholds=iou_thresholds,form='pascal_voc')
        final_scores.append(image_precision)

    return np.mean(final_scores)

def show_result(sample_id, preds, gt_boxes):
    sample = cv2.imread(f'../input/global-wheat-detection/train/{sample_id}.jpg', cv2.IMREAD_COLOR)
    sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    for pred_box in preds:
        cv2.rectangle(
            sample,
            (pred_box[0], pred_box[1]),
            (pred_box[2], pred_box[3]),
            (220, 0, 0), 2
        )

    for gt_box in gt_boxes:    
        cv2.rectangle(
            sample,
            (gt_box[0], gt_box[1]),
            (gt_box[2], gt_box[3]),
            (0, 0, 220), 2
        )

    ax.set_axis_off()
    ax.imshow(sample)
    ax.set_title("RED: Predicted | BLUE - Ground-truth")


# In[ ]:


# Bayesian Optimize

from skopt import gp_minimize, forest_minimize
from skopt.utils import use_named_args
from skopt.plots import plot_objective, plot_evaluations, plot_convergence, plot_regret
from skopt.space import Categorical, Integer, Real

def log(text):
    print(text)

def optimize(space, all_predictions, n_calls=10):
    @use_named_args(space)
    def score(**params):
        log('-'*10)
        log(params)
        final_score = calculate_final_score(all_predictions, **params)
        log(f'final_score = {final_score}')
        log('-'*10)
        return -final_score

    return gp_minimize(func=score, dimensions=space, n_calls=n_calls)


# In[ ]:


from utils.datasets import *
from utils.utils import *

def makePseudolabel():
    source = '../input/global-wheat-detection/test/'
    weights = WEIGHTS
    
    imagenames =  os.listdir(source)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load model
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    model.to(device).eval()
    
    dataset = LoadImages(source, img_size=1024)

    path2save = 'train2017/'
    
    if not os.path.exists('convertor/fold0/labels/'+path2save):
        os.makedirs('convertor/fold0/labels/'+path2save)
    if not os.path.exists('convertor/fold0/images/{}'.format(path2save)):
        os.makedirs('convertor/fold0/images/{}'.format(path2save))
    
    for path, img, img0, vid_cap in dataset:
        image_id = os.path.basename(path).split('.')[0]
        img = img.transpose(1,2,0) # [H, W, 3]
        
        enboxes = []
        enscores = []
        
        # only rot, no flip
        if is_ROT:    
            for i in range(4):
                img1 = TTAImage(img, i)
                boxes, scores = detect1Image(img1, img0, model, device, aug=False)
                for _ in range(3-i):
                    boxes = rotBoxes90(boxes, *img.shape[:2])            
                enboxes.append(boxes)
                enscores.append(scores) 
        
        # flip
        boxes, scores = detect1Image(img, img0, model, device, aug=is_AUG)
        enboxes.append(boxes)
        enscores.append(scores) 
            
        boxes, scores, labels = run_wbf(enboxes, enscores, image_size=1024, iou_thr=best_iou_thr, skip_box_thr=best_skip_box_thr)
        boxes = boxes.astype(np.int32).clip(min=0, max=1024)
        
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        
        indices = scores >= best_score_threshold
        boxes = boxes[indices]
        scores = scores[indices]
        
        lineo = ''
        for box in boxes:
            x1, y1, w, h = box
            xc, yc, w, h = (x1+w/2)/1024, (y1+h/2)/1024, w/1024, h/1024
            lineo += '0 %f %f %f %f\n'%(xc, yc, w, h)
            
        fileo = open('convertor/fold0/labels/'+path2save+image_id+".txt", 'w+')
        fileo.write(lineo)
        fileo.close()
        sh.copy("../input/global-wheat-detection/test/{}.jpg".format(image_id),'convertor/fold0/images/{}/{}.jpg'.format(path2save,image_id))
            


# In[ ]:


if PSEUDO or VALIDATE:
    convertTrainLabel()


# In[ ]:


if PSEUDO:
    # this gives worse results
    '''
    if VALIDATE and is_TEST:
        all_predictions = validate()
        for score_threshold in tqdm(np.arange(0, 1, 0.01), total=np.arange(0, 1, 0.01).shape[0]):
            final_score = calculate_final_score(all_predictions, score_threshold)
            if final_score > best_final_score:
                best_final_score = final_score
                best_score_threshold = score_threshold

        print('-'*30)
        print(f'[Best Score Threshold]: {best_score_threshold}')
        print(f'[OOF Score]: {best_final_score:.4f}')
        print('-'*30)
    ''' 
    makePseudolabel()
    
    if is_TEST:
        get_ipython().system('python train.py --weights {WEIGHTS} --img 1024 --batch 2 --epochs {EPO} --data {DATA} --cfg {CONFIG}')
    else:
        pass
        #!python train.py --weights {WEIGHTS} --img 1024 --batch 2 --epochs 1 --data {DATA} --cfg {CONFIG}
    
    


# In[ ]:


if VALIDATE and is_TEST:
    all_predictions = validate()
    
    # Bayesian Optimization: worse results.
    '''
    space = [
        Real(0, 1, name='iou_thr'),
        Real(0.25, 1, name='skip_box_thr'),
        Real(0, 1, name='score_threshold'),
    ]

    opt_result = optimize(
        space, 
        all_predictions,
        n_calls=50,
    )
    
    best_final_score = -opt_result.fun
    best_iou_thr = opt_result.x[0]
    best_skip_box_thr = opt_result.x[1]
    best_score_threshold = opt_result.x[2]


    print('-'*13 + 'WBF' + '-'*14)
    print("[Baseline score]", calculate_final_score(all_predictions, 0.6, 0.43, 0))
    print(f'[Best Iou Thr]: {best_iou_thr:.3f}')
    print(f'[Best Skip Box Thr]: {best_skip_box_thr:.3f}')
    print(f'[Best Score Thr]: {best_score_threshold:.3f}')
    print(f'[Best Score]: {best_final_score:.4f}')
    print('-'*30)
    
    '''
    
    for score_threshold in tqdm(np.arange(0, 1, 0.01), total=np.arange(0, 1, 0.01).shape[0]):
        final_score = calculate_final_score(all_predictions, best_iou_thr, best_skip_box_thr, score_threshold)
        if final_score > best_final_score:
            best_final_score = final_score
            best_score_threshold = score_threshold

    print('-'*30)
    print(f'[Best Score Threshold]: {best_score_threshold}')
    print(f'[OOF Score]: {best_final_score:.4f}')
    print('-'*30)
    


# In[ ]:


def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)

def detect():
    source = '../input/global-wheat-detection/test/'
    weights = 'weights/best.pt'
    if not os.path.exists(weights):
        weights = WEIGHTS
    
    imagenames =  os.listdir(source)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load model
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    model.to(device).eval()
    
    dataset = LoadImages(source, img_size=1024)

    results = []
    fig, ax = plt.subplots(5, 2, figsize=(30, 70))
    count = 0
    
    for path, img, img0, vid_cap in dataset:
        image_id = os.path.basename(path).split('.')[0]
        img = img.transpose(1,2,0) # [H, W, 3]
        
        enboxes = []
        enscores = []
        
        # only rot, no flip
        if is_ROT:    
            for i in range(4):
                img1 = TTAImage(img, i)
                boxes, scores = detect1Image(img1, img0, model, device, aug=False)
                for _ in range(3-i):
                    boxes = rotBoxes90(boxes, *img.shape[:2])            
                enboxes.append(boxes)
                enscores.append(scores) 
        
        # flip
        boxes, scores = detect1Image(img, img0, model, device, aug=is_AUG)
        enboxes.append(boxes)
        enscores.append(scores) 
            
        boxes, scores, labels = run_wbf(enboxes, enscores, image_size=1024, iou_thr=best_iou_thr, skip_box_thr=best_skip_box_thr)    
        boxes = boxes.astype(np.int32).clip(min=0, max=1024)
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        
        indices = scores >= best_score_threshold
        boxes = boxes[indices]
        scores = scores[indices]
        
        if count<10:
            img_ = cv2.imread(path)  # BGR
            for box, score in zip(boxes,scores):
                cv2.rectangle(img_, (box[0], box[1]), (box[2]+box[0], box[3]+box[1]), (220, 0, 0), 2)
                cv2.putText(img_, '%.2f'%(score), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
            ax[count%5][count//5].imshow(img_)
            count+=1
            
        result = {
            'image_id': image_id,
            'PredictionString': format_prediction_string(boxes, scores)
        }

        results.append(result)
    return results


# In[ ]:


results = detect()
test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
test_df.to_csv('submission.csv', index=False)
test_df.head()


# In[ ]:


get_ipython().system('rm -rf convertor')

