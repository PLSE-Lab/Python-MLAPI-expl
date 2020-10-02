#!/usr/bin/env python
# coding: utf-8

# This is a framework that I have been working on. It commits, so it works to an extent; however, it receives a submission scoring error. The framework runs multiple models, averages predictions depending on overlap, and runs a CNN on the final outcome. If you use, please cite or ask to team up.

# # Import Statements

# In[ ]:


import pandas as pd
import numpy as np
import cv2
import os
import re
from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
from matplotlib import pyplot as plt
import os
import sys
sys.path.insert(0, "../input/weightedboxesfusion")
from ensemble_boxes import *


# # File Locations

# In[ ]:


DIR_INPUT = '/kaggle/input/global-wheat-detection'
DIR_TRAIN = f'{DIR_INPUT}/train'
DIR_TEST = f'{DIR_INPUT}/test'
DIR_WEIGHTS = '/kaggle/input/first-group-of-models/'


# # Apply Test Function

# In[ ]:


def applytest(model,epoch_number):
    DIR_INPUT = '/kaggle/input/global-wheat-detection'
    DIR_TEST = f'{DIR_INPUT}/test'
    
    test_df = os.listdir(DIR_TEST)
    
    for i,img in enumerate(test_df):
        test_df[i]=test_df[i][:-4]
    
    class WheatTestDataset(Dataset):
    
        def __init__(self, dataframe, image_dir, transforms=None):
            super().__init__()
    
            self.image_ids = dataframe
            self.df = dataframe
            self.image_dir = image_dir
            self.transforms = transforms
    
        def __getitem__(self, index: int):
    
            
            DIR_INPUT = '/kaggle/input/global-wheat-detection'
            DIR_TEST = f'{DIR_INPUT}/test'
            image_id = os.listdir(DIR_TEST)
            image_id = image_id[index]
            #image_id = self.image_ids[index]
            #records = self.df[self.df['image_id'] == image_id]
    
            image = cv2.imread(DIR_TEST+'/'+image_id, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image /= 255.0
    
            if self.transforms:
                sample = {
                    'image': image,
                }
                sample = self.transforms(**sample)
                image = sample['image']
    
            return image, image_id
    
        def __len__(self) -> int:
            return len(self.image_ids)
        
    # Albumentations
    def get_test_transform():
        return A.Compose([
            # A.Resize(512, 512),
            ToTensorV2(p=1.0)
        ])
    
    
    
    x = model.to(device)
    
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    test_dataset = WheatTestDataset(test_df, DIR_TEST, get_test_transform())
    
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_fn
    )
    
    def format_prediction_string(boxes, scores):
        pred_strings = []
        for j in zip(scores, boxes):
            pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))
    
        return " ".join(pred_strings)
    
    detection_threshold = 0.5
    results = []
    
    allboxes = []
    allimage_ids = []
    allscores = []
    for images, image_ids in test_data_loader:
        #Batch Size = 1, so only one image per iteration in this loop
        print(image_ids)
        images = list(image.to(device) for image in images)
        outputs = model(images)
        
            
        #We want the boxes, how to store the boxes?
        #boxes is just a list of boxes for each images, so this is just a list of one
        sample = images[0].permute(1,2,0).cpu().numpy()
        boxes = outputs[0]['boxes'].data.cpu().numpy()
        scores = outputs[0]['scores'].data.cpu().numpy()
        
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        scores = scores[scores >= detection_threshold]
        allimage_ids.append(image_ids)
        allboxes.append(boxes)
        allscores.append(scores)
    
    return allboxes, allimage_ids, allscores


# # Apply Test to All Test Images

# In[ ]:


path = DIR_WEIGHTS
model_dir = os.listdir(path)

boxes = []
ids = []
scores = []
for i,model_weights in enumerate(model_dir):
    if i >= 0:
        #load a model; pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
            
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            
        num_classes = 2  # 1 class (wheat) + background
            
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
            
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            
        # Load the trained weights
        model.load_state_dict(torch.load(path+model_weights, map_location=torch.device('cpu')))
        model.eval()
        tboxes, tids, tscores = applytest(model,i)
        boxes.append(tboxes)
        ids.append(tids)
        scores.append(tscores)


# # Bounding Box Functions

# In[ ]:


def overlapped(r1,r2):
    overlap = 0
    #Top Right
    
    if r1[0]<=r2[2] and r1[0]>=r2[0]:
        if r1[3]>=r2[3] and r1[3]<=r2[1]:
            overlap = 1
    
    #Top Left
    if r1[2]>=r2[0] and r1[2]<=r2[2]:
        if r1[3]<=r2[1] and r1[3]>=r2[3]:
            overlap = 1
    
    #Bottom Right
    if r2[2]<=r1[2] and r2[2]>=r1[0]:
        if r2[3]<=r1[1] and r2[3]>=r1[3]:
            overlap = 1
    
    #Bottom Left
    if r2[0]<=r1[2] and r2[0]>=r1[0]:
        if r2[3] <= r1[1] and r2[1] >= r1[3]:
            overlap = 1
    
    #Other
    if r1[0]>=r2[0] and r1[0]<=r2[2]:
        if r2[1]<=r1[1] and r2[1]>=r1[3]:
            overlap = 1
            
    #Other
    if r1[0]>=r2[0] and r1[0]<=r2[2]:
        if r1[1]<=r2[1] and r1[1]>=r2[3]:
            overlap = 1
    
    return overlap

def rec_overlap(r1,r2):
    p1x = max(r1[0],r2[0])
    p1y = min(r1[1],r2[1])
    p2x = min(r1[2],r2[2])
    p2y = max(r1[3],r2[3])
    
    r3 = list([p1x,p1y,p2x,p2y])
    overlap = abs(p1x-p2x)*abs(p1y-p2y)
    
    area = abs(r1[0]-r1[2])*abs(r1[1]-r1[3])
    poverlap = overlap/area
    
    return poverlap, r3


def groupboxes(boxes):
    #1040 = size of image
    boxes = np.asarray([boxes[:,0], 1040 - boxes[:,1], boxes[:,2], 1040 - boxes[:,3]])
    boxes = boxes.transpose()
    n = boxes.shape[0]
    areas = np.zeros([n,n])

    for i in range(0,n):
        control = 1
        for j in range(0,n):
            r0 = list([boxes[i,0],boxes[i,1],boxes[i,2],boxes[i,3]])
            r1 = list([boxes[j,0],boxes[j,1],boxes[j,2],boxes[j,3]])
            [p,r2] = rec_overlap(r0,r1)
    
            overlap = overlapped(r0,r1)
            
            if p <= 1 and overlap == 1 and p != 0:
                if i >= 0:
                    areas[i,j]=p

    group = np.zeros([n,1])
    group_count = 1

    for i in range(0,n):
        for j in range(0,n):
            if areas[i,j] > .5:
                if areas [j,i] > .5:
                    if group[i] == 0:
                        if group[j] == 0:
                            group[i] = group_count
                            group[j] = group_count
                            group_count = group_count + 1
                            continue
                        
                        group[i] = group[j]
                        continue
                    
                    if group[j] == 0:
                        group[j] = group[i]
                        continue
                    
                    group[group==group[j]]=group[i]
    
    return areas,group
                
def average_boxes(boxes,group,scores):
    u = np.unique(group)
    ab = []
    ts = []
    for i in range(0,u.shape[0]):
        tgroup,temp = np.where(group==u[i])
        gboxes = boxes[tgroup.astype(int),:]
        tscores = scores[tgroup.astype(int)]
        tscores = np.mean(tscores)
        
        p1 = np.mean(gboxes[:,0])
        p2 = np.mean(gboxes[:,1])
        p3 = np.mean(gboxes[:,2])
        p4 = np.mean(gboxes[:,3])
        p4 = p4-p2
        p3 = p3-p1
        
        average_box = list([p1,p2,p3,p4])
        ab.append(average_box)
        ts.append(tscores)
    
    ab = np.asarray(ab)
    ts = np.asarray(ts)
    return ab,ts

def run_wbf(boxes,scores,labels,image_size=1024, iou_thr=0.55, skip_box_thr=0.7, weights=None):
#     boxes =boxes/(image_size-1)
#    labels0 = [np.ones(len(scores[idx])) for idx in range(scores.shape[0])]
    labels0 = labels
    boxes, scores, labelst = weighted_boxes_fusion(boxes, scores, labels0, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
#     boxes = boxes*(image_size-1)
#     boxes = boxes
    return boxes, scores, labelst


# # Formating Prediction Strings

# In[ ]:


def format_prediction_string(boxes, scores):
        pred_strings = []
        for j in zip(scores, boxes):
            pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))
    
        return " ".join(pred_strings)


# # Apply CNN

# In[ ]:


from random import random
from tensorflow.keras.models import load_model
saved_model = load_model("../input/wheatcnn/mymodel_30k.h5")


# In[ ]:


import pylab


# In[ ]:


def applypadding(img):
    h = img.shape[0]
    w = img.shape[1]
    x = round(random()*(224 - h))
    y = round(random()*(224 - w))
    
    imgtemp = np.zeros((224,224,3))
    try:
        imgtemp[x:x+h,y:y+w,:] = img
    except:
        try:
            imgtemp[0:224,y:y+w,:] = img[0:224,:,:]
        except:
            try:
                imgtemp[x:x+h,0:224,:] = img[:,0:224,:]
            except:
                imgtemp[0:224,0:224,:] = img[0:224,0:224,:]
    return imgtemp

def applycnn(img_sample,model,bb,scores):
    imgs = []
    print(bb.shape)
    count = 0
    for box in bb:
        count += 1
        temp_img = img_sample[box[1]:box[1]+box[3],box[0]:box[0]+box[2],:]
        temp_img = applypadding(temp_img)
        imgs.append(temp_img)
    
    pylab.imshow(temp_img)
    temp_imgs = imgs
    imgs = np.asarray(imgs)
    outputs = saved_model.predict(imgs)
    
    #for i,output in enumerate(outputs):
        #t = Image.fromarray(temp_imgs[i].astype(np.uint8))
        #t.save('/kaggle/working/'+str(output)+'.png')
    return outputs
    


# # Averaging Bounding Boxes/Scores

# In[ ]:


import os
imgs = os.listdir(DIR_TEST)
results = []
ab_s = []
ts_s = []
ab_nocnn = []
ts_nocnn = []
for i,img in enumerate(imgs):
    print(i)
    tboxes = []
    tscores = []
    
    #j = number of models,i = image
    for j in range(0,len(boxes)):
        tboxes.append(boxes[j][i])
        tscores.append(scores[j][i])
    
    #Concatenates list of boxes and scores extracted from multiple models
    bboxes = np.vstack(tboxes)
    iscores = np.hstack(tscores)
    labels = np.ones(iscores.shape[0])
    
    bboxes = [bboxes.tolist()]
    iscores = [iscores.tolist()]
    labels = [labels.tolist()]
    
    ab,ts,labels = run_wbf(bboxes,iscores,labels)
    #areas,group = groupboxes(bboxes)
    #ab,ts = average_boxes(bboxes,group,iscores)
    
    ab = ab.astype(int)
    
    for j in range(0,ab.shape[0]):
        ab[j][2] = ab[j][2]-ab[j][0]
        ab[j][3] = ab[j][3]-ab[j][1]
    
    ab_nocnn.append(ab)
    ts_nocnn.append(ts)
    check = 1
    if check == 1:
        try:
            sample = cv2.imread(DIR_TEST+'/'+img)
            pylab.imshow(sample)
            #Next line was not shadowed last time
            outputs = applycnn(sample,saved_model,ab,ts)
            count = 0
            for i,output in enumerate(outputs):
                if output < .2:
                    ab = np.delete(ab,count,axis=0)
                    ts = np.delete(ts,count,axis=0)
                else:
                    count = count + 1
        except:
            print('Did not work')

    
    ab_s.append(ab)
    ts_s.append(ts)
    
    result = {
        'image_id': img[:-4],
        'PredictionString': format_prediction_string(ab,ts)
    }
    results.append(result)


# # Test Images

# In[ ]:


for i in range(0,9):
    print("-------------CNN---------------")
    
    sample = cv2.imread(DIR_TEST+'/'+imgs[9-i], cv2.IMREAD_COLOR)
    sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    boxes = ab_s[9-i]
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    for box in boxes:
        cv2.rectangle(sample,
                      (box[0], box[1]),
                      (box[0]+box[2], box[1]+box[3]),
                      (220, 0, 0), 2)

    ax.set_axis_off()
    ax.imshow(sample)

    print("-------------No CNN---------------")
    
    sample = cv2.imread(DIR_TEST+'/'+imgs[9-i], cv2.IMREAD_COLOR)
    sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    boxes = ab_nocnn[9-i]
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    for box in boxes:
        cv2.rectangle(sample,
                      (box[0], box[1]),
                      (box[0]+box[2], box[1]+box[3]),
                      (220, 0, 0), 2)

    ax.set_axis_off()
    ax.imshow(sample)


# # Submit

# In[ ]:


test_df = pd.DataFrame(results, columns = ['image_id', 'PredictionString'])


# In[ ]:


test_df.to_csv('submission.csv', index=False)

