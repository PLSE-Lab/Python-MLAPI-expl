#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('ls .')


# In[ ]:


get_ipython().system('pip install ../input/packages/webcolors-1.11.1-py3-none-any.whl ')


# In[ ]:


get_ipython().system('cp -r ../input/efficientdet-code/EfficientDet .')


# In[ ]:


get_ipython().system('mv ../input/weights2/efficientdet-d3_47_160500.pth ./EfficientDet/weights')


# In[ ]:


cd EfficientDet


# In[ ]:


get_ipython().system('ls .')


# In[ ]:


"""
Simple Inference Script of EfficientDet-Pytorch
"""
import os
import time
import torch
from torch.backends import cudnn
from matplotlib import colors

from backbone import EfficientDetBackbone
import cv2
import numpy as np
import pandas as pd

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box

from ensemble_boxes import *

threshold = 0.2
iou_threshold = 0.7
compound_coef = 3

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True
force_input_size = 1024  # set None to use default size

img_path = '../../input/global-wheat-detection/test'
obj_list = ['wheat']
P_img_ext = ["jpg", "png", "bmp"]
model_path = 'weights/efficientdet-d3_47_160500.pth'

# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

color_list = standard_to_bgr(STANDARD_COLORS)
# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size


def file_list(path, allfile):
    filelist = os.listdir(path)

    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            file_list(filepath, allfile)
        else:
            if filepath.split(".")[-1] in P_img_ext:
                allfile.append(filepath.strip())
    return allfile


def display(preds, imgs, img_save_name, imshow=True, imwrite=False):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])
            plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj,score=score,color=color_list[get_index_label(obj, obj_list)])

        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)

        if imwrite:
            cv2.imwrite('test/{}_mark.jpg'.format(img_save_name), imgs[i])


model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list), ratios=anchor_ratios,
                             scales=anchor_scales)
model.load_state_dict(torch.load(model_path))
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()


img_list = []
img_list = file_list(img_path, img_list)

# commit submission
submission = []

for img in img_list:
    print("Processing --->", img)
    prediction_string = []

    ori_imgs, framed_imgs, framed_metas = preprocess(img, max_size=input_size)

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    with torch.no_grad():
        features, regression, classification, anchors = model(x)
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()
        out = postprocess(x,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)

    out = invert_affine(framed_metas, out)
    
    # filter abnormal box
    if len(out[0]['rois']) != 0:
        index_d = []
        for idx, box in enumerate(out[0]['rois']):
            x1, y1, x2, y2 = box
            if x2-x1>400 and y2 -y1 >400:
                index_d.append(idx)

        out[0]['rois'] = np.delete(out[0]['rois'], index_d, axis=0)
        out[0]['scores'] = np.delete(out[0]['scores'], index_d)
        out[0]['class_ids'] = np.delete(out[0]['class_ids'], index_d)

    # WBF
    boxes_list = (1.0/1024*out[0]['rois']).tolist()
    scores_list = out[0]['scores'].tolist()
    labels_list = out[0]['class_ids'].tolist()
    boxes, scores, labels = weighted_boxes_fusion([boxes_list], [scores_list], [labels_list], weights=None, iou_thr=0.45, skip_box_thr=0.01)

    out[0]['rois'] = np.array(boxes)*1024
    out[0]['scores'] = np.array(scores)
    out[0]['labels'] = np.array(labels)    
    
    #display(out, ori_imgs, img_name_save, imshow=False, imwrite=True)

    if len(out[0]['rois']) == 0:
        prediction_string.append("")
    else:
        for j in range(len(out[0]['rois'])):
            x1, y1, x2, y2 = out[0]['rois'][j].astype(np.int)
            score = float(out[0]['scores'][j])

            x = x1
            y = y1
            w = x2 - x1
            h = y2 - y1
            s = float(score)
            prediction_string.append("{} {} {} {} {}".format(s, x, y, w, h))
            
    img_name_save = os.path.basename(img)[:-4]
    prediction_string = " ".join(prediction_string)
    submission.append([img_name_save,prediction_string])

#print(submission)
sample_submission = pd.DataFrame(submission, columns=["image_id", "PredictionString"])
sample_submission.to_csv('../submission.csv', index=False)


# In[ ]:


cd ..


# In[ ]:


get_ipython().system('rm -rf EfficientDet')


# In[ ]:


get_ipython().system('ls .')

