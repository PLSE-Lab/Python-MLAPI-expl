#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import sys
sys.path.insert(0, "../input/weightedboxesfusion")
from ensemble_boxes import *
import glob


# In[ ]:


# !ls ../input/yolov5git/yolov5/utils/


# In[ ]:


get_ipython().system('cp -r ../input/yolov5tta/yolov5tta/* .')
get_ipython().system('cp -r ../input/yolov5git/yolov5/utils/* ./utils/')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'import argparse\n\nfrom utils.datasets import *\nfrom utils.utils import *\n\n\ndef detect(save_img=False):\n    weights, imgsz = opt.weights,opt.img_size\n    source = \'../input/global-wheat-detection/test/\'\n    \n    # Initialize\n    device = torch_utils.select_device(opt.device)\n    half = False\n    # Load model\n    models = []\n    for w in weights:\n        models.append(torch.load(w, map_location=device)[\'model\'].to(device).float().eval())\n\n\n    dataset = LoadImages(source, img_size=1024)\n\n    # Get names and colors\n\n    # Run inference\n    t0 = time.time()\n    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img\n    all_path=[]\n    all_bboxex =[]\n    all_score =[]\n    for path, img, im0s, vid_cap in dataset:\n        img = torch.from_numpy(img).to(device)\n        img = img.half() if half else img.float()  # uint8 to fp16/32\n        img /= 255.0  # 0 - 255 to 0.0 - 1.0\n        if img.ndimension() == 3:\n            img = img.unsqueeze(0)\n\n        # Inference\n        t1 = torch_utils.time_synchronized()\n        bboxes_2 = []\n        score_2 = []\n        for model in models:\n            pred = model(img, augment=opt.augment)[0]\n            pred = non_max_suppression(pred, 0.4, opt.iou_thres,fast=True, classes=None, agnostic=False)\n            t2 = torch_utils.time_synchronized()\n\n            bboxes = []\n            score = []\n            # Process detections\n            for i, det in enumerate(pred):  # detections per image\n                p, s, im0 = path, \'\', im0s\n                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh\n                if det is not None and len(det):\n                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()\n                    for c in det[:, -1].unique():\n                        n = (det[:, -1] == c).sum()  # detections per class\n\n                    for *xyxy, conf, cls in det:\n                        if True:  # Write to file\n                            xywh = torch.tensor(xyxy).view(-1).numpy()  # normalized xywh\n#                             xywh[2] = xywh[2]-xywh[0]\n#                             xywh[3] = xywh[3]-xywh[1]\n                            bboxes.append(xywh)\n                            score.append(conf)\n            bboxes_2.append(bboxes)\n            score_2.append(score)\n        all_path.append(path)\n        all_score.append(score_2)\n        all_bboxex.append(bboxes_2)\n    return all_path,all_score,all_bboxex\n\n\n\nclass opt:\n    weights_folds = list(glob.glob("../input/yolov5pth/*"))\n    weights_folds.sort()\n    weights = weights_folds[4:]\n    img_size = 1024\n    conf_thres = 0.1\n    iou_thres = 0.94\n    augment = True\n    device = \'0\'\n    classes=None\n    agnostic_nms = True\n        \nopt.img_size = check_img_size(opt.img_size)\n\n\nwith torch.no_grad():\n    res = detect()')


# In[ ]:


def run_wbf(boxes,scores, image_size=1024, iou_thr=0.4, skip_box_thr=0.34, weights=None):
    labels0 = [np.ones(len(scores[idx])) for idx in range(len(scores))]
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels0, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    return boxes, scores, labels


# In[ ]:


all_path,all_score,all_bboxex = res


# In[ ]:


results =[]
def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))
    return " ".join(pred_strings)
for row in range(len(all_path)):
    image_id = all_path[row].split("/")[-1].split(".")[0]
    boxes = all_bboxex[row]
    scores = all_score[row]
    boxes, scores, labels = run_wbf(boxes,scores)
    boxes = (boxes*1024/1024).astype(np.int32).clip(min=0, max=1023)
    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
    result = {'image_id': image_id,'PredictionString': format_prediction_string(boxes, scores)}
    results.append(result)
get_ipython().system('rm -rf *')
test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
test_df.to_csv('submission.csv', index=False)
test_df.head()


# In[ ]:


size = 300
idx =-1
font = cv2.FONT_HERSHEY_SIMPLEX 
image = image = cv2.imread(all_path[idx], cv2.IMREAD_COLOR)
fontScale = 1
color = (255, 0, 0) 

# Line thickness of 2 px 
thickness = 2
for b,s in zip(boxes,scores):
    image = cv2.rectangle(image, (b[0],b[1]), (b[0]+b[2],b[1]+b[3]), (255,0,0), 1) 
    image = cv2.putText(image, '{:.2}'.format(s), (b[0]+np.random.randint(3),b[1]), font,  
                   fontScale, color, thickness, cv2.LINE_AA)
plt.figure(figsize=[20,20])
plt.imshow(image[:,:,::-1])
plt.show()


# In[ ]:




