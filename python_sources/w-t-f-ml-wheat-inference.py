#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install --no-deps ../input/wtfml/


# In[ ]:


import os
import ast

import pandas as pd
import numpy as np

import albumentations
import argparse
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from sklearn import metrics
from sklearn.model_selection import train_test_split
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from wtfml.engine import RCNNEngine
from wtfml.data_loaders.image import RCNNLoader

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from PIL import Image
from PIL import ImageFile
import matplotlib.patches as patches


# In[ ]:


def format_prediction_string(boxes, scores):
    # function taken from: https://www.kaggle.com/arunmohan003/fasterrcnn-using-pytorch-baseline
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)


def collate_fn(batch):
    return tuple(zip(*batch))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=False, 
            pretrained_backbone=False
        )
        in_features = self.base_model.roi_heads.box_predictor.cls_score.in_features
        self.base_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    def forward(self, images, targets):
        if targets is None:
            return self.base_model(images, targets)
        else:
            output = self.base_model(images, targets)
            if isinstance(output, list):
                return output
            loss = sum(loss for loss in output.values())
            return loss


# In[ ]:


class args:
	data_path = "../input/global-wheat-detection"
	epochs = 30
	device = "cuda"


# In[ ]:


model = Model()
model.to(args.device)

model.load_state_dict(
    torch.load('../input/wtfwheat/model.bin')
)

mean = (0., 0., 0.)
std = (1, 1, 1)
aug = albumentations.Compose(
    [albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)]
)

test_df = pd.read_csv(os.path.join(args.data_path, "sample_submission.csv"))
test_df.loc[:, "bbox"] = ["[0, 0, 10, 10]"] * len(test_df)
test_df.bbox = test_df.bbox.apply(ast.literal_eval)
test_df = test_df.groupby('image_id')['bbox'].apply(list).reset_index(name='bboxes')
test_image_ids = test_df.image_id.values

images = test_df.image_id.values.tolist()
images = [os.path.join(args.data_path, "test", i + ".jpg") for i in images]
targets = test_df.bboxes.values

test_dataset = RCNNLoader(
    image_paths=images, 
    bounding_boxes=targets, 
    augmentations=aug
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn
)

prediction_strings = []
predictions = RCNNEngine.predict(test_loader, model, device=args.device)
for p in predictions:
    boxes = p['boxes'].numpy()
    scores = p['scores'].numpy()

    boxes = boxes[scores >= 0.5].astype(np.int32)
    scores = scores[scores >= 0.5]

    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
    prediction_strings.append(format_prediction_string(boxes, scores))

sample = pd.read_csv(os.path.join(args.data_path, "sample_submission.csv"))
sample.loc[:, "PredictionString"] = prediction_strings
sample.loc[:, "image_id"] = test_image_ids
sample.to_csv("submission.csv", index=False)


# In[ ]:


for idx in range(len(sample)):
    npimage = np.array(Image.open(os.path.join(args.data_path, "test", sample.image_id.values[idx] + ".jpg")))
    boxes = sample.PredictionString.values[idx]
    boxes = [x for i, x in enumerate(boxes.split()) if i % 5 != 0]
    boxes = np.array(boxes).reshape(-1, 4).astype(float)

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    for b in boxes:
        ax.add_patch(
         patches.Rectangle(
            (b[0], b[1]),
            b[2],
            b[3],
            fill=False,
            color='red')
        )
    ax.set_axis_off()
    ax.imshow(npimage)


# In[ ]:




