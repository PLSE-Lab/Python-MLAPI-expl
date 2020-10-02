#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
# import module we'll need to import our custom module
import os
from shutil import copyfile
# copy our file into the working directory (make sure it has .py suffix)
for dirname, _, filenames in os.walk('../input/maskrcnn-utils/'):
    for filename in filenames:
        print(filename)
        copyfile(src = "../input/maskrcnn-utils/{:s}".format(filename), dst = "../working/{:s}".format(filename))
        
from engine import train_one_epoch, evaluate
import utils
import transforms as T
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.utils.data
from PIL import Image, ImageFile
import pandas as pd
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


# In[ ]:


def _get_instance_segmentation_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# In[ ]:


import collections
import os
import numpy as np


class OpenDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, df_path, height, width, transforms=None):
        self.transforms = transforms
        self.image_dir = image_dir
        self.df = pd.read_csv(df_path)
        self.height = height
        self.width = width
        self.image_info = collections.defaultdict(dict)
        
        # Filling up image_info is left as an exercise to the reader
        

    def __getitem__(self, idx):
        # load images ad masks
        img_path = self.image_info[idx]["image_path"]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.width, self.height), resample=Image.BILINEAR)
        
        # processing part and extraction of boxes is left as an exercise to the reader

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.image_info)


# In[ ]:


num_classes = 10 # define number of classes here
device = torch.device('cuda:0')

dataset_train = OpenDataset("../input/train/", "../input/train.csv", 128, 128, transforms=None)

model_ft = get_instance_segmentation_model(num_classes)
model_ft.to(device)

data_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=4, shuffle=True, num_workers=8,
    collate_fn=utils.collate_fn)

params = [p for p in model_ft.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=5,
                                               gamma=0.1)
num_epochs = 8
for epoch in range(num_epochs):
    train_one_epoch(model_ft, optimizer, data_loader, device, epoch, print_freq=10)
    lr_scheduler.step()

torch.save(model_ft.state_dict(), "model.bin")


# #### I have left certain parts as exercise to the reader. This model is quite fast and should give good results if used properly. In case of any questions, feel free to ask. Upvotes are appreciated if this helps you.
