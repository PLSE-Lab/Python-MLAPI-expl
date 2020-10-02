#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Download TorchVision repo to use some files from
# references/detection
get_ipython().system('git clone https://github.com/pytorch/vision.git')

get_ipython().system('cp /kaggle/working/vision/references/detection/utils.py /kaggle/working/')
get_ipython().system('cp /kaggle/working/vision/references/detection/transforms.py /kaggle/working/')
get_ipython().system('cp /kaggle/working/vision/references/detection/coco_eval.py /kaggle/working/')
get_ipython().system('cp /kaggle/working/vision/references/detection/engine.py /kaggle/working/')
get_ipython().system('cp /kaggle/working/vision/references/detection/coco_utils.py /kaggle/working/')


# In[ ]:


pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'


# In[ ]:


import torch
import torchvision
import torch.nn
import pandas
import re
import os
import zipfile
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image
from engine import train_one_epoch, evaluate
import transforms as T
import utils
import matplotlib.patches as patches


# In[ ]:


DIR = "/kaggle/input/global-wheat-detection/"
pd_train = pandas.read_csv(DIR+"train.csv")


# # **COLLECTING DATA TO ARRAY**

# In[ ]:


new_bbox = []

image_id = pd_train["image_id"].to_numpy()
source = pd_train['source'].to_numpy()
bbox = pd_train["bbox"].to_numpy()

for b in bbox:
    #save new bbox
    box = b.strip("[]").split(",")
    x,y,w,h = np.array(box).astype(float)
    new_bbox.append([float(box[0]),float(box[1]),float(box[2]),float(box[3])])
    


# In[ ]:


class dictImage():
    def __init__(self, DIR, image_id, bbox):
        self.DIR = DIR
        self.image_id = image_id
        self.bbox = bbox #like a coordinate from image
        self.dict_image = {}
        self.list_data = []
    
    def buildDict(self):
        self.dict_image = {}
        idx = 0
        for image_id, bbox in zip(self.image_id, self.bbox):
            path = self.DIR+"/train/"+ image_id+".jpg"
            
            if image_id not in self.dict_image:
                self.dict_image[image_id] = {}
                self.dict_image[image_id]["path"] =  path
                self.dict_image[image_id]["bbox"] = [bbox]
            else:
                self.dict_image[image_id]["bbox"].append(bbox)

    def convertToListData(self):
        self.list_data = []
        for key, val in self.dict_image.items():
            self.list_data.append(val)


# In[ ]:


data_train = dictImage(DIR, image_id, new_bbox)
data_train.buildDict()
data_train.convertToListData()


# In[ ]:


len(data_train.list_data)


# # BUILD DATASET

# In[ ]:


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
transf_train = get_transform(train=True)
transf_test = get_transform(train=False)


# In[ ]:


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, list_data, transforms=None):
        self.transforms = transforms
        self.list_data = list_data
        self.reduce_dim = 0.25
    
    def getMasks(self, path, bbox):
        sv = True
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        nw = int(self.reduce_dim * img.shape[0])
        nh = int(self.reduce_dim * img.shape[1])
        img = cv2.resize(img,(nw,nh)) #reduce dimention of image for reduce time consuming
        
        masks = np.zeros_like(img)
        x,y,w,h =  bbox
        x1 = int(x * self.reduce_dim)
        x2 = int((x+w) * self.reduce_dim)
        y1 = int(y * self.reduce_dim)
        y2 = int((y+h) * self.reduce_dim)
        
        masks[y1:y2, x1:x2] = 1 #remember that arrays always start from row to column or y to x
        
        if(x1 == x2):
            sv = False
        if(y1 == y2):
            sv = False
        
        xy = [x1,y1,x2,y2]
        
        return masks, xy, sv

    def __getitem__(self, idx):
        data = self.list_data[idx]
        path = data['path']
        
        #reduce image dimention to reduce time processing
        img = Image.open(path).convert("RGB")
        nw = int(self.reduce_dim * img.size[0])#new width
        nh = int(self.reduce_dim * img.size[1])#new height
        img = img.resize((nw,nh))#new dimention image
        
        masks = []
        n_bbox = []
        lbl = []
        
        for bbox in data['bbox']:
            
            bbox = np.array(bbox).astype(int)
            m, box, sv = self.getMasks(path, bbox) #sv variable to validate data whether it is used or not
            if sv == True:
                lbl.append(1)
                masks.append(m)
                n_bbox.append(box)
                
        target = {}
        target["boxes"] = torch.as_tensor(n_bbox, dtype=torch.float32)
        target["labels"] = torch.as_tensor(lbl, dtype=torch.int64)
        target["masks"] = torch.as_tensor(masks, dtype=torch.uint8)
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
    
    def __len__(self):
        return len(self.list_data)


# In[ ]:


torch.manual_seed(1)
dataset = BuildDataset(data_train.list_data, transf_train)

# define training data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=5, shuffle=True, num_workers=6,
    collate_fn=utils.collate_fn)


# In[ ]:


dataset[0]


# # **DEFINE MODEL**

# In[ ]:


def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


# In[ ]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 2#background and wheat

# get the model using our helper function
model = get_instance_segmentation_model(num_classes)
# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)
print("device",device)


# **LET'S TRAIN DATA**

# I only did training for 2 epochs. I have previously tried training 5 epochs but I did not have enough time because I was limited to 9 hours for this assignment

# In[ ]:


# let's train it for 10 epochs
num_epochs = 2

for epoch in range(num_epochs):
    # train for one epoch, printing every 100 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=20)
    # update the learning rate
    lr_scheduler.step()


# In[ ]:


def showImage(img, boxes, scores):
    img = img.mul(255) 
    img = img.permute(1,2,0).byte().cpu().numpy()
    fig,ax = plt.subplots(1)
    fig.set_figheight(8)
    fig.set_figwidth(8)
    ax.imshow(img)
    
    for box, score in zip(boxes, scores):
        if score >= 0.5:
            x1,y1,x2,y2= box
            x = x1 
            y = y1
            w = x2 - x1
            h = y2-y1
            rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
    plt.show()


# In[ ]:


model.eval()
dir_test = DIR+"test/"

list_test = os.listdir(dir_test)
for f in list_test:
    pth = dir_test + f
    # pth = data_train.list_data[100]['path']
    img, _ = transf_test(Image.open(pth),None)
    img = img.unsqueeze(0).cuda()
    predict = model(img)
    boxes = predict[0]['boxes']
    scores = predict[0]['scores']
    img = img.squeeze(0)
    showImage(img, boxes, scores)


# In[ ]:




