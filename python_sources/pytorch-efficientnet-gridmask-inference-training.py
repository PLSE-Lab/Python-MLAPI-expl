#!/usr/bin/env python
# coding: utf-8

# Due to Limited resources i am doing inference only, model training code is there you can uncomment the calling of train function and run it.
# 
# This model includes EfficientNet with GridMask.
# Most of the code taken from https://www.kaggle.com/abhishek/melanoma-detection-with-pytorch/
# if you find it useful please upvote.
# Thanks to Abhishek Thakur for awesome kernel.

# In[ ]:


get_ipython().system('pip install wtfml')
get_ipython().system('pip install pretrainedmodels')


# In[ ]:


get_ipython().system('pip install --user --upgrade efficientnet-pytorch')
get_ipython().system('pip install --user --upgrade albumentations')


# In[ ]:


import os
import torch 
import albumentations

import numpy as np
import pandas as pd

import torch.nn as nn
from sklearn import metrics
from sklearn import model_selection
from torch.nn import functional as F

from wtfml.utils import EarlyStopping
from wtfml.engine import Engine
from wtfml.data_loaders.image import ClassificationLoader

import pretrainedmodels
from efficientnet_pytorch import EfficientNet


# In[ ]:


class EfficientNetB1(nn.Module):
    def __init__(self, pretrained):
        super(EfficientNetB1, self).__init__()

        if pretrained is True:
            self.base_model = EfficientNet.from_pretrained("efficientnet-b7")
        else:
            self.base_model = EfficientNet.from_name("efficientnet-b7")
        
        self.l0 = nn.Linear(2560, 1)

    def forward(self, image, targets):
        batch_size, _, _, _ = image.shape
        x = self.base_model.extract_features(image)
        x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)
        out = self.l0(x)
        loss = nn.BCEWithLogitsLoss()(out, targets.view(-1, 1).type_as(x))
        return out, loss


# In[ ]:


import pandas as pd

import torch

from PIL import Image
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.augmentations import functional as F1
from PIL import Image, ImageOps, ImageEnhance

from albumentations.core.transforms_interface import DualTransform

class GridMask(DualTransform):
    """GridMask augmentation for image classification and object detection.

    Args:
        num_grid (int): number of grid in a row or column.
        fill_value (int, float, lisf of int, list of float): value for dropped pixels.
        rotate ((int, int) or int): range from which a random angle is picked. If rotate is a single int
            an angle is picked from (-rotate, rotate). Default: (-90, 90)
        mode (int):
            0 - cropout a quarter of the square of each grid (left top)
            1 - reserve a quarter of the square of each grid (left top)
            2 - cropout 2 quarter of the square of each grid (left top & right bottom)

    Targets:
        image, mask

    Image types:
        uint8, float32

    Reference:
    |  https://arxiv.org/abs/2001.04086
    |  https://github.com/akuxcw/GridMask
    """

    def __init__(self, num_grid=3, fill_value=0, rotate=0, mode=0, always_apply=False, p=0.5):
        super(GridMask, self).__init__(always_apply, p)
        if isinstance(num_grid, int):
            num_grid = (num_grid, num_grid)
        if isinstance(rotate, int):
            rotate = (-rotate, rotate)
        self.num_grid = num_grid
        self.fill_value = fill_value
        self.rotate = rotate
        self.mode = mode
        self.masks = None
        self.rand_h_max = []
        self.rand_w_max = []

    def init_masks(self, height, width):
        if self.masks is None:
            self.masks = []
            n_masks = self.num_grid[1] - self.num_grid[0] + 1
            for n, n_g in enumerate(range(self.num_grid[0], self.num_grid[1] + 1, 1)):
                grid_h = height / n_g
                grid_w = width / n_g
                this_mask = np.ones((int((n_g + 1) * grid_h), int((n_g + 1) * grid_w))).astype(np.uint8)
                for i in range(n_g + 1):
                    for j in range(n_g + 1):
                        this_mask[
                             int(i * grid_h) : int(i * grid_h + grid_h / 2),
                             int(j * grid_w) : int(j * grid_w + grid_w / 2)
                        ] = self.fill_value
                        if self.mode == 2:
                            this_mask[
                                 int(i * grid_h + grid_h / 2) : int(i * grid_h + grid_h),
                                 int(j * grid_w + grid_w / 2) : int(j * grid_w + grid_w)
                            ] = self.fill_value
                
                if self.mode == 1:
                    this_mask = 1 - this_mask

                self.masks.append(this_mask)
                self.rand_h_max.append(grid_h)
                self.rand_w_max.append(grid_w)

    def apply(self, image, mask, rand_h, rand_w, angle, **params):
        h, w = image.shape[:2]
        mask = F1.rotate(mask, angle) if self.rotate[1] > 0 else mask
        mask = mask[:,:,np.newaxis] if image.ndim == 3 else mask
        image *= mask[rand_h:rand_h+h, rand_w:rand_w+w].astype(image.dtype)
        return image

    def get_params_dependent_on_targets(self, params):
        img = params['image']
        height, width = img.shape[:2]
        self.init_masks(height, width)

        mid = np.random.randint(len(self.masks))
        mask = self.masks[mid]
        rand_h = np.random.randint(self.rand_h_max[mid])
        rand_w = np.random.randint(self.rand_w_max[mid])
        angle = np.random.randint(self.rotate[0], self.rotate[1]) if self.rotate[1] > 0 else 0

        return {'mask': mask, 'rand_h': rand_h, 'rand_w': rand_w, 'angle': angle}

    @property
    def targets_as_params(self):
        return ['image']

    def get_transform_init_args_names(self):
        return ('num_grid', 'fill_value', 'rotate', 'mode')


# 

# In[ ]:


# create train folds
df = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")
df["kfold"] = -1
df = df.sample(frac=1).reset_index(drop=True)
y = df.target.values
kf = model_selection.StratifiedKFold(n_splits=5)
for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
    df.loc[v_, 'kfold'] = f

df.to_csv("train_folds.csv", index=False)


# 

# In[ ]:


def train(fold):
    training_data_path = "../input/siic-isic-224x224-images/train/"
    df = pd.read_csv("/kaggle/working/train_folds.csv")
    device = "cuda"
    epochs = 1  #increase it to 50
    train_bs = 8
    valid_bs = 4
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    model = EfficientNetB1(pretrained=True)
    model.to(device)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_aug = albumentations.Compose(
    [
        albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        albumentations.ShiftScaleRotate(shift_limit=0.0625,
                                        scale_limit=0.1, rotate_limit=15),
        albumentations.Flip(p=0.5),
        albumentations.OneOf([
                    GridMask(num_grid=3, mode=0, rotate=15),
                    GridMask(num_grid=3, mode=2, rotate=15),
                ], p=0.75)
    ]
    )
    
    valid_aug = albumentations.Compose(
    [
        albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
        
    ]
    )
    train_images = df_train.image_name.values.tolist()
    train_images = [os.path.join(training_data_path, i+".png" ) for i in train_images]
    train_targets = df_train.target.values
    valid_images = df_valid.image_name.values.tolist()
    valid_images = [os.path.join(training_data_path, i+".png") for i in valid_images]
    valid_targets = df_valid.target.values
    
    train_dataset = ClassificationLoader(
            image_paths = train_images,
            targets = train_targets,
            resize = None,
            augmentations = train_aug,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_bs, shuffle=True, num_workers=4
    )
    
    valid_dataset = ClassificationLoader(
        image_paths = valid_images,
        targets = valid_targets,
        resize = None,
        augmentations = valid_aug,
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size = valid_bs, shuffle=False, num_workers=4
    )
    
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=3,
                threshold=0.001,
                mode="max",
    )
    
    es = EarlyStopping(patience=5, mode="max")
    for epoch in range(epochs):
        train_loss = Engine.train(train_loader, model, optimizer, device=device)
        predictions, valid_loss = Engine.evaluate(valid_loader, model, device=device)
        predictions = np.vstack((predictions)).ravel()
        auc = metrics.roc_auc_score(valid_targets, predictions)
        print(f"Epoch = {epoch}, AUC = {auc}")
        scheduler.step(auc)
        es(auc, model,model_path=f"model_fold_{fold}.bin")
        if es.early_stop:
            print("Early Stopping")
            break


# In[ ]:


def predict(fold):
    test_data_path = "../input/siic-isic-224x224-images/test/"
    df = pd.read_csv("../input/siim-isic-melanoma-classification/test.csv")
    device = "cuda"
    if fold == 1:
        model_path = f"../input/pytorch-efficientnet-gridmask/model_fold_{fold}.bin"
    else:
        model_path = f"../input/modelsmelanoma/model_fold_{fold}.bin"
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
        ]
    )
    images = df.image_name.values.tolist()
    images = [os.path.join(test_data_path, i+".png") for i in images]
    targets = np.zeros(len(images))
    test_dataset = ClassificationLoader(
        image_paths = images,
        targets = targets,
        resize = None,
        augmentations = aug,
    )
    test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=16, shuffle=False,
            num_workers=4
    )
    model = EfficientNetB1(pretrained=False)
    model.load_state_dict(torch.load(model_path),strict=False)
    model.to(device)
    
    predictions = Engine.predict(test_loader, model, device=device)
    predictions = np.vstack((predictions)).ravel()
    return predictions


# In[ ]:


import gc
gc.collect()


# In[ ]:


# train(0)


# Uncomment the below calling of train function.

# In[ ]:


# train(1)


# In[ ]:


# train(2)


# In[ ]:


# train(3)


# In[ ]:


# train(4)


# In[ ]:





# In[ ]:


p1 = predict(0)
p2 = predict(1)
p3 = predict(2)
p4 = predict(3)
p5 = predict(4)


# In[ ]:


predictions = (p1+p2+p3+p4+p5)/5 
sample = pd.read_csv("../input/siim-isic-melanoma-classification/sample_submission.csv")
sample.loc[:,"target"] = predictions
sample.to_csv("submission.cv", index=False)


# In[ ]:


sample.head()


# In[ ]:




