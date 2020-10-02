#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install ../input/pretrainedmodels/pretrainedmodels-0.7.4/pretrainedmodels-0.7.4/ > /dev/null')
package_path = '../input/unetmodelscripts/'
import sys
sys.path.append(package_path)


# In[ ]:


import pdb
import os
import cv2
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from albumentations import (VerticalFlip, HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.pytorch import ToTensor
import torch.utils.data as data
from resnet_unet import Unet


# In[ ]:


#https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# In[ ]:


class TestDataset(Dataset):
    '''Dataset for test prediction'''
    def __init__(self, root, df, mean, std,TTA=None):
        self.root = root
        df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
        self.fnames = df['ImageId'].unique().tolist()
        self.num_samples = len(self.fnames)
        if TTA=="h":
            self.transform = Compose(
                [
                    HorizontalFlip(p=1),
                    Normalize(mean=mean,std=std,p=1),
                    ToTensor(),
                ]
            )
        elif TTA=="v":
            self.transform = Compose(
                [
                    VerticalFlip(p=1),
                    Normalize(mean=mean,std=std,p=1),
                    ToTensor(),
                ]
            )            
        else:
            self.transform = Compose(
                [
                    Normalize(mean=mean, std=std, p=1),
                    ToTensor(),
                ]
            )

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.root, fname)
        image = cv2.imread(path)
        images = self.transform(image=image)["image"]
        return fname, images

    def __len__(self):
        return self.num_samples


# In[ ]:


def post_process(probability, threshold, min_size):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((256, 1600), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


# In[ ]:


sample_submission_path = '../input/severstal-steel-defect-detection/sample_submission.csv'
test_data_folder = "../input/severstal-steel-defect-detection/test_images"


# In[ ]:


# initialize test dataloader
best_threshold = 0.4
num_workers = 2
batch_size = 1
print('best_threshold', best_threshold)
min_size = 3500
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
df = pd.read_csv(sample_submission_path)
testset = DataLoader(
    TestDataset(test_data_folder,df,mean,std),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)
testset_hTTA = DataLoader(
    TestDataset(test_data_folder,df,mean,std,TTA="h"),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)
testset_vTTA = DataLoader(
    TestDataset(test_data_folder,df,mean,std,TTA="v"),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)


# In[ ]:


# Initialize mode and load trained weights
predictions = []
# encoder_name = ["resnet18","resnet34"]
encoder_name = ["resnet34"]
model_name = ["uresnet34v3"]
model_dir = ["resnet34"]
models = []
for i in range(len(model_name)):
    ckpt_path = "../input/{}/{}.pth".format(model_dir[i],model_name[i])
    device = torch.device("cuda")
    model = Unet(encoder_name=encoder_name[i],encoder_weights = None,classes=4,activation=None)
    model.to(device)
    model.eval()
    state = torch.load(ckpt_path,map_location=lambda storage,loc:storage)
    model.load_state_dict(state["state_dict"])
    models.append(model)


# In[ ]:


# start prediction
predictions = []
for i, (batch,batch_hTTA,batch_vTTA) in enumerate(tqdm(zip(testset,testset_hTTA,testset_vTTA))):
    fnames, images = batch
    fnames, images_hTTA = batch_hTTA
    fnaems, images_vTTA = batch_vTTA
    batch_preds = 0
    for model in models:
        pred = torch.sigmoid(model(images.to(device))).detach().cpu().numpy()
        pred_hTTA = torch.sigmoid(model(images_hTTA.to(device))).detach().cpu().numpy()
        pred_vTTA = torch.sigmoid(model(images_vTTA.to(device))).detach().cpu().numpy()
        batch_preds += (pred+pred_hTTA[:,:,:,::-1]+pred_vTTA[:,:,::-1,:])/3
    batch_preds/= len(models)
    
    for fname, preds in zip(fnames, batch_preds):
        for cls, pred in enumerate(preds):
            pred, num = post_process(pred, best_threshold, min_size)
            rle = mask2rle(pred)
            name = fname + f"_{cls+1}"
            predictions.append([name, rle])

# save predictions to submission.csv
df = pd.DataFrame(predictions, columns=['ImageId_ClassId', 'EncodedPixels'])
df.to_csv("submission.csv", index=False)


# In[ ]:


# df.head(50)

