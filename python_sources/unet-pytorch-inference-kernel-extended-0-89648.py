#!/usr/bin/env python
# coding: utf-8

# ## Extension notes
# 
# This notebook is an extension of [Rishabh's](https://www.kaggle.com/rishabhiitbhu) [inference kernel](https://www.kaggle.com/rishabhiitbhu/unet-pytorch-inference-kernel). <br>
# If you find this notebook useful, don't forget to give his notebook an upvote as well. <br>
# I want to thank Rishabh for not only providing the original kernel, but also assisting me with my questions in the comments.
# 
# The notebook is configured to submit results of a U-net architecture with (1) resnet and (2) se_resnet encoders. <br>
# The following encoders are supported in this notebook:
# 
# * (1) resnet18, resnet34, resnet50, resnet101, resnet152
# * (2) senet154, se_resnet50, se_resnet101, se_resnet152, se_resnext50_32x4d, se_resnext101_32x4d
# 
# You can train a model offline with any of the above encoders and submit the results using this notebook.
# 
# Four locally trained models are available in this notebook:
# 
# 1. resnet18_20_epochs.pth the model from the original notebook
# 2. senet50_20_epochs.pth a U-net using a pretrained se_resnet50 encoder.
# 3. senext50_30_epochs.pth a U-net using a pretrained se_resnext50_32x4d.
# 4. senext50_30_epochs_high_threshold.pth a U-net using a pretrained se_resnext50_32x4d encoder where the base_threshold was set to 0.8.
# 
# For the U-net model with the senext50 encoder, setting the base_threshold from 0.5 to 0.8 <br>
# improved the score from 0.88776 to 0.89648 leaving everything else the same. 
# 

# ### package_path instructions
# 
# Change the *package_path*:
# 
# * to *'../input/resnetunetmodelcode'* if you use a (1) resnet encoder 
# * to *'../input/senetunetmodelcode'* if you use a (2) senet encoder.

# In[ ]:


get_ipython().system('pip install ../input/pretrainedmodels/pretrainedmodels-0.7.4/pretrainedmodels-0.7.4/ > /dev/null')
# package_path = '../input/resnetunetmodelcode'
package_path = '../input/senetunetmodelcode'
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
from albumentations import (Normalize, Compose)
# from albumentations.torch import ToTensor was renamed to:
from albumentations.pytorch import ToTensor
import torch.utils.data as data
from senet_unet_model_code import Unet


# In[ ]:


os.listdir('../input')


# ### Available models
# 
# This notebook contains three model.pth files that were trained locally. 

# In[ ]:


get_ipython().system('ls ../input/pretrainedmodels')
get_ipython().system('ls ../input/resnetunetmodelcode')
get_ipython().system('ls ../input/resnetmodels')
get_ipython().system('ls ../input/senetunetmodelcode')
get_ipython().system('ls ../input/senetmodels')


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
    def __init__(self, root, df, mean, std):
        self.root = root
        df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
        self.fnames = df['ImageId'].unique().tolist()
        self.num_samples = len(self.fnames)
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
best_threshold = 0.5
num_workers = 2
batch_size = 4
print('best_threshold', best_threshold)
min_size = 3500
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
df = pd.read_csv(sample_submission_path)
testset = DataLoader(
    TestDataset(test_data_folder, df, mean, std),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)


# ### Model selection instructions
# 
# If you want to use a model included in this notebook:
# 
# 1. Uncomment the corresponding ckpt_path.
# 2. Write the encoder name (see above) into the Unet() call.
# 

# In[ ]:


# Initialize mode and load trained weights
# ckpt_path = "../input/resnetmodels/resnet18_20_epochs.pth"
# ckpt_path = "../input/senetmodels/senet50_20_epochs.pth"
ckpt_path = "../input/senetmodels/senext50_30_epochs.pth"
device = torch.device("cuda")
# change the encoder name in the Unet() call.
model = Unet('se_resnext50_32x4d', encoder_weights=None, classes=4, activation=None)
model.to(device)
model.eval()
state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
model.load_state_dict(state["state_dict"])


# In[ ]:


# start prediction
predictions = []
for i, batch in enumerate(tqdm(testset)):
    fnames, images = batch
    batch_preds = torch.sigmoid(model(images.to(device)))
    batch_preds = batch_preds.detach().cpu().numpy()
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


df.head()


# ### Refrences (from Rishabh's original notebook):
# 
# Few kernels from which I've borrowed some code:
# 
# * https://www.kaggle.com/amanooo/defect-detection-starter-u-net
# * https://www.kaggle.com/go1dfish/clear-mask-visualization-and-simple-eda
# 
# A big thank you to all those who share their code on Kaggle, I'm nobody without you guys. I've learnt a lot from fellow kagglers, special shout-out to [@Abhishek](https://www.kaggle.com/abhishek), [@Yury](https://www.kaggle.com/deyury), [@Heng](https://www.kaggle.com/hengck23), [@Ekhtiar](https://www.kaggle.com/ekhtiar), [@lafoss](https://www.kaggle.com/iafoss), [@Siddhartha](https://www.kaggle.com/meaninglesslives), [@xhulu](https://www.kaggle.com/xhlulu), and the list goes on..

# Do upvote if you liked my kernel :)
