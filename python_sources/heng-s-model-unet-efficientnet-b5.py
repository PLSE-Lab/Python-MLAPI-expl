#!/usr/bin/env python
# coding: utf-8

# This is an inference kernel of [this](https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/111457#latest-647456) discussion. I followed Heng's steps and tried to replicate his results. I trained the model for around 18 hours and this kernel does the inference of that model. This kernel successfully commits but when I submit I get `Kaggle Error`. See version4 of this kernel.Maybe I am missing something; hopefully you can correct it, which is why I am opensourcing the trained model for you to experiment. The trained model is a traced version, which means you can load it by simply `model = torch.jit.load(ckpt).cuda()`

# In[ ]:


import os
import cv2
import torch
import pandas as pd
import numpy as np
import glob
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from albumentations import (Normalize, Resize, Compose)
#from albumentations.torch import ToTensor
from albumentations.pytorch.transforms import ToTensor
import torch.utils.data as data
import torchvision.models as models
import torch.nn as nn
from torch.nn import functional as F


# In[ ]:


class SteelDataset(Dataset):
    def __init__(self, df, augment=None):

        
        df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
        self.fnames = df['ImageId'].unique().tolist()
        

    def __len__(self):
        return len(self.fnames)


    def __getitem__(self, index):
        image_id = self.fnames[index]
        image = cv2.imread(test_data_folder + '/%s'%(image_id), cv2.IMREAD_COLOR)
        return image_id, image
    


# In[ ]:


get_ipython().system('ls ../input')


# In[ ]:


sample_submission_path = '../input/severstal-steel-defect-detection/sample_submission.csv'
test_data_folder = "../input/severstal-steel-defect-detection/test_images"


# In[ ]:


def null_collate(batch):
    batch_size = len(batch)

    input = []
    infor = []
    for b in range(batch_size):
        input.append(batch[b][1])
        infor.append(batch[b][0])

    input = np.stack(input).astype(np.float32)/255
    input = input.transpose(0,3,1,2)
    
    input = torch.from_numpy(input).float()
    
    return infor, input


# In[ ]:


df = pd.read_csv(sample_submission_path)
test_dataset = SteelDataset(df)

test_loader = DataLoader(
            test_dataset,
            batch_size  = 2,
            drop_last   = False,
            num_workers = 0,
            pin_memory  = True,
            collate_fn  = null_collate
    )


# In[ ]:



#test time augmentation  -----------------------
def null_augment   (input): return input
def flip_lr_augment(input): return torch.flip(input, dims=[2])
def flip_ud_augment(input): return torch.flip(input, dims=[3])

def null_inverse_augment   (logit): return logit
def flip_lr_inverse_augment(logit): return torch.flip(logit, dims=[2])
def flip_ud_inverse_augment(logit): return torch.flip(logit, dims=[3])

augment = (
        (null_augment,   null_inverse_augment   ),
        (flip_lr_augment,flip_lr_inverse_augment),
        (flip_ud_augment,flip_ud_inverse_augment),
    )


# In[ ]:


TEMPERATE=0.5
######################################################################################
def probability_mask_to_probability_label(probability):
    batch_size,num_class,H,W = probability.shape
    probability = probability.permute(0, 2, 3, 1).contiguous().view(batch_size,-1, 5)
    value, index = probability.max(1)
    probability = value[:,1:]
    return probability


def remove_small_one(predict, min_size):
    H,W = predict.shape
    num_component, component = cv2.connectedComponents(predict.astype(np.uint8))
    predict = np.zeros((H,W), np.bool)
    for c in range(1,num_component):
        p = (component==c)
        if p.sum()>min_size:
            predict[p] = True
    return predict

def remove_small(predict, min_size):
    for b in range(len(predict)):
        for c in range(4):
            predict[b,c] = remove_small_one(predict[b,c], min_size[c])
    return predict


# In[ ]:


def do_evaluate_segmentation(net, test_loader, augment=[]):

    #----

    #def sharpen(p,t=0):
    def sharpen(p,t=TEMPERATE):
        if t!=0:
            return p**t
        else:
            return p


    test_num  = 0
    test_id   = []
    #test_image = []
    test_probability_label = [] # 8bit
    test_probability_mask  = [] # 8bit
    test_truth_label = []
    test_truth_mask  = []

    #start = timer()
    for t, (fnames, input) in enumerate(tqdm(test_loader)):

        batch_size,C,H,W = input.shape
        input = input.cuda()

        with torch.no_grad():
            net.eval()

            num_augment = 0
            if 1: #  null
                logit =  net(input)
                probability = torch.softmax(logit,1)

                probability_mask = sharpen(probability,0)
                num_augment+=1

            if 'flip_lr' in augment:
                logit = net(torch.flip(input,dims=[3]))
                probability  = torch.softmax(torch.flip(logit,dims=[3]),1)

                probability_mask += sharpen(probability)
                num_augment+=1

            if 'flip_ud' in augment:
                logit = net(torch.flip(input,dims=[2]))
                probability = torch.softmax(torch.flip(logit,dims=[2]),1)

                probability_mask += sharpen(probability)
                num_augment+=1

            #---
            probability_mask = probability_mask/num_augment
            probability_label = probability_mask_to_probability_label(probability_mask)

        probability_mask = (probability_mask.data.cpu().numpy()*255).astype(np.uint8)
        probability_label = (probability_label.data.cpu().numpy()*255).astype(np.uint8)

        test_id.extend([i for i in fnames])

        test_probability_mask.append(probability_mask)
        test_probability_label.append(probability_label)
        
    test_probability_mask = np.concatenate(test_probability_mask)
    test_probability_label = np.concatenate(test_probability_label)
    
    
    return test_probability_label, test_probability_mask, test_id


# In[ ]:


get_ipython().system('ls ../input/henge5')


# In[ ]:


ckpt_file = '../input/henge5/trace_model_swa.pth'
net = torch.jit.load(ckpt_file).cuda()


# In[ ]:


probability_label, probability_mask, image_id = do_evaluate_segmentation(net, test_loader, augment=['null'])


# In[ ]:


del net
gc.collect()


# In[ ]:


#value = np.max(probability_mask,1,keepdims=True)
#value = probability_mask*(value==probability_mask)
probability_mask = probability_mask[:,1:] #remove background class


# In[ ]:


threshold_label      = [ 0.70, 0.8, 0.50, 0.70,]
threshold_mask_pixel = [ 0.6, 0.8, 0.5, 0.6,]
threshold_mask_size  = [ 1,  1,  1,  1,]


# In[ ]:


predict_label = probability_label>(np.array(threshold_label)*255).astype(np.uint8).reshape(1,4)
predict_mask  = probability_mask>(np.array(threshold_mask_pixel)*255).astype(np.uint8).reshape(1,4,1,1)


# In[ ]:


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


image_id_class_id = []
encoded_pixel = []
for b in range(len(image_id)):
    for c in range(4):
        image_id_class_id.append(image_id[b]+'_%d'%(c+1))

        if predict_label[b,c]==0:
            rle=''
        else:
            rle = mask2rle(predict_mask[b,c])
        encoded_pixel.append(rle)


# In[ ]:


df = pd.DataFrame(zip(image_id_class_id, encoded_pixel), columns=['ImageId_ClassId', 'EncodedPixels'])
df.to_csv('submission.csv', index=False)


# In[ ]:


df.head()


# In[ ]:



def summarise_submission_csv(df):


    text = ''
    df['Class'] = df['ImageId_ClassId'].str[-1].astype(np.int32)
    df['Label'] = (df['EncodedPixels']!='').astype(np.int32)
    num_image = len(df)//4
    num = len(df)

    pos = (df['Label']==1).sum()
    neg = num-pos


    pos1 = ((df['Class']==1) & (df['Label']==1)).sum()
    pos2 = ((df['Class']==2) & (df['Label']==1)).sum()
    pos3 = ((df['Class']==3) & (df['Label']==1)).sum()
    pos4 = ((df['Class']==4) & (df['Label']==1)).sum()

    neg1 = num_image-pos1
    neg2 = num_image-pos2
    neg3 = num_image-pos3
    neg4 = num_image-pos4


    text += 'compare with LB probing ... \n'
    text += '\t\tnum_image = %5d(1801) \n'%num_image
    text += '\t\tnum  = %5d(7204) \n'%num
    text += '\n'

    text += '\t\tpos1 = %5d( 128)  %0.3f\n'%(pos1,pos1/128)
    text += '\t\tpos2 = %5d(  43)  %0.3f\n'%(pos2,pos2/43)
    text += '\t\tpos3 = %5d( 741)  %0.3f\n'%(pos3,pos3/741)
    text += '\t\tpos4 = %5d( 120)  %0.3f\n'%(pos4,pos4/120)
    text += '\n'

    text += '\t\tneg1 = %5d(1673)  %0.3f  %3d\n'%(neg1,neg1/1673, neg1-1673)
    text += '\t\tneg2 = %5d(1758)  %0.3f  %3d\n'%(neg2,neg2/1758, neg2-1758)
    text += '\t\tneg3 = %5d(1060)  %0.3f  %3d\n'%(neg3,neg3/1060, neg3-1060)
    text += '\t\tneg4 = %5d(1681)  %0.3f  %3d\n'%(neg4,neg4/1681, neg4-1681)
    text += '--------------------------------------------------\n'
    text += '\t\tneg  = %5d(6172)  %0.3f  %3d \n'%(neg,neg/6172, neg-6172)
    text += '\n'

    if 1:
        #compare with reference
        pass

    return text


# In[ ]:


## print statistics ----
text = summarise_submission_csv(df)
print(text)


# In[ ]:


def rle2mask(mask_rle, shape=(1600,256)):
   '''
   mask_rle: run-length as string formated (start length)
   shape: (width,height) of array to return 
   Returns numpy array, 1 - mask, 0 - background

   '''
   s = mask_rle.split()
   starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
   starts -= 1
   ends = starts + lengths
   img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
   for lo, hi in zip(starts, ends):
       img[lo:hi] = 1
   return img.reshape(shape).T


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('submission.csv')[:60]
df['Image'] = df['ImageId_ClassId'].map(lambda x: x.split('_')[0])
df['Class'] = df['ImageId_ClassId'].map(lambda x: x.split('_')[1])

for row in df.itertuples():
    img_path = os.path.join(test_data_folder, row.Image)
    img = cv2.imread(img_path)
    mask = rle2mask(row.EncodedPixels, (1600, 256))         if isinstance(row.EncodedPixels, str) else np.zeros((256, 1600))
    if mask.sum() == 0:
        continue
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 60))
    axes[0].imshow(img/255)
    axes[1].imshow(mask*60)
    axes[0].set_title(row.Image)
    axes[1].set_title(row.Class)
    plt.show()


# The results are extremely sensitive to the thresholds. So play with the thresholds and see the change on LB. You can also try adding Test time Augmentation(TTA) and see if it performs better on LB. Good luck and Happy Kaggling
