#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2
import skimage.io
from tqdm.notebook import tqdm
import zipfile
import pandas as pd
import numpy as np
import shutil


# In[ ]:


get_ipython().system('mkdir -p /root/.kaggle/')
get_ipython().system('cp ../input/my-kaggle-api/kaggle.json /root/.kaggle/')
get_ipython().system('chmod 600 /root/.kaggle/kaggle.json')


# In[ ]:


get_ipython().system('mkdir -p /tmp/panda_dataset')


# In[ ]:


get_ipython().system('ls /tmp/')


# In[ ]:


data = '''{
  "title": "Panda_Dataset_medium_25_256_256",
  "id": "raghaw/panda-dataset-medium-25-256-256",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ]
}
'''
text_file = open("/tmp/panda_dataset/dataset-metadata.json", 'w+')
n = text_file.write(data)
text_file.close()


# In[ ]:


TRAIN = '../input/prostate-cancer-grade-assessment/train_images/'
MASKS = '../input/prostate-cancer-grade-assessment/train_label_masks/'
OUT_TRAIN = '/tmp/panda_dataset/train_medium_25_256_256.zip'
OUT_MASKS = '/tmp/panda_dataset/masks_medium_25_256_256.zip'
sz = 256
N = 25


# In[ ]:


def tile(img, mask):
    result = []
    shape = img.shape
    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz
    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                constant_values=255)
    mask = np.pad(mask,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                constant_values=0)
    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    mask = mask.reshape(mask.shape[0]//sz,sz,mask.shape[1]//sz,sz,3)
    mask = mask.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    if len(img) < N:
        mask = np.pad(mask,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=0)
        img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]
    img = img[idxs]
    mask = mask[idxs]
    for i in range(len(img)):
        result.append({'img':img[i], 'mask':mask[i], 'idx':i})
    return result


# In[ ]:


x_tot,x2_tot = [],[]
names = [name[:-5] for name in os.listdir(TRAIN)]
with zipfile.ZipFile(OUT_TRAIN, 'w') as img_out, zipfile.ZipFile(OUT_MASKS, 'w') as mask_out:
    for name in tqdm(names):
        img = skimage.io.MultiImage(os.path.join(TRAIN,name+'.tiff'))[1]
        mask_exist = os.path.isfile(os.path.join(MASKS,name+'_mask.tiff'))
        if mask_exist:
            mask = skimage.io.MultiImage(os.path.join(MASKS,name+'_mask.tiff'))[1]
        else:
            mask = np.zeros_like(img)
        tiles = tile(img,mask)
        for t in tiles:
            img,mask,idx = t['img'],t['mask'],t['idx']
            x_tot.append((img/255.0).reshape(-1,3).mean(0))
            x2_tot.append(((img/255.0)**2).reshape(-1,3).mean(0)) 
            #if read with PIL RGB turns into BGR
            img = cv2.imencode('.png',cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1]
            img_out.writestr(f'{name}_{idx}.png', img)
            if mask_exist:
                mask = cv2.imencode('.png',mask[:,:,0])[1]
                mask_out.writestr(f'{name}_{idx}.png', mask)


# In[ ]:


#image stats
img_avr =  np.array(x_tot).mean(0)
img_std =  np.sqrt(np.array(x2_tot).mean(0) - img_avr**2)
print('mean:',img_avr, ', std:', np.sqrt(img_std))


# In[ ]:


get_ipython().system('sleep 10')


# In[ ]:


get_ipython().system('unzip -q /tmp/panda_dataset/train_medium_25_256_256.zip -d /tmp/train_images')


# In[ ]:


train_files = os.listdir("/tmp/train_images")


# In[ ]:


len(train_files)


# In[ ]:


files = set([file.split("_")[0] for file in train_files])


# In[ ]:


len(files)


# In[ ]:


df = pd.read_csv("../input/prostate-cancer-grade-assessment/train.csv")


# In[ ]:


df.shape


# In[ ]:


if len(files) != df.shape[0]:
    df = df[df.image_id.isin(files)]
    df.to_csv("/tmp/panda_dataset/train.csv", index = False)
else:
    shutil.copy2("../input/prostate-cancer-grade-assessment/train.csv", "/tmp/panda_dataset/train.csv")


# In[ ]:


get_ipython().system('sleep 30')


# In[ ]:


get_ipython().system('ls -l /tmp/panda_dataset')


# In[ ]:


get_ipython().system('kaggle datasets create -p /tmp/panda_dataset')


# In[ ]:


get_ipython().system('rm -rf /tmp/train_images')

