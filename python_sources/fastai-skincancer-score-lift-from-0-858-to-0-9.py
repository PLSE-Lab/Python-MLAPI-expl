#!/usr/bin/env python
# coding: utf-8

# **FastAI implementation of Skin Cancer Classification competition with 512 * 512 sized images
# 
# Update Log - Change in image size from 128 to 324 lifted score from 0.858 to 0.9. Data cleaning and Null data handling with further boost score**

# In[ ]:


import glob
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from fastai import *
from fastai.vision import *


# In[ ]:


train_df = pd.read_csv('../input/siim-isic-melanoma-classification-jpeg512/train.csv')
test_df = pd.read_csv('../input/test-files/test.csv')
submission_df = pd.read_csv('../input/test-files/sample_submission.csv')


# In[ ]:


tfrm = get_transforms(do_flip = True, flip_vert = True)


# In[ ]:


# Check if only jpg files in test folder
assert len(glob.glob('../input/siim-isic-melanoma-classification-jpeg512/test512/*.jpg')) == len(os.listdir('../input/siim-isic-melanoma-classification-jpeg512/test512/'))


# In[ ]:


# Check if only jpg files in train folder
assert len(glob.glob('../input/siim-isic-melanoma-classification-jpeg512/train512/*.jpg')) == len(os.listdir('../input/siim-isic-melanoma-classification-jpeg512/train512/'))


# In[ ]:


test_df.image_name = test_df.image_name.apply(lambda file : file+'.jpg')


# In[ ]:


train_df.image_name = train_df.image_name.apply(lambda file : file+'.jpg')


# In[ ]:


test_imgs = ImageList.from_df(test_df, path = '../input/siim-isic-melanoma-classification-jpeg512', folder = 'test512')


# In[ ]:


np.random.seed(42)
src = ImageList.from_df(train_df, path = '../input/siim-isic-melanoma-classification-jpeg512', folder = 'train512')                      .split_by_rand_pct(0.2)                      .label_from_df(cols = -1)                      .add_test(test_imgs)
                      


# In[ ]:


src


# In[ ]:


data = src.transform(tfrm, padding_mode = 'reflection', size = 324, resize_method = ResizeMethod.SQUISH).databunch(bs = 32, device = None)          .normalize(imagenet_stats)
          #.databunch(bs = 32, device = torch.device('cuda:0'))\
          


# In[ ]:


data.show_batch(3)


# In[ ]:


data.classes


# In[ ]:


learn = cnn_learner(data=data, base_arch=models.resnet101, metrics=[FBeta(beta=1, average='macro'), accuracy],
                    callback_fns=ShowGraph)


# In[ ]:


learn.summary()


# In[ ]:


learn.model_dir = '/kaggle/output/'


# In[ ]:


for obj in gc.get_objects():
    if torch.is_tensor(obj):
        del obj
gc.collect()
torch.cuda.empty_cache()


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


learn.model_dir = '/kaggle/working/'


# In[ ]:


learn.save('baseline')


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(4, 1e-5)


# In[ ]:


learn.save('version1')


# In[ ]:


learn.summary()


# In[ ]:


test = os.listdir(Path('../input/siim-isic-melanoma-classification-jpeg512/test512'))
test.sort(key=lambda f: int(re.sub('\D', '', f)))

with open('/kaggle/working/submission.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['image_name', 'target'])
    
    for image_file in test:
        image = os.path.join(Path('../input/siim-isic-melanoma-classification-jpeg512/test512'), image_file) 
        image_name = Path(image).stem

        img = open_image(image)
        pred_class,pred_idx,outputs = learn.predict(img)
        target = float(outputs[1])

        
        writer.writerow([image_name, target])

