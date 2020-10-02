#!/usr/bin/env python
# coding: utf-8

# **This is gonna be my 1st attempt to Age estimation, any improvement or advice would be most  welcome**

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *
from fastai.metrics import error_rate
from fastai import *
from fastai.vision import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import random
import os


# In[ ]:


bs = 64


# In[ ]:


path = Path("../input/utkface-new/utkface_aligned_cropped")
path_test=('../input/utkface-new/UTKface_Aligned_cropped/crop_part1')
path


# In[ ]:


path.ls()


# In[ ]:


im =PIL.Image.open(path/'UTKFace/26_1_1_20170117201805678.jpg.chip.jpg')


# In[ ]:


im


# In[ ]:


path_labels = path
path_img = path/'UTKFace'


# In[ ]:


fnames = get_image_files(path_img)
fnames[:5]


# In[ ]:


def extract_age(filename):
    return float(filename.stem.split('_')[0])


# In[ ]:


extract_age(path_img/'28_0_0_20170117202521375.jpg.chip.jpg')


# In[ ]:


np.random.seed(2)
ds=get_transforms(do_flip=False,max_rotate=0,max_zoom=1, max_lighting=0, max_warp=0, p_affine=0, p_lighting=0 )


# In[ ]:


fn_paths = path_img.ls(); fn_paths[:2]


# In[ ]:


def extract_age(filename):
    return float(filename.stem.split('_')[0])


# In[ ]:


def load_face_data(img_size, batch_size,path):
    tfms = get_transforms(max_warp=0.)
    return (ImageList.from_folder(path)
            .random_split_by_pct(0.2, seed=666)
            .label_from_func(extract_age)
            .transform(tfms, size=img_size)
            .databunch(bs=batch_size))


# In[ ]:


data = load_face_data(224, 256,path)


# In[ ]:


data.show_batch(rows=3, figsize=(7,7))


# In[ ]:


age=[extract_age(i) for i in path_img.ls()]
plt.figure(figsize=(10, 5))
plt.plot(*zip(*sorted(Counter(age).items())), '.:')
plt.title('Number of Images by Age')
plt.ylabel('count')
plt.xlabel('age')
plt.grid()


# In[ ]:


class AgeModel(nn.Module):
    def __init__(self):
        super().__init__()
        layers = list(models.resnet34(pretrained=True).children())[:-2]
        layers += [AdaptiveConcatPool2d(),Flatten()]
        layers += [nn.Linear(1024, 16), nn.ReLU(), nn.Linear(16,1)]
        self.agemodel = nn.Sequential(*layers)
    def forward(self, x):
        return self.agemodel(x).squeeze() # could add 116*torch.sigmoid


# In[ ]:


model = AgeModel()


# In[ ]:


learn = Learner(data, model, loss_func = F.l1_loss, model_dir="/tmp/model/")
learn=learn.load("/kaggle/input/training/ageTraining")


# In[ ]:


learn.split([model.agemodel[2],model.agemodel[-3]])


# In[ ]:


learn.layer_groups[-1]


# In[ ]:


learn.freeze_to(-1)


# In[ ]:





# In[ ]:


learn.fit_one_cycle(1)


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(6,max_lr =1e-3)


# In[ ]:


learn.unfreeze()

learn.lr_find(); learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(12, max_lr = slice(1e-6,1e-5))


# In[ ]:


learn.save('/kaggle/working/ageTraining')


# In[ ]:


Path('/kaggle/working').ls()


# In[ ]:


os.chdir(r'/kaggle/working')
from IPython.display import FileLink
FileLink(r'ageTraining.pth')


# In[ ]:


learn.show_results(rows=4)


# In[ ]:


img = data.train_ds[0][0]
img


# In[ ]:



learn.model.eval()


# In[ ]:


data1 = learn.data.train_ds[0][0]


# In[ ]:


data1


# In[ ]:


pred = learn.predict(data)


# In[ ]:




