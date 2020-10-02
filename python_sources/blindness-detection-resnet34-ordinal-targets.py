#!/usr/bin/env python
# coding: utf-8

# This is an extension of [this](https://www.kaggle.com/kageyama/fork-of-fastai-blindness-detection-resnet34) notebook, which extends upon [this](https://www.kaggle.com/kageyama/fastai-blindness-detection-resnet34) notebook.
# 
# In this experiment, I try ordinal variables using [this](https://arxiv.org/abs/0704.1028) technique. Basically, I transform the targets to look like multilabel classification, then apply this method for making predictions:
# 
# > "...our methods scans output nodes in the order O1, O2,....,OK. It stop when the output of a node is smaller than the predefined threshold T (e.g. 0.5) or no nodes left. The index k of the last node Ok whose output is bigger than T is the predicted category of the data point."
# 
# So basically, I'll apply sigmoid to the model's outputs, then threshold at 0.5 and find the position one before the first zero.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

from sklearn.metrics import confusion_matrix
from fastai import *
from fastai.vision import *
from fastai.callbacks import Callback

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2                  
import numpy as np  
from tqdm import tqdm
import os                   
from random import shuffle  
from zipfile import ZipFile
from PIL import Image
from sklearn.utils import shuffle

print(os.listdir("../input"))


# In[ ]:


# copy pretrained weights for resnet34 to the folder fastai will search by default
Path('/tmp/.cache/torch/checkpoints/').mkdir(exist_ok=True, parents=True)
get_ipython().system("cp '../input/resnet34/resnet34.pth' '/tmp/.cache/torch/checkpoints/resnet34-333f7ec4.pth'")


# In[ ]:


df_train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
df_test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

x_train = df_train['id_code']
y_train = df_train['diagnosis']


# In[ ]:


df_train.diagnosis.hist()


# In[ ]:


import torch
import torch.utils.data
import torchvision


# In[ ]:


print(os.listdir("../input/aptos2019-blindness-detection/")) 


# In[ ]:


def get_label(diagnosis):
    return ','.join([str(i) for i in range(diagnosis + 1)])


# In[ ]:


df_train['label'] = df_train.diagnosis.apply(get_label)


# In[ ]:


df_train.head(10)


# In[ ]:


# create image data bunch
data = ImageDataBunch.from_df('./', 
                              df=df_train, 
                              valid_pct=0.2,
                              folder="../input/aptos2019-blindness-detection/train_images",
                              suffix=".png",
                              ds_tfms=get_transforms(flip_vert=True, max_warp=0),
                              size=224,
                              bs=64, 
                              num_workers=0,
                             label_col='label', label_delim=',').normalize(imagenet_stats)


# In[ ]:


# check classes
print(f'Classes: \n {data.classes}')


# In[ ]:


# show some sample images
data.show_batch(rows=3, figsize=(7,6))


# 

# In[ ]:


def get_preds(arr):
    mask = arr == 0
    return np.clip(np.where(mask.any(1), mask.argmax(1), 5) - 1, 0, 4)


# In[ ]:


last_output = torch.tensor([
    [1.7226, 1.7226, 1.7226, 1.7226, 1.7226],
    [0, 0, 0, 0, 1.7226],
    [0.12841, -7.6266, -6.3899, -2.1333, -0.48995],
    [0.68119, 1.7226, -1.9895, -0.097746, 0.53576]
])
arr = (torch.sigmoid(last_output) > 0.5).numpy(); arr


# In[ ]:


# Test output
assert (get_preds(arr) == np.array([4, 0, 0, 1])).all()


# In[ ]:


class ConfusionMatrix(Callback):
    "Computes the confusion matrix."

    def on_train_begin(self, **kwargs):
        self.n_classes = 0

    def on_epoch_begin(self, **kwargs):
        self.cm = None

    def on_batch_end(self, last_output:Tensor, last_target:Tensor, **kwargs):
        preds = torch.tensor(get_preds((torch.sigmoid(last_output) > 0.5).cpu().numpy()))
        
        targs = torch.tensor(get_preds(last_target.cpu().numpy()))

        if self.n_classes == 0:
            self.n_classes = last_output.shape[-1]
            self.x = torch.arange(0, self.n_classes)
        
        cm = ((preds==self.x[:, None]) & (targs==self.x[:, None, None])).sum(dim=2, dtype=torch.float32)
        if self.cm is None: self.cm =  cm
        else:               self.cm += cm

    def on_epoch_end(self, **kwargs):
        self.metric = self.cm
        

@dataclass
class KappaScore(ConfusionMatrix):
    "Compute the rate of agreement (Cohens Kappa)."
    weights:Optional[str]=None      # None, `linear`, or `quadratic`

    def on_epoch_end(self, last_metrics, **kwargs):
        sum0 = self.cm.sum(dim=0)
        sum1 = self.cm.sum(dim=1)
        expected = torch.einsum('i,j->ij', (sum0, sum1)) / sum0.sum()
        if self.weights is None:
            w = torch.ones((self.n_classes, self.n_classes))
            w[self.x, self.x] = 0
        elif self.weights == "linear" or self.weights == "quadratic":
            w = torch.zeros((self.n_classes, self.n_classes))
            w += torch.arange(self.n_classes, dtype=torch.float)
            w = torch.abs(w - torch.t(w)) if self.weights == "linear" else (w - torch.t(w)) ** 2
        else: raise ValueError('Unknown weights. Expected None, "linear", or "quadratic".')
        k = torch.sum(w * self.cm) / torch.sum(w * expected)
        return add_metrics(last_metrics, 1-k)


# In[ ]:


accuracy


# In[ ]:


kappa = KappaScore(weights="quadratic")

# build model (use resnet34)
learn = create_cnn(data, models.resnet34, metrics=[kappa, accuracy_thresh], model_dir="/tmp/model/")


# In[ ]:


# learn.lr_find()
# learn.recorder.plot()


# In[ ]:


# first time learning
learn.fit_one_cycle(6, 1e-2)


# In[ ]:


# save stage
learn.save('stage-1')


# In[ ]:


# search appropriate learning rate
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()


# In[ ]:


# second time learning
learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-5 ))


# In[ ]:


# save stage
learn.save('stage-2')


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


sample_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
sample_df.head()


# In[ ]:


learn.data.add_test(ImageList.from_df(sample_df,'../input/aptos2019-blindness-detection',folder='test_images',suffix='.png'))


# In[ ]:


preds, y = learn.get_preds(DatasetType.Test)


# In[ ]:


preds


# In[ ]:


sample_df.diagnosis = get_preds((preds > 0.5).cpu().numpy())
sample_df.head(10)


# In[ ]:


sample_df.to_csv('submission.csv',index=False)

