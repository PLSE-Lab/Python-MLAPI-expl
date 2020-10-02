#!/usr/bin/env python
# coding: utf-8

# <center><h1> A Tutorial for Beginners. </h1></center>
# 
# 
# <center><h2> On How to train A neural network for image Segmentation using Fast.ai and Transfer Learning</h2></center>
# 
# 
# <center><h3> We will use a pretrained resnet model </h3></center>
# 

# <center><h3> Please Upvote if you like it. </h3></center>

# In[ ]:


from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as immg
import gc
import numpy as np
from scipy import signal
from scipy import misc
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


path = Path('/kaggle/input/chest-xray-with-masks-for-image-segmentation')


# In[ ]:


path.ls()


# # Data

# In[ ]:


fnames = get_image_files(path/'train')
fnamesMask = get_image_files(path/'masks')


# In[ ]:


path_lbl = path/'masks'
path_img = path/'train'
get_y_fn = lambda x: path_lbl/f'{x.stem}_mask{x.suffix}'       # Function to get masks for a image
codes = np.array([0,1])


# In[ ]:


sns.set_style('darkgrid')


# ## Function to show chest X_ray with Mask

# In[ ]:


def show_chest(f):  # f = file_name
  img_a = immg.imread(f)
  img_a_mask = immg.imread(get_y_fn(f))
  plt.figure(1,figsize=(20,8))
  plt.subplot(121)
  plt.imshow(img_a);plt.title('Chest X Ray');plt.axis('off')
  plt.subplot(122)
  plt.imshow(img_a,alpha=0.9);
  plt.imshow(img_a_mask,alpha=0.3);plt.title('Chest X-Ray with Lung mask');plt.axis('off')
  plt.show()


# ## A sample X-ray with Mask

# In[ ]:


show_chest(fnames[50])


# In[ ]:


gc.collect()


# ## Creating A DatabLock for the model

# In[ ]:


data = (SegmentationItemList.from_folder(path=path/'train')  # Location from path
        .split_by_rand_pct(0.2)                          # Split for train and validation set
        .label_from_func(get_y_fn, classes=codes)      # Label from a above defined function
        .transform(get_transforms(), size=128, tfm_y=True)   # If you want to apply any image Transform
        .databunch(bs=32)                                   # Batch size  please decrese batch size if cuda out of memory
        .normalize(imagenet_stats))            # Normalise with imagenet stats


# In[ ]:


data.show_batch(rows=3,figsize=(20,8));


# In[ ]:


len(data.train_ds),len(data.valid_ds)


# # Model

# * Metrics for lung mask
# * You can Also use classic **dice** if you want its fast.ai builtin metric

# In[ ]:


name2id = {v:k for k,v in enumerate(codes)}
void_code = -1

def accuracy_mask(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()


# In[ ]:


metrics = accuracy_mask
wd=1e-2    # wd = weight decay


# In[ ]:


learn = unet_learner(data, models.resnet18, metrics = [accuracy_mask], wd = wd, bottle=True, model_dir = '/kaggle/working/')


# ## Finding a suitable learning rate for our model
# 
# * With help fast.ai **learning rate finder** function

# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


gc.collect()     # to clear the cache


# In[ ]:


lr =1e-3              # Learning Rate


# In[ ]:


learn.fit_one_cycle(5, slice(lr) )            # Model traing for 10 epochs


# ## To check results of our trained model

# In[ ]:


learn.show_results(rows=3, figsize=(12,16))


# In[ ]:


learn.save('stage-1-big')  # saving the model 
learn.load('stage-1-big');  # loading the model
learn.unfreeze()


# ## Export the model

# In[ ]:


learn.export('/kaggle/working/chest_mask.pkl')


# ### Load the model  and predict

# * Function to make a prediction and Overlap the chest Images with Predicted lung Mask

# In[ ]:


def chest_predict(f):
  pred=learn.predict(open_image(f))[0]
  im = Image.open(f)
  pred.show(figsize=(6,6),alpha=0.9,title='Chest X Ray with predicted mask',cmap='winter')
  plt.imshow(np.asarray(im.resize((128, 128))),alpha=0.7)


# ### Prediction

# In[ ]:


chest_predict(fnames[450])


# ## Model Summary

# In[ ]:


print(learn.summary())


# In[ ]:




