#!/usr/bin/env python
# coding: utf-8

# # Importance heat maps
# 
# **Heat maps are awesome!**
# 
# I don't know anything about pneumonia diagnoses from chest x-rays, so I am going to use deep learning to show me what I should be looking at. 
# 
# This will be using Class Activation Maps (CAMs). I have no knowledge of, so I when I came across this dataset, I was really interested to see if deep learning could be used to show what areas of the x-ray are indicative of pneumonia

# In[1]:


from pathlib import Path
import pandas as pd
import numpy as np
from fastai.core import *
from fastai.conv_learner import *
from fastai.plots import *


# In[2]:


PATH = Path('../input/chest-xray-pneumonia/chest_xray/chest_xray/')


# In[3]:


iter_norm = iter([img for img in (PATH/'val/NORMAL').iterdir()  if '.jpeg' in str(img)])
iter_pneum = iter([img for img in (PATH/'val/PNEUMONIA').iterdir()  if '.jpeg' in str(img)])
fig, axess = plt.subplots(nrows=8, ncols=2,figsize=(20,60))
for axes in axess:
    axes[0].imshow(open_image(next(iter_norm)))
    axes[0].set_title('NORMAL')
    axes[1].imshow(open_image(next(iter_pneum)))
    axes[1].set_title('PNEUMONIA')


# Looking at the images, I do not have a scoobies on which what constitutes normal, or having pneumonia

# In[4]:


train = list((PATH/'train').iterdir())
train_imgs = [img for dirImg in train if dirImg.is_dir() for img in dirImg.iterdir()  if '.jpeg' in str(img)]
train_imgs = {str(img):int('PNEUMONIA' in str(img)) for img in train_imgs} 
train_df = pd.DataFrame(A([[k,v] for k,v in train_imgs.items()]), columns=['image','label'])

test = list((PATH/'test').iterdir())
test_imgs = [img for dirImg in test if dirImg.is_dir() for img in dirImg.iterdir()  if '.jpeg' in str(img)]
test_imgs = {str(img):int('PNEUMONIA' in str(img)) for img in test_imgs} 
test_df = pd.DataFrame(A([[k,v] for k,v in test_imgs.items()]), columns=['image','label'])

df = train_df.append(test_df)

df.to_csv('images.csv', index=False)

val_idxs = df[df.image.apply(lambda x: '/test/' in x)].index; len(val_idxs)


# # Make a model
# 
# Just going to use a standard resnet34 and fine tune it. The only difference is that we are going to use a CNNs instead of fully-connected layers in order to preserve the geometry so we can create a heat map

# In[24]:


arch = resnet34
tfms = tfms_from_model(arch, 400, aug_tfms=transforms_basic)
data = ImageClassifierData.from_csv('','', 'images.csv',bs=64,tfms=tfms, val_idxs=val_idxs)


# In[25]:


get_ipython().run_line_magic('mkdir', '/tmp/.torch/')
get_ipython().run_line_magic('mkdir', '/tmp/.torch/models/')
get_ipython().system('cp ../input/resnet34/resnet34.pth /tmp/.torch/models/resnet34-333f7ec4.pth')


# In[26]:


def conv_block(n_in, n_out):
    return nn.Sequential(nn.Dropout2d(),
                  nn.Conv2d(n_in, n_out,3, padding=1),
                  nn.BatchNorm2d(n_out),
                  nn.ReLU()
                 )


# In[27]:


m = arch(True)
m = nn.Sequential(*children(m)[:-2], 
                  conv_block(512,256),
                  nn.Conv2d(256, 2, 3, padding=1), 
                  nn.AdaptiveAvgPool2d(1), Flatten(), 
                  nn.LogSoftmax())


# In[28]:


learn = ConvLearner.from_model_data(m, data)


# In[29]:


learn.freeze_to(-5)
learn.fit(3e-2,2,cycle_len=1, cycle_mult=2)


# **Getting a test set accuracy of 85%**, but not really too worried about how good this is. It just has to be good enough to recognise the key features

# ## Create some heat maps!

# In[30]:


from matplotlib.patheffects import Stroke, Normal
def draw_outline(o, lw):
    o.set_path_effects([matplotlib.patheffects.Stroke(
        linewidth=lw, foreground='black'), matplotlib.patheffects.Normal()])

def draw_rect(ax, b, color='white'):
    patch = ax.add_patch(matplotlib.patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor=color, lw=2))
    draw_outline(patch, 4)

from PIL import Image, ImageEnhance
def increase_contrast(dx):
    image = Image.fromarray(np.uint8(dx*255))
    contrast = ImageEnhance.Contrast(image)
    contrast = contrast.enhance(2)
    dx_cont = np.array(contrast.getdata())
    return dx_cont.reshape([512,512,3])/255


# In[31]:


class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()


# In[85]:


def create_heat_map(x):
    sfs = [SaveFeatures(l) for l in [m[-3],m[-4], m[-5], m[-6]]]
    py = m(Variable(T(x)))
    py = np.exp(to_np(py)[0])
    for o in sfs: o.remove()

    feat = np.maximum(0,to_np(sfs[1].features[0]))

    dx = data.val_ds.denorm(x)[0]
    f2=np.dot(np.rollaxis(feat,0,3), py)
    f2-=f2.min()
    f2/=f2.max()
    return dx, f2


# In[86]:


x,y = data.val_ds[500]
x,y = x[None], A(y)[None,None]
vx = Variable(T(x), requires_grad=True)


# In[87]:


dx, f2 = create_heat_map(x)
fig, ax = plt.subplots(ncols=2, figsize=(20,10))
ax[0].imshow(dx)
ax[0].imshow(scipy.misc.imresize(f2, dx.shape), alpha=0.5, cmap='hot');
ax[1].imshow(dx)
draw_rect(ax[1], [120,200,100,100])


# Looks like the middle there is a patch which is particularly strong.. let's zoom in on it

# In[88]:


clipped = dx[50:250,100:300]


# In[89]:


clipped[None,:].shape


# In[90]:


c = data.val_ds.transform(clipped)


# In[91]:


preds = learn.predict_array(c[None,:])


# In[92]:


dx, f2 = create_heat_map(c[None])
fig, ax = plt.subplots(ncols=2, figsize=(20,10))
ax[0].imshow(dx)
ax[0].imshow(scipy.misc.imresize(f2, dx.shape), alpha=0.5, cmap='hot');
ax[1].imshow(dx)


# Well that certainly confirms that it is a stongly matching area.. I am still none the wiser to what makes the difference between xrays with pneumonia and without, but hey, it was interesting!

# Credit to fast.ai lesson 7 for this approach, and pretty much all of the code
