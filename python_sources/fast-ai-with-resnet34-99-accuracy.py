#!/usr/bin/env python
# coding: utf-8

# # Training Monkey Recognition with Fast.ai and ResNet34
# Monkey classification performed using a pre-trained model with ImageNet (1.2 million images and 1000 classes). The pre-trained ResNet34 is a CNN model, a varied version of the model that won the 2015 ImageNet competition.
# 
# [ResNet-34](https://www.kaggle.com/pytorch/resnet34/home): Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
# [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
# 
# This method achieved a **high accuracy of 98%-100% on validation** (depending on seed), wrongly classifying only 1~3 monkeys out of the 272 validation examples, proving ResNet34 to also be well-adapted to a general purpose as detecting feature differences in monkey species.

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
import torch
import glob
import os
import pathlib
import matplotlib.pyplot as plt
from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *


# In[ ]:


PATH = "../input/10-monkey-species/"
sz=224


# ## Viewing some monkeys
# Let's first view the different species names along with an image from that class.

# In[ ]:


labels = np.array(pd.read_csv("../input/10-monkey-species/monkey_labels.txt", header=None, skiprows=1).iloc[:,2])
labels = [labels[i].strip() for i in range(len(labels))]
labels_name = ['Class: %d, %s'%(i,labels[i]) for i in range(10)]


# In[ ]:


def plots(ims, figsize=(12,6), rows=3, titles=None):
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows+1, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i])


# In[ ]:


imgs = []
for i in range(10):
    file = os.listdir(f'{PATH}training/training/n%d'%i)
    img = plt.imread(f'{PATH}training/training/n%d/{file[0]}'%i)
    imgs.append(img)

plots(imgs, titles=labels_name, rows=4, figsize=(16,15))


# ## Load ResNet34
# If working without internet connection, the following steps are necessary to import the resnet34 weights. Otherwise, these are not necessary.

# In[ ]:


cache_dir = os.path.expanduser(os.path.join('~', '.torch'))
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
models_dir = os.path.join(cache_dir, 'models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)


# In[ ]:


get_ipython().system('cp ../input/resnet34/resnet34.pth /tmp/.torch/models/resnet34-333f7ec4.pth')


# ## Train Model
# ResNet weights are already powerful enough to achieve high classification accuracy without needing a large number of epochs (we use only 2 in this case), and is able to arrive at 99% accuracy of classifying the 10 monkey species within a couple of GPU-minutes.

# In[ ]:


arch = resnet34
data = ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch, sz), trn_name='training/training', val_name='validation/validation')
data.path = pathlib.Path('.') 
learn = ConvLearner.pretrained(arch, data, precompute=False)
learn.fit(0.01, 2)


# ## Predictions for Validation Set
# Model is allowed to run on validation set and predictions for their classes are made.

# In[ ]:


log_preds = predict(learn.model,learn.data.val_dl)
preds = np.argmax(log_preds, axis=1)


# In[ ]:


def correct(is_correct): return np.where((preds == data.val_y)==is_correct)[0]

def load_img_id(ds, idx): return np.array(PIL.Image.open(PATH+ds.fnames[idx]))

def plot_val_with_title(idxs, title):
    imgs = [load_img_id(data.val_ds,x) for x in idxs]
    title_probs = ['Prediction: %d, Truth: %d'%(preds[x],data.val_y[x]) for x in idxs]
    print(title)
    return plots(imgs, rows=1, titles=title_probs, figsize=(16,8))


# ## Random Examples of Correctly Classified Monkeys

# In[ ]:


plot_val_with_title(np.random.choice(correct(True),3,replace=False), "Correctly classified")


# ## All Instances of Incorrectly Classified Monkeys
# We can view all the instances in which a monkey was wrongly classified. 

# In[ ]:


plot_val_with_title(correct(False), "All Incorrectly classified")


# ## Confusion Matrix
# We can also observe the confusion matrix to see the correct/incorrectly classified instances of all classes.

# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(data.val_y, preds)


# In[ ]:


plot_confusion_matrix(cm, data.classes)


# ## Testing on a Random Pictures
# Here, a few pictures of monkeys are obtained from the internet. We check if the network is able to correctly identify their species. For fun, we include also a picture of a monkey not from any of the 10 above species.

# In[ ]:


pltimgs = [plt.imread('../input/monkeys/'+name) for name in os.listdir('../input/monkeys/')]


# In[ ]:


plots(pltimgs, titles=os.listdir('../input/monkeys/'), rows=2, figsize=(16,15))


# In[ ]:


trn_tfms, val_tfms = tfms_from_model(arch,sz)
test_imgs = [val_tfms(open_image('../input/monkeys/'+name)) for name in os.listdir('../input/monkeys/')]
learn.precompute=False
test_pred = learn.predict_array(test_imgs)
test_pred = np.argmax(test_pred, axis=1)
test_pred


# In[ ]:


plots(pltimgs, titles=['Predict: '+labels[i] for i in test_pred], rows=2, figsize=(16,15))


# Except for the gorilla which was identified, the other monkeys are all correctly classified.

# In[ ]:




