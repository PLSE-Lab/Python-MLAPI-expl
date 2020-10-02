#!/usr/bin/env python
# coding: utf-8

# # What is malaria?
# 
# > **Malaria spreads when a mosquito becomes infected with the disease after biting an infected person, and the infected mosquito then bites a noninfected person. The malaria parasites enter that person's bloodstream and travel to the liver. When the parasites mature, they leave the liver and infect red blood cells. **
# > * Malaria is a life-threatening disease caused by a parasite? that is transmitted through the bite of infected female Anopheles mosquitoes.
# * The parasite that causes malaria is a microscopic, single-celled organism called Plasmodium.
# * Malaria is predominantly found in the tropical and sub-tropical areas of Africa, South America and Asia.
# * There are six different species of malaria parasite that cause malaria in humans: Plasmodium falciparum, Plasmodium vivax, Plasmodium ovale curtisi, Plasmodium ovale wallikeri, Plasmodium malariae and the very rare Plasmodium knowlesi.
# 
# 
# ![](http://i.imgur.com/LmNjmjQ.jpg)
# 
# 1. Healthy red blood cell; 2. Malaria parasites developing within infected red blood cells; 3. Malaria parasites about to burst out of red blood cell.
# 
# 

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns

from torch import nn, optim
from torchvision import transforms, models, datasets

from fastai.callbacks import *
from sklearn.metrics import roc_curve, auc
from fastai.vision import *

sns.set(style='whitegrid')
plt.style.use('seaborn-darkgrid')
import os
print(os.listdir("../input/cell_images/cell_images/"))


# # How malaria transmitted?
# 
# * Malaria is transmitted via the bite of the female Anopheles mosquito.
# * These mosquitos most commonly bite between dusk and dawn.
# * If a mosquito bites a person already infected with the malaria parasite it can suck up the parasite in the blood and then spread the parasite on to the next person they bite.
# 
# 
# ![](https://www.mayoclinic.org/-/media/kcms/gbs/patient-consumer/images/2013/08/26/10/13/ds00475_im00175_mcdc7_malaria_transmitthu_jpg.jpg)

# In[ ]:


path = Path('../input/cell_images/cell_images/')
np.random.seed(42)

data = ImageDataBunch.from_folder(path, train='train', valid_pct=0.2, ds_tfms=get_transforms(),size=224, bs=128, num_workers=0).normalize(imagenet_stats)
data


# In[ ]:


data.classes


# ![](https://static1.squarespace.com/static/5b1399b63e2d09fc93010364/t/5c12f9e9575d1fee96a4f3f3/1544747519916/MalariaCells.jpg?format=750w)

# # Visualizing the Image

# In[ ]:


data.show_batch(4, figsize=(15,10))


# # Transfer Learning Using Resnet50

# In[ ]:


learn = create_cnn(data, models.resnet50 , model_dir="/tmp/model/", metrics=[accuracy, error_rate])
learn


# # Training the model

# In[ ]:


learn.fit_one_cycle(6, 1e-2, pct_start=0.05,callbacks=[SaveModelCallback(learn, every='improvement', monitor='accuracy')])
learn.recorder.plot_losses()
plt.show()


# # Fine Tuning

# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)
Learning_rate = learn.recorder.min_grad_lr
print(Learning_rate)
plt.show()


# In[ ]:


learn.fit_one_cycle(3, Learning_rate, callbacks=[SaveModelCallback(learn, every='improvement', monitor='accuracy')])


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(4, figsize=(10,8), heatmap=False)
plt.show()


# In[ ]:


interp.plot_confusion_matrix(figsize=(10, 8))
plt.show()
interp.most_confused()


# In[ ]:


learn.show_results(ds_type=DatasetType.Valid)


# # Reference
# https://www.yourgenome.org/facts/what-is-malaria
# 
# https://www.nature.com/scitable/nated/topicpage/how-the-malaria-parasite-remodels-and-takes-132627374
# 
# https://www.cdc.gov/malaria/about/biology/index.html
# 
# https://www.mayoclinic.org/diseases-conditions/malaria/multimedia/malaria-transmission-cycle/img-20006373
# 
