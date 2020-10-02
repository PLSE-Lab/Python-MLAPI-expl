#!/usr/bin/env python
# coding: utf-8

# # **ResNet34 fine tune with 1 cycle policy- 87% Acc**
# ### Yulin Chen, PhD
# #### 03/08/2019
# 
# * **1. Introduction**
# * **2. Data Preprocessing**
#     * 2.1 Data visualization
#     * 2.2 Computing image statistics
#     * 2.3 Data standardization and data augmentation
# * **3. Convolutional Neural Network with Fastai**
#     * 3.1 Define the base model with ResNet34
#     * 3.2 Show a failure example of fixed learning rate
#     * 3.3 Evaluate learning rate and weight decay with One-Cycle-Policy
#     * 3.4 Unfreeze the head and train the model
#     * 3.5 Unfreeze all the layers and fine-tune the model
# * **4. Evaluate the model**
#     * 4.1 Confusion list
#     * 4.2 Gradient-weighted Class Activation Mapping

# # **1. Introduction**
# The Car data set contains 8,144 car images and 196 classes of car Make, Year, and Model. This gives ~40 images per class in average, which is a relative few images compared to the number of classifications. So, I choose to start with a pre-trained ResNet34, and avoid too complicated models to ease the overfitting issue. The model is first trained only the head layers, and followed by fine tuning the weights of the base model. The overfitting is the biggest corcerned in this project. Data augmentation and One-cycle-policy are used to reduce overfitting. One cycle policy also help prevent the model get stuck at local minima.
# 
# This Notebook follows three main parts:
# 
# The data preprocessing  
# The CNN modeling 
# The results analysis  
# 
# The model is built on fastai due to its easiness of implementing learning rate sweep and 1 cycle policy.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os


from fastai.vision import *
from fastai.metrics import error_rate
from fastai.vision import *
from torchvision.models import *
from glob import iglob
import cv2
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import os
from IPython.display import Image

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


#  # **2. Data Preprocessing**

# ## 2.1 Data visualization
# The size of the Images in the dataset are not consistent, but fastai will take care of that when feeding images to ImageDataBuntch.

# In[ ]:


data_dir='../input/stanford-car-dataset-by-classes-folder/car_data/car_data'
train_dir = "../input/stanford-car-dataset-by-classes-folder/car_data/car_data/train/*/*.jpg"
data_path = Path(data_dir)


# In[ ]:



fig, ax = plt.subplots(1,5, figsize=(20,4))
fig.suptitle('car image examples',fontsize=20)
# choose some images to plot
cnt = 1
plt.figure(1)

for img_path in iglob(train_dir):
    img = cv2.imread(img_path)
    plt.subplot(1,5,cnt)
    plt.imshow(img)
    #ax[0,cnt].imshow(img)
    cnt += 1
    if cnt > 5:
        break


# ## 2.2 Compute image statistics
# Calculating statistics will give channel averages of [0.454952, 0.460148, 0.470733], and standard deviation of [0.302969, 0.294664, 0.295581]. Ths info will later apply to the standardization of our dataset.

# In[ ]:



# As we count the statistics, we can check if there are any completely black or white images
x_tot = np.zeros(3)
x2_tot = np.zeros(3)
cnt = 0

for img_path in iglob(train_dir):
    imagearray = cv2.imread(img_path).reshape(-1,3)/255.
    x_tot += imagearray.mean(axis=0)
    x2_tot += (imagearray**2).mean(axis=0)
    cnt += 1
    
channel_avr = x_tot/cnt
channel_std = np.sqrt(x2_tot/cnt - channel_avr**2)
channel_avr,channel_std


# Note: I just noticed there is a method  `data.batch_stats()` can probably do the same thing.

# ## 2.3 Data standardization and augmentation
# There are couple of ways we can use to avoid overfitting; more data, augmentation, regularization and less complex model architectures. The image augmentations that I use are defined as follow: 
# 
# * random rotation (< 30 degree)
# * random flip (horizontal)
# * random lighting (< 10%)
# * random zoom (< 10%)
# 
# Then I load the images to an ImageDataBunch which standardize the image size to be 224, and the data are ready for training. 

# In[ ]:


# Create ImageDataBunch using fastai data block API
batch_size = 64
data = ImageDataBunch.from_folder(data_path,  
                                  valid_pct=0.2,
                                  ds_tfms=get_transforms(do_flip=True,flip_vert=False, max_rotate=30, max_zoom=0.1, max_lighting=0.1),
                                  size=224,
                                  bs=batch_size, 
                                  num_workers=0).normalize([tensor([0.454952, 0.460148, 0.470733]), tensor([0.302969, 0.294664, 0.295581])])
                                  # Normalize with training set stats. These are means and std's of each three channel and we calculated these previously in the stats step.


# # **3. Convolutional Neural Network with Fastai**
# ## 3.1 Define the base model with ResNet34
# In ML production pipeline, it is a good idea to start with a relatively simple model. With a simple model, we can very quickly see if there are some unexpected problems like bad data quality that will make any further investments into the model tuning not worth it. As shown in figure below (ref). Here I use a pretrained convnet model and transfer learning to adjust the weights to the data. Going for a deeper model architecture will start overfitting faster, especially when we only have ~40 images per classification in average. Here I use a ResNet34 model, which is implemented with double- or triple- layer skips as a residual component, in order to prevent vanishing gradient when network get deep.
# 
# I use Fast.ai V1 software library that is built on PyTorch. What I like about Fast.ai is that it includes many recent advancements in deep learning research, for example: Learning rate finder, discriminative learning rates, batchnorm freezing, one-cycle policy, etc.  
# 
# [*reference: https://arxiv.org/abs/1605.07678*](https://arxiv.org/abs/1605.07678)

# In[ ]:


Image('../input/screen-shots/Screen Shot_model summary.png')


# In[ ]:


def getLearner():
    return cnn_learner(data, resnet34, pretrained=True, path='.', metrics=accuracy, ps=0.5, callback_fns=ShowGraph)
learner = getLearner()


# In[ ]:


# some trick to make sure the pretrained weight gets downloaded correctly
get_ipython().system('cp ../input/resnet34/resnet34.pth /tmp/.cache/torch/checkpoints/resnet34-333f7ec4.pth')


# In[ ]:


def getLearner():
    return cnn_learner(data, resnet34, pretrained=True, path='.', metrics=accuracy, ps=0.5, callback_fns=ShowGraph)
learner = getLearner()


# ## 3.2 Show a failure example with fixed learning rate training (lr = 0.003)
# I do a fixed learning rate training on the model. Without optimizing learning rate, you can see the model gets stuck in local optima after couple epochs.

# In[ ]:


Image('../input/screen-shots/Screen Shot_resnet34_fit.png')


# After 10 epochs of training at learning rate 0.003, the accuracy reaches ~62%. We can see that from the plotted losses, both training and validation loss gets flattened out. The model get stuck at saddle point and gradient descent becomes very small.Large learning rate might be able to drive the model out of local minima, so we try to use one-cycle-policy.

# ## 3.3 Evaluate learning rate and weight decay with One-Cycle-Policy
# One cycle policy is proposed by Leslie Smith, arXiv, April 2018. The policy cycles the learning rate between lower bound and upper bound during a complete tra. This approach can help get out from the saddle point (local minimum). A cycle is an iteration where we go from lower bound learning rate to higher bound and back to lower bound. when learning rate is higher, the learning rate works as regularisation method and keep network from overfitting. This helps the network to avoid steep areas of loss and land better flatter minima. In addititon, Fastai library has implemented a training function for one cycle policy that can be used with only a few lines of code.
# 
# First, we find the optimal learning rate and weight decay values. The optimal learning rate is just before the base of the loss and before the start of divergence. It is important that choosing the maximum learning rate at the point where the loss is still descending.
# 
# As for the weight decay that is the regularization L2 penalty of the weight applied at the optimizer, Leslie proposes to select the largest one that will still let us train at a high learning rate. Also, smaller datasets and architectures seem to require larger values for weight decay while larger datasets and deeper architectures seem to require smaller values.  I do a small grid search with 1e-4, 1e-5 and 1e-6 weight decays.

# In[ ]:


# We can use lr_find with different weight decays and record all losses so that we can plot them on the same graph
# Number of iterations is by default 100, but at this low number of itrations, there might be too much variance
# from random sampling that makes it difficult to compare WD's. I recommend using an iteration count of at least 300 for more consistent results.
lrs = []
losses = []
wds = [1e-6, 1e-5, 1e-4]
iter_count = 300

for wd in wds:
    learner = getLearner() #reset learner - this gets more consistent starting conditions
    learner.lr_find(wd=wd, num_it=iter_count)
    lrs.append(learner.recorder.lrs)
    losses.append(learner.recorder.losses)


# In[ ]:


_, ax = plt.subplots(1,1)
min_y = 4
max_y = 7
for i in range(len(losses)):
    ax.plot(lrs[i], losses[i])
    min_y = min(np.asarray(losses[i]).min(), min_y)
ax.set_ylabel("Loss")
ax.set_xlabel("Learning Rate")
ax.set_xscale('log')
#ax ranges may need some tuning with different model architectures 
ax.set_xlim((1e-4,3e-1))
ax.set_ylim((min_y - 0.02,max_y))
ax.legend(wds)
ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))


# We want to select the largest weight decay that gets to a low loss and has the highest learning rate before shooting up. From the grid search of WD within range of 1e-4 ~ 1e-6, we don't see much difference. So I choose to use WD = 1e-5 to proceed. I select the learning rate around 1e-2 where it is close to the bottom but still descending.
# 
# ## 3.4 Unfreeze all the layers and fine-tune the model
# I train only the heads while keeping the rest of the model frozen. Otherwise, the random initialization of the head weights could harm the relatively well-performing pre-trained weights of the model. After the heads have adjusted and the model somewhat works, we can continue to train the rest of the weights. Fastai already take care of the part that freeze the base weight and unfreeze the head when loading a pretrained model. By default, Fastai cut the base model at the last convolutional layer and add:
# * an AdaptiveConcatPool2d layer,
# * a Flatten layer,
# * blocks of [BatchNorm, Dropout, Linear, ReLU] layers.

# In[ ]:


max_lr = 1e-2
wd = 1e-2
# 1cycle policy
learner_one_cycle = getLearner()
learner_one_cycle.fit_one_cycle(cyc_len=10, max_lr=max_lr, wd=wd)


# The accuracy of the model that is only trained the top layer gets ~75% after 10 epochs, which is much better than using fixed learning rate.

# In[ ]:


learner_one_cycle.recorder.plot_lr()


# We can see that the learning rate starts from lower and reaches the `max_lr` in the middle. Then it slows back down near the end. The idea is that we start with a low warm-up learning rate and gradually increase it to high. The higher rate is having a regularizing effect as it won't allow the model to settle for sharp and narrow local minima but pushes for wider and more stable one.
# 
# In the middle of our cycle, we start to lower the learning rate as we are hopefully in a good stable area. This means that we start to look for the minima within that area.

# In[ ]:


# before we continue, lets save the model at this stage
learner_one_cycle.save('resnet34_stage1', return_path=True)


# ## 3.5 Finetuning the baseline model
# Next, I unfreeze all the trainable parameters from the base model and continue its training.
# 
# The model already performs well, as I unfreeze the bottom layers that have been pre-trained with a large number of general images to detect common shapes and patterns, all weights are mostly adjusted. We should now train with much lower learning rates.

# In[ ]:


# unfreeze and run learning rate finder again
learner_one_cycle.unfreeze()
learner_one_cycle.lr_find(wd=wd)

# plot learning rate finder results
learner_one_cycle.recorder.plot()


# In[ ]:


# Now, smaller learning rates. This time we define the min and max lr of the cycle
learner_one_cycle.fit_one_cycle(cyc_len=10, max_lr=slice(5e-5,5e-4))
# Save the finetuned model
learner_one_cycle.save('resnet34_stage2')


# # **4. Evaluate the model**
# After another 10 epoches of training, the model is able to reach 87% of accuracy at the validation set. At the end of the training epochs,  the validation performance starts seperating from the training performance. This means that the model starts overfitting during the small learning rates. This is a good place to stop.  

# ## 4.1 Confusion list
# Confusion matrix can help us understand if there are certain classes that get predicted wrong more easily than others. The number of classifications are too large to plot with confusion matrix. I print out the list instead. `most_confused` method gives us sorted descending list of largest non-diagonal entries of **confusion matrix, presented as actual, predicted, number of occurrences**.   
#   
# The confusion list indicates that 'Dodge Caliber Wagon' with slightly high frenquency of wrong prediction.

# In[ ]:


# predict the validation set with our model
interp = ClassificationInterpretation.from_learner(learner_one_cycle)
interp.most_confused()


# ## 4.2 Gradient-weighted Class Activation Mapping (Grad-CAM)  
# This method produces a coarse localization map highlighting the areas that the model considers important for the classification decision. The visual explanation gives transparency to the model making it easier to notice if it has learned the wrong things. More details can refer to the [paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf).  
#   
#   The followings are Grad-CAM with 9 highest loss in the validation dataset.

# In[ ]:


interp.plot_top_losses(9, figsize=(20,20))

