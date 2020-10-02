#!/usr/bin/env python
# coding: utf-8

# Loading All Libraries to get desire functionality. 
# * **fastaI** for overall model fitting , training and testing
# * **pathlib** for making string to Path so that can work with Python OS Module
# * **os** for getting working of Operating System like creating directory , getting files etc
# * **pandas** to open CSV as Data Frames and Data Table Operation
# * **seaborn and matplotlib** for ploting graphs
# * **PIL** *Python Image Library* for opening images and some image operation
# 
# 
# **matplotlib inline is method to work with graph inside notebook**
# 

# In[ ]:


from fastai.utils import *
from fastai.vision import *
from fastai.callbacks import *
from pathlib import Path
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import PIL
from torch.utils import model_zoo

get_ipython().run_line_magic('matplotlib', 'inline')


# Checking our input directory using ****ls**** Command
# 
# *CMD Commands / DOS Command start with ! (marks of exclamiation). *

# In[ ]:


get_ipython().system('ls ../input')


# Setting Path Input path , from which our data loader will open images . data_path is Python pathlib object which is reciving input path as a string. **train_df** is a pandas data frame created with train.csv files. Previous mentioned files have training images id and lables. **test_df** pandas frame have label of test image dataset from sample_submission.csv .

# In[ ]:


data_path = Path("../input/")
train_df = pd.read_csv(data_path/'train.csv')
test_df = pd.read_csv(data_path/'sample_submission.csv')


# To get Visualization about clases are balanced or not.

# In[ ]:


sns.countplot('has_cactus', data=train_df)
plt.title('Classes', fontsize=15)
plt.show()


# As clases are unblanced one class have 3x more images than other so we have to equalize both classes. We can perform two apporches 
# 1. Decrease the size of larger class.
# 2. Increase the size of shorter class.
# We will perform 2nd approch in later cell to balance classes.

# Let plot some images from both classes. Images are open by **Python Image Library** , you can use **OpenCV** also which is comprehensive library having Computer Vision and Image Processing operation. We do not have require any rich image processing operation till now so we think PIL is enough till now.

# In[ ]:


plt.figure(figsize=(8,6))

i = 0
sample = train_df.sample(12)
for row in sample.iterrows():
    img_name = row[1][0]
    img = PIL.Image.open(data_path/'train'/'train'/img_name)
    i += 1
    plt.subplot(3,4,i)
    title = 'Not Cactus (0)'
    if row[1][1] == 1:
        title = 'Cactus (1)'
    plt.title(title, fontsize=10)
    plt.imshow(img)
    plt.axis('off')

plt.subplots_adjust(top=0.90)
plt.suptitle('Sample of Images', fontsize=16)
plt.show()


# In[ ]:


df1 = train_df[train_df.has_cactus==0].copy()
df2 = df1.copy()
train_df = train_df.append([df1, df2], ignore_index=True)


# A quick plot verifies that the dataset is now very close to being balanced.

# In[ ]:


sns.countplot('has_cactus', data=train_df)
plt.title('Oversampled Classes', fontsize=15)
plt.show()


# # Data Augmentation
# Fast.ai has a powerful set of transformations [built into the library](https://docs.fast.ai/vision.transform.html). The has default setting which have experimentation been shown to be a good starting point for regular photos. These default augmentations include flipping on the horizontal axis, rotating, zooming, changing the lighting, and warping and are applied randomly on each photo during a training epoch.
# 
# Since this kernel uses aerial images, I enable flipping on the vertical axis by setting `flip_vert=True`. A trick I picked up from looking at some of the other fast.ai kernels (like this one by [Alexander Milekhin](https://www.kaggle.com/kenseitrg/simple-fastai-exercise)) is to upscale the original 32x32 images to 128x128, so the images are large enough for the fast.ai rotation, zooming, and warping transformations to be useful.

# In[ ]:


tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=20, max_lighting=0.3, max_warp=0.2, max_zoom=1.2)


# I then use the [data block api](https://docs.fast.ai/data_block.html) to finish constructing the dataset, holding back 20 percent of the training images as a validation set, labeling the images from the `train_df` dataframe, and adding the test images for ease of use later.

# In[ ]:


test_images = ImageList.from_df(test_df, path=data_path/'test', folder='test')

src = (ImageList.from_df(train_df, path=data_path/'train', folder='train')
       .split_by_rand_pct(0.2)
       .label_from_df()
       .add_test(test_images))


# When performing 1cycle learning, its recommended to use a large batch size. However, for this dataset using large batch sizes of 1024 or 2048 performed worse than smaller batch sizes. After some experimentation, I concluded that a batch size of 256 works well for training the model. 
# 
# When applying transformations that changes the image size, such as perspective warping, the best performing method of dealing with squaring the now warped image is to reflect the image data to fill the edges. There is a bug in the underlying pytorch reflection method which prevents it from being used on all datasets but given the regularity of these images reflection works fine.
# 
# Since I am using transfer learning via a ResNet-34 trained on ImageNet, I need to normalize the Aerial Cactus images with `imagenet_stats` to match the pre-trained model.

# In[ ]:


data = (src.transform(tfms, 
                     size=128,
                     resize_method=ResizeMethod.PAD, 
                     padding_mode='reflection')
        .databunch(bs=256)
        .normalize(imagenet_stats))


# A quick sanity check shows that both classes have loaded correctly.

# In[ ]:


data.classes, data.c


# Plotting `show_batch` gives a visual example of the images with the data augmentation from the fast.ai transformer applied.

# In[ ]:


data.show_batch(rows=3, figsize=(9,9))


# # Training the Model
# Behind the scenes, the fast.ai `cnn_learner` has stripped out the last few layers of the ResNet-34 and replaced them with a few untrained layers which ends in a linear layer to predict the two classes. The pretrained ResNet-34 model is frozen and is not allowed to change while the newly created layers will be trained to predict cactus or not cactus. After training the new layers, the pretrained ResNet-34 layers can be unfrozen and the whole model trained on the dataset.
# 
# The `cnn_learner` has sensible defaults for hyperparameters such as dropout, weight decay, and momentum. After some experimentation I concluded that the fast.ai defaults perform pretty well on this dataset and have left them be.
# 
# Since this kernel is being evaluated on the [area under the ROC curve](http://en.wikipedia.org/wiki/Receiver_operating_characteristic) (AUROC) between the predicted probabilities and the observed targets, I will include `AUROC` as a metric which the learner will output while training.

# In[ ]:



learn = cnn_learner(data,
                    models.resnet101,pretrained=True,
                    metrics=[accuracy, AUROC()],path='',model_dir='work')


# In[ ]:


get_ipython().system('ls')


# The learning rate hyperparameter is one of the most important hyperparameters to set, and the `lr_find` method provides a graphical way to set a good learning rate. After plotting the losses, one looks for the steepest slope where the loss is decreasing the fastest. There are multiple selections that could be made, and I have chosen the steepest looking slope right before the incline decreases at 1e-3.

# In[ ]:


learn.lr_find(stop_div=True)
learn.recorder.plot(suggestion=True)


# I will train the frozen layers of the model using `fit_one_cycle` for five epochs.

# In[ ]:


lr = 3.98E-04
learn.fit_one_cycle(5, lr)


# The learner has a built in method for plotting the training and validation loss.

# In[ ]:


learn.recorder.plot_losses()


# ## A Digression on Fit One Cycle
# `fit_one-cycle` works by taking the learning rate, dividing it by ten, and then gradually increasing and then decreasing the learning rate as training progresses. A simplified version of the idea behind 1cycle learning is increasing the learning rate allows the model to escape any suboptimal local minima. Then decreasing the learning rate assists the model in selecting a good minimum. The momentum of the model is adjusted in the opposite direction so the model does not overshoot while at the highest learning rate, but still has enough momentum to find a new minima at the lower learning rates.
# 
# The charts below show the cyclical learning rate and momentum during the training of the frozen ResNet-34 model. 
# 
# For more details on 1cycle learning, check out [Sylvain Gugger's post](https://sgugger.github.io/the-1cycle-policy.html) on the subject.

# In[ ]:


learn.recorder.plot_lr(show_moms=True)


# ## Training the Unfrozen Model
# For many tasks, the output of the ResNet-34 model after five epochs would be satisfactory. But should the initial training not result in such highly accurate model, you can unfreeze the frozen layers using the `unfreeze` method of `learner` and then run `lr_find` to select a discriminative learning rate. The learning rate will be lower for the first layers of the model and then increase for the last layers of the model. A rule of thumb for selecting a good learning rate for the whole model is to pick the higher slice to be ten times before the loss jumps and then set the lower slice to be the original learning rate divided by five or ten.
# 
# In this case, I won't be using the normal procedure and instead will be using `freeze_to(1)`. This keeps the first half of the pre-trained ResNet-34 model frozen while allowing the second half to train on this dataset. The rational is the first layers are already good at picking out lines, shapes, combinations of shapes, and objects of increasing complexity. While the second half is good at identifying the wide variety of objects in ImageNet, so training the second half on this dataset will allow the model to specialize on this competition's desert fauna. 

# In[ ]:


learn.save('step-1')


# In[ ]:


get_ipython().system('ls work')


# While the default fast.ai settings worked well for the initial frozen training, I will set the weight decay `wd` to 0.1 and a range of dropout `ps` to be 0.6 and 0.4 for the unfrozen portions of the model. For this training, the model does not steadily progress in accuracy, AUROC, or training and validation loss. I will use the `SaveModelCallback` to keep track of the best performing model as determined by validation loss and use it at the end of training.
# 
# I will create a new learner with these settings and then load the weights trained in the previous step.

# In[ ]:


learn = cnn_learner(data,
                    models.resnet101, pretrained=True,
                    metrics=[accuracy, AUROC()], 
                    callback_fns=[partial(SaveModelCallback)],
                    wd=0.1,
                    ps=[0.9, 0.6, 0.4],
                    path = '')
learn = learn.load('../work/step-1')


# Next use `lr_find` again to to select a discriminative learning rate.

# In[ ]:


learn.freeze_to(4)
learn.lr_find()
learn.recorder.plot()


# In this case the optimal learning rates will not differ too much: 4e-4 and 2e-4.

# In[ ]:


learn.fit_one_cycle(3)


# In[ ]:


learn.fit_one_cycle(7)


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix(figsize=(2,2))


# In[ ]:


interp.plot_top_losses(4, figsize=(6,6))


# In[ ]:


probability, classification = learn.get_preds(ds_type=DatasetType.Test)
test_df.has_cactus = probability.numpy()[:, 0]
test_df.head()


# In[ ]:


test_df.to_csv("submission.csv", index=False)


# In[ ]:




