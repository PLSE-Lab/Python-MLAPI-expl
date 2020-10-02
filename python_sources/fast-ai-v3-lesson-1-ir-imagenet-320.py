#!/usr/bin/env python
# coding: utf-8

# **If you like this Kernel enough to fork it, please remember to upvote as well!**

# This is the first notebook from version 3 of the fast.ai course! For more information you can visit https://www.fast.ai/.

# # Lesson 1 - Custom Modified - Imagenet 320

# In this lesson we will build our first image classifier from scratch, and see if we can achieve world-class results. Let's dive in!
# 
# Every notebook starts with the following three lines; they ensure that any edits to libraries you make are reloaded here automatically, and also that any charts or images displayed are shown in this notebook.

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# We import all the necessary packages. We are going to work with the [fastai V1 library](http://www.fast.ai/2018/10/02/fastai-ai/) which sits on top of [Pytorch 1.0](https://hackernoon.com/pytorch-1-0-468332ba5163). The fastai library provides many useful functions that enable us to quickly and easily build neural networks and train our models.

# In[ ]:


from fastai import *
from fastai.vision import *


# If you're using a computer with an unusually small GPU, you may get an out of memory error when running this notebook. If this happens, click Kernel->Restart, uncomment the 2nd line below to use a smaller *batch size* (you'll learn all about what this means during the course), and try again.

# In[ ]:


bs = 64
# bs = 16   # uncomment this line if you run out of memory even after clicking Kernel->Restart


# ## Looking at the data

# Imagenette	Based on Deng et al., 2009. 	320 px.	A subset of 10 easily classified classes from Imagenet: tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute
# https://github.com/fastai/imagenette
# 
# We are going to use the `untar_data` function to which we must pass a URL as an argument and which will download and extract the data.

# In[ ]:


help(untar_data)


# In[ ]:


path = untar_data(URLs.IMAGENETTE_320); path


# In[ ]:


path.ls()


# In[ ]:


path_img = path/'train'
print("Here are the labels...  I don't know what n02979186 translates to, but let's find out.")
path_img.ls()


# Fix the labels by renaming the folders...
# 

# In[ ]:


path_img.ls()
import os
os.rename(str(path_img) + '/n01440764', str(path_img) + '/Trench_Fish')
os.rename(str(path_img) + '/n02102040', str(path_img) + '/English_Springer_Dog')
os.rename(str(path_img) + '/n02979186', str(path_img) + '/Cassette_Player')
os.rename(str(path_img) + '/n03000684', str(path_img) + '/Chainsaw')
os.rename(str(path_img) + '/n03028079', str(path_img) + '/Church')
os.rename(str(path_img) + '/n03394916', str(path_img) + '/French_Horn')
os.rename(str(path_img) + '/n03417042', str(path_img) + '/Garbage_Truck')
os.rename(str(path_img) + '/n03425413', str(path_img) + '/Gas_Pump')
os.rename(str(path_img) + '/n03445777', str(path_img) + '/Golf_Ball')
os.rename(str(path_img) + '/n03888257', str(path_img) + '/Parachute')
path_img.ls()


# The first thing we do when we approach a problem is to take a look at the data. We _always_ need to understand very well what the problem is and what the data looks like before we can figure out how to solve it. Taking a look at the data means understanding how the data directories are structured, what the labels are and what some sample images look like.
# 
# The main difference between the handling of image classification datasets is the way labels are stored. In this particular dataset, labels are stored in the filenames themselves. We will need to extract them to be able to classify the images into the correct categories. Fortunately, the fastai library has a handy function made exactly for this, `ImageDataBunch.from_name_re` gets the labels from the filenames using a [regular expression](https://docs.python.org/3.6/library/re.html).

# In[ ]:


help(get_image_files)
fnames = get_image_files(path_img, True, True)
fnames[:5]


# In[ ]:


np.random.seed(2)
pat = re.compile(r'/([^/]+)/[^/]+_\d+.JPEG$')


# In[ ]:


data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs, num_workers=0
                                  ).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3, figsize=(10,10))


# In[ ]:


print(data.classes)
len(data.classes),data.c


# ## Training: resnet34

# Now we will start training our model. We will use a [convolutional neural network](http://cs231n.github.io/convolutional-networks/) backbone and a fully connected head with a single hidden layer as a classifier. Don't know what these things mean? Not to worry, we will dive deeper in the coming lessons. For the moment you need to know that we are building a model which will take images as input and will output the predicted probability for each of the categories (in this case, it will have 37 outputs).
# 
# We will train for 4 epochs (4 cycles through all our data).

# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate)


# In[ ]:


learn.fit_one_cycle(8)


# In[ ]:


learn.save('stage-1')


# ## Results

# Let's see what results we have got. 
# 
# We will first see which were the categories that the model most confused with one another. We will try to see if what the model predicted was reasonable or not. In this case the mistakes look reasonable (none of the mistakes seems obviously naive). This is an indicator that our classifier is working correctly. 
# 
# Furthermore, when we plot the confusion matrix, we can see that the distribution is heavily skewed: the model makes the same mistakes over and over again but it rarely confuses other categories. This suggests that it just finds it difficult to distinguish some specific categories between each other; this is normal behaviour.

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(18,18), heatmap=True)


# In[ ]:


interp.plot_confusion_matrix(figsize=(18,18), dpi=60)


# In[ ]:


interp.most_confused(min_val=1)


# ## Unfreezing, fine-tuning, and learning rates

# Since our model is working as we expect it to, we will *unfreeze* our model and train some more.

# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(2)
learn.load('stage-1');


# Weird...  Unfreezing and adding more cycles caused the error rate to go from 0.007 to 0.07?  
# 
# What if I do 6 cycles, unfreeze and fit two more cycles?  Is that also bad because two "fit_one_cycle" calls will result in more image re-use than one bigger fit_one_cycle?

# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(2)
learn.load('stage-1')


# Maybe lr_find() has an effect on re-training?  When we're super accurate, it's hard to tell if it's just the random numbers or something I'm doing.

# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-5,1e-4))


# So far I'm not getting better results after unfreezing (but it was better when I chose the max learning rate).  
# 
# Maybe a different data set would have different results.  Best results were from the initial training, with error rate of 0.007.

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(18,18), heatmap=True)


# ## Training: resnet50

# Now we will train in the same way as before but with one caveat: instead of using resnet34 as our backbone we will use resnet50 (resnet34 is a 34 layer residual network while resnet50 has 50 layers. It will be explained later in the course and you can learn the details in the [resnet paper](https://arxiv.org/pdf/1512.03385.pdf)).
# 
# Basically, resnet50 usually performs better because it is a deeper network with more parameters. Let's see if we can achieve a higher performance here. To help it along, let's us use larger images too, since that way the network can see more detail. We reduce the batch size a bit since otherwise this larger network will require more GPU memory.

# In[ ]:


data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(),
                                   size=299, bs=bs//2, num_workers=0).normalize(imagenet_stats)


# In[ ]:


learn = cnn_learner(data, models.resnet50, metrics=error_rate)


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# Let's fit_one_cycle 4 times.  (8 times didn't make a difference)
# 

# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


learn.save('stage-1-50')


# Fine tuning didn't make a difference, it was still jumping between 0.007 and 0.0065

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(18,18), heatmap=False)
interp.plot_top_losses(9, figsize=(18,18), heatmap=True)

