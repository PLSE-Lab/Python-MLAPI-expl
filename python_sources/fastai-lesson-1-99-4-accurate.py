#!/usr/bin/env python
# coding: utf-8

# ## Code from fastai 2018 DL1 Lesson 1

# I just replaced the dataset and changed the learning rate and augmentation.

# In[ ]:


# Put these at the top of every notebook, to get automatic reloading and inline plotting
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# This file contains all the main external libs we'll use
from fastai.imports import *


# In[ ]:


from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *


# `PATH` is the path to your data - if you use the recommended setup approaches from the lesson, you won't need to change this. `sz` is the size that the images will be resized to in order to ensure that the training runs quickly. We'll be talking about this parameter a lot during the course. Leave it at `224` for now.
# 
# `TMP_PATH` and `MODEL_PATH` are required because otherwise the learn object would try to write on the read-only data directory.

# In[ ]:


#PATH = "data/dogscats/"
TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model/"
PATH = "../input/lego brick images/LEGO brick images/"
sz=224


# It's important that you have a working NVidia GPU set up. The programming framework used to behind the scenes to work with NVidia GPUs is called CUDA. Therefore, you need to ensure the following line returns `True` before you proceed. If you have problems with this, please check the FAQ and ask for help on [the forums](http://forums.fast.ai).

# In[ ]:


torch.cuda.is_available()


# In addition, NVidia provides special accelerated functions for deep learning in a package called CuDNN. Although not strictly necessary, it will improve training performance significantly, and is included by default in all supported fastai configurations. Therefore, if the following does not return `True`, you may want to look into why.

# In[ ]:


torch.backends.cudnn.enabled


# Testing that the data can be read.

# In[ ]:


os.listdir(PATH)


# In[ ]:


os.listdir(f'{PATH}valid')


# In[ ]:


files = os.listdir(f'{PATH}valid/3673 Peg 2M')[:5]
files


# In[ ]:


img = plt.imread(f'{PATH}valid/3004 Brick 1x2/{files[1]}')
plt.imshow(img);


# ## First model

# We're going to use a <b>pre-trained</b> model, that is, a model created by some one else to solve a different problem. Instead of building a model from scratch to solve a similar problem, we'll use a model trained on ImageNet (1.2 million images and 1000 classes) as a starting point. The model is a Convolutional Neural Network (CNN), a type of Neural Network that builds state-of-the-art models for computer vision. We'll be learning all about CNNs during this course.
# 
# We will be using the <b>resnet34</b> model. resnet34 is a version of the model that won the 2015 ImageNet competition. Here is more info on [resnet models](https://github.com/KaimingHe/deep-residual-networks). We'll be studying them in depth later, but for now we'll focus on using them effectively.
# 

# In[ ]:


# Uncomment the below if you need to reset your precomputed activations
# shutil.rmtree(f'{PATH}tmp', ignore_errors=True)


# In[ ]:


arch=resnet34
data = ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch, sz))
learn = ConvLearner.pretrained(arch, data, precompute=True, tmp_name=TMP_PATH, models_name=MODEL_PATH)
learn.fit(0.02, 3)


# ## Choosing a learning rate

# The *learning rate* determines how quickly or how slowly you want to update the *weights* (or *parameters*). Learning rate is one of the most difficult parameters to set, because it significantly affects model performance.
# 
# The method `learn.lr_find()` helps you find an optimal learning rate. It uses the technique developed in the 2015 paper [Cyclical Learning Rates for Training Neural Networks](http://arxiv.org/abs/1506.01186), where we simply keep increasing the learning rate from a very small value, until the loss stops decreasing. We can plot the learning rate across batches to see what this looks like.
# 
# We first create a new learner, since we want to know how to set the learning rate for a new (untrained) model.

# In[ ]:


learn = ConvLearner.pretrained(arch, data, precompute=True, tmp_name=TMP_PATH, models_name=MODEL_PATH)


# In[ ]:


lrf=learn.lr_find()


# Our `learn` object contains an attribute `sched` that contains our learning rate scheduler, and has some convenient plotting functionality including this one:

# In[ ]:


learn.sched.plot_lr()


# Note that in the previous plot *iteration* is one iteration (or *minibatch*) of SGD. In one epoch there are 
# (num_train_samples/num_iterations) of SGD.
# 
# We can see the plot of loss versus learning rate to see where our loss stops decreasing:

# In[ ]:


learn.sched.plot()


# The loss is still clearly improving at lr=1e-1 (0.1), so that's what we use. Note that the optimal learning rate can change as we train the model, so you may want to re-run this function from time to time.

# ## Improving our model

# ### Data augmentation

# If you try training for more epochs, you'll notice that we start to *overfit*, which means that our model is learning to recognize the specific images in the training set, rather than generalizing such that we also get good results on the validation set. One way to fix this is to effectively create more data, through *data augmentation*. This refers to randomly changing the images in ways that shouldn't impact their interpretation, such as horizontal flipping, zooming, and rotating.
# 
# We can do this by passing `aug_tfms` (*augmentation transforms*) to `tfms_from_model`, with a list of functions to apply that randomly change the image however we wish. The Lego images are rendered from different orientations. We can use the `transforms_top_down` augmentation because having the Legos upside down is fine. We can also specify random zooming of images up to specified scale by adding the `max_zoom` parameter.

# In[ ]:


tfms = tfms_from_model(resnet34, sz, aug_tfms=transforms_top_down, max_zoom=1.1)


# In[ ]:


def get_augs():
    data = ImageClassifierData.from_paths(PATH, bs=2, tfms=tfms, num_workers=1)
    x,_ = next(iter(data.aug_dl))
    return data.trn_ds.denorm(x)[1]


# In[ ]:


ims = np.stack([get_augs() for i in range(6)])


# In[ ]:


plots(ims, rows=2)


# Let's create a new `data` object that includes this augmentation in the transforms.

# In[ ]:


data = ImageClassifierData.from_paths(PATH, tfms=tfms)
learn = ConvLearner.pretrained(arch, data, precompute=True, tmp_name=TMP_PATH, models_name=MODEL_PATH)


# In[ ]:


learn.precompute=False


# ### Fine-tuning and differential learning rate annealing

# Now that we have a good final layer trained, we can try fine-tuning the other layers. To tell the learner that we want to unfreeze the remaining layers, just call (surprise surprise!) `unfreeze()`.

# In[ ]:


learn.unfreeze()


# Note that the other layers have *already* been trained to recognize imagenet photos (whereas our final layers where randomly initialized), so we want to be careful of not destroying the carefully tuned weights that are already there.
# 
# Generally speaking, the earlier layers (as we've seen) have more general-purpose features. Therefore we would expect them to need less fine-tuning for new datasets. For this reason we will use different learning rates for different layers: the first few layers will be at 1e-4, the middle layers at 1e-3, and our FC layers we'll leave at 2e-1. We refer to this as *differential learning rates*, although there's no standard name for this techique in the literature that we're aware of.

# In[ ]:


lr=np.array([1e-4,1e-3,2e-1])


# In[ ]:


# On a GTX 1080 Ti this takes about 8 minutes
learn.fit(lr, 4, cycle_len=1, cycle_mult=2)


# In[ ]:


learn.sched.plot_lr()
learn.sched.plot_loss()


# It seems that the model is not ovefitting yet. Let's save the model.

# In[ ]:


learn.save('224_all')


# In[ ]:


learn.load('224_all')


# There is something else we can do with data augmentation: use it at *inference* time (also known as *test* time). Not surprisingly, this is known as *test time augmentation*, or just *TTA*.
# 
# TTA simply makes predictions not just on the images in your validation set, but also makes predictions on a number of randomly augmented versions of them too (by default, it uses the original image along with 4 randomly augmented versions). It then takes the average prediction from these images, and uses that. To use TTA on the validation set, we can use the learner's `TTA()` method.

# In[ ]:


log_preds,y = learn.TTA()
probs = np.mean(np.exp(log_preds),0)


# In[ ]:


accuracy_np(probs, y)


# I generally see about a 10-20% reduction in error on this dataset when using TTA at this point, which is an amazing result for such a quick and easy technique!
# 
# This time TTA didn't seem to help as the dataset is very clean. The Legos 

# ## Review: easy steps to train a world-class image classifier

# 1. precompute=True
# 1. Use `lr_find()` to find highest learning rate where loss is still clearly improving
# 1. Train last layer from precomputed activations for 1-2 epochs
# 1. Train last layer with data augmentation (i.e. precompute=False) for 2-3 epochs with cycle_len=1
# 1. Unfreeze all layers
# 1. Set earlier layers to 3x-10x lower learning rate than next higher layer
# 1. Use `lr_find()` again
# 1. Train full network with cycle_mult=2 until over-fitting
