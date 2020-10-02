#!/usr/bin/env python
# coding: utf-8

# # <div align="center">Multi-label Classification using FastAi Library</div>
# 
# ## **Context**
# Applying Fastai library on apparel image dataset and creating a multi-label classification model based on what I learned from Jeremy Howard's [lesson 3 of the fastai course](https://course.fast.ai/videos/?lesson=3).
# 
# ## **Dataset**
# 
# While searching the internet for a good dataset to apply the multi-label classification on, I stumbled upon pyimagesearch's  [multi-label classification with keras's](https://www.pyimagesearch.com/2018/05/07/multi-label-classification-with-keras/) article, and Adrian used a small simple dataset containing 3 clothing categories. But to expand on the dataset, I combined it with [trolukovich's dataset](https://www.kaggle.com/trolukovich/apparel-images-dataset) and my own by scraping Google and Bing using [cwerner's fastclass](https://github.com/cwerner/fastclass) package. Now it contains 8 different apparel categories in 9 different colours. It is published on Kaggle under the name [Apparel Dataset](https://www.kaggle.com/kaiska/apparel-dataset)
# 
# If you want to create your own image set, I highly recommend using Christian Warner's [fastclass package](https://github.com/cwerner/fastclass), he explains how to use it in a [short article](https://www.christianwerner.net/tech/Build-your-image-dataset-faster/).
# Additionally, there is a [tutorial](https://www.pyimagesearch.com/2018/04/09/how-to-quickly-build-a-deep-learning-image-dataset/) on pyimagesearch which helps you build an image dataset by scraping bing, but it uses a more difficult approach and requires bing API which, if you are not a student, will require you to input your credit card information along side phone verification.
# 
# For this kernel, I will be applying the fastai library to classify the apparel and its colour within an image.

# To add the image dataset to your kaggle kernel, simply click on **File** and then **Add or upload data** from within a kernel you are editing, then paste [`kaiska/apparel-dataset`](https://kaggle.com/kaiska/apparel-dataset) in the search box and click `Add`.
# 
# ![add apparel dataset](https://i.imgur.com/pfI09eB.gif)

# ## Loading relevant libraries
# 
# Every notebook starts with the following three lines; they ensure that any edits to libraries you make are reloaded here automatically, and 
# also that any charts or images displayed are shown in this notebook.

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# We import all the necessary packages. We are going to work with the [fastai V1 library](http://www.fast.ai/2018/10/02/fastai-ai/) which sits on top of [Pytorch 1.0](https://hackernoon.com/pytorch-1-0-468332ba5163). The fastai library provides many useful functions that enable us to quickly and easily build neural networks and train our models.

# In[ ]:


from fastai import *
from fastai.vision import *

import os
import sys
import shutil
import requests
# from PIL import Image
from io import BytesIO


# ### Moving files 
# 
# Let's first copy our dataset to `/kaggle/working/` to be able to apply changes to the dataset without having to change the directory later. This is because the input directory on kaggle is read-only.

# In[ ]:


# copy dataset to working (to enable manipulating the directory)
path = '/kaggle/input/apparel-dataset/'   
dest = '/kaggle/working/dataset/'
shutil.copytree(path, dest, copy_function = shutil.copy)  


# In this dataset, each picture can have multiple labels. If we take a look at the folder names, we see that each folder contains two labels seperated by an underscore.

# In[ ]:


os.listdir('/kaggle/working/dataset/')


# ### Creating DataBunch
# 
# To put this in a `DataBunch` while using the [data block API](https://docs.fast.ai/data_block.html), we then need to be using ImageList (and not ImageDataBunch). This will make sure the model created has the proper loss function to deal with the multiple classes. Also, the main difference for using `ImageList` over `ImageDataBunch` is that the later has pre-set constrains, while using `ImageList` gives you [more flexibility](https://forums.fast.ai/t/dataset-creation-imagedatabunch-vs-imagelists/45427/2).

# In[ ]:


tfms = get_transforms()

img_src = '/kaggle/working/dataset/'
src = (ImageList.from_folder(img_src) #set image folder
       .split_by_rand_pct(0.2) #set the split of training and validation to 80/20
       .label_from_folder(label_delim='_')) #get label names from folder and split by underscore

data = (src.transform(tfms, size=256) #set image size to 256
        .databunch(num_workers=0).normalize(imagenet_stats))


# Using `data.show_batch()` we can have a look at some of our pictures with the labels associated with them. Since this is a multi-label classification data, there are multiple labels associated with each of our images separated by ;.

# In[ ]:


data.show_batch(rows=3, figsize=(12,9))
print(f"""Classes in our data: {data.classes}\n
Number of classes: {data.c}\n
Training Dataset Length: {len(data.train_ds)}\n
Validation Dataset Length: {len(data.valid_ds)}""")


# To create a Learner we use [`cnn_learner`](https://docs.fast.ai/vision.learner.html#cnn_learner) instead of `create_cnn` which has been deprecated. Our base architecture is `resnet50` for this classification, but the metrics are a little bit different; we will use accuracy_thresh instead of accuracy.
# 
# When dealing with single classification problems, we determined the prediction for a given class by simply picking the prediction that had the highest accuracy, but for this problem, each activation can be 0 or 1. `accuracy_thresh` selects the ones that are above a certain threshold (0.5 being the default) and compares them to the ground truth.
# 
# Since there are 17 possible classes, we're going to have one probability for each of those. But then we're not just going to pick out one of those 17, we're going to pick out n of those 17. So what we do is, we compare each probability to some threshold. Then we say anything that's higher than that threshold, we're going to assume that the models saying it does have that feature. So we can pick that threshold.
# 
# For this kernel, we will adjust the threshold to 0.2.

# In[ ]:


acc_02 = partial(accuracy_thresh, thresh=0.2)
learn = cnn_learner(data, models.resnet50, metrics=acc_02, model_dir='/kaggle/working/models')


# To train a model properly, we should first train it without unfreezing the learner as that will train the head weights to better understand and categories our images. I recommend reading [Poonam's](https://forums.fast.ai/t/why-do-we-need-to-unfreeze-the-learner-everytime-before-retarining-even-if-learn-fit-one-cycle-works-fine-without-learn-unfreeze/41614/5) advice on this.

# In[ ]:


learn.fit_one_cycle(5)


# In[ ]:


learn.save('stage-1-rn50')


# Now we unfreeze the model, and we check our learning rate to find the best learning rate with the minimal loss so we can retrain our model on more restricted learning rate.

# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# As you can see, the learning rate is at its lowest loss between 1e<sup>-5</sup> and 1e<sup>-4</sup>. We add the constraints around those numbers to our next cycle and train the model again.

# In[ ]:


learn.fit_one_cycle(5, slice(3e-5, 5e-4))


# Now we have a better accuracy overall for our model and at a point where it will perform well on most given images. We save the model and prepare to use it for product.

# In[ ]:


learn.save('stage-2-rn50')


# It's a good rule of thumb to save your models as you go along. Particularly, you want to know if it is before or after the unfreeze (stage 1 or 2), what size you were training on, and what architecture were you training on. That way you can always load the models without having to retrain them, help you go back and experimenting pretty easily.

# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


# If you need to load a model, use the funciton below
# learn.load('/kaggle/input/multilabel-models/models/stage-2-rn50')


# ## Predictions
# ### Predicting a test set
# 
# To predict a testset, we simply refer to the [fastai documentation](https://docs.fast.ai/basic_train.html#Learner.get_preds).
# We simply need use `load_learner` method and point it to the path of `export.pkl` file, and define the test folder. Then just run `learn.get_preds()` to get the predictions.
# 
# Note here that the predictions are ordered based on the labels order. You can get the list labels by running `learn.data.c2i`.

# In[ ]:


learn = load_learner('/kaggle/input/multilabel-models/models/', 
                     test=ImageList.from_folder('/kaggle/input/apparel/black_pants')) #loading from training set as an example only
preds,_ = learn.get_preds(ds_type=DatasetType.Test)


# ### Predicting imges from URL or local file 
# 
# There are two main ways to predict images, either by uploading a file or by reading an already hosted image.
# 
# * To predict a hosted image, we will simply load this image, convert it to an image file and predict.
# * To predict a local file, simply open the image using `open_image(path_to_img)`
# 
# Below is the code on how to do it.

# In[ ]:


"""
Get the prediction labels and their accuracies, then return the results as a dictionary.

[obj] - tensor matrix containing the predicted accuracy given from the model
[learn] - fastai learner needed to get the labels
[thresh] - minimum accuracy threshold to returning results
"""
def get_preds(obj, learn, thresh = 15):
    labels = []
    # get list of classes from Learner object
    for item in learn.data.c2i:
        labels.append(item)

    predictions = {}
    x=0
    for item in obj:
        acc= round(item.item(), 3)*100
#         acc= int(item.item()*100) # no decimal places
        if acc > thresh:
            predictions[labels[x]] = acc
        x+=1
        
    # sorting predictions by highest accuracy
    predictions ={k: v for k, v in sorted(predictions.items(), key=lambda item: item[1], reverse=True)}

    return predictions


# In[ ]:


from io import BytesIO
import requests

url = "https://live.staticflickr.com/8188/28638701352_1aa058d0c6_b.jpg" 
response = requests.get(url).content #get request contents

img = open_image(BytesIO(response)) #convert to image
# img = open_image(path_to_img) #for local image file

img.show() #show image
_, _, pred_pct = learn.predict(img) #predict while ignoring first 2 array inputs
print(get_preds(pred_pct, learn))


# ## Deployment
# 
# In order to deploy your trained model online, I recommend checking out fastai's tutorial on how to [deploy on Render](https://course.fast.ai/deployment_render.html). It's a free and easy method which will take no more than 30 minutes to set up.
# You will need:
# * A google drive or dropbox account for uploading model files.
# * A github account to fork fastai's repository which uses starlette.
# * A render account to host the code.
# 
# If you are interested in deploying using other methods, there are 5 other ways listed on fastai.
# 

# To [export](https://docs.fast.ai/basic_train.html#Learner.export) the state of the Learner, simply use `learn.export()`. This will be needed to predict new images when deploying your trained model online. By default, the exported learner file name is `export.pkl`

# In[ ]:


learn.export()

