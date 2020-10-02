#!/usr/bin/env python
# coding: utf-8

# # Cyber Analytics Club
# # Evening 1: Building your very own image classifier using Fast.ai

# ## Agenda
# 
# This evening session is a mirror of Jeremy Howard and the Fast.ai course content (lecture 2). Thank you Fast.ai. There are some minor tweaks by yours truly to ensure the session is even more easier for folks with no Python skills.
# 
# 1. Introduction & Setup
# 2. Data Gathering
# 3. Preprocessing & Modeling
# 4. Performance Tuning & Evaluation
# 5. Data Cleaning & Retraining
# 6. Next time (maybe): Deployment

# ### Prerequisites
# 
# * Grown mindset and attitude
# * An idea of what code looks like
# 
# Moving forward with this club, a basic understanding of Python will go a long way. A great resource to start this journey is Google's introduction Python course:
# [Google's Python Class](https://developers.google.com/edu/python/)
# 
# This very course is where it all began for me 3 years ago when I decided to start coding again.

# ### Fast.ai resources
# 
# This session wouldn't be possible without the fantastic Fast.ai offerings, as well as, Sanyam Bhutani who authored the Kaggle version of Lesson 2. For more information access the below links:[](http://)

# [Lesson Video Link](https://course.fast.ai/videos/?lesson=2)
# 
# [Lesson resources and updates](https://forums.fast.ai/t/lesson-2-official-resources-and-updates/28630)
# 
# [Lesson chat](https://forums.fast.ai/t/lesson-2-chat/28722)
# 
# [Further discussion thread](https://forums.fast.ai/t/lesson-2-further-discussion/28706)

# # Lab: Creating your own dataset from Google Images
# ## Example "Never forget the Aussie-made Commodore Classifier"
# More information:
# [Holden Commodore Wikipedia](https://en.wikipedia.org/wiki/Holden_Commodore)

# References:
# * Lecture 2, Practical Deep Learning for Coders, Francisco Ingham and Jeremy Howard 2019. For complete info on the course, visit course.fast.ai
# * Google Images data set inspired by [Adrian Rosebrock](https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/)*

# In[ ]:


from fastai import *
from fastai.vision import *


# In[ ]:





# ## Data Gathering
# 
# **Steps**
# 
# Here we will:
# 
# * Go to images.google.com and search the classes of images you are keen to train a model on
# * For each image class you search, scroll down for a little bit to ensure you have enough images in view, then run the following commands in Chrome:
#     ```
#     First:
#     Press Ctrl + Shift + J in Windows/Linux and Cmd + Opt + J in Mac
#     ```
#     ```
#     Then:
#     urls = Array.from(document.querySelectorAll('.rg_di .rg_meta')).map(el=>JSON.parse(el.textContent).ou);
#     window.open('data:text/csv;charset=utf-8,' + escape(urls.join('\n')));
# ```
# * Now you have some labeled URLs, come back to your kaggle kernel and click "+ Add Data" then click "upload". Then name your dataset and upload the relevant csv files you just created.
# 
# 

# ### Define labels

# In[ ]:


# Exclude first gen commodores, they are giving me headaches

classes = ['andy', 'rafael', 'joker']


# In[ ]:


get_ipython().system('ls -la')


# In[ ]:


# Explain your data set in one word
data_name = "tennis"


# In[ ]:


for model in classes:
    folder = model
    file = model + '.csv'
    path = Path("data/" + data_name)
    print(file)
    dest = path/folder
    dest.mkdir(parents=True, exist_ok=True)
    get_ipython().system('cp ../input/* {path}/')
    download_images(path/file, dest, max_pics=299)


# Then we can remove any images that can't be opened:

# In[ ]:


for c in classes:
    print(c)
    verify_images(path/c, delete=True, max_size=500)


# ## Preprocessing & Modeling

# ### Prepare image data

# In[ ]:


np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=224, num_workers=0).normalize(imagenet_stats)


# In[ ]:



# If you already cleaned your data, run this cell instead of the one before
# np.random.seed(42)
# data = ImageDataBunch.from_csv(path, folder=".", valid_pct=0.2, csv_labels='cleaned.csv',
#         ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)


# In[ ]:


data.classes


# ### View data

# In[ ]:


data.show_batch(rows=3, figsize=(9,12))


# ### Training

# ### Transfer Learning using ResNet trained on ImageNet
# 
# Transfer learning is a technique where you use a model trained on a very large dataset (usually ImageNet in computer vision) and then adapt it to your own dataset. The idea is that it has learned to recognize many features on all of this data, and that you will benefit from this knowledge, especially if your dataset is small, compared to starting from a randomly initialized model. It has been proved in this article on a wide range of tasks that transfer learning nearly always give better results.
# 
# The fastai library includes several pretrained models from torchvision, namely:
# 
# * resnet18, resnet34, resnet50, resnet101, resnet152
# * squeezenet1_0, squeezenet1_1
# * densenet121, densenet169, densenet201, densenet161
# * vgg16_bn, vgg19_bn
# * alexnet

# In[ ]:


learner = cnn_learner(data, models.resnet34, metrics=error_rate)


# In[ ]:


learner.fit_one_cycle(6)


# In[ ]:


learner.save('stage-1')


# In[ ]:


learner.unfreeze()


# ## Performance Tuning & Evaluation

# ### Tuning

# In[ ]:


learner.lr_find()


# In[ ]:


learner.recorder.plot()


# In[ ]:


learner.fit_one_cycle(2, max_lr=slice(1e-4,1e-3))


# In[ ]:


learner.save('stage-2')


# ### Interpretation 

# In[ ]:


learner.load('stage-2');


# In[ ]:


interp = ClassificationInterpretation.from_learner(learner)


# In[ ]:


interp.most_confused(min_val=2)[:10]


# In[ ]:


interp.plot_confusion_matrix()


# ## Cleaning Up
# 
# Some of our top losses aren't due to bad performance by our model. There are images in our data set that shouldn't be.
# 
# Using the `ImageCleaner` widget from `fastai.widgets` we can prune our top losses, removing photos that don't belong.

# In[ ]:


from fastai.widgets import *


# First we need to get the file paths from our top_losses. We can do this with `.from_toplosses`. We then feed the top losses indexes and corresponding dataset to `ImageCleaner`.
# 
# Notice that the widget will not delete images directly from disk but it will create a new csv file `cleaned.csv` from where you can create a new ImageDataBunch with the corrected labels to continue training your model.

# Note: Please Set the Number of images to a number that you'd like to view:
# ex: ```n_imgs=100```

# In[ ]:


ds, idxs = DatasetFormatter().from_toplosses(learner, n_imgs=100)


# In[ ]:


ImageCleaner(ds, idxs, path)


# Flag photos for deletion by clicking 'Delete'. Then click 'Next Batch' to delete flagged photos and keep the rest in that row. ImageCleaner will show you a new row of images until there are no more to show. In this case, the widget will show you images until there are none left from top_losses.ImageCleaner(ds, idxs)
# 
# You can also find duplicates in your dataset and delete them! To do this, you need to run .from_similars to get the potential duplicates' ids and then run ImageCleaner with duplicates=True. The API works in a similar way as with misclassified images: just choose the ones you want to delete and click 'Next Batch' until there are no more images left.

# In[ ]:


ds, idxs = DatasetFormatter().from_similars(learner)


# Remember to recreate your ImageDataBunch from your cleaned.csv to include the changes you made in your data!

# ### Observations of Performance
# * Commodore VX and VT are very similar
# * Commodore VY and VZ are very similar
# * Quality Commodore images via Google Images can be hard with lots of rubbish
# * May need a lot more quality images to be a robust solution

# In[ ]:





# ## Deployment: Putting your model in production
# > 
# You probably want to use CPU for inference, except at massive scale (and you almost certainly don't need to train in real-time). If you don't have a GPU that happens automatically. You can test your model on CPU like so:

# In[ ]:


#import fastai
#fastai.defaults.device = torch.device('cpu')


# In[ ]:


#img = open_image(path/'black'/'00000021.jpg')
#img


# In[ ]:


#classes = ['black', 'grizzly', 'teddys']


# In[ ]:


#data2 = ImageDataBunch.single_from_classes(path, classes, tfms=get_transforms(), size=224).normalize(imagenet_stats)


# In[ ]:


#learn = create_cnn(data2, models.resnet34).load('stage-2')


# In[ ]:


#pred_class,pred_idx,outputs = learn.predict(img)
#pred_class


# So you might create a route something like this ([thanks](https://github.com/simonw/cougar-or-not) to Simon Willison for the structure of this code):
# 
# ```
# 
# @app.route("/classify-url", methods=["GET"])
# async def classify_url(request):
#     bytes = await get_bytes(request.query_params["url"])
#     img = open_image(BytesIO(bytes))
#     _,_,losses = learner.predict(img)
#     return JSONResponse({
#         "predictions": sorted(
#             zip(cat_learner.data.classes, map(float, losses)),
#             key=lambda p: p[1],
#             reverse=True
#         )
#     })
#     
#     ```
#     
# 

# (This [example](https://www.starlette.io/) is for the Starlette web app toolkit.)

# ## Things that can go wrong

# - Most of the time things will train fine with the defaults
# - There's not much you really need to tune (despite what you've heard!)
# - Most likely are
#   - Learning rate
#   - Number of epochs

# ### Learning rate (LR) too low

# In[ ]:


#learn = create_cnn(data, models.resnet34, metrics=error_rate)


# In[ ]:


#learn.fit_one_cycle(5, max_lr=1e-5)


# In[ ]:


#learn.recorder.plot_losses()


# As well as taking a really long time, it's getting too many looks at each image, so may overfit.

# ### Too few epochs

# In[ ]:


#learn = create_cnn(data, models.resnet34, metrics=error_rate, pretrained=False)


# In[ ]:


#learn.fit_one_cycle(1)


# ### Too many epochs

# In[ ]:


# np.random.seed(42)
# data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.9, bs=32, 
#        ds_tfms=get_transforms(do_flip=False, max_rotate=0, max_zoom=1, max_lighting=0, max_warp=0
#                              ),size=224, num_workers=4).normalize(imagenet_stats)


# In[ ]:


# learn = create_cnn(data, models.resnet50, metrics=error_rate, ps=0, wd=0)
# learn.unfreeze()


# In[ ]:


# learn.fit_one_cycle(40, slice(1e-6,1e-4))

