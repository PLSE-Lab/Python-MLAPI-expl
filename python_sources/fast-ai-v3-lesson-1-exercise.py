#!/usr/bin/env python
# coding: utf-8

# # Part 1 - exploring data sources

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

# We are going to look at a number of data sets to see the different ways that the data is structured and how to extract labels.
# 
# We are going to use the `untar_data` function to which we must pass a URL as an argument and which will download and extract the data.

# In[ ]:


#help(URLs)


# In[ ]:


#help(untar_data)


# The PLANET_SAMPLE data set contains images of the earth and a csv file that contains the labels.  Let's look at how to work with it.

# In[ ]:


path = untar_data(URLs.PLANET_SAMPLE); path


# In[ ]:


#path.ls()


# We see that there is a `train` directory and a `labels.csv` file.  Define some names to make life easier.

# In[ ]:


path_anno = path
path_img = path/'train'


# 
# 
# To begin we take a look at the `lables.csv`
# 
# Since it is a `.csv` we import the csv module into python.  And why not just print out the rows.

# In[ ]:


#import csv
#with open(path/'labels.csv') as csvDataFile:
#    csvReader = csv.reader(csvDataFile)
#    for row in csvReader:
#        print(row)


# OK.  We have two fields per row.  A file name and a list of tags?

# In[ ]:


#help(csv)


# We can create arrays to hold the values

# In[ ]:


fileName = []
labels = []

with open(path/'labels.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        fileName.append(row[0])
        labels.append(row[1])
#print(fileName)
#print(labels)


# Can we get a list of files that say, have the attribute `clear`?

# In[ ]:


clearFileNames = []
with open(path/'labels.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        if 'clear' in row[1]:
            clearFileNames.append(row[0])
#print(clearFileNames)


# And can we get a list of files that do not have atrribute clear?
# 

# In[ ]:


nonClearFileNames = []
with open(path/'labels.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        if 'clear' not in row[1] and row[0] != 'image_name':
            nonClearFileNames.append(row[0])
#print(nonClearFileNames)
nonClearFileNames[:5]


# So we have split our data into two parts, clear and non clear.  Now we want to set up a databunch based on that classification.  To do that we create a DataFrame that will give us the data structure to pass to our DataBunch.  Pandas helps us becaus we have a concat that can take two DataFrame and join them together.

# In[ ]:


#doc(DataFrame)
clearFileNames = []
nonClearFileNames = []
with open(path/'labels.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        if 'clear' in row[1] and row[0] != 'image_name':
            clearFileNames.append(row[0])
        if 'clear' not in row[1] and row[0] != 'image_name':
            nonClearFileNames.append(row[0])
clearDF = pd.DataFrame({'name':clearFileNames,'label':'clear'})
nonClearDF = pd.DataFrame({'name':nonClearFileNames,'label':'non clear'})
labelledData = pd.concat([clearDF, nonClearDF], ignore_index=True)


# In[ ]:


#doc(ImageDataBunch.from_df)
tfms = get_transforms()
data = ImageDataBunch.from_df(path/'train',labelledData, size=224, suffix='.jpg', ds_tfms=tfms, bs=bs).normalize(imagenet_stats)


# Let's take a look at what that gives us.

# In[ ]:


data.show_batch(rows=3, figsize=(9,9))


# OK.  Now train a model.

# In[ ]:


learn = create_cnn(data, models.resnet34, metrics=error_rate)


# In[ ]:


learn.fit_one_cycle(4)


# As we discussed in the meeting, since the training loss is still decreasing we can probably do better.  Let's try another round of training.

# In[ ]:


learn.fit_one_cycle(4)


# Keep going....

# In[ ]:


learn.fit_one_cycle(4)


# now it looks like the train_loss has plateaued as has the error rate. So let's save this model.

# In[ ]:


learn.save('Stage 1')


# # Results
# 
# Let's see what we got. 
# 
# Is the model making reasonable predictions?

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(9,9),heatmap=True)


# In[ ]:


interp.plot_top_losses(9, figsize=(9,9),heatmap=False)


# In[ ]:


#doc(interp.plot_top_losses)


# ## Unfreezing, fine-tuning, and learning rates
# 
# Since our model is working as we expect it to, we will *unfreeze* our model and train some more.

# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(1)


# In[ ]:


learn.load('Stage 1');


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-3))


# Using the Learining Rate graph and adjusting the paramaters of the max_lr slice and taking only the final descending portion of the graph gets me to a 6% error rate, which is improved from the first pass where it was ~10%.  Let's take a look at what it gets confused about now, and what it looks at while getting confused.

# In[ ]:


interp.plot_top_losses(9, figsize=(9,9),heatmap=True)


# The images it got wrong before it still gets wrong, but at least it is looking at different spots to come up with that incorrect conclusion?

# In[ ]:


learn.save('Stage_2')


# ## Training: resnet50

# Now we will train in the same way as before but with one caveat: instead of using resnet34 as our backbone we will use resnet50 (resnet34 is a 34 layer residual network while resnet50 has 50 layers. It will be explained later in the course and you can learn the details in the [resnet paper](https://arxiv.org/pdf/1512.03385.pdf)).
# 
# Basically, resnet50 usually performs better because it is a deeper network with more parameters. Let's see if we can achieve a higher performance here. To help it along, let's us use larger images too, since that way the network can see more detail. We reduce the batch size a bit since otherwise this larger network will require more GPU memory.

# In[ ]:


tfms = get_transforms()
data = ImageDataBunch.from_df(path/'train',labelledData, size=299, suffix='.jpg', ds_tfms=tfms, bs=bs//2, num_workers=0).normalize(imagenet_stats)


# In[ ]:


learn = create_cnn(data, models.resnet50, metrics=error_rate)


# In[ ]:


learn.fit_one_cycle(8)


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.save('stage-1-50')


# Let's see if full fine-tuning helps:

# redo our training with the new learning rates.  If it does not improve revert.

# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(8, max_lr=slice(5e-6,5e-2))


# Tried a few values for this.  Got down to 6% with 8 epochs.

# In[ ]:


learn.save('stage-1-50-v2')
learn.load('stage-1-50')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.most_confused(min_val=2)
interp.plot_top_losses(9, figsize=(9,9),heatmap=True)


# A totaly different set of images that are confusing to the model.
