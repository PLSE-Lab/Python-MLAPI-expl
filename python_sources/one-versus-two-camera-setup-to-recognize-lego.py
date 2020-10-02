#!/usr/bin/env python
# coding: utf-8

# # One versus two camera setup to recognize LEGO bricks
# This simple kernel is to show the advantage of a two camera versus one camera setup. 
# 
# Version 2 of the [Images of LEGO bricks dataset](https://www.kaggle.com/joosthazelzet/lego-brick-images) are rendered with 2 different camera positions as explained in the [dataset creation article](https://www.kaggle.com/joosthazelzet/how-to-create-a-lego-bricks-dataset-using-maya). 
# This kernel will show how to create a Resnet34 model to recognize LEGO bricks. Next, the one camera versus two camera setup is demonstrated and shown how the two camera approach results in a much smaller error rate. 
# 
# In order to run this kernel you need to enable under the Kaggle NoteBook Settings: 
# - 'Internet' to on to be able to download the basic Resnet34 model 
# - 'GPU' to On to speed up the learning process

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *
from fastai.metrics import error_rate
from pathlib import Path
from ipywidgets import IntProgress
from IPython.display import display


# In[ ]:


path = '/kaggle/input/lego-brick-images'
imagePath = path +'/dataset'


# In[ ]:


#Test if the dataset images are found
fnames = get_image_files(imagePath)
fnames[:4]


# In[ ]:


# Test the regular expression to filter out the classification name. 
# The space at the end is trimmed during import into ImageDataBunch, so don't care.
import re
re.search(r'([^/]+) ', fnames[0].name)[0] 


# The Resnet34 model is trained using all images regardless the 2 camera orientation. During the validation phase the two cameras will be considered.

# In[ ]:


data = ImageDataBunch.from_name_re(imagePath, 
                                   fnames, 
                                   r'/([^/]+) ', 
                                   ds_tfms=get_transforms(), 
                                   size=224, 
                                   bs=64
                                  ).normalize(imagenet_stats)


# In[ ]:


#Check the number of training anf validation items.
#The validation.txt file is not used as input, the ImageDataBunch does this.
len(data.train_ds.x.items), len(data.valid_ds.x.items)


# In[ ]:


#Check the data classes
print([len(data.classes), data.classes])


# In[ ]:


#Be sure to enable under Kaggle NoteBook Settings: 'Internet' to On and 'GPU' to On
learn = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir='/kaggle/output')


# In[ ]:


lr_find(learn)
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(10, slice(6e-3), pct_start=0.9)


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(10, max_lr=slice(1e-6,1e-4))
learn.freeze()


# ## Result with one camera
# The result of the final training epoch shows a error rate of ~5%. 
# Let's take a look which images go most wrong:  

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()
interp.plot_top_losses(9, figsize=(15,11))


# If you examine these images then you could easily see what goes wrong. These viewpoints can occur with several bricks.
# The confusion matrix makes even more clear:

# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# For example, check row '3046 roof corner inside tile 2x2' with column '3003 brick 2x2' where the cell shows 13 errors. Let look up images of these 2 bricks what is going on:  

# In[ ]:


fig, ax = plt.subplots(1,2)
ax[0].imshow(plt.imread(f'{imagePath}/3046 roof corner inside tile 2x2 007R.png'));
ax[1].imshow(plt.imread(f'{imagePath}/3003 brick 2x2 000L.png'));


# With a little fantasy you can image why from some viewpoint angles will become an issue if only one camera is used.
# 
# Let's now invoke the two cameras and see what this improves.

# ## Two camera recognition
# ![](http://)Let's first start with a verification of how th error rate is determined on the final epoch to ensure we understand how this is calculated and to avoid we use different methods:

# In[ ]:


#Determine the error rate with one camera as verification. 
#This must be equal to the last outcome of training epoch.

prg = IntProgress(min=0, max=len(data.valid_ds.x.items)) # instantiate the progress bar
display(prg) # display the progress bar

err = 0
for f in data.valid_ds.x.items:
    cat = f.name[:-9]
    pred_class,pred_idx,outputs = learn.predict(open_image(f))
    pred_cat = learn.data.classes[pred_class.data.item()]
    if pred_cat != cat:
        err += 1
    prg.value += 1
    
print(f'Error rate with one camera: {err/len(data.valid_ds.x.items)}')


# The error rate is the same as last epoch. We are on the right track here.
# 
# There is always an image pair in the two camera setup. Let's select the images of one camera and lookup the twin image of the other camera.
# The ImageDataBunch object used during the training, selected 20% of all images regardless of camera orientation. The selection is at random, but statistically this should be a close to 50%/50% even split:

# In[ ]:


fnamesR = [f for f in data.valid_ds.x.items if f.name[-5:] == 'R.png']
fnamesL = [f for f in data.valid_ds.x.items if f.name[-5:] == 'L.png']
print([len(fnamesR), len(fnamesL)])


# The smallest set must be leading and selected to ensure always an matching twin can be found. So for the two camera we end up with a validation set of 3976 paired images:

# In[ ]:


if len(fnamesR) < len(fnamesL):
    suffix = 'L'
    fnames2 = fnamesR
else:
    suffix = 'R'
    fnames2 = fnamesL
print([suffix, len(fnames2)])


# Now it is time for the final number crunch:

# In[ ]:


#Determine the error rate with two cameras

prg = IntProgress(min=0, max=len(fnames2)) # instantiate the progress bar
display(prg) # display the progress bar

err = 0
for fA in fnames2:
    fB = Path(f'{imagePath}/{fA.name[:-5]}{suffix}.png')
    cat = fA.name[:-9]
    pred_classA,pred_idxA,outputsA = learn.predict(open_image(fA))
    pred_catA = learn.data.classes[pred_classA.data.item()]
    pred_classB,pred_idxB,outputsB = learn.predict(open_image(fB))
    pred_catB = learn.data.classes[pred_classB.data.item()]
    outputs = outputsA+outputsB
    arr = outputs.numpy()
    maxval = np.amax(arr)
    maxind = np.where(arr == maxval)[0][0]
    if data.classes[maxind] != cat:
        err += 1
    prg.value += 1
print(f'Error validation set with two cameras: {err/len(fnames2)}')


# And that is a real improvement going to <1%! 

# # Conclusion
# It does pay off to use a multiple camera setup.
