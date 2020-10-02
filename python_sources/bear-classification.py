#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai import *
from fastai.vision import *


# ## **Get a list of URLs**
# 
# **Search and scroll**
# 
# Go to Google Images and search for the images you are interested in. The more specific you are in your Google Search, the better the results and the less manual pruning you will have to do.
# 
# Scroll down until you've seen all the images you want to download, or until you see a button that says 'Show more results'. All the images you scrolled past are now available to download. To get more, click on the button, and continue scrolling. The maximum number of images Google Images shows is 700.
# 
# It is a good idea to put things you want to exclude into the search query, for instance if you are searching for the Eurasian wolf, "canis lupus lupus", it might be a good idea to exclude other variants:
# 
# "canis lupus lupus" -dog -arctos -familiaris -baileyi -occidentalis
# 
# You can also limit your results to show only photos by clicking on Tools and selecting Photos from the Type dropdown.
# 
# **Download into file**
# 
# Now you must run some Javascript code in your browser which will save the URLs of all the images you want for you dataset.
# 
# Press CtrlShiftJ in Windows/Linux and CmdOptJ in Mac, and a small window the javascript 'Console' will appear. That is where you will paste the JavaScript commands.
# 
# You will need to get the urls of each of the images. You can do this by running the following commands:
# 
# ```
# urls = Array.from(document.querySelectorAll('.rg_di .rg_meta')).map(el=>JSON.parse(el.textContent).ou);
# window.open('data:text/csv;charset=utf-8,' + escape(urls.join('\n')));
# ```
# 
# **Create directory and upload urls file into your server**
# 
# Choose an appropriate name for your labeled images. You can run these steps multiple times to grab different labels.

# **Note:** You can download the urls locally and upload them to kaggle using:
# 
# 
# Here, I have uploaded the urls for 
#  - Teddy
#  - Grizzly
#  - Black 

# In[ ]:


classes = ['teddys','grizzly','black']


# In[ ]:


folder = 'black'
file = 'urls_black.txt'


# In[ ]:


path = Path('data/bears')
dest = path/folder
dest.mkdir(parents=True, exist_ok=True)


# In[ ]:


get_ipython().system('cp ../input/* {path}/')


# In[ ]:


download_images(path/file, dest, max_pics=200)


# In[ ]:


folder = 'teddys'
file = 'urls_teddys.txt'


# In[ ]:


path = Path('data/bears')
dest = path/folder
dest.mkdir(parents=True, exist_ok=True)


# In[ ]:


download_images(path/file, dest, max_pics=200)


# In[ ]:


folder = 'grizzly'
file = 'urls_grizzly.txt'


# In[ ]:


path = Path('data/bears')
dest = path/folder
dest.mkdir(parents=True, exist_ok=True)


# In[ ]:


download_images(path/file, dest, max_pics=200)


# Then we can remove any images that can't be opened:

# In[ ]:


for c in classes:
     print(c)
     verify_images(path/c, delete=True, max_size=500)


# ## View data

# In[ ]:


np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=64, num_workers=0).normalize(imagenet_stats)


# In[ ]:


data.classes


# In[ ]:


data.show_batch(rows=3, figsize=(3,4))


# In[ ]:


learn = create_cnn(data, models.resnet34, metrics=error_rate)


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


learn.save('stage-1')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(2, max_lr=slice(1e-4,1e-3))


# In[ ]:


learn.save('stage-2')


# ## Interpretation

# In[ ]:


learn.load('stage-2');


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix()


# In[ ]:


learn = cnn_learner(data, models.resnet34).load('stage-2')


# In[ ]:


learn.export()


# In[ ]:


path.ls()


# In[ ]:


from IPython.display import FileLinks
FileLinks('.')

