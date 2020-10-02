#!/usr/bin/env python
# coding: utf-8

# # Car Classification
# 
# We'll first set some default settings and import our required tools.
# 

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from fastai import *
from fastai.vision import *

bs = 64


# # Data
# 
# In order to train our Car Classifier DL model, we'll need some relevant data. Since we are using a CNN, we need a bunch of labeled car images. I followed the steps [here](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson2-download.ipynb/) to get some car images from Google Images. I then loaded the URLs of the images into Kaggle as a custom dataset, and from there into my file system. 

# In[ ]:


cars = ['Accord','Altima','Charger', 'Corolla','Tesla_s','Civic']


# In[ ]:


path = Path('../data/car')

for car in cars:
    (path/car).mkdir(parents=True, exist_ok=True)

path.ls()


# In[ ]:


for car in cars:
    download_images(Path('../input')/f'{car.lower()}', path/car, max_pics=500)


# In[ ]:


for car in cars:
    print(car)
    verify_images(path/car,delete =True,max_size=500)


# In[ ]:


np.random.seed(123)
data = ImageDataBunch.from_folder(path,train=".",valid_pct=0.2,ds_tfms=get_transforms(),
                                  size=224,bs=bs,num_workers=0).normalize(imagenet_stats)


# In[ ]:


print((len(data.classes),data.c))
data.show_batch(rows=3, figsize=(7,6))


# In[ ]:


learn = create_cnn(data, models.resnet50, metrics=error_rate)


# In[ ]:


learn.fit_one_cycle(8)
learn.save('stage-1')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.plot_confusion_matrix()


# In[ ]:


interp.most_confused(min_val=2)


# In[ ]:


learn.unfreeze()
for i in range(8):
    learn.fit_one_cycle(2)
    learn.save('stage-'+str(i+2))


# In[ ]:


#learn.load('stage-1')


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(4, max_lr=slice(6e-5,4e-4))


# In[ ]:


learn.save('lr-optimized')


# In[ ]:


interp=ClassificationInterpretation.from_learner(learn)
interp.most_confused(min_val=2)
interp.plot_confusion_matrix()


# # Data Cleaning

# In[ ]:


from fastai.widgets import *


# First we need to get the file paths from our top_losses. We can do this with .from_toplosses. We then feed the top losses indexes and corresponding dataset to ImageCleaner.
# 
# Notice that the widget will not delete images directly from disk but it will create a new csv file cleaned.csv from where you can create a new ImageDataBunch with the corrected labels to continue training your model.

# In[ ]:


ds, idxs = DatasetFormatter().from_toplosses(learn, ds_type=DatasetType.Valid)
ImageCleaner(ds, idxs, path)


# In[ ]:


#After cleaning data, run this cell
np.random.seed(42)
csvpath=Path(path/'cleaned.csv')
data = ImageDataBunch.from_csv(".", folder=".", valid_pct=0.2, csv_labels=csvpath,
    ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)


# In[ ]:


#After cleaning data, run this cell
np.random.seed(42)
csvpath=Path(path/'cleaned.csv')
data = ImageDataBunch.from_csv(".", folder=".", valid_pct=0.2, csv_labels=csvpath,
    ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)


# In[ ]:




