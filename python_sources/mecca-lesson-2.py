#!/usr/bin/env python
# coding: utf-8

# Remember to turn on GPU and Internet in the settings tabs.

# In[ ]:


# settings
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# load libraries
from fastai import *
from fastai.vision import *
import pandas as pd
import cv2


# ### Load data
# If you download data from internet : Remember to turn on the Internet settings

# In[ ]:


size = 16 # ssize of input images
bs = 64 # batch size
tfms = get_transforms()


# How to load data the right way : [Link](https://docs.fast.ai/data_block.html)

# In[ ]:


get_ipython().system(' ls ../input/fashionmnist/data/data')


# In[ ]:


path = Path('../input/fashionmnist/data/data')


# In[ ]:


cv2.imread(str((path/'train/Coat').ls()[0])).shape


# In[ ]:


# Load data to DataBunch
data = ImageDataBunch.from_folder(path,train='train',test='test',
                                 ds_tfms=tfms, size=size, bs=bs,valid_pct=.2).normalize(imagenet_stats)
data


# In[ ]:


data.show_batch(rows=3)


# ### Create your learner

# In[ ]:


model = models.resnet18


# In[ ]:


data.path = '/tmp/.torch/models'


# In[ ]:


learn = cnn_learner(data, model, metrics=accuracy,callback_fns=[ShowGraph])


# In[ ]:


learn.summary()


# ## Training begin

# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.save("stage-1")


# In[ ]:


lr = 2e-2


# In[ ]:


learn.fit_one_cycle(9,slice(lr))


# In[ ]:


learn.fit_one_cycle(4,slice(lr))


# In[ ]:


learn.unfreeze()


# In[ ]:


lr = lr /100
learn.fit_one_cycle(4,slice(lr))


# In[ ]:


accuracy(*learn.TTA())


# In[ ]:


learn.save('stage-2')


# ### Training stage 2

# In[ ]:


learn.load('stage-2')
pass


# In[ ]:


size = 28


# In[ ]:


# Load data to DataBunch
data = ImageDataBunch.from_folder(path,train='train',test='test',
                                 ds_tfms=tfms, size=size, bs=bs,valid_pct=.2).normalize(imagenet_stats)
data


# In[ ]:


learn.data = data


# In[ ]:


learn.freeze()
learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr = 3e-4


# In[ ]:


learn.fit_one_cycle(5,slice(lr))


# In[ ]:


learn.fit(6)


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(5,slice(1e-4))


# In[ ]:


accuracy(*learn.TTA())


# In[ ]:


learn.save('stage-3')


# In[ ]:


get_ipython().system(' wget https://www.uni-regensburg.de/Fakultaeten/phil_Fak_II/Psychologie/Psy_II/beautycheck/english/prototypen/w_sexy_gr.jpg')


# In[ ]:


from PIL import Image
import cv2


# In[ ]:


from fastai.vision import Image,pil2tensor


# In[ ]:


def array2tensor(x):
    """ Return an tensor image from cv2 array """
    x = cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
    return Image(pil2tensor(x,np.float32).div_(255))


# In[ ]:


get_ipython().system('wget https://i.pinimg.com/originals/c0/41/8f/c0418f5967b642a1f01864409fb2f86a.jpg')


# In[ ]:


get_ipython().system(' ls')


# In[ ]:


from PIL import Image as I


# In[ ]:


img = cv2.imread('c0418f5967b642a1f01864409fb2f86a.jpg')


# In[ ]:


I.open('c0418f5967b642a1f01864409fb2f86a.jpg')


# In[ ]:


img = array2tensor(img)


# In[ ]:


learn.predict(img)


# # Interpretation

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


learn.export('')


# In[ ]:


learn =  load_model_from_export('')


# In[ ]:


learn.predict('')


# In[ ]:


interp.most_confused(min_val=2)


# # DATA LOADING

# ### Planet

# In[ ]:


planet = untar_data(URLs.PLANET_TINY)
planet_tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)


# In[ ]:


pd.read_csv(planet/"labels.csv").head()


# In[ ]:


data = ImageDataBunch.from_csv(planet, folder='train',csv_labels='labels.csv', size=128, suffix='.jpg', label_delim = ' ', ds_tfms=planet_tfms)


# In[ ]:


data.show_batch(rows=2, figsize=(9,7))


# ### Download from google

# In[ ]:


# folders = ['tesla','lambo','audi']
# files = ['urls_tesla.csv','urls_lambo.csv','urls_audi.csv']


# In[ ]:


# urls = """
# https://st.motortrend.com/uploads/sites/10/2017/09/2018-audi-r8-coupe-angular-front.png
# https://www.cstatic-images.com/car-pictures/xl/usc90aus061a021001.png
# https://st.motortrend.com/uploads/sites/10/2015/11/2015-audi-rs7-hatchback-angular-front.png
# https://upload.wikimedia.org/wikipedia/commons/d/d2/2018_Audi_A7_S_Line_40_TDi_S-A_2.0.jpg
# """


# In[ ]:


# pd.DataFrame(urls.strip().split('\n')).to_csv(path/'urls_audi.csv',index=False)


# In[ ]:


# for file,folder in zip(files,folders):
#     path = Path('data/cars')
#     dest = path/folder
#     dest.mkdir(parents=True, exist_ok=True)
#     download_images(path/file, dest, max_pics=200)


# In[ ]:


#download_images(path/file, dest, max_pics=200)

