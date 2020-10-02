#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from fastai.vision import *


# In[3]:


path = untar_data(URLs.BIWI_HEAD_POSE)


# In[4]:


cal = np.genfromtxt(path/'01'/'rgb.cal', skip_footer=6); cal


# In[5]:


fname = '09/frame_00667_rgb.jpg'


# In[7]:


def img2txt_name(f): return path/f'{str(f)[:-7]}pose.txt'


# In[8]:


img = open_image(path/fname)
img.show()


# In[9]:


ctr = np.genfromtxt(img2txt_name(fname), skip_header=3); ctr


# In[10]:


def convert_biwi(coords):
    c1 = coords[0] * cal[0][0]/coords[2] + cal[0][2]
    c2 = coords[1] * cal[1][1]/coords[2] + cal[1][2]
    return tensor([c2,c1])

def get_ctr(f):
    ctr = np.genfromtxt(img2txt_name(f), skip_header=3)
    return convert_biwi(ctr)

def get_ip(img,pts): return ImagePoints(FlowField(img.size, pts), scale=True)


# In[11]:


get_ctr(fname)


# In[12]:


ctr = get_ctr(fname)
img.show(y=get_ip(img, ctr), figsize=(6, 6))


# In[13]:


data = (PointsItemList.from_folder(path)
        .split_by_valid_func(lambda o: o.parent.name=='13')
        .label_from_func(get_ctr)
        .transform(get_transforms(), tfm_y=True, size=(120,160))
        .databunch().normalize(imagenet_stats)
       )


# In[14]:


data.show_batch(3, figsize=(9,6))


# In[15]:


learn = cnn_learner(data, models.resnet34)


# In[16]:


learn.lr_find()
learn.recorder.plot()


# In[17]:


lr = 2e-2


# In[19]:


learn.fit_one_cycle(3, slice(lr))


# In[20]:


learn.save('/kaggle/working/stage-1')


# In[21]:


learn.show_results()


# In[22]:


tfms = get_transforms(max_rotate=20, max_zoom=1.5, max_lighting=0.5, max_warp=0.4, p_affine=1., p_lighting=1.)

data = (PointsItemList.from_folder(path)
        .split_by_valid_func(lambda o: o.parent.name=='13')
        .label_from_func(get_ctr)
        .transform(tfms, tfm_y=True, size=(120,160))
        .databunch().normalize(imagenet_stats)
       )


# In[23]:


def _plot(i,j,ax):
    x,y = data.train_ds[0]
    x.show(ax, y=y)

plot_multi(_plot, 3, 3, figsize=(8,6))


# In[ ]:




