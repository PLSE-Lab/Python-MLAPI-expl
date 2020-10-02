#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastai import *
from fastai.vision import *


# In[2]:


path= "../input/train/"


# In[3]:


df = pd.read_csv('../input/train/train.csv')
df.head()


# In[4]:


src = ImageList.from_csv(path, 'train.csv', folder='images').split_by_rand_pct(0.2, seed = 2)


# In[5]:


tfms = get_transforms(max_rotate=20, max_lighting=0.4, max_warp=0.4,
                      p_affine=1., p_lighting=1.)


# In[6]:


def get_data(size, bs, padding_mode='reflection'):
    return (src.label_from_df()
               .transform(tfms, size=size, padding_mode=padding_mode)
               .databunch(bs=bs, num_workers=0).normalize(imagenet_stats))


# In[7]:


data = get_data(224, 16, 'zeros')


# In[8]:


def _plot(i,j,ax):
    x,y = data.train_ds[3]
    x.show(ax, y=y)

plot_multi(_plot, 3, 3, figsize=(8,8))


# In[9]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate, bn_final=True, model_dir='/tmp/models')


# In[12]:


learn.fit_one_cycle(3, slice(1e-2), pct_start=0.8)


# In[13]:


learn.unfreeze()


# In[14]:


learn.lr_find()


# In[15]:


learn.recorder.plot(suggestion=True)


# In[16]:


learn.fit_one_cycle(2, max_lr=slice(2e-5,1e-4))


# In[17]:


data = get_data(352,8)
learn.data = data


# In[18]:


learn.fit_one_cycle(2, max_lr=slice(2e-5,1e-4))


# In[19]:


learn.save('/kaggle/working/352')


# In[20]:


data = get_data(352,16)


# In[21]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate, bn_final=True, model_dir='/tmp/models').load('/kaggle/working/352')


# In[23]:


idx=0
x,y = data.valid_ds[idx]
x.show()
# data.valid_ds.y[idx]


# In[24]:


k = tensor([
    [0.  ,-5/3,1],
    [-5/3,-5/3,1],
    [1.  ,1   ,1],
]).expand(1,3,3,3)/6


# In[25]:


k.shape


# In[26]:


from fastai.callbacks.hooks import *


# In[27]:


k.shape


# In[28]:


t = data.valid_ds[0][0].data; t.shape


# In[29]:


t[None].shape


# In[30]:


edge = F.conv2d(t[None], k)


# In[31]:


show_image(edge[0], figsize=(5,5));


# In[33]:


m = learn.model.eval();


# In[34]:


xb,_ = data.one_item(x)
xb_im = Image(data.denorm(xb)[0])
xb = xb.cuda()


# In[35]:


def hooked_backward(cat=y):
    with hook_output(m[0]) as hook_a: 
        with hook_output(m[0], grad=True) as hook_g:
            preds = m(xb)
            preds[0,int(cat)].backward()
    return hook_a,hook_g


# In[36]:


hook_a,hook_g = hooked_backward()


# In[37]:


acts  = hook_a.stored[0].cpu()
acts.shape


# In[38]:


avg_acts = acts.mean(0)
avg_acts.shape


# In[39]:


def show_heatmap(hm):
    _,ax = plt.subplots()
    xb_im.show(ax)
    ax.imshow(hm, alpha=0.6, extent=(0,352,352,0),
              interpolation='bilinear', cmap='magma');


# In[40]:


show_heatmap(avg_acts)


# In[10]:


learn.model


# In[11]:


learn.summary()


# In[ ]:




