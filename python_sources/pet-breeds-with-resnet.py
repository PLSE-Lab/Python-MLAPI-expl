#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.utils.collect_env import show_install
show_install()


# In[ ]:


from fastai.callbacks.tracker import *


# In[ ]:


import matplotlib.pyplot as plt
from fastai.vision import *
import random 
import gc


# In[ ]:


def reset_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
#     tf.set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
reset_seed()


# In[ ]:


PATH = "../input/"
# TMP_PATH = "/tmp/tmp"
# MODEL_PATH = "/tmp/model/"


# In[ ]:


path = Path(PATH)
path_anno = path/'annotations/annotations'
path_img = path/'images/images'


# In[ ]:


fnames = get_image_files(path_img)


# In[ ]:


def get_data(bs,size,seed=42):
    reset_seed(seed)
    pat = r'/([^/]+)_\d+.jpg$'
    return ImageDataBunch.from_name_re(path_img,fnames,pat,valid_pct=0.15,ds_tfms = get_transforms(),
                                      size=size,bs=bs).normalize(imagenet_stats)


# In[ ]:


def create_learner(data,is_fp16=False,seed=42):
    reset_seed(seed)
    learn = cnn_learner(data,models.resnet50,metrics=accuracy,path = os.getcwd(),
                       callback_fns=[partial(SaveModelCallback, monitor='val_loss',mode='min',every='improvement')])
    if is_fp16:
        learn = learn.to_fp16()
    return learn

def create_learner_lr_find(data,is_fp16=False,seed=42):
    reset_seed(seed)
    learn = cnn_learner(data,models.resnet50,metrics=accuracy,path = os.getcwd())
    if is_fp16:
        learn = learn.to_fp16()
    return learn


# # Stage 1 size 224

# In[ ]:


# del data
# del learn
# gc.collect()


# In[ ]:


data= get_data(32,224)


# In[ ]:


learn = create_learner_lr_find(data,True)
learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn = create_learner(data,True)


# In[ ]:


reset_seed()
learn.freeze()
learn.fit_one_cycle(13,max_lr=2.8e-03)
learn.recorder.plot_losses()


# In[ ]:


learn.save('224-stage1')


# In[ ]:


# learn.load('224-bs32-13epochs-stage1');


# In[ ]:


from sklearn.metrics import accuracy_score
y_pred = learn.get_preds(with_loss=True)
print(accuracy_score(learn.data.valid_ds.y.items,np.argmax(to_np(y_pred[0]),axis=1)))
print(y_pred[2].mean())


# # Stage 2 size 224

# In[ ]:


# del data
# del learn
# gc.collect()


# In[ ]:


data= get_data(32,224)

learn = create_learner_lr_find(data,True)
# learn.load('224-stage1');


# In[ ]:


learn.load('224-stage1');
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn = create_learner(data,True)
learn.load('224-stage1');
learn.unfreeze()
learn.fit_one_cycle(3,max_lr=slice(1e-06,4e-06))
learn.recorder.plot_losses()


# In[ ]:


learn.save('224-stage2')


# # Stage 1 size 300

# In[ ]:


del data
del learn
gc.collect()


# In[ ]:


data= get_data(20,300)


# In[ ]:


learn = create_learner_lr_find(data,True)
learn.load('224-stage2');


# In[ ]:


learn.freeze()
learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.freeze()
learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn = create_learner(data,True)
learn.load('224-stage2');


# In[ ]:


learn = create_learner(data,True)
learn.load('224-stage2');
learn.freeze()
learn.fit_one_cycle(3,max_lr=1e-05)
learn.recorder.plot_losses()


# In[ ]:


# learn = create_learner(data,True)
# learn.load('224-stage2');
# learn.freeze()
# learn.fit_one_cycle(3,max_lr=2e-05)
# learn.recorder.plot_losses()


# In[ ]:


learn.save('300-stage1')


# # Stage 2

# In[ ]:


del data
del learn
gc.collect()


# In[ ]:


data= get_data(15,300)

learn = create_learner_lr_find(data,True)
learn.load('300-stage1');


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn = create_learner(data,True)
learn.load('300-stage1');
learn.unfreeze()
learn.fit_one_cycle(6,max_lr=slice(1e-06,1.2e-04))
learn.recorder.plot_losses()


# In[ ]:


# learn = create_learner(data,True)
# learn.load('300-stage1');
# learn.unfreeze()
# learn.fit_one_cycle(4,max_lr=slice(1e-06,1.2e-04))
# learn.recorder.plot_losses()


# In[ ]:


learn.save('300-stage2');


# In[ ]:


learn.data.c

