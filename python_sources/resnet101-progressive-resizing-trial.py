#!/usr/bin/env python
# coding: utf-8

# #### Fork of older version of [this kernel](https://www.kaggle.com/khursani8/fast-ai-starter-resnet34)

# In[ ]:


from fastai.vision import *
from fastai.metrics import KappaScore, accuracy
from fastai.callbacks.tracker import ReduceLROnPlateauCallback, SaveModelCallback
from fastai.callbacks import MixedPrecision
import sys, os, gc
sys.path.insert(0, '../input/aptos2019-blindness-detection')


# In[ ]:


# copy pretrained weights for resnet34 to the folder fastai will search by default
Path('/tmp/.cache/torch/checkpoints/').mkdir(exist_ok=True, parents=True)


# In[ ]:


get_ipython().system("cp '../input/resnet101/resnet101.pth' '/tmp/.cache/torch/checkpoints/resnet101-5d3b4d8f.pth'")


# In[ ]:


PATH = Path('../input/aptos2019-blindness-detection')
df = pd.read_csv(PATH/'train.csv')


# In[ ]:


tfms = get_transforms(
    do_flip=True,
    flip_vert=True,
    max_rotate=360,
    max_warp=0,
    max_zoom=1.1,
    max_lighting=0.1,
    p_lighting=0.5
)


# In[ ]:


def return_train_data(size=56, bs=256):
    return ImageDataBunch.from_df(
        df=df,
        folder='train_images',
        path=PATH,
        valid_pct=0.2,
        ds_tfms=tfms,
        size=size,
        bs=bs,
        suffix=".png",
        resize_method=ResizeMethod.SQUISH,
        padding_mode='zeros'
    ).normalize(imagenet_stats)


# In[ ]:


# Small image data
train_data = return_train_data()


# ### 56*256

# In[ ]:


kappa = KappaScore()
kappa.weights = "quadratic"
# loss_func = LabelSmoothingCrossEntropy()
learn = cnn_learner(
    train_data,models.resnet101,metrics=[accuracy,kappa], model_dir='/kaggle', pretrained=True
                   ).mixup()
# learn = load_learner(path="../input/retinopathy-model", file="stage_1.pkl").to_fp32()

# learn.data = train_data


# In[ ]:


# Callbacks
reduce_lr = ReduceLROnPlateauCallback(learn=learn, monitor='kappa_score', factor=1e-3, patience=2, min_delta=0.0002)
save_mod = SaveModelCallback(learn, every='improvement', monitor='kappa_score', name='best')


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot(suggestion=True)
min_grad_lr = learn.recorder.min_grad_lr


# In[ ]:


learn.fit_one_cycle(5, min_grad_lr, callbacks=[
    reduce_lr, save_mod
])


# In[ ]:


learn.load("best");
learn.export(f"/kaggle/working/model_bs_{train_data.batch_size}.pkl")


# In[ ]:


del train_data
gc.collect(); gc.collect()
learn.load("best");


# ### 112*128

# In[ ]:


train_data = return_train_data(size=112, bs=128)
learn.data = train_data
learn.unfreeze()
learn.lr_find()


# In[ ]:


learn.recorder.plot(suggestion=True)
min_grad_lr = learn.recorder.min_grad_lr


# In[ ]:


learn.fit_one_cycle(5, min_grad_lr, callbacks=[reduce_lr, save_mod])


# In[ ]:


learn.load("best");
learn.export(f"/kaggle/working/model_bs_{train_data.batch_size}.pkl")


# In[ ]:


del train_data
gc.collect(); gc.collect()
learn.load("best");


# ### 224*64

# In[ ]:


train_data = return_train_data(size=224, bs=64)
learn.data = train_data
learn.freeze()
learn.lr_find()


# In[ ]:


learn.recorder.plot(suggestion=True)
min_grad_lr = learn.recorder.min_grad_lr


# In[ ]:


learn.fit_one_cycle(5, min_grad_lr, callbacks=[reduce_lr, save_mod])


# In[ ]:


learn.load("best");
learn.export(f"/kaggle/working/model_bs_{train_data.batch_size}.pkl")


# In[ ]:


del train_data
gc.collect(); gc.collect()
learn.load("best");


# ### 336*32

# In[ ]:


train_data = return_train_data(size=336, bs=32)
learn.data = train_data
##### UNFREEZING #######
learn.unfreeze()
learn.lr_find()


# In[ ]:


learn.recorder.plot(suggestion=True)
min_grad_lr = learn.recorder.min_grad_lr


# In[ ]:


learn.fit_one_cycle(5, min_grad_lr, callbacks=[reduce_lr, save_mod])


# In[ ]:


learn.load("best");
learn.export(f"/kaggle/working/model_bs_{train_data.batch_size}.pkl")


# In[ ]:


del train_data
gc.collect(); gc.collect()
learn.load("best");


# ### Make Predictions

# In[ ]:


sample_df = pd.read_csv(PATH/'sample_submission.csv')
sample_df.head()


# In[ ]:


learn.data.add_test(ImageList.from_df(sample_df,PATH,folder='test_images',suffix='.png'))


# In[ ]:


preds,y = learn.get_preds(DatasetType.Test)


# In[ ]:


sample_df.diagnosis = preds.argmax(1)
sample_df.head()


# In[ ]:


sample_df.to_csv('submission.csv',index=False)

