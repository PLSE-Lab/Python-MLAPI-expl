#!/usr/bin/env python
# coding: utf-8

# # Image Classification on Simpson's data using fastai
# 
# Using fastai is so easy, and building baseline solutions using fastai is "fast". So, in this notebook I present the baseline model based on fastai lesson notebooks.
# 
# Steps I followed:
# * Create dabunch for 128x128 resolution.
# * Train with frozen resnet50 model.
# * Train with unfrozen resnet50 model..
# * Create new databunch of 256x256 resolution.
# * Train with frozen resnet50 model. (with weights from previous stage)
# * Train with unfrozen resnet50 model..
# 
# Then, it's a matter of tweaking the architectures.

# In[ ]:


from fastai.vision import *
from fastai.metrics import accuracy, fbeta


# In[ ]:


path = Path("../input")


# ## Creating dataset of 128x128

# In[ ]:


src = (ImageList.from_folder(path/'train')
                .split_by_rand_pct()
                .label_from_folder()
                .add_test(Path('../input/test/test').ls()))


# In[ ]:


data = (src.transform(get_transforms(), size=128)
           .databunch(path=Path("../"))
           .normalize(imagenet_stats))


# In[ ]:


data.show_batch(3)


# In[ ]:


data.batch_stats()


# ## Creating learner with resnet50

# In[ ]:


arch = models.resnet34


# In[ ]:


learn = cnn_learner(data, arch, metrics=accuracy).to_fp16()


# ## Finding optimal lr and training with frozen weights

# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr = 5e-2


# In[ ]:


learn.fit_one_cycle(6, max_lr=slice(lr))


# In[ ]:


learn.save('stage-1')


# In[ ]:


test_probs, _ = learn.get_preds(ds_type=DatasetType.Test)
test_preds = [data.classes[pred] for pred in np.argmax(test_probs.numpy(), axis=-1)]


# In[ ]:


fnames = [f.name[:-4] for f in learn.data.test_ds.items]
df = pd.DataFrame({'id':fnames, 'predicted_class':test_preds}, columns=['id', 'predicted_class'])
df['id'] = df['id'].astype(str) + '.jpg'
df.to_csv('submission-1.csv', index=False)


# <a href='submission-1.csv'>Download Submission 1</a>

# ## Finding optimal lr and training with un-frozen weights

# In[ ]:


learn.load('stage-1');


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5, max_lr=slice(5e-6, lr/10))


# In[ ]:


learn.save('stage-2')


# In[ ]:


test_probs, _ = learn.get_preds(ds_type=DatasetType.Test)
test_preds = [data.classes[pred] for pred in np.argmax(test_probs.numpy(), axis=-1)]


# In[ ]:


fnames = [f.name[:-4] for f in learn.data.test_ds.items]
df = pd.DataFrame({'id':fnames, 'predicted_class':test_preds}, columns=['id', 'predicted_class'])
df['id'] = df['id'].astype(str) + '.jpg'
df.to_csv('submission-2.csv', index=False)


# <a href='submission-2.csv'>Download Submission 2</a>

# In[ ]:


learn.load('stage-2');


# ## Creating 256x256 dataset

# In[ ]:


data = (src.transform(get_transforms(), size=256)
           .databunch(path=Path("../"))
           .normalize(imagenet_stats))


# In[ ]:


learn.data = data
learn = learn.to_fp16()


# ## Finding optimal lr and training with frozen weights

# In[ ]:


learn.freeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr = 5e-3


# In[ ]:


learn.fit_one_cycle(5, max_lr=slice(lr))


# In[ ]:


learn.save('stage-256-1')


# In[ ]:


test_probs, _ = learn.get_preds(ds_type=DatasetType.Test)
test_preds = [data.classes[pred] for pred in np.argmax(test_probs.numpy(), axis=-1)]


# In[ ]:


fnames = [f.name[:-4] for f in learn.data.test_ds.items]
df = pd.DataFrame({'id':fnames, 'predicted_class':test_preds}, columns=['id', 'predicted_class'])
df['id'] = df['id'].astype(str) + '.jpg'
df.to_csv('submission-3.csv', index=False)


# <a href='submission-3.csv'>Download Submission 3</a>

# ## Finding optimal lr and training with un-frozen weights

# In[ ]:


learn.load('stage-256-1');


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5, max_lr=slice(1e-5, 1e-4))


# In[ ]:


learn.save('stage-256-2')


# In[ ]:


test_probs, _ = learn.get_preds(ds_type=DatasetType.Test)
test_preds = [data.classes[pred] for pred in np.argmax(test_probs.numpy(), axis=-1)]


# In[ ]:


fnames = [f.name[:-4] for f in learn.data.test_ds.items]
df = pd.DataFrame({'id':fnames, 'predicted_class':test_preds}, columns=['id', 'predicted_class'])
df['id'] = df['id'].astype(str) + '.jpg'
df.to_csv('submission-4.csv', index=False)


# <a href='submission-4.csv'>Download Submission 4</a>

# ## Analyse results and optimize training

# In[ ]:


preds,y,losses = learn.get_preds(with_loss=True)
interp = ClassificationInterpretation(learn.to_fp32(), preds, y, losses)


# In[ ]:


interp.plot_top_losses(6, figsize=(15, 15))


# In[ ]:


interp.most_confused(min_val=2)


# ## Applying TTA

# In[ ]:


test_probs, _ = learn.TTA(ds_type=DatasetType.Test)
test_preds = [data.classes[pred] for pred in np.argmax(test_probs.numpy(), axis=-1)]


# In[ ]:


fnames = [f.name[:-4] for f in learn.data.test_ds.items]
df = pd.DataFrame({'id':fnames, 'predicted_class':test_preds}, columns=['id', 'predicted_class'])
df['id'] = df['id'].astype(str) + '.jpg'
df.to_csv('submission-5.csv', index=False)


# <a href='submission-5.csv'>Download Submission 5</a>

# In[ ]:


interp.plot_confusion_matrix(figsize=(20,20), normalize=True, )


# In[ ]:




