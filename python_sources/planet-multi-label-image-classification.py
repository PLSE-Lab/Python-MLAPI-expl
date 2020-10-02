#!/usr/bin/env python
# coding: utf-8

# # Planet: Multi-label classification 
# 
# This kernel will show how  to classify the multi-labled image data of planet with fastai v1.0.48+.
# 
# 
# ref : [fast-ai-v3-lesson-3-planet](https://www.kaggle.com/hortonhearsafoo/fast-ai-v3-lesson-3-planet)

# ## 1. Prepare Env

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# 
# ### To get the latest vertion of fastai (optional)

# In[ ]:


get_ipython().system('pip install git+https://github.com/fastai/fastai')


# In[ ]:


from fastai.vision import *


# ### check the detail of enviroment

# In[ ]:


#import fastai
#print(fastai.__version__)
import fastai.utils.collect_env; 
fastai.utils.collect_env.show_install(1)
#1.0.47 dev pass


# In[ ]:


path = Path('/kaggle/input/')
path.ls()


# In[ ]:


get_ipython().system('ls /kaggle/input')


# ### count files

# In[ ]:


get_ipython().system('ls /kaggle/input/train-jpg -l |grep "^-"|wc -l')
get_ipython().system('ls /kaggle/input/test-jpg-v2 -l |grep "^-"|wc -l')


# ## 2. Prepare Training Data 

# In[ ]:


df = pd.read_csv(path/'train_v2.csv')
df.head()


# ### data augment

# In[ ]:


tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0., max_rotate=15.)


# ### read training data

# In[ ]:


np.random.seed(42)
src = (ImageList.from_csv(path, 'train_v2.csv', folder='train-jpg', suffix='.jpg')#.use_partial_data(0.01)
       .split_by_rand_pct(0.2)
       .label_from_df(label_delim=' '))
data = (src.transform(tfms, size=128)
        .databunch(num_workers=0).normalize(imagenet_stats))


# ### check the data object

# In[ ]:


data.show_batch(rows=3, figsize=(12,9))


# ## 3. Create Network

# In[ ]:


arch = models.densenet121


# In[ ]:


acc_02 = partial(accuracy_thresh, thresh=0.19)
f_score = partial(fbeta, thresh=0.19)
learn = cnn_learner(data, arch, metrics=[acc_02, f_score], model_dir='/kaggle/working/')


# ## 4. Train Network

# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr = 0.01


# In[ ]:


learn.fit_one_cycle(5, slice(lr))


# In[ ]:


learn.save('stage-1-rn50')


# ### fine tune the network

# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5, slice(1e-5, lr/5))


# In[ ]:


learn.save('stage-2-rn50')


# In[ ]:


data = (src.transform(tfms, size=256)
        .databunch(num_workers=0).normalize(imagenet_stats))

learn.data = data
data.train_ds[0][0].shape


# In[ ]:


learn.freeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr=1e-2/2


# In[ ]:


learn.fit_one_cycle(5, slice(lr))


# In[ ]:


learn.save('stage-1-256-rn50')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(5, slice(1e-5, lr/5))


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.save('stage-2-256-rn50')


# In[ ]:


learn.export(fname='/kaggle/working/export.pkl',destroy=True)


# ## 5. Predict

# In[ ]:


#test = ImageList.(from_folder(path/'test-jpg').add(ImageList.from_folder(path/'test-jpg-additional')) #two folder
test = (ImageList.from_folder(path/'test-jpg-v2'))#.use_partial_data(0.01))
len(test)


# In[ ]:


learn_test = load_learner('/kaggle/working/', test=test, num_workers=0, bs=1)
preds, _ = learn_test.get_preds(ds_type=DatasetType.Test)
preds_tta, _ = learn_test.TTA(ds_type=DatasetType.Test)
#preds = np.mean(np.exp(log_preds))


# In[ ]:


print(preds[:5])


# ## 6. Submission

# In[ ]:


thresh = 0.15
labelled_preds = [' '.join([learn_test.data.classes[i] for i,p in enumerate(pred) if p > thresh]) for pred in preds]
fnames = [f.name[:-4] for f in learn_test.data.test_ds.x.items]
df = pd.DataFrame({'image_name':fnames, 'tags':labelled_preds}, columns=['image_name', 'tags'])
df.to_csv('submission_015.csv', index=False)


# In[ ]:


thresh = 0.18
labelled_preds = [' '.join([learn_test.data.classes[i] for i,p in enumerate(pred) if p > thresh]) for pred in preds]
fnames = [f.name[:-4] for f in learn_test.data.test_ds.x.items]
df = pd.DataFrame({'image_name':fnames, 'tags':labelled_preds}, columns=['image_name', 'tags'])
df.to_csv('submission_018.csv', index=False)


# In[ ]:


thresh = 0.19
labelled_preds = [' '.join([learn_test.data.classes[i] for i,p in enumerate(pred) if p > thresh]) for pred in preds]
fnames = [f.name[:-4] for f in learn_test.data.test_ds.x.items]
df = pd.DataFrame({'image_name':fnames, 'tags':labelled_preds}, columns=['image_name', 'tags'])
df.to_csv('submission_019.csv', index=False)


# In[ ]:


thresh = 0.20
labelled_preds = [' '.join([learn_test.data.classes[i] for i,p in enumerate(pred) if p > thresh]) for pred in preds]
fnames = [f.name[:-4] for f in learn_test.data.test_ds.x.items]
df = pd.DataFrame({'image_name':fnames, 'tags':labelled_preds}, columns=['image_name', 'tags'])
df.to_csv('submission_020.csv', index=False)


# In[ ]:


thresh = 0.21
labelled_preds = [' '.join([learn_test.data.classes[i] for i,p in enumerate(pred) if p > thresh]) for pred in preds]
fnames = [f.name[:-4] for f in learn_test.data.test_ds.x.items]
df = pd.DataFrame({'image_name':fnames, 'tags':labelled_preds}, columns=['image_name', 'tags'])
df.to_csv('submission_021.csv', index=False)


# In[ ]:





# In[ ]:




