#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai import *
from fastai.vision import *


# In[ ]:


path=Path('/kaggle/input/plant-pathology-2020-fgvc7')
path.ls()


# In[ ]:


df=pd.read_csv(path/'train.csv')
df.head()
test1 = pd.read_csv(path/'test.csv')
cols = df.columns.tolist()[1:];cols


# In[ ]:


tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)


# In[ ]:


test = (ImageList.from_df(test1,path,folder='images',suffix='.jpg',cols='image_id'))


# In[ ]:


np.random.seed(42)
src=(ImageList.from_csv(path,'train.csv',folder='images',suffix='.jpg')
    .split_by_rand_pct(0.2)
    .label_from_df(cols=cols,label_cls = MultiCategoryList))


# In[ ]:


data = (src.transform(tfms, size=229)
        .add_test(test)
        .databunch().normalize(imagenet_stats))


# In[ ]:


data.save('/kaggle/working/plant_data.pkl')


# In[ ]:


data.show_batch(rows=3, figsize=(12,9))


# In[ ]:


len(data.train_ds),len(data.valid_ds),len(data.test_ds)


# In[ ]:


arch = models.resnet50


# In[ ]:


acc_02 = partial(accuracy_thresh, thresh=0.2)
learn = cnn_learner(data, arch, metrics=acc_02,model_dir='/kaggle/working')


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


lr=1e-2
learn.fit_one_cycle(5,slice(lr))


# In[ ]:


learn.save('stage_1')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5, slice(1e-5, lr/10))


# In[ ]:


learn.save('stage-2')


# In[ ]:


data = (src.transform(tfms, size=299)
        .databunch().normalize(imagenet_stats))

learn.data = data
data.train_ds[0][0].shape


# In[ ]:


data.show_batch(rows=3, figsize=(12,9))


# In[ ]:


learn.freeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr=1e-3
learn.fit_one_cycle(5, slice(lr))
learn.save('stage-2')


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(5, slice(lr))


# In[ ]:


learn.save('plant1')


# In[ ]:


learn.export('/kaggle/working/plant.pkl')


# In[ ]:


preds = learn.get_preds(DatasetType.Test)


# In[ ]:


test = pd.read_csv(path/'test.csv')
test_id = test['image_id'].values
submission = pd.DataFrame({'image_id': test_id})
submission = pd.concat([submission, pd.DataFrame(preds[0].numpy() , columns =cols)], axis=1)

submission.to_csv('submission_plant12.csv', index=False)
submission.head(10)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




