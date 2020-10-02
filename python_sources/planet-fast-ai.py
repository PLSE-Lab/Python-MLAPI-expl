#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
start_time = time.time()

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *
from fastai import *


# In[ ]:


from pathlib import Path
path = Path('../input/')


# In[ ]:


df = pd.read_csv(path/'train_v2.csv')
df.head()


# In[ ]:


tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)


# In[ ]:


np.random.seed(42)
src = (ImageItemList.from_csv(path, 'train_v2.csv', folder='train-jpg', suffix='.jpg')
       .random_split_by_pct(0.2)
       .label_from_df(label_delim=' '))


# In[ ]:


data = (src
        .transform(tfms, size=128)
        .databunch()
        .normalize(imagenet_stats)
       )


# In[ ]:


data.show_batch(rows = 3)


# In[ ]:


arch = models.resnet50


# In[ ]:


acc_02 = partial(accuracy_thresh, thresh=0.2)
f_score = partial(fbeta, thresh=0.2)
learn = create_cnn(data, arch, metrics=[acc_02, f_score], path='../working/')


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(3, slice(2.29E-02,0.01))


# In[ ]:


learn.save('stage-1-rn50')


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(3, slice(7.59E-05))


# In[ ]:


learn.save('stage-2-rn50')
learn.export()


# > We can do transfer learning by changing the size of the data and learning using same **learner**. But Kaggle kernel gets Runtime error as of no sufficient memory

# In[ ]:


test = ImageItemList.from_folder(path/'test-jpg-v2').add(ImageImageList.from_folder(path/'test-jpg-additional'))
len(test)


# In[ ]:


learn = load_learner('../working/', test=test)
preds, _ = learn.get_preds(ds_type=DatasetType.Test)


# In[ ]:


thresh = 0.2
labelled_preds = [' '.join([learn.data.classes[i] for i,p in enumerate(pred) if p > thresh]) for pred in preds]


# In[ ]:


submission = pd.read_csv(path/'sample_submission_v2.csv')
submission['tags'] = labelled_preds
submission.to_csv('fastai_resnet50.csv')


# In[ ]:


print('Kernel Runtime: {0} minutes '.format((time.time() - start_time)/60.0))

