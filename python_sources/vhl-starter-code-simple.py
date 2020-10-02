#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Allows you to save your models somewhere without
# copying all the data over.
# Trust me on this.

get_ipython().system('mkdir input')
get_ipython().system('cp /kaggle/input/train_labels.csv input')
get_ipython().system('cp /kaggle/input/sample_submission.csv input')
get_ipython().system('ln -s /kaggle/input/train/ input/train')
get_ipython().system('ln -s /kaggle/input/test/ input/test')


# In[ ]:


from fastai.vision import *


# In[ ]:


db = (ImageItemList.from_csv(csv_name='train_labels.csv', path='input', folder='train', suffix='.tif')
        .random_split_by_pct()
        .label_from_df()
        .transform(get_transforms(flip_vert=True), size=64)
        .add_test_folder('test')
        .databunch(bs=32)
        .normalize(imagenet_stats))


# In[ ]:


db.show_batch(4, figsize=(12,12), ds_type=DatasetType.Test)


# In[ ]:


learn = create_cnn(db, models.resnet34, metrics=[error_rate])


# In[ ]:


lr_find(learn)
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(1)


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix()


# In[ ]:


interp.plot_top_losses(4)


# In[ ]:


probs, _ = learn.get_preds(DatasetType.Test)


# In[ ]:


preds = probs.argmax(1)


# In[ ]:


test_df = pd.read_csv('./input/sample_submission.csv')
test_df['id'] = [i.stem for i in db.test_ds.items]
test_df['label'] = preds


# In[ ]:


test_df.to_csv('submission.csv', index=False)


# In[ ]:




