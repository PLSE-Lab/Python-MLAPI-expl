#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai.vision import *


# In[ ]:


path = Path('../input/geekhub-ds-2019-challenge')
labels = pd.read_csv('../input/geekhub-ds-2019-challenge/train_labels.csv').set_index('Id')
display(labels.transpose())


# ## 1. The Data

# In[ ]:


np.random.seed(42)
torch.manual_seed(42) 
# Get the data
data = (ImageList.from_folder(path/'train/train')  # Where to find the data? 
    .split_by_rand_pct(valid_pct=0.10)              # How to split in train/valid?
    .label_from_func(lambda o: labels.loc[int(o.name.split('.')[0])].values[0]) #How to label?
    .add_test(ImageList.from_folder(path/'test/test'))  # Add test data (no labels)
    .transform(tfms=get_transforms(), size=224)    # Data augmentation?
    .databunch(bs=16)                              # Databunch?
    .normalize(imagenet_stats))                    # Normalize?


# In[ ]:


data


# ## 2. Training

# In[ ]:


working_dir = Path('/kaggle/working')
learn = cnn_learner(data, models.resnet152, metrics=accuracy, model_dir=working_dir).to_fp16()


# In[ ]:


learn.lr_find();
learn.recorder.plot()


# ### 2.1 Fitting only the top layer

# In[ ]:


learn.fit_one_cycle(10, max_lr=1e-3)


# In[ ]:


learn.save('stage-1')


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.recorder.plot_lr(show_moms=True)


# ### 2.2 Fitting all the layers

# In[ ]:


learn.unfreeze()
learn.lr_find(stop_div=False, num_it=100)
learn.recorder.plot()


# In[ ]:


learn.load('stage-1')
learn.unfreeze()
learn.fit_one_cycle(4, max_lr=slice(1e-5,1e-4))


# In[ ]:


learn.to_fp32().save('stage-2')


# ## 3. Evaluation

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# In[ ]:


interp.plot_top_losses(16, figsize=(15,18))


# ## 4. Predictions

# ### 4.1 Predict on Test

# In[ ]:


preds = learn.get_preds(ds_type=DatasetType.Test)


# ### 4.2 Submission

# In[ ]:


ids_test = [int(o.parts[-1].split('.')[-2]) for o in data.test_ds.items]
pred_sub_test = pd.DataFrame({'Id':ids_test, 'Category': preds[0].argmax(axis=1)})
pred_sub_test['Category'] = pred_sub_test.Category.map({0: 'daisy', 1: 'dandelion', 2: 'rose', 3: 'sunflower', 4: 'tulip'})
# Save output
pred_sub_test.to_csv('csv_test.csv', index=False)

