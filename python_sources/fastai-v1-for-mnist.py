#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai.vision import *
from fastai.callbacks import *
from tqdm import tqdm_notebook
from cv2 import cv2 as cv


# In[ ]:


cv.__version__


# # 1 Data

# In[ ]:


path = Path('../input')
path.ls()


# In[ ]:


train_df = pd.read_csv(path/'train.csv')
test_df = pd.read_csv(path/'test.csv')
sub = pd.read_csv(path/'sample_submission.csv')


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


sub.head()


# In[ ]:


get_ipython().system('ls /kaggle/working')


# In[ ]:


working_dir = Path('/kaggle/working/mnist_data')
train_dir = (working_dir/'train')
test_dir = (working_dir/'test')


# In[ ]:


working_dir.mkdir(exist_ok=True)
train_dir.mkdir(exist_ok=True)
test_dir.mkdir(exist_ok=True)
working_dir.ls()


# In[ ]:


labels = train_df['label'].unique()
labels


# In[ ]:


for i in tqdm_notebook(labels):
    label_i_path = train_dir/f'{i}'
    label_i_path.mkdir(exist_ok=True)


# In[ ]:


def save_array2image(save_dir, index, array):
    plt.imsave(save_dir/f'{index}.jpg', array)


# In[ ]:


len_train = len(train_df)
for i in tqdm_notebook(range(len_train)):
    row = train_df.iloc[i]
    label = row['label'] 
    image_array = row[1:]
    image_array = np.reshape(np.array(image_array), (28, 28))
    save_dir = train_dir/str(label)
    save_array2image(save_dir, f'train_{i}', image_array)


# In[ ]:


get_ipython().system('ls /kaggle/working/mnist_data/train')


# In[ ]:


len_test = len(test_df)
for i in tqdm_notebook(range(len_test)):
    image_array = test_df.iloc[i]
    image_array = np.reshape(np.array(image_array), (28, 28))
    save_dir = test_dir
    save_array2image(save_dir, f'test_{i}', image_array)


# In[ ]:


get_ipython().system('ls /kaggle/working/mnist_data/test')


# In[ ]:


tfms = get_transforms(do_flip=False, max_rotate=20, max_zoom=1.2, max_lighting=0.1, max_warp=0.1)


# In[ ]:


test_src = ImageList.from_folder(extensions='.jpg', path='./mnist_data/test')
test_src


# In[ ]:


get_ipython().run_cell_magic('time', '', "train_src = ImageList.from_folder(extensions='.jpg', path='./mnist_data/train')\ntrain_src = train_src.split_by_rand_pct(0.2)\ntrain_src = train_src.label_from_folder()\ntrain_src = train_src.add_test(test_src)\ntrain_src = train_src.transform(tfms)\n\ntrain_data = train_src.databunch().normalize()\ntrain_data")


# In[ ]:


train_data.show_batch(rows= 5, ds_type=DatasetType.Test)


# # 2 models, loss, optimizer

# ## 2.1 model 1 - resnet

# In[ ]:


esc = partial(EarlyStoppingCallback, monitor='accuracy', min_delta=0.03, patience=3)
# smc = partial(SaveModelCallback, every='improvement', monitor='accuracy', name='best')


# In[ ]:


def train_for_leaner(learn):
    # freeze cycle
    learn.freeze()
    learn.lr_find()
    learn.recorder.plot(suggestion=True)
    lr_fr = learn.recorder.min_grad_lr
    
    learn.fit(3, lr_fr)
    
    # unfreeze cycle
    learn.unfreeze()
    learn.lr_find()
    learn.recorder.plot(suggestion=True)
    lr_un = learn.recorder.min_grad_lr
    
    learn.fit(5, slice(lr_un, lr_fr/10))
    learn.fit(10, slice(lr_un/3, lr_fr/30))
    return learn


# In[ ]:


learner_1 = cnn_learner(train_data, models.resnet101, metrics=accuracy, ps=0.6, wd = 0.01, callback_fns=esc).mixup().to_fp16()


# In[ ]:


learner_1 = train_for_leaner(learner_1)


# In[ ]:


learner_1.show_results(ds_type=DatasetType.Valid)


# In[ ]:


pred_1, _ = learner_1.get_preds(ds_type = DatasetType.Test)
pred_1.shape


# In[ ]:


pred_1_cat = np.argmax(pred_1, axis=1)
pred_1_cat = pred_1_cat.reshape(-1, 1)


# ## 2.2 model 2 - densenet

# In[ ]:


learner_2 = cnn_learner(train_data, models.densenet121, metrics=accuracy, ps=0.6, wd = 0.005).mixup().to_fp16()


# In[ ]:


learner_2 = train_for_leaner(learner_2)


# In[ ]:


learner_2.show_results(ds_type=DatasetType.Test)


# In[ ]:


pred_2, _ = learner_2.get_preds(ds_type = DatasetType.Test)
pred_2_cat = np.argmax(pred_2, axis=1)
pred_2_cat = np.array(pred_2_cat.reshape(-1, 1))


# # 3 stacking

# In[ ]:


pred = pred_1 * 0.5 + pred_2 * 0.5
pred_cat = np.argmax(pred, axis=1)
pred_cat = np.array(pred_cat.reshape(-1, 1))
pred_cat


# # 4 Submission

# In[ ]:


sub['Label'] = pred_cat
sub.head(20)


# In[ ]:


sub.to_csv('submission_mnist.csv', index=False)


# In[ ]:


sub_test = pd.read_csv('submission_mnist.csv')
sub_test.head()


# In[ ]:


get_ipython().system('ls .')


# In[ ]:




