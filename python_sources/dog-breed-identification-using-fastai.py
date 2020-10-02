#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import matplotlib as plt


# In[ ]:


from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
import os


# In[ ]:


# Checking if GPU is available
torch.cuda.is_available()


# In[ ]:


torch.backends.cudnn.enabled


# In[ ]:


PATH='data/'
size = 224 # size of the image
arch = resnext101_64
bs = 64 # made it bigger so that more are trained and takes less time hopefully


# In[ ]:


label_csv = os.path.join(PATH, 'labels.csv')
no_of_records = len(list(open(label_csv)))-1 # to exclude the header
# Creating a validation set from the given dataset as no separate validation dataset provided.
val_idxs = get_cv_idxs(no_of_records, val_pct=0.25) # keeping 25% as validation set


# In[ ]:


len(val_idxs)


# # Initial Exploration

# In[ ]:


os.listdir(PATH)


# In[ ]:


import pandas as pd
label_df = pd.read_csv(label_csv)


# In[ ]:


label_df.head()


# In[ ]:


# Get summary of each breed
pvt_table = label_df.pivot_table(index=['breed'], values='id', aggfunc=len).sort_values('id', ascending=False)
pvt_table


# In[ ]:


trg_pix = os.path.join(PATH, 'train')
pix_list = os.listdir(trg_pix)
pix_list


# In[ ]:


img = PIL.Image.open(os.path.join(PATH, 'train', pix_list[-1]))
img


# In[ ]:


img.size


# # Appling Transformation and creating a model

# In[ ]:


tfms = tfms_from_model(arch, size, aug_tfms=transforms_side_on, max_zoom=1.1)
data = ImageClassifierData.from_csv(PATH, 'train', 
                 f'{PATH}labels.csv', test_name='test', 
                 val_idxs=val_idxs, suffix='.jpg', tfms=tfms, bs=bs)


# In[ ]:


# No of images
n = len(list(open(label_csv)))-1
n


# In[ ]:


val_idxs = get_cv_idxs(n) # Creating validation set, default is 20%


# In[ ]:


file_name = PATH+data.trn_ds.fnames[0]
file_name


# In[ ]:


img = PIL.Image.open(file_name)
img


# In[ ]:


img.size


# In[ ]:


# Also need to check the size of each image as it also has effect on the chosen model
# Creating a dictionary of file name with its dimension
size_d = {k: PIL.Image.open(PATH+k).size for k in data.trn_ds.fnames}
size_d


# In[ ]:


# Creating a list of tuple of row, col size which can then be used in graph
row_sz, col_sz = list(zip(*size_d.values()))


# In[ ]:


# Convert row_sz to numpy array which is easier to manipulate
row_sz = np.array(row_sz)
plt.hist(row_sz) # row is the width of the image


# In[ ]:


# There are few images more than 2000 rows, so lets filter them out
plt.hist(row_sz[row_sz<1000])# Can be seen in the histogram that most images are row of 500


# In[ ]:


col_sz = np.array(col_sz)
col_sz


# In[ ]:


plt.hist(col_sz)


# In[ ]:


plt.hist(col_sz[col_sz<1000])


# # Initial Model

# In[ ]:


# most of the images are 500 x 500
# Start with small image size
def get_data(sz, bs):
    tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on,
                           max_zoom=1.1)
    data = ImageClassifierData.from_csv(PATH, 'train', 
               f'{PATH}labels.csv', test_name='test', num_workers=4,
               val_idxs=val_idxs, suffix='.jpg', tfms=tfms, bs=bs)

    return data if sz>300 else data.resize(340, 'tmp')


# ## Precompute

# In[ ]:


# Getting the data and creating the neural network with precompute=True i.e. use the validation from the pre-trained network
# I am using resnext101_64 which has already been trained on 1000 classes on 1.2 million images
# With precompute the model can be trained quickly.
data = get_data(size, bs)
learn = ConvLearner.pretrained(arch, data, precompute=True)


# In[ ]:


# Finding the learning rate to find the highest learning rate where loss is still clearly improving.
learn.lr_find() # The iteration stops when lr becomes very big and is of no use anymore.


# In[ ]:


# Plot showing increase in learning rate with increase in number of iterations
learn.sched.plot_lr()


# In[ ]:


# Graph between learning rate and the loss. 0.01 seems to be the best choice because though 0.1 is the minimal loss
# but loss starts to increase after that point so discarded.
learn.sched.plot()


# In[ ]:


# Training the last layer from precomputed activations
# Going for lr =/0.055 for another model to see if there is any improvement in accuracy
learn.fit(0.055, 5) # Here we receive 91.9% accuracy however there seems to me underfitting of the model


# ## Augment the data

# In[ ]:


from sklearn import metrics


# In[ ]:


# Using drop out by using parameter ps 
learn = ConvLearner.pretrained(arch, data, precompute=True, ps=0.5) # This refers to dropping 50% of neurons at random. This prevents overfitting.


# In[ ]:


# Training the model again after the dropout
learn.fit(1e-2, 2) # The accuracy has dropped a bit but is expected as 50% of random neurons have been dropped above


# In[ ]:


# Further improving the model by setting precompute = False.
learn.precompute=False


# In[ ]:


# Retraining the model after setting precompute=False and including new parameter called cycle_len
# This will take longer time as the activation is being recalculated from scratch. 
learn.fit(1e-2, 5, cycle_len=1)


# In[ ]:


learn.save('224_pre')


# In[ ]:


learn.load('224_pre')


# ## Increase the size of image

# In[ ]:


learn.set_data(get_data(299, 64))
learn.freeze() # The last layer is frozen so that activation remains the same while image size has changed


# In[ ]:


# Retraining the model now
learn.fit(1e-2, 3, cycle_len=1)


# In[ ]:


# Increasing the size has decreased training loss and vlaidation loss and slightly improved accuracy
# There could be chance that learning rate is changing before actually able to reach the correct value,
# So to find better learning rate, increase the length of iteration. this is done by cycle_mult parameter
# There will be 7 epochs by the learner .(1+2*1 +2*2 )epoch=7 epoch
learn.fit(1e-2, 3, cycle_len=1, cycle_mult=2)


# In[ ]:


# Looks like a good model as training loss and validation loss are comparable with good accuracy


# ## Applying Time Test Augmentation

# In[ ]:


# Does TTA helps
log_preds, y = learn.TTA()
probs = np.mean(np.exp(log_preds), 0)
accuracy_np(probs, y), metrics.log_loss(y, probs)


# In[ ]:


# Not much difference in accuracy


# In[ ]:


learn.save('299_pre')


# In[ ]:


learn.load('299_pre')


# # Prediction on test set now

# In[ ]:


log_preds_test, y_test = learn.TTA(is_test=True)
probs_test = np.mean(np.exp(log_preds_test), 0)


# # Creating file for submission

# In[ ]:


df = pd.DataFrame(probs_test)
df.columns = data.classes
# insert clean ids - without folder prefix and .jpg suffix - of images as first column
df.insert(0, "id", [e[5:-4] for e in data.test_ds.fnames])


# In[ ]:


comp_name='dog_breed_iden'
df.to_csv(f"sub_{comp_name}_{str(arch.__name__)}.csv", index=False)

