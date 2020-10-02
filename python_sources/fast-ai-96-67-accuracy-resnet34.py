#!/usr/bin/env python
# coding: utf-8

# Checking input directory and loading data and libraries

# In[ ]:


get_ipython().system('pip install fastai==0.7.0 --no-deps')
get_ipython().system('pip install torch==0.4.1 torchvision==0.2.1')


# In[ ]:


# verify dataset is there
get_ipython().system('ls ../input')


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# load fastai libraries
from fastai.imports import *
from fastai.torch_imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *


# In[ ]:


# load additional libraries
import h5py


# In[ ]:


# verify GPU
print(torch.cuda.is_available(), torch.backends.cudnn.enabled)


# > # Load and inspect data

# In[ ]:


# function to load the data into appropriate format
def load_dataset(path_to_train, path_to_test):
    train_dataset = h5py.File(path_to_train)
    train_x = np.array(train_dataset['train_set_x'][:])
    train_y = np.array(train_dataset['train_set_y'][:])

    test_dataset = h5py.File(path_to_test)
    test_x = np.array(test_dataset['test_set_x'][:])
    test_y = np.array(test_dataset['test_set_y'][:])

    # y reshaped
    train_y = train_y.reshape((1, train_x.shape[0]))
    test_y = test_y.reshape((1, test_y.shape[0]))

    return train_x, train_y, test_x, test_y


# In[ ]:


# defining input patt and reading the data using the defined function
PATH = "../input"
X_train, Y_train, X_test, Y_test = load_dataset(f"{PATH}/train_happy.h5", f"{PATH}/test_happy.h5")
# swap dimensions (Andrew Ng likes them flipped around)
Y_train = Y_train.T.squeeze()
Y_test = Y_test.T.squeeze()


# In[ ]:


# always check the shapes of your input while doing deep learning (or anything invlving tensors)!
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)


# In[ ]:


# visualize a training example
plt.imshow(X_train[0])


# In[ ]:


# size of an image in train
X_train[0].shape


# # Model

# In[ ]:


# setup architecture
arch = resnet34
sz = 48
bs = 32


# In[ ]:


# function to apply tranformations on data according to paramters selected
def get_data(sz, bs):
    tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
    return ImageClassifierData.from_arrays(path = 'tmp/',
                                           trn=(X_train, Y_train),
                                           val=(X_test, Y_test),
                                           bs=bs,
                                           classes=Y_train,
                                           tfms=tfms)


# In[ ]:


data = get_data(sz, bs)


# In[ ]:


# run learner with precompute enabled
learn = ConvLearner.pretrained(arch, data, precompute=True)


# In[ ]:


# find optimal learning rate
lrf = learn.lr_find()
learn.sched.plot_lr()


# In[ ]:


learn.sched.plot()


# In[ ]:


lr = 0.028


#  I tried with a smaller number of epochs and cycle_mult and the model was underfitting (train error < valid error) and so increased the cycle_mult so that gradient descent doesn't get stuck in a non-optimal or less optimal region.

# In[ ]:


learn.fit(lr, 5, cycle_len=1, cycle_mult=3)


# In[ ]:


learn.sched.plot_loss()


# Using Jeremy Howard's technique to reduce overfitting by first training on a smaller size of images( 48) and then increasing the size and training again (effectively increasing the kind and number of training examples.

# In[ ]:


learn.set_data(get_data(64, bs))


# In[ ]:


learn.freeze()


# In[ ]:


# fitting final model again after using new examples
learn.fit(0.028, 5, cycle_len=1, cycle_mult=3)


# In[ ]:


# add test time augmentation to see if it helps
log_preds,y = learn.TTA()
probs = np.mean(np.exp(log_preds),0)


# In[ ]:


accuracy_np(probs, y)


# # Results

# ### Confusion Matrix

# In[ ]:


preds = np.argmax(probs, axis=1)
probs = probs[:,1]


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, preds)


# In[ ]:


classes = np.unique(Y_train)
plot_confusion_matrix(cm, classes)


# ### Incorrect predictions

# In[ ]:


def load_img_id(idx):
    #print(idx)
    img = X_test[idx].reshape(64,64,3)
    return img

def plot_val_with_title(idxs, title):
    print(idxs)
    imgs = [load_img_id(x) for x in idxs]
    title_probs = [(preds[x], y[x]) for x in idxs]
    print(title)
    return plots(imgs, rows=4, titles=title_probs, figsize=(16,8)) if len(imgs)>0 else print('Not Found.')


# In[ ]:


# count incorrect predictions
incorrect_digits = np.where(preds != y)[0]
len(incorrect_digits)


# In[ ]:


# visualize incorrect predictions
plot_val_with_title(incorrect_digits, "Incorrect digits (prediction, label)")


# In[ ]:


get_ipython().run_cell_magic('capture', '', '!apt-get install zip\n!zip -r output.zip tmp\n!rm -rf  tmp/*')


# In[ ]:




