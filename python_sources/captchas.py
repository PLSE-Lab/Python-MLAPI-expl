#!/usr/bin/env python
# coding: utf-8

# # Captchas

# In[ ]:


get_ipython().system('nvidia-smi')


# In[1]:


from fastai.vision import *
import os
print(os.listdir("../input/samples/samples")[:10])


# In[4]:


path = Path(r'../input/samples/')


# In[ ]:


from IPython.display import Image
Image(filename='../input/samples/samples/bny23.png')


# ## Multilabel Classification

# In[ ]:


def label_from_filename(path):
    label = [char for char in path.name[:-4]]
    return label


# In[ ]:


data = (ImageList.from_folder(path)
        .split_by_rand_pct(0.2)
        .label_from_func(label_from_filename)
        .transform(get_transforms(do_flip=False))
        .databunch()
        .normalize()
       )
data.show_batch(3)


# In[ ]:


acc_02 = partial(accuracy_thresh, thresh=0.2)


# In[ ]:


learn = learn = cnn_learner(data, models.resnet18, model_dir='/tmp', metrics=acc_02)
lr_find(learn)
learn.recorder.plot()


# In[ ]:


lr = 5e-2
learn.fit_one_cycle(5, lr)


# In[ ]:


learn.unfreeze()
lr_find(learn)
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(15, slice(1e-3, lr/5))


# ### Misclassifications

# In[ ]:


def show_extremes(learn, k=3, thresh=0.2, show_losses=True):
    preds, y, losses = learn.get_preds(with_loss=True)
    losses = losses.view(preds.shape).sum(dim=1)
    
    if show_losses: sort_idx = np.argsort(losses.numpy())[::-1]
    else: sort_idx = np.argsort(losses.numpy())
        
    imgs = learn.data.valid_ds
    
    fig, ax = plt.subplots(ncols=k, nrows=k, figsize=(10,10))

    for i,axis in zip(sort_idx, ax.flatten()):
        img, actual_label = imgs[i]
        actual_label = ''.join(set(str(actual_label).split(';')))

        pred = preds[i]
        show_image(img, ax=axis)
        pred_label = [c for p,c in zip(pred>=thresh, learn.data.classes) if p]
        loss = losses[i]

        title = f'{"".join(pred_label)} | {str(actual_label)} | {loss.item():.4f}'
        axis.set_title(title)
    fig.suptitle('prediction | actual | loss', fontsize=16)
    plt.show()


# In[ ]:


show_extremes(learn, show_losses=True)


# In[ ]:


show_extremes(learn, show_losses=False)


# # Single char classification

# In[ ]:


def char_from_path(path): return path.name[2]


# In[ ]:


data = (ImageList.from_folder(path)
        .split_by_rand_pct(0.2)
        .label_from_func(char_from_path)
        .transform(get_transforms(do_flip=False))
        .databunch()
        .normalize()
       )
data.show_batch(3)


# In[ ]:


learn = learn = cnn_learner(data, models.resnet18, model_dir='/tmp', metrics=accuracy, ps=.2)
lr_find(learn)
learn.recorder.plot()


# In[ ]:


lr = 1e-2
learn.fit_one_cycle(10, lr)


# In[ ]:


learn.unfreeze()
lr_find(learn)
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(15, slice(5e-4, lr/5))


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(10,10))


# # Helpers
# 
# Recognition of the 5 characters of each captcha is done as a regression task. There are 5 characters and 19 possibilities for each characters. Each captcha can be represented by a 19x5 Matrix (concatenation of 5 one-hot encoded vectors). This matrix is flattened to a 95 dimensional vector. This vector is the target for the regression.
# The following methods handle
# - The encoding from each possible character to the onehot-encoded vector
# - The encoding of each captcha to the (flattened) onehot-encoded vector
# - The decoding of the flattened vector to readable captcha characters
# 
# The loss function for the regression is MSE. While usefull for seeing if we are moving towards the right direction, it's not as useful for judging if it's solving the captcha recognition task. To make the progress towards the goal visible, we take each prediction vector from the validation set, transform it into the original onehot-matrix, convert the columns to characters and then compare this character list to the actual label.
# 
# The ratio of correctly classified captchas over the total amount of captchas is calculated in 'label_accuracy'.

# In[5]:


#convert label
labels = [[char for char in code.name[:-4]] for code in (path/'samples').glob('*.png')]
labels = set([letter for label in labels for letter in label])
print(len(labels), 'different labels were found')

encoding_dict = {l:e for e,l in enumerate(labels)}
decoding_dict = {e:l for l,e in encoding_dict.items()}

code_dimension = len(labels)
captcha_dimension = 5

def to_onehot(filename):
    code = filename.name[:-4]
    onehot = np.zeros((code_dimension, captcha_dimension))
    for column, letter in enumerate(code):
        onehot[encoding_dict[letter], column] = 1
    return onehot.reshape(-1)

def decode(onehot):
    onehot = onehot.reshape(code_dimension, captcha_dimension)
    idx = np.argmax(onehot, axis=0)
    return [decoding_dict[i.item()] for i in idx]

def label_accuracy(preds, actuals):
    pred = torch.unbind(preds)
    act = torch.unbind(actuals)
    
    valid = 0
    total = 0
    
    for left,right in zip(pred,act):
        total+=1
        p = decode(left)
        a = decode(right)
        if p==a: valid += 1

    return torch.tensor(valid/total).cuda()

def c_acc(idx, preds, actuals):
        pred = torch.unbind(preds)
        act = torch.unbind(actuals)

        valid = 0
        total = 0

        for left,right in zip(pred,act):
            total+=1
            p = decode(left)
            a = decode(right)
            if p[idx]==a[idx]: valid += 1

        return torch.tensor(valid/total).cuda()
    
def char_accuracy(n):
    return partial(c_acc, n)


# In[6]:


#The captchas are already the result of some sort of transformation, so I'll try not using any additional ones
data = (ImageList.from_folder(path)
        .split_by_rand_pct(0.2)
        .label_from_func(to_onehot, label_cls = FloatList)
        .transform(get_transforms(do_flip=False))
        .databunch()
        .normalize()
       )


# In[7]:


#the labels are already in onehot-encoded form at this point
data.show_batch(3)


# In[8]:


learn = cnn_learner(data, models.resnet18, model_dir='/tmp',
                    metrics=[label_accuracy, char_accuracy(0),char_accuracy(1),char_accuracy(2),char_accuracy(3),char_accuracy(4)],
                   ps=0.)


# In[10]:


lr_find(learn)
learn.recorder.plot()


# In[11]:


lr = 5e-2
learn.fit_one_cycle(5, lr)


# The validation loss is sinking, but not a single captcha is correctly (completely) classified at this point.

# In[12]:


learn.save('pretrained')


# In[9]:


learn.load('pretrained')


# In[13]:


learn.unfreeze()
lr_find(learn)
learn.recorder.plot()


# In[27]:


#learn.save('unfr_1')
learn.load('unfr_1')


# In[28]:


learn.fit_one_cycle(80, slice(1e-3, 1e-2), pct_start=0.90)


# In[23]:


learn.save('trained_96pct')


# * 96% on resnet 18 with 80 epochs of 1cycle. 92% with 50 epochs. Check for overfitting
# * Segments characters from right to left. Middle character and left-most character are a problem

# 

# In[ ]:


dat, lbl = learn.data.valid_ds[10]
get_ipython().run_line_magic('pinfo2', 'dat.show')


# In[ ]:


def show_preds(learn, k=3):
    for i in range(10):
        dat, lbl = learn.data.valid_ds[10+i]
        dat.show()
        plt.title(' '.join(decode(learn.predict(dat)[0].data)))
        plt.show()

