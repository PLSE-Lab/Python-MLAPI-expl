#!/usr/bin/env python
# coding: utf-8

# * I orginally had a model with **98.6%** accuracy using **Tensorflow Keras**. 
# * For this notebook, I tried to reimplement the same model using **Pytorch and FastAI** library as well as try out transfer learning using a pretrained **ResNet50** model.

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *


# In[ ]:


path = Path('/kaggle/input/Kannada-MNIST')
path.ls()


# In[ ]:


train_df = pd.read_csv(path/'train.csv')
train_df.head()


# In[ ]:


test_df = pd.read_csv(path/'test.csv')
test_df.head()


# # Preprocessing

# In[ ]:


# Create an index column of each example to use as filename
train_df['id'] = train_df.index
train_df.head()


# In[ ]:


# Do the same thing for test set
test_df['label'] = 0
test_df.head()


# In[ ]:


# Customized ItemList for pixel values
# Reference: https://www.kaggle.com/heye0507/fastai-1-0-with-customized-itemlist

class PixelImageItemList(ImageList):
    def open(self,id):
        regex = re.compile(r'\d+')
        id = re.findall(regex, id)
        df = self.inner_df[self.inner_df.id.values == int(id[0])]
        df_id = df[df.id.values == int(id[0])]
        img_pixel = df_id.drop(labels=['label','id'], axis=1).values
        img_pixel = img_pixel.reshape(28,28)
        img_pixel = np.stack((img_pixel,)*3, axis=-1)
        return vision.Image(pil2tensor(img_pixel, np.float32).div_(255))


# In[ ]:


# Data augmentation
tfms = get_transforms(do_flip=False, max_zoom=0.1)
# tfms = ([*rand_pad(padding=3, size=28, mode='zeros')], [])


# In[ ]:


src = (PixelImageItemList.from_df(train_df, '.' , cols='id')
                         .split_by_rand_pct()
                         .label_from_df(cols='label')
                         .transform(tfms, size=28))


# In[ ]:


data = src.databunch(bs=64).normalize()


# In[ ]:


# Add the test data into the data source
test = PixelImageItemList.from_df(test_df, path='.', cols='id')
data.add_test(test)


# In[ ]:


data.show_batch(rows=3, figsize=(9, 9))


# **First let's try the default ResNet50 model from FastAI library**

# In[ ]:


arch = models.resnet50
metrics = [error_rate, accuracy]


# In[ ]:


learn = cnn_learner(data, arch, metrics=metrics)


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


min_grad_lr = learn.recorder.min_grad_lr
learn.fit_one_cycle(4, slice(min_grad_lr))


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


min_grad_lr = learn.recorder.min_grad_lr
learn.fit_one_cycle(2, slice(min_grad_lr))


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.show_results()


# * So with the pretained ResNet50 model, we are able to achieve **99.5%** accuracy!!!
# * Now let's try defining our own model architecture using Pytorch

# In[ ]:


def conv2(ni, nf, stride): return conv_layer(ni, nf, stride = stride, ks = 3)


# In[ ]:


model = nn.Sequential(
    conv2(3, 32, 1),  # 28 
    conv2(32, 32, 2), # 14
    nn.Dropout(0.2),
    conv2(32, 64, 1), # 14
    conv2(64, 64, 2), # 7
    nn.Dropout(0.3),
    conv2(64, 128, 1), # 7
    conv2(128, 128, 2), # 4
    nn.Dropout(0.4),
    Flatten(),
    nn.Linear(128 * 4 * 4, 10)
)


# In[ ]:


xb, yb = data.one_batch()
print(xb.shape)
print(yb.shape)


# In[ ]:


learn_custom = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=metrics)


# In[ ]:


learn_custom.summary()


# In[ ]:


learn_custom.lr_find()
learn_custom.recorder.plot(suggestion=True)


# In[ ]:


min_grad_lr = learn_custom.recorder.min_grad_lr
learn_custom.fit_one_cycle(5, slice(min_grad_lr))


# In[ ]:


learn_custom.recorder.plot_losses()


# * After 5 epochs, our own model outperformed the pretrained ResNet50 model, reaching **99.7%** accuracy!!!
# * For the submission file, we will use the predictions from our own model.

# In[ ]:


# Get the predictions and create the "submission.csv" file
preds, _ = learn_custom.TTA(ds_type=DatasetType.Test)
y = torch.argmax(preds, dim=1)

submission = pd.DataFrame({ 'id': range(0, test_df.shape[0]), 'label': y })
submission.to_csv('submission.csv',index=False)


# In[ ]:


# Double check the submission
submission.head()

