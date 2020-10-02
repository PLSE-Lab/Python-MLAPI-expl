#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *


# In[ ]:


path = Path('/kaggle/input/aerial-cactus-identification/')
path.ls()


# In[ ]:


# Unzip the training and testing images
from zipfile import ZipFile

with ZipFile(path/'train.zip', 'r') as archive:
    archive.extractall()

with ZipFile(path/'test.zip', 'r') as archive:
    archive.extractall()


# In[ ]:


train_df = pd.read_csv(path/'train.csv')
test_df = pd.read_csv(path/'sample_submission.csv')
train_df.head()


# In[ ]:


img_path = Path('/kaggle/working')
img_path.ls()


# In[ ]:


# Data augmentation
tfms = get_transforms()


# In[ ]:


test_img = ImageList.from_df(test_df, path=img_path, folder='test')
data = ImageDataBunch.from_df(path=img_path/'train', df=train_df, label_col=1, bs=32, size=32, ds_tfms=tfms)
data.add_test(test_img)


# In[ ]:


data.show_batch(rows=5, figsize=(9,9))


# In[ ]:


def train(arch):
    # Define learner
    learn = cnn_learner(data, arch, metrics=[error_rate, accuracy])
    learn.model_dir = '/kaggle/working'

    # Train
    learn.lr_find()
    learn.recorder.plot(suggestion=True)
    min_grad_lr = learn.recorder.min_grad_lr
    if min_grad_lr < 1e-5:
        min_grad_lr = 3e-3
    learn.fit_one_cycle(5, slice(min_grad_lr))
    
    learn.recorder.plot_losses()
    
    return learn


# In[ ]:


# ResNet50
resnet50_learner = train(models.resnet50)


# In[ ]:


# DenseNet121
densenet121_learner = train(models.densenet121)


# In[ ]:


# VGG19
vgg_learner = train(models.vgg19_bn)


# In[ ]:


# Average the predicted results from all the models
def ensemble_predition(test_img):
    img = open_image(img_path/'test'/test_img)
    
    resnet50_predicition = resnet50_learner.predict(img)
    densenet121_predicition = densenet121_learner.predict(img)
    vgg_predicition = vgg_learner.predict(img)
    
    # Ensemble average
    prediction = (resnet50_predicition[2] + densenet121_predicition[2] + vgg_predicition[2]) / 3
    
    # Prediction results
    predicted_label = torch.argmax(prediction).item()
    
    return predicted_label


# In[ ]:


# Prepare test predictions

resnet50_predicition, _ = resnet50_learner.TTA(ds_type=DatasetType.Test)
densenet121_predicition, _ = densenet121_learner.TTA(ds_type=DatasetType.Test)
vgg_predicition, _ = vgg_learner.TTA(ds_type=DatasetType.Test)

# Average the predicted results from all the models
ens_test_preds = (resnet50_predicition + densenet121_predicition + vgg_predicition) / 3


# In[ ]:


# Create "submission.csv" file
test_df.has_cactus = ens_test_preds[:, 0]
test_df.to_csv('submission.csv', index=False)


# In[ ]:


# Double check the submission
test_df.head()

