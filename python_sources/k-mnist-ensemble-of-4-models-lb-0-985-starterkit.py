#!/usr/bin/env python
# coding: utf-8

# <font size=6 color='#000'>Introduction</font>
# <br>
# 
# In this kernel we will be exploring different model architectures, their performance (against one another) and create an ensembled submission of all of our models and submit it to the Leaderboard.
# 
# I've picked 4 of my best performing models so far for this kernel, everything coded up using the **fast.ai** library and PyTorch.
# 
# > **And... Please don't forget to smash that UPVOTE button.**
# 
# So lets jump right into it...
# 
# 
# <img src="https://thewhiskylounge.com/wp-content/uploads/2015/12/twl-img-event-blending-workshop-01.jpg" width="600" height="400">
# 
# ## What are we doing in this kernel?
# 
# * **1. Getting Started**
# 
# * **2. Loading Data + EDA**
# 
# * **3. Data Processing**
# 
# * **4. Defining our Models**
# 
# * **5. Training and Inference**
# 
# * **6. Comparing Model Performances**
# 
# * **7. Creating Average Ensembled Submission**
# 

# <font size=5 color='red'>Show your appreciation by giving this kernel an UPVOTE</font>

# ## Getting Started
# Here we import all the requires libraries and utillity functions.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import seaborn as sns

from fastai.vision import *
from fastai.callbacks import SaveModelCallback

from radam_optimizer_pytorch import RAdam
from torch.nn import Conv2d
from torch.optim import Adam

import os
PATH = Path('../input/Kannada-MNIST/')
os.listdir(PATH)


# Set the same random seeds for all libraries to ensure reproducibility.

# In[ ]:


# Setting Global Random Seed
def random_seed(seed_value, use_cuda):  
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda: torch.cuda.manual_seed_all(seed_value) # gpu 

random_seed(42, True)


# ## Load Data + EDA

# The images are given in standard format with a CSV for refenrence. Let's read that in first. The CSV file contains pixel values for all individual pixels in the image. The Image dimension is 28x28 so we have 784 pixel values.

# In[ ]:


train_csv = pd.read_csv(PATH/'train.csv')
train_csv.head().T


# The labels aren't given in the CSV file so we'll go fetch that.

# In[ ]:


def get_data_labels(csv,label):
    fileraw = pd.read_csv(csv)
    labels = fileraw[label].to_numpy()
    data = fileraw.drop([label],axis=1).to_numpy(dtype=np.float32).reshape((fileraw.shape[0],28,28))
    data = np.expand_dims(data, axis=1)
    return data, labels

train_data, train_labels = get_data_labels(PATH/'train.csv','label')
test_data, test_labels = get_data_labels(PATH/'test.csv','id')
other_data, other_labels = get_data_labels(PATH/'Dig-MNIST.csv','label')


# In[ ]:


print(f' Train:\tdata shape {train_data.shape}\tlabel shape {train_labels.shape}\n Test:\tdata shape {test_data.shape}\tlabel shape {test_labels.shape}\n Other:\tdata shape {other_data.shape}\tlabel shape {other_labels.shape}')


# In[ ]:


plt.title(f'Training Label: {train_labels[43]}')
plt.imshow(train_data[43,0],cmap='gray');


# Wait don't get confused that's not a 2. Remember the numbers are in a language called Kannada which is one of many languages spoken in the southern part of India.

# ## Processing Data for Training

# Splitting the Full Train set into
# 
# **80% - Training**
# 
# **20% - Validation**

# In[ ]:


np.random.seed(42)
ran_20_pct_idx = (np.random.random_sample(train_labels.shape)) < .2

train_80_labels = train_labels[np.invert(ran_20_pct_idx)]
train_80_data = train_data[np.invert(ran_20_pct_idx)]

valid_20_labels = train_labels[ran_20_pct_idx]
valid_20_data = train_data[ran_20_pct_idx]


# Now we create a fastai databunch. I strongly recommend reading the documentation here [docs.fast.ai](https://www.docs.fast.ai) if you have any problems understanding what's happening in the next few lines.

# In[ ]:


class ArrayDataset(Dataset):
    "Dataset for numpy arrays based on fastai example: "
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.c = len(np.unique(y))
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        return self.x[i], self.y[i]


# In[ ]:


train_ds = ArrayDataset(train_80_data,train_80_labels)
valid_ds = ArrayDataset(valid_20_data,valid_20_labels)
other_ds = ArrayDataset(other_data, other_labels)
test_ds = ArrayDataset(test_data, test_labels)


# In[ ]:


bs = 64 # Batch Size
data = DataBunch.create(train_ds, valid_ds, test_ds=test_ds, bs=bs)


# ## Define Models

# We create a models directory for fast.ai to save and load our models.

# In[ ]:


get_ipython().system('mkdir models')


# In[ ]:


MODEL_DIR = Path('../working/models/')


# **Model 1** in our experiment is just a small custom Convolutional Neural Net that has previously worked very well for me in the original MNIST dataset. The architecture of this model is pretty straight forward.
# 
# If you have troubles understanding how CNNs are coded in PyTorch you can check out some of my other tutorial kernels listed below
# * [MNIST: Introduction to Computer Vision with PyTorch](https://www.kaggle.com/abhinand05/in-depth-guide-to-convolutional-neural-networks)
# 
# * [In-Depth Guide to Convolutional Neural Networks](https://www.kaggle.com/abhinand05/in-depth-guide-to-convolutional-neural-networks)

# In[ ]:


class ConvNet1(nn.Module):
    def __init__(self):
        super(ConvNet1, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64)        
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        
        self.drop_out = nn.Dropout(0.2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(4608, 128)
        self.fc2 = nn.Linear(128, 10)
        self.bn1d = nn.BatchNorm1d(128)
        self.output = nn.Softmax(dim=1)
        
    def forward(self, x):
        # conv layers
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.drop_out(self.layer3(out))
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.drop_out(self.layer6(out))
        out = out.view(out.shape[0], -1)
#         print(out.shape) # Life Saving Debuggung Step
        # FC Layer 1
        out = self.fc1(out)
        out = self.relu(out)
        out = self.bn1d(out)
        out = self.drop_out(out)
        # Output layer
        out = self.fc2(out)
        out = self.output(out)
        return out


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move Model to GPU
conv_net_1 = ConvNet1()
conv_net_1 = conv_net_1.to(device)


# **Model 2** is also one of those that has worked well on the original MNIST Dataset.

# In[ ]:


class ConvNet2(nn.Module):
    def __init__(self):
        super(ConvNet2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout(0.5)
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.drop_out(out)
        out = self.fc2(out)
        return out


# In[ ]:


# Move to GPU
conv_net_2 = ConvNet2()
conv_net_2 = conv_net_2.to(device)


# **Model 3**

# In[ ]:


# Helper Function for Model 3
def conv2(ni,nf,stride=2,ks=3): return conv_layer(ni,nf,stride=stride,ks=ks)


# In[ ]:


conv_net_3 = nn.Sequential(
    conv2(1,32,stride=1,ks=3),
    conv2(32,32,stride=1,ks=3),
    conv2(32,32,stride=2,ks=5),
    nn.Dropout(0.4),
    
    conv2(32,64,stride=1,ks=3),
    conv2(64,64,stride=1,ks=3),
    conv2(64,64,stride=2,ks=5),
    nn.Dropout(0.4),
    
    Flatten(),
    nn.Linear(3136, 128),
    relu(inplace=True),
    nn.BatchNorm1d(128),
    nn.Dropout(0.4),
    nn.Linear(128,10)
)


# **Model 4: (ResNet18)**
# 
# Throwing proven architectures like ResNet18 into the mix might help to improve our ensemble. 

# In[ ]:


get_ipython().system('ls ../input/pytorch-pretrained-models')


# We load the ResNet Model and change the in_channels to 1 because this dataset only contains Greyscale images.

# In[ ]:


rn18 = models.resnet18(pretrained=False)
rn18.load_state_dict(torch.load('../input/pytorch-pretrained-models/resnet18-5c106cde.pth'))
rn18.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)


# <font size=5 color='red'>Show your appreciation by giving this kernel an UPVOTE if you liked.</font>

# ## Training and Inference

# Now we train the models using fastai library one by one.

# ### Model - 1

# In[ ]:


learner1 = Learner(data, 
                  conv_net_1, 
                  metrics=accuracy, 
                  model_dir=MODEL_DIR,
                  opt_func=Adam,
                  loss_func=nn.CrossEntropyLoss()
                 )


# In[ ]:


learner1.lr_find()
learner1.recorder.plot(suggestion=True)


# In[ ]:


get_ipython().run_cell_magic('time', '', "learner1.fit_one_cycle(50, \n                      slice(1e-03),\n                      callbacks=[SaveModelCallback(learner1, \n                                                   every='improvement', \n                                                   monitor='accuracy', \n                                                   name='best_model_1')]\n                     ) ")


# In[ ]:


learner1.recorder.plot_losses(skip_start=800)


# In[ ]:


learner1.recorder.plot_metrics(skip_start=800)


# In[ ]:


learner1.load('best_model_1')
preds, ids = learner1.get_preds(DatasetType.Test)
y = torch.argmax(preds, dim=1)

submission_1 = pd.DataFrame({ 'id': ids,'label': y })
submission_1.to_csv("submission.csv", index=False)
submission_1.to_csv("submission_1.csv", index=False)


# ### Model - 2

# In[ ]:


learner2 = Learner(data, 
                  conv_net_2, 
                  metrics=accuracy, 
                  model_dir=MODEL_DIR,
                  opt_func=RAdam,
                  loss_func=nn.CrossEntropyLoss()
                 )


# In[ ]:


learner2.lr_find()
learner2.recorder.plot(suggestion=True)


# In[ ]:


get_ipython().run_cell_magic('time', '', "learner2.fit_one_cycle(50, \n                      slice(1e-03),\n                      callbacks=[SaveModelCallback(learner2, \n                                                   every='improvement', \n                                                   monitor='accuracy', \n                                                   name='best_model_2')]\n                     ) ")


# In[ ]:


learner2.recorder.plot_losses(skip_start=800)


# In[ ]:


learner2.recorder.plot_metrics(skip_start=800)


# In[ ]:


learner2.load('best_model_2')
preds, ids = learner2.get_preds(DatasetType.Test)
y = torch.argmax(preds, dim=1)

submission_2 = pd.DataFrame({ 'id': ids,'label': y })
submission_2.to_csv("submission_2.csv", index=False)


# ### Model - 3

# In[ ]:


learner3 = Learner(data, 
                  conv_net_3, 
                  metrics=accuracy, 
                  model_dir=MODEL_DIR,
                  opt_func=Adam,
                  loss_func=nn.CrossEntropyLoss()
                 )


# In[ ]:


learner3.lr_find()
learner3.recorder.plot(suggestion=True)


# In[ ]:


get_ipython().run_cell_magic('time', '', "learner3.fit_one_cycle(50, \n                      slice(8e-03),\n                      callbacks=[SaveModelCallback(learner3, \n                                                   every='improvement', \n                                                   monitor='accuracy', \n                                                   name='best_model_3')]\n                     ) ")


# In[ ]:


learner3.recorder.plot_losses(skip_start=800)


# In[ ]:


learner3.recorder.plot_metrics(skip_start=800)


# In[ ]:


learner3.load('best_model_3')
preds, ids = learner3.get_preds(DatasetType.Test)
y = torch.argmax(torch.exp(preds), dim=1)

submission_3 = pd.DataFrame({ 'id': ids,'label': y })
submission_3.to_csv("submission_3.csv", index=False)


# ### Model - 4

# In[ ]:


learner4 = Learner(data, 
                  rn18, 
                  metrics=accuracy, 
                  model_dir=MODEL_DIR,
                  opt_func=Adam,
                  loss_func=nn.CrossEntropyLoss()
                 )


# In[ ]:


learner4.lr_find()
learner4.recorder.plot(suggestion=True)


# In[ ]:


get_ipython().run_cell_magic('time', '', "learner4.fit_one_cycle(50, \n                      slice(1e-03),\n                      callbacks=[SaveModelCallback(learner4, \n                                                   every='improvement', \n                                                   monitor='accuracy', \n                                                   name='best_model_4')]\n                     ) ")


# In[ ]:


learner4.recorder.plot_losses(skip_start=800)


# In[ ]:


learner4.recorder.plot_metrics(skip_start=800)


# In[ ]:


learner4.load('best_model_4')
preds, ids = learner4.get_preds(DatasetType.Test)
y = torch.argmax(torch.exp(preds), dim=1)

submission_4 = pd.DataFrame({ 'id': ids,'label': y })
submission_4.to_csv("submission_4.csv", index=False)


# ## Comparing Models

# We extract the data stored in `learner.recorder` after training to plot a few graphs to evaluate the performance of our models.

# In[ ]:


flatten = lambda l: [np.float32(item) for sublist in l for item in sublist]
metrics_list_1 = flatten(learner1.recorder.metrics)
metrics_list_2 = flatten(learner2.recorder.metrics)
metrics_list_3 = flatten(learner3.recorder.metrics)
metrics_list_4 = flatten(learner4.recorder.metrics)


# In[ ]:


losses_1 = pd.DataFrame({'loss':learner1.recorder.val_losses, 'accuracy': metrics_list_1})
losses_2 = pd.DataFrame({'loss':learner2.recorder.val_losses, 'accuracy': metrics_list_2})
losses_3 = pd.DataFrame({'loss':learner3.recorder.val_losses, 'accuracy': metrics_list_3})
losses_4 = pd.DataFrame({'loss':learner4.recorder.val_losses, 'accuracy': metrics_list_4})

fig, ax = plt.subplots(1,1,figsize=(14, 6))
ax.set(xlabel='Epochs Processed', ylabel='Loss', title='Comparing Validation Losses')
# losses_1['loss'].sort_index().plot(ax=ax)
losses_2['loss'].sort_index().plot(ax=ax)
losses_3['loss'].sort_index().plot(ax=ax)
losses_4['loss'].sort_index().plot(ax=ax)

ax.legend(['Model 2', 'Model 3', 'Model 4'])


# Since Model 1 has validation losses in a completely different range we plot that below seperately.
# 
# > But don't think it's not a good model just yet, it gave me scores close to 0.985 in Public LB and as you can see in the forthcoming plot, it has a pretty good validation accuracy as well which looks pretty stable.

# In[ ]:


fig, ax = plt.subplots(1,1,figsize=(14, 6))
ax.set(xlabel='Epochs Processed', ylabel='Loss', title='Validation Losses for Model 1')

losses_1['loss'].sort_index().plot(ax=ax)
ax.legend(['Model 1'])


# In[ ]:


fig, ax = plt.subplots(1,1,figsize=(14, 6))
ax.set(xlabel='Epochs Processed', ylabel='Loss', title='Comparing Validation Accuracy')
losses_1['accuracy'].sort_index().plot(ax=ax)
losses_2['accuracy'].sort_index().plot(ax=ax)
losses_3['accuracy'].sort_index().plot(ax=ax)
losses_4['accuracy'].sort_index().plot(ax=ax)

ax.legend(['Model 1', 'Model 2', 'Model 3', 'Model 4'])


# In[ ]:


preds_1 = (submission_1.label.value_counts()).rename('Model_1')
preds_2 = (submission_2.label.value_counts()).rename('Model_2')
preds_3 = (submission_3.label.value_counts()).rename('Model_3')
preds_4 = (submission_4.label.value_counts()).rename('Model_4')

preds_data = pd.concat([preds_1, preds_2, preds_3, preds_4], axis=1)
preds_data['category'] = preds_data.index
preds_data = pd.melt(preds_data, id_vars='category', var_name='model', value_name='preds')

fig = sns.catplot(x='category', y='preds', hue='model',data=preds_data, kind='bar', height=4, aspect=3)
fig.set(title='Distribution of predictions for each model per category')


# ## Creating Average Ensembled Submission

# I'm using a simple averaged ensemble and creating a submission file.

# In[ ]:


blended_preds = np.round((submission_1['label'] + submission_2['label'] + 
                          submission_3['label'] + submission_4['label'])/4)

blended_submission = pd.DataFrame({'id': ids, 'label': blended_preds})
blended_submission['label'] = blended_submission['label'].astype(np.uint8)
blended_submission.to_csv("blended_submission.csv", index=False)


# <font size=5 color='red'>Show your appreciation by giving this kernel an UPVOTE if you liked.</font>
