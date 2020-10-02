#!/usr/bin/env python
# coding: utf-8

# This kernel is forked by [Introduction to CNN Keras - 0.997 (top 6%](https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6).

# # 1. Preparation

# In[ ]:


get_ipython().system('pip install skorch')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetClassifier
import torch
from torch.optim import Adam
from torchvision.models import resnet18


# In[ ]:


seaborn.set(style="darkgrid", context="notebook", palette="muted")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[ ]:


seed = 7
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# # 2. Handling Data
# * `train`, `test` : CSV Row Data
# * `x_train` : Training Input Data
# * `y_train` : Training Label Data
# * `x_test` : Testing Input Data

# ## Load Images

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


print(train.shape)
train.head()


# In[ ]:


print(test.shape)
test.head()


# ## Create Training Images `x_train` and Testing Images `x_test`
# We separate pixels from `train`, while not separate from `test` because it has only pixels.

# In[ ]:


x_train = train.drop(["label"], axis=1) 
x_test = deepcopy(test)
x_train.shape


# ## Scale `x_train` and `x_test` from [0, 255] to [0, 1]

# In[ ]:


seaborn.distplot(x_train[4:5], kde=False, rug=True)


# In[ ]:


x_train = x_train / 255.0
x_test = x_test / 255.0


# In[ ]:


seaborn.distplot(x_train[4:5], kde=False, rug=True)


# ## Reshape `x_train` and `x_test` from (1, 784) to (1, 28, 28)
# Convert to (channel x width x height).
# Gray-scale is defined by only 1 channel.

# In[ ]:


print("train.shape=%s, test.shape=%s" % (x_train.shape, x_test.shape))


# In[ ]:


x_train = x_train.values.reshape(-1, 1, 28, 28)
x_test = x_test.values.reshape(-1, 1, 28, 28)


# In[ ]:


print("train.shape=%s, test.shape=%s" % (x_train.shape, x_test.shape))


# ## Convert `x_train` and `x_test` from `numpy.array` to `torch.FloatTensor`

# In[ ]:


x_train = torch.from_numpy(x_train).type('torch.FloatTensor')
x_test = torch.from_numpy(x_test).type('torch.FloatTensor')


# ## Create Training Labels `y_train`
# We separate labels from `train`.

# In[ ]:


y_train = train["label"]


# ### Convert `y_train` from `pandas.DataFrame` to `torch.LongTensor`

# In[ ]:


y_train = torch.Tensor(y_train).type('torch.LongTensor')


# ## Try Showing 4th Training Image

# In[ ]:


y_train[3]


# In[ ]:


plt.imshow(x_train[3][0,:,:])


# # 3. Training and Prediction

# ## Crerate and Modify ResNet18

# In[ ]:


network = resnet18()
network


# By checking `torchvision.models.resnet18`, we have to modify as follows.  
# * Change `conv1` layer from 3 channels to 1
# * Change `fc` layer from 1000 output features to 10
# * Add a softmax layer after `fc` layer

# In[ ]:


network.conv1 = torch.nn.Conv2d(1, 64,
                                kernel_size=(7, 7),
                                stride=(2, 2),
                                padding=(3, 3),
                                bias=False)
network.fc = torch.nn.Linear(in_features=512,
                             out_features=10,
                             bias=True)
network.add_module("softmax",
                   torch.nn.Softmax(dim=-1))
network


# ## Train

# In[ ]:


network.zero_grad()
classifier = NeuralNetClassifier(
    network,
    max_epochs=20,
    lr=0.01,
    batch_size=256,
    optimizer=torch.optim.Adam,
    device=device,
    criterion=torch.nn.CrossEntropyLoss,
    train_split=None
)
classifier.fit(x_train, y_train)


# ## Predict from Training Images

# In[ ]:


pred_train = classifier.predict(x_train)
pred_train.shape


# ## Check the 5th Training Image Prediction

# In[ ]:


pred_train[4]


# In[ ]:


plt.imshow(x_train[4][0,:,:])


# # 4. Evaluating Training

# ## Plot Confusion Matrix

# In[ ]:


cm = confusion_matrix(y_train.numpy(), pred_train) 
cm_df = pd.DataFrame(cm, columns=np.unique(y_train.numpy()), index = np.unique(y_train.numpy()))
cm_df.index.name = "True Label"
cm_df.columns.name = "Predicted Label"
cm_df


# In[ ]:


seaborn.heatmap(cm_df,
                annot=True,
                cmap="Blues",
                fmt="d")


# ## Check Top 6 Error-Predicted Images

# We crerate all error-predicted images, and narrow 6 down.

# In[ ]:


errors = (pred_train - y_train.numpy() != 0)
pred_train_errors = pred_train[errors]
x_train_errors = x_train.numpy()[errors]
y_train_errors = y_train.numpy()[errors]


# In[ ]:


pred_train_errors = pred_train_errors[:6]
x_train_errors = x_train_errors[:6]
y_train_errors = y_train_errors[:6]


# In[ ]:


fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
for row in range(2):
    for col in range(3):
        idx = 3 * row + col
        ax[row][col].imshow(x_train_errors[idx][0])
        args = (pred_train_errors[idx], y_train_errors[idx])
        title = "Predict:%s,True:%s" % args
        ax[row][col].set_title(title)


# # 5. Submitting

# ## Predict from Testing Images

# In[ ]:


pred_test = classifier.predict(x_test)
pred_test.shape


# ## Check the 5th Testing Image Prediction

# In[ ]:


pred_test[4]


# In[ ]:


plt.imshow(x_test[4][0,:,:])


# ## Create CSV File

# In[ ]:


result = pd.DataFrame({"ImageId" : range(1,28001),
                       "Label" : pred_test})
result.to_csv("result.csv",index=False)

