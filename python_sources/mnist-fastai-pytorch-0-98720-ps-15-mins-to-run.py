#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai import *
from fastai.vision import *
DATAPATH = Path('/kaggle/input/Kannada-MNIST/')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


def get_data_labels(csv,label):
    fileraw = pd.read_csv(csv)
    labels = fileraw[label].to_numpy()
    data = fileraw.drop([label],axis=1).to_numpy(dtype=np.float32).reshape((fileraw.shape[0],28,28))
    data = np.expand_dims(data, axis=1)
    return data, labels


# Process the training, testing and 'other' datasets, and then check to ensure the arrays look reasonable.

# In[ ]:


train_data, train_labels = get_data_labels(DATAPATH/'train.csv','label')
test_data, test_labels = get_data_labels(DATAPATH/'test.csv','id')
other_data, other_labels = get_data_labels(DATAPATH/'Dig-MNIST.csv','label')


# In[ ]:


print(f' Train:\tdata shape {train_data.shape}\tlabel shape {train_labels.shape}\n Test:\tdata shape {test_data.shape}\tlabel shape {test_labels.shape}\n Other:\tdata shape {other_data.shape}\tlabel shape {other_labels.shape}')

Display a labelled image:
# In[ ]:


plt.title(f'Training Label: {train_labels[5]}')
plt.imshow(train_data[8,0],cmap='gray');


# In[ ]:


np.random.seed(60)
#np.random.randint(low = 0, high = 255, size = 1)

ran_10_pct_idx = (np.random.random_sample(train_labels.shape)) < .001

train_90_labels = train_labels[np.invert(ran_10_pct_idx)]
train_90_data = train_data[np.invert(ran_10_pct_idx)]

valid_10_labels = train_labels[ran_10_pct_idx]
valid_10_data = train_data[ran_10_pct_idx]


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


train_ds = ArrayDataset(train_90_data,train_90_labels)
valid_ds = ArrayDataset(valid_10_data,valid_10_labels)
other_ds = ArrayDataset(other_data, other_labels)
test_ds = ArrayDataset(test_data, test_labels)


# In[ ]:


bs = 256
databunch = DataBunch.create(train_ds, valid_ds, test_ds=test_ds, bs=bs)


# In[ ]:


def conv2(ni,nf,stride=2,ks=5): return conv_layer(ni,nf,stride=stride,ks=ks)

# Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and batchnorm (if `bn`) layers."


# In[ ]:


best_architecture = nn.Sequential(
    conv2(1,32,stride=1,ks=5),
    conv2(32,32,stride=1,ks=5),
    conv2(32,32,stride=2,ks=8),
    nn.Dropout(0.4),
    
    conv2(32,64,stride=1,ks=5),
    conv2(64,64,stride=1,ks=5),
    conv2(64,64,stride=2,ks=5),
    nn.Dropout(0.4),
    
    Flatten(),
    nn.Linear(3136, 256),
    relu(inplace=True),
    nn.BatchNorm1d(256),
    nn.Dropout(0.4),
    nn.Linear(256,10)
)


# In[ ]:


learn = Learner(databunch, best_architecture, loss_func = nn.CrossEntropyLoss(), metrics=[accuracy] )


# In[ ]:


learn.fit_one_cycle(42)


# In[ ]:


preds, ids = learn.get_preds(DatasetType.Test)
y = torch.argmax(preds, dim=1)


# In[ ]:


submission = pd.DataFrame({ 'id': ids,'label': y })
submission.to_csv(path_or_buf ="submission.csv", index=False)

