#!/usr/bin/env python
# coding: utf-8

# # My Digit Recognizer Kaggle competition
# ### with Fastai v1

# # Getting the Data from CSV into DataBunch

# In[ ]:


from fastai.vision import *


# In[ ]:


from fastai.callbacks import *


# In[ ]:


path = Path('.')
datapath=path/'../input/digit-recognizer'
train_csv = datapath/'train.csv'
test_csv = datapath/'test.csv'


# In[ ]:


data = pd.read_csv(train_csv)


# In[ ]:


y = data.label.values
X = torch.tensor(data.drop('label', axis = 1).values)


# ### Randomly Split onto Training and Validation sets

# In[ ]:


rand_idx = torch.randperm(X.shape[0])
split_ratio = 0.8
split = int(X.shape[0] * split_ratio)
train_idxs = rand_idx[:split]
test_idxs  = rand_idx[split:]

X_train = X[train_idxs]
X_valid = X[test_idxs]
y_train = y[train_idxs]
y_valid = y[test_idxs]
X_train.shape, X_valid.shape


# In[ ]:


def tensor2Images(x,channels=3):
    assert channels == 3 or channels == 1, "Channels: 1 - mono, or 3 - RGB"
    return [Image(x[i].reshape(-1,28,28).repeat(channels, 1, 1)/255.) for i in range(x.shape[0])]


# In[ ]:


class MNISTImageList(ImageList):
    "`ImageList` of Images stored as in `items` as tensor."

    def open(self, fn):
        "No file associated to open"
        pass

    def get(self, i):
        res = self.items[i]
        self.sizes[i] = sys.getsizeof(res)
        return res


# #### Get LabelLists

# In[ ]:


til = MNISTImageList(tensor2Images(X_train,3))


# In[ ]:


til[0]


# In[ ]:


train_ll = LabelList(MNISTImageList(tensor2Images(X_train,3)),CategoryList(y_train, ['0','1','2','3','4','5','6','7','8','9']))
valid_ll = LabelList(MNISTImageList(tensor2Images(X_valid,3)),CategoryList(y_valid, ['0','1','2','3','4','5','6','7','8','9']))


# In[ ]:


ll = LabelLists(path,train_ll,valid_ll)
ll


# In[ ]:


data = pd.read_csv(test_csv)
Xtest = torch.tensor(data.values)
test_il = ItemList(tensor2Images(Xtest))


# In[ ]:


ll.add_test(test_il)


# In[ ]:


assert len(ll.train.x)==len(ll.train.y)
assert len(ll.valid.x)==len(ll.valid.y)


# #### Get ImageDataBunch from LabelLists

# In[ ]:


tfms = get_transforms(do_flip=False)


# In[ ]:


dbch = ImageDataBunch.create_from_ll(ll, bs=128, val_bs=256, ds_tfms=tfms)


# In[ ]:


dbch.sanity_check()


# In[ ]:


dbch.show_batch(rows=3, figsize=(4,4))


# # Training

# ## ResNet34

# In[ ]:


learn = cnn_learner(dbch,models.resnet50,metrics=accuracy,callback_fns=[CSVLogger,ShowGraph,SaveModelCallback])


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(10,max_lr=1e-2)


# In[ ]:


learn.save('stage1')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find(start_lr=1e-9)
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(10,max_lr=slice(1e-6,1e-5))


# In[ ]:


learn.load('bestmodel');


# In[ ]:


learn.lr_find(start_lr=1e-9)
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(10,max_lr=slice(8e-6,3e-5))


# In[ ]:


learn.load('bestmodel');


# In[ ]:


learn.recorder.plot_metrics()


# In[ ]:


learn.lr_find(start_lr=1e-10)
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(10,max_lr=slice(1e-8,2e-8))


# In[ ]:


learn.load('bestmodel');


# In[ ]:


learn.lr_find(start_lr=1e-10)
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(10,max_lr=slice(1e-6,1e-5))


# In[ ]:


learn.load('bestmodel');


# In[ ]:


learn.show_results(ds_type=DatasetType.Train, rows=4, figsize=(6,8))


# In[ ]:


learn.summary()


# In[ ]:


learn.export()


# In[ ]:


learn.get_preds


# ## Custom Model

# In[ ]:


# model = nn.Sequential(
#         nn.Conv2d(784, 64, kernel_size=(3,3)),
#         nn.ReLU(),
#         nn.MaxPool2d((2,2)),
#         nn.Dropout(0.5),
#         nn.Conv2d(64, 32, kernel_size=(3,3)),
#         nn.ReLU(),
#         nn.MaxPool2d((2,2)),
#         nn.Dropout(0.25),
#         nn.Flatten(),
#         nn.Linear(128,64),
#         nn.Dropout(0.5),
#         nn.Linear(64,10),
#         nn.Softmax()    
#     )


# In[ ]:


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x *(torch.tanh(F.softplus(x)))


# In[ ]:


model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        Mish(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.25),
        nn.Linear(128,64),
        Mish(),
        nn.BatchNorm1d(64),
        nn.Dropout(0.25),
        nn.Linear(64,32),
        Mish(),
        nn.BatchNorm1d(32),
        nn.Dropout(0.25),
        nn.Linear(32,10),
        nn.Softmax(dim=1)    
    )


# In[ ]:


model


# In[ ]:


opt_func = functools.partial(torch.optim.AdamW, betas=(0.9, 0.99))


# In[ ]:


learn = Learner(dbch, model, opt_func=opt_func, metrics=accuracy)


# In[ ]:


learn.summary()


# In[ ]:


learn.lr_find(start_lr=1e-8)
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(10,max_lr=5e-2)


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.lr_find(start_lr=1e-8)
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(10,max_lr=1e-2)


# In[ ]:


learn.recorder.plot_metrics()


# In[ ]:


learn.lr_find()#start_lr=1e-8)
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(10,max_lr=1e-3)


# In[ ]:


learn.recorder.plot_metrics()


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(20,max_lr=1e-3)


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.lr_find(start_lr=1e-9)
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(10,max_lr=1e-5)


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.lr_find(start_lr=1e-9)
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(10,max_lr=1e-5)


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.fit_one_cycle(10,max_lr=1e-3)


# In[ ]:


learn.recorder.plot_metrics()228


# # Inference

# In[ ]:


data = pd.read_csv(test_csv)
Xtest = torch.tensor(data.values)
test_il = ItemList(tensor2Images(Xtest))


# In[ ]:


test_il


# In[ ]:


test_il[0]


# In[ ]:


learn = load_learner(path,test=test_il)


# In[ ]:


learn


# In[ ]:


preds,y = learn.get_preds(ds_type=DatasetType.Test)


# In[ ]:


y = preds.argmax(dim=1)


# In[ ]:


assert len(y)==len(test_il)


# In[ ]:


res = pd.DataFrame(y,columns=['Label'],index=range(1, 28001))


# In[ ]:


res.index.name = 'ImageId'


# In[ ]:


res.head()


# In[ ]:


import datetime
today = datetime.datetime.today()


# In[ ]:


t = today.strftime("%Y-%m-%d-%H.%M.%S")


# In[ ]:


res.to_csv(f'submission-{t}')


# In[ ]:


get_ipython().system('ls')


# In[ ]:




