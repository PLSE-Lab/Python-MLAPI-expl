#!/usr/bin/env python
# coding: utf-8

# In this notebook, we'll be looking at utilizing the `fastai2` library to train a xresnet model from scratch (as no pretrained models exist currently), and comparing how it's overall accuracy is along with how to utilize a few more techniques in the `fastai2` library to speed up training *and* improve accuracy

# ## Exploring the Data Format
# 
# First things first, let's explore how our data is given to us! Let's import from the `fastai2` `vision` library to get us some helper libraries:

# In[ ]:


get_ipython().system('pip install fastai2 --quiet')


# In[ ]:


from fastai2.vision.all import *


# To make things easier, we'll set a `Path` to our data:

# In[ ]:


path = Path('../input/plant-pathology-2020-fgvc7')


# Now our labels are stored inside of `train.csv`, so let's open it up and look inside:

# In[ ]:


train_df = pd.read_csv(path/'train.csv')


# In[ ]:


train_df.head()


# So we have an `image_id` which would correspond to a particular file name, and our labels are given as one of four binary values! That doesn't help us very much though, as we *just* want to use the class (healthy, multiple_diseases, rust, or scab). How do we do that?

# In[ ]:


train_df.iloc[0, 1:]


# So here we have a row of our data's labels. To get the one that matches we'll want to see who's value is equal to 1:

# In[ ]:


train_df.iloc[0,1:][train_df.iloc[0, 1:]==1].index[0]


# While this is great, I prefer working in numpy as it can be faster, and the larger the dataset the more small times compact. 

# In[ ]:


get_ipython().run_cell_magic('timeit', '', '_ = train_df.iloc[0,1:][train_df.iloc[0, 1:]==1].index[0]')


# ## Working with NumPy
# 
# First let's convert our `DataFrame` to NumPy:

# In[ ]:


df_np = train_df.to_numpy()


# Now as we lost the particular column names here, let's find and replace our encoded `y`'s with their names. We can do this with a dictionary of their index:

# In[ ]:


index2name = {1:'healthy',
      2:'multiple_diseases',
      3:'rust',
      4:'scab'}


# In[ ]:


df_np[0]


# Now let's time how long it takes to do just what we did above. We'll use `np.where` to find the particular index in the row where the value is equal to 1, and then return the `index2name` of our label:

# In[ ]:


get_ipython().run_cell_magic('timeit', '', 'idx = np.where(df_np[1]==1)[0][0]\ny = index2name[idx]')


# *Considerably* faster here! Alright! Now let's begin to build some `DataLoaders` for us to use!

# ## DataBlock and DataLoaders
# 
# One nice benefit (and probably my favorite feature) of `fastai2` is the improved `DataBlock` API. Essentially we lay out a plan for us to use for our data and fit everything together!
# 
# For our particular classification problem, we'll want to use an `ImageBlock` and `CategoryBlock`, meaning that we have our inputs as images and outputs as categories:

# In[ ]:


dblock = DataBlock(blocks=(ImageBlock, CategoryBlock))


# ### `get_x`
# 
# Next we'll want to tell our `DataBlock` how to get our `x`'s (or our filenames). Our `x`'s are in that first index of our `NumPy` array, so let's make a `get_x` function. 
# 
# But first, how do we know what gets passed? We can make something like so:

# In[ ]:


def get_x(fn): print(fn)


# And pass this to our `DataBlock`. Let's see what it does:

# In[ ]:


dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                  get_x=get_x)


# We can call `dblock.summary()` here and pass in what we expect our inputs to be (our `NumPy` array) and see what it gives us:

# In[ ]:


dblock.summary(df_np)


# It'll return a very long stack trace error, I've pasted below what occurs before it to save on space:
# 
# ```
# Setting-up type transforms pipelines
# Collecting items from [['Train_0' 0 0 0 1]
#  ['Train_1' 0 1 0 0]
#  ['Train_2' 1 0 0 0]
#  ...
#  ['Train_1818' 1 0 0 0]
#  ['Train_1819' 0 0 1 0]
#  ['Train_1820' 0 0 0 1]]
# Found 1821 items
# 2 datasets of sizes 1457,364
# Setting up Pipeline: get_x -> PILBase.create
# ['Train_1422' 0 0 1 0]
# Setting up Pipeline: Categorize
# ```

# The part we're interested in is that `get_x` call we see there. We can see it prints out a row from our `NumPy` array. Now that we know this, let's build our proper `get_x`!
# 
# We'll want it to return a `Path` to our filename. We'll add a prefix and a suffix to it to allow so:

# In[ ]:


def get_x(row): return path/Path('images/'+row[0]+'.jpg')


# Now if we pass in a row from our array, we should get the respective `Path`:

# In[ ]:


get_x(df_np[0])


# And if we wanted to verify it works, we can call `PILImage.create` to try to open that `Path` with Pillow:

# In[ ]:


PILImage.create(get_x(df_np[0]))


# ### `get_y`
# 
# Now that we can grab our `x`'s, let's setup how to grab our `y`'s. It'll look very similar to our `get_x`, bringing in what we did earlier. Just for a reminder, here's how we grabbed a particular label based on a row:

# In[ ]:


idx = np.where(df_np[1]==1)[0][0]
y = index2name[idx]


# In[ ]:


def get_y(row):
    idx = np.where(row==1)[0][0]
    return index2name[idx]


# It should be just as simple as replacing `df_np[1]` with `row`. Let's try it out:

# In[ ]:


get_y(df_np[0])


# Perfect! Let's integrate both of these into our `DataBlock`:

# In[ ]:


dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                  get_x=get_x,
                  get_y=get_y)


# ## Splitting the Data and Augmentation

# ### Splitting
# 
# Next we need to tell `fastai2` how we want to split our data. We have a variety of options available to us. For our first example we'll split randomly and have 20% of the data as our validation set. Later we'll look at K-Fold Cross Validation and we'll want to use a set of index's instead.
# 
# `fastai2`'s split methods look something like so:

# In[ ]:


splitter = RandomSplitter(valid_pct=0.2, seed=42)


# So we can pass in a `valid_pct` as well as a seed for reproducability. 

# In[ ]:


splitter


# Wait, it's a function? Yes! Let's try passing to it a list of indexs, say 1-10 and see what happens:

# In[ ]:


idxs = list(range(1,11)); idxs


# In[ ]:


splitter(idxs)


# And we can see the first array is our training indexs and the second is our valid indexs. It looks strange, this is because it's an `L` type array, something custom for `fastai2`. For more on `L` read [here](http://fastcore.fast.ai/foundation#L)

# ### Augmentation
# 
# For our augmentations, we care about `item` and `batch` transforms. 
# 
# Item transforms are performed on the CPU and done on an individual image-by-image basis, and batch transforms are done on the GPU and done, well, on batches!
# 
# In general, we want our item transforms to simply prepare the data for getting into a batch, such as ensuring all images are the same size. Let's do that here. We'll set `item_tfms` to be a `RandomResizedCrop`, and our `batch_tfms` will have some random augmentation along with normalizing our data

# In[ ]:


item_tfms = RandomResizedCrop(224)
batch_tfms=[*aug_transforms(size=224, flip_vert=True),
                                   Normalize]


# Now one thing we may want to do is since we're training from scratch with `xresnet`, we'll want to calculate the normalization statistics which can get pretty heavy. Can `fastai2` help us with this? Yes! Let's see how!

# ## Calculating our Normalization Statistics
# 
# When we use `Normalize` as in our `batch_tfms`, we can either call it normally or we can use `from_stats()` and pass in some statistics to use. If we don't it will automatically calculate them based on the first batch of data. Let's see that in action:

# In[ ]:


dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                  get_x=get_x,
                  get_y=get_y,
                  splitter=splitter,
                  item_tfms=item_tfms,
                  batch_tfms=batch_tfms)


# To build our `DataLoaders` we call `dblock.dataloaders()` and pass in a batch size along with how `get_x` is expecting it's input:

# In[ ]:


dls = dblock.dataloaders(df_np, bs=64)


# To get the `Normalization` statistics we want to look inside the `after_batch` of our training dataloader like so:

# In[ ]:


dls.train.after_batch


# We can see that `Normalize` lives here. To get those statistics we simply grab the `mean` and `std`:

# In[ ]:


norm = dls.train.after_batch.normalize


# In[ ]:


norm.mean, norm.std


# So we can see that it calculated our statistics for us! Now how can we use that to our advantage? 
# 
# Let's make one big giant `DataLoader` that has all of our data. First we'll make a dataset of all of our data. We'll want to pass in `PILImage.create` and `Categorize` to our `tfms`. As you can see, the first array specifies our `x` and the second our `y`:

# In[ ]:


dset = Datasets(items=df_np, tfms=[[get_x, PILImage.create], [get_y, Categorize]])


# In[ ]:


dset[0]


# And now let's make it into a `DataLoader` by providing the `after_item` and `after_batch` from earlier, along with specifying our device as `cuda` so `Normalize` can work on the GPU. We'll use a special type of `DataLoader` called the `TfmdDL`:

# In[ ]:


dl = TfmdDL(dset, after_item=dls.train.after_item,
                     after_batch=[IntToFloatTensor(), Normalize()],
                     bs=64, device='cuda')


# We're not quite done yet, but almost! The last thing we want to do is make our batch size equal to the size of our dataset, this way the `Normalize` is done over the entire thing. Let's do that now:
# 
# * Note: This will take a bit to run, this is expected. Most of the time, simply running off of one batch of data has been found to be good enough (Sylvain and Jeremy found), but I wanted to show how it could be done

# In[ ]:


dl = TfmdDL(dset, after_item=dls.train.after_item,
                     after_batch=[IntToFloatTensor(), Normalize()],
                     bs=len(dset), device='cuda')


# And now let's take a look!

# In[ ]:


norm = dl.after_batch.normalize


# In[ ]:


norm.mean, norm.std


# As we can see, not *too* far off from what we got earlier off of one batch, but now we know the dataset's statistics! Let's store those away:

# In[ ]:


norm.mean.flatten()


# In[ ]:


plant_norm = (norm.mean.flatten(), norm.std.flatten()); plant_norm


# ## The Final DataLoaders
# 
# Now we have all the pieces in place, let's put them together! This is what our `DataBlock` looks like now, notice the change to `Normalize`:

# In[ ]:


item_tfms = RandomResizedCrop(224)
batch_tfms=[*aug_transforms(size=224, flip_vert=True),
                                   Normalize.from_stats(*plant_norm)]


# In[ ]:


dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                  get_x=get_x,
                  get_y=get_y,
                  splitter=splitter,
                  item_tfms=item_tfms,
                  batch_tfms=batch_tfms)


# And now we can make our `DataLoaders` and look at a batch!

# In[ ]:


dls = dblock.dataloaders(df_np, bs=32)


# In[ ]:


dls.show_batch()


# ## Training and `xresnet`
# 
# Now that we've got the hard part out of the way, let's get to training a model! For this notebook we'll be showing off how to use the `xresnet` (from the Bag Of Tricks paper) and all of the features you can include with it, along with utlizing the new `ranger` optimizer function and Flat Cosine Annealing fit function built into `fastai2`!

# ### `xresnet` and `ranger`
# 
# `xresnet` is built upong a variety of papers and is the result of the `ImageWoof`/`ImageNette` competitions [here](github.com/fastai/imagenette). Essentially a super difficult subset of the full ImageNet to allow for quick test of your ideas!
# 
# Currently, `xresnet` allows these customizations:
#   * Dropout
#   * Self-Attention
#   * Activation Class
#   
# For our model, we'll utilize the self-attention along with the Mish activation function. Let's see how to build our model:

# In[ ]:


net = xresnet50(pretrained=False, act_cls=Mish, sa=True, n_out=dls.c)


# Now let's get our optimizer function. `ranger` is a mix between the `LookAhead` optimizer along with `RAdam`. `fastai2` has a convience function to use for this, but first let's see what it actually looks like:

# In[ ]:


@delegates(RAdam)
def ranger(p, lr, mom=0.95, wd=0.01, eps=1e-6, **kwargs):
    "Convenience method for `Lookahead` with `RAdam`"
    return Lookahead(RAdam(p, lr=lr, mom=mom, wd=wd, eps=eps, **kwargs))


# Looks straight forward enough! Let's make our `opt_func`:

# In[ ]:


opt_func = ranger


# ### Training
# 
# Now we can start training. In general we'll perform the following steps:
# 
# 1. Build our `Learner` by passing in our `DataLoaders`, a model, a loss function, some metrics, and our optimizer
# 
# 2. Find a good learning rate via `lr_find()`
# 
# 3. Fit our model, in this case with `fit_flat_cos`
# 
# For our metrics, we'll use some `roc` scores:

# In[ ]:


from sklearn.metrics import roc_auc_score

def roc_auc(preds, targs, labels=range(4)):
    targs = np.eye(4)[targs]
    return np.mean([roc_auc_score(targs[:,i], preds[:,i]) for i in labels])

def healthy_roc_auc(*args):
    return roc_auc(*args, labels=[0])

def multiple_diseases_roc_auc(*args):
    return roc_auc(*args, labels=[1])

def rust_roc_auc(*args):
    return roc_auc(*args, labels=[2])

def scab_roc_auc(*args):
    return roc_auc(*args, labels=[3])


# To use custom metrics, we'll want to wrap them inside of an `AccumMetric` like so:

# In[ ]:


metric = partial(AccumMetric, flatten=False)


# In[ ]:


metrics=[
            error_rate,
            metric(healthy_roc_auc),
            metric(multiple_diseases_roc_auc),
            metric(rust_roc_auc),
            metric(scab_roc_auc),
            metric(roc_auc)]


# And now let's build our `Learner`!

# In[ ]:


learn = Learner(dls, net, opt_func=ranger, loss_func=CrossEntropyLossFlat(),
               metrics=metrics)


# Next we'll call `learn.lr_find()` to help us find a decent learning rate to use:

# In[ ]:


learn.lr_find()


# We'll probably want a learning rate of about 1e-3, so we'll try to use that. First we'll fit with `fit_flat_cos` and then we'll look at what's really going on:

# In[ ]:


learn.fit_flat_cos(5, 1e-3)


# Very impressive result in simply 5 epochs, ~73% accuracy. But why `fit_flat_cos`? Not the infamous `fit_one_cycle`?
# 
# Surprisingly, `ranger` does not like this schedule. Why? Let's compare the two.

# ## One Cycle vs Flat Cosine Annealing
# 
# 
# We can build a `synth_learner` to simulate training:

# In[ ]:


from fastai2.test_utils import synth_learner
one_cycle = synth_learner()
one_cycle.fit_one_cycle(1)


# And now we can plot that scheduler:

# In[ ]:


one_cycle.recorder.plot_sched()


# We can see we follow a cycle, the learning rate starts low, goes high, then goes low again. How about `fit_flat_cos`?

# In[ ]:


flat_cos = synth_learner()
flat_cos.fit_flat_cos(1)


# In[ ]:


flat_cos.recorder.plot_sched()


# *Very* different. What's going on? We train at a consistant learning rate before performing a Cosine Anneal at some `pct_start` value, which by default is 75% of the batches.The reason for this is `ranger` already includes the warm up phase that One Cycle had, so we can simply skip it all together. It was found that this trained *much* better when used. Now let's get back to training

# ## Training Cont.
# 
# Now let's resume training. We'll try fitting for 5 more epochs at a slightly lower learning rate, before calling it there:

# In[ ]:


learn.fit_flat_cos(5, 1e-4)


# As you can see, we got down to a little over 20% error, great! So what's next?

# ## Test-Time Augmentation
# 
# `fastai2` has the ability to perform what is known as Test-Time Augmentation. What that essentially means is we gather a number of sets of predictions from our model. One of which is simply just running through our validation set, the other `n` have the *training* augmentation applied to them. As the augmentations can have a degree of randomness to them, this is why we do it four times. In general, combining these four allows for a higher accuracy. Does that occur here as well? Let's find out!

# To use Test-Time Augmentation, simply call `learn.tta()`. You can then pass in a particular `DataLoader` you want to use, if not it will use the validation `DataLoader` by default (or `ds_idx=1`). When we explore Cross Validation we'll see how to perform inference on the test set:

# In[ ]:


preds, targs = learn.tta(ds_idx=1, n=4)


# `tta` will return the predictions and the targets for the data. If no targets are given (such as a test `DataLoader`), they will simply be blank.
# 
# Now let's try to calculate the `error_rate` of those values!

# In[ ]:


error_rate(preds, targs)


# We can see we improved our results a little bit! `tta` in general helps rather than hinders, so it is *always* worth a shot trying. Just remember you're always getting 4 times as many predictions when you do it! (by default)

# ## Cross Validation
# 
# Now that we've got all the building blocks, time to put our faith in the CV! But how do we set it up? Let's take a look. 
# 
# To utilize `StratifiedKFold` we're going to want to grab all of our labels from the dataset. We'll utilize our `Dataset` we made earlier:

# In[ ]:


from sklearn.model_selection import StratifiedKFold


# In[ ]:


train_lbls = []
for _, lbl in dset:
    train_lbls.append(lbl)


# To use our fold, we'll also want our images. This is as simple as:

# In[ ]:


imgs = df_np[:,0]


# We're going to write a few functions that allow us to perform Progressive Resizing along with Pre-Sizing as we train. This entails training a model at a low image size for a few epochs, then a higher, until finally the highest we want to train at. And Pre-Sizing is utilizing our RandomResizeCrop on a larger image before cropping it to a smaller one later. We'll also want our `DataBlock` to use an `IndexSplitter` now, as this is what our KFold will give us:

# In[ ]:


def get_data(idx, size=224, bs=64):
    dblock = DataBlock(blocks    = (ImageBlock, CategoryBlock),
                       get_x=get_x,
                       get_y=get_y,
                       splitter=IndexSplitter(idx),
                       item_tfms=RandomResizedCrop(size+64),
                       batch_tfms=[*aug_transforms(size=size, flip_vert=True),
                                   Normalize.from_stats(*plant_norm)],
                      )
    return dblock.dataloaders(train_df, bs=bs)


# The last thing we need is how to make our `test_dl` with our test data. First let's grab the test `DataFrame`:

# In[ ]:


test_df = pd.read_csv(path/'test.csv')


# Now we'll want to make this into a NumPy array so it's formatted similar to what we did earlyer

# In[ ]:


test_np = test_df.to_numpy()


# To make sure it works, we'll try to make a `test_dl` with our `DataLoaders` from earlier:

# In[ ]:


test_dl = learn.dls.test_dl(test_np)


# In[ ]:


test_dl.show_batch()


# Perfect! Now we have everything we need. Lets build our script. We'll include EarlyStopping too as we only want to use the best models during training:

# In[ ]:


import gc


# So let's go over what's happening in this script before we run it. We get the data at a size of 128x128 and train for five epochs, keeping track of the ROC AUC score. Then we save that model and clear our memory before training at 256, loading in that initial save, and training for ten epochs. Finally we train for five more at a size of 448. Then we perform our `tta` and add it to `test_preds`. Let's run it:

# In[ ]:


test_preds = []
skf = StratifiedKFold(n_splits=3, shuffle=True)
for _, val_idx in skf.split(imgs, np.array(train_lbls)):
    dls = get_data(val_idx, 128, 32)
    net = xresnet50(pretrained=False, act_cls=Mish, sa=True, n_out=dls.c)
    learn = Learner(dls, net, opt_func=ranger, loss_func=CrossEntropyLossFlat(),
               metrics=metrics)
    learn.fit_flat_cos(5, 4e-3, cbs=EarlyStoppingCallback(monitor='roc_auc'))
    learn.save('initial')
    del learn
    torch.cuda.empty_cache()
    gc.collect()
    
    dls = get_data(val_idx, 256, 16)
    learn = Learner(dls, net, opt_func=ranger, loss_func=CrossEntropyLossFlat(),
               metrics=metrics)
    learn.load('initial');
    learn.fit_flat_cos(10, 1e-3, cbs=EarlyStoppingCallback(monitor='roc_auc'))
    learn.save('stage-1')
    del learn
    torch.cuda.empty_cache()
    gc.collect()
    dls = get_data(val_idx, 448, 8)
    learn = Learner(dls, net, opt_func=ranger, loss_func=CrossEntropyLossFlat(),
               metrics=metrics)
    learn.fit_flat_cos(5, slice(1e-5, 1e-4), cbs=EarlyStoppingCallback(monitor='roc_auc'))
    tst_dl = learn.dls.test_dl(test_df)
    y, _ = learn.tta(dl=tst_dl)
    test_preds.append(y)
    del learn
    torch.cuda.empty_cache()
    gc.collect()


# ## Submitting Predictions
# 
# Great, we now have this set of 3 predictions, what do we do from there?
# 
# First we'll want to average all of those together to ensemble our model like so:

# In[ ]:


tot = test_preds[0]
for i in test_preds[1:]:
    tot += i


# We'll check our total number of predictions just to verify it is still 3:

# In[ ]:


len(test_preds)


# Now we can divide it by 3:

# In[ ]:


tot = tot / 3


# Finally we'll submit our predictions. To do so we do:

# In[ ]:


subm = pd.read_csv(path/'sample_submission.csv')


# In[ ]:


subm.iloc[:, 1:] = tot
subm.to_csv("submission.csv", index=False, float_format='%.2f')


# And we're done! I hope this helps you get more familiar with how to take `fastai2` to the next level. If you found this useful please don't forget to upvote, and ask away in the comments if you have any questions. Thanks!
