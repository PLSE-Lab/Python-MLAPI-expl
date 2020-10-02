#!/usr/bin/env python
# coding: utf-8

# # fastai MultiLabel Classification using Kfold Cross Validation

# The problem I have considered is Multi Label classification. In addition to having multiple labels in each image, the other challenge in this problem is the existence of rare classes and combinations of different classes. So in this situation normal split or random split doesnt work because you can end up putting rare cases in the validation set and your model will never learn about them. The stratification present in the scikit-learn is also not equipped to deal with multilabel targets. 

# I have specifically choosen this problem because we may learn some techniques on the way, which we otherwise would not have thought of.
# 
# **There may be better or easy way of doing kfold cross validation but I have done it keeping in mind how to implement using fastai**, so if you know some better way so please mail or tweet the idea, i will try to implement and give you credit.

# ## Install all the necessary libraries
# 
# I am using fastai2 so import that. 
# 

# In[ ]:


get_ipython().system('pip install -q fastai2')


# ### Cross Validation

# Cross-validation, how I see it, is the idea of minimizing randomness from one split by makings n folds, each fold containing train and validation splits. You train the model on each fold, so you have n models. Then you take average predictions from all models, which supposedly give us more confidence in results.
# These we will see in following code. I found iterative-stratification package that provides scikit-learn compatible cross validators with stratification for multilabel data.

# **My opinion**: 
# 
# ---
# 
# In my opinion it's more important to make one right split, especially because CV takes n times more to train. Then why did I do it??
# 
# I wanted to explore classification using cross validation using fastai, which I didn't find many resources to learn. So if I write this blog it may help people.
# 
# fastai has no cross validation split(may be) in their library to work like other functions they provide. It may be because cross validation takes time, so may be it not that useful.
# 
# But still in this condition I feel its worth exploring using fastai.
# 
# 
# 
# 
# 
# 
# 
# 

# so what is **stratification**??
# 
# The splitting of data into folds may be governed by criteria such as ensuring that each fold has the same proportion of observations with a given categorical value, such as the class outcome value. This is called stratified cross-validation

# In[ ]:


get_ipython().system('pip install -q iterative-stratification')


# In[ ]:


from fastai2.vision.all import *
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


# Here dataset is of Zero to GANs - Human Protein Classification inclass jovian.ml hosted competition

# In[ ]:


path = Path('../input/jovian-pytorch-z2g/Human protein atlas')

train_df = pd.read_csv(path/'train.csv')

train_df['Image'] = train_df['Image'].apply(str) + ".png"

train_df['Image'] = "../input/jovian-pytorch-z2g/Human protein atlas/train/" + train_df['Image']

train_df.head()


# The method I use here is if we have column called fold and with fold number it would be helpfull to split data using that.
# 
# fastai has IndexSplitter in datablock api so this would be helpful.
# 
# 

# In[ ]:


strat_kfold = MultilabelStratifiedKFold(n_splits=3, random_state=42, shuffle=True)
train_df['fold'] = -1
for i, (_, test_index) in enumerate(strat_kfold.split(train_df.Image.values, train_df.iloc[:,1:].values)):
    train_df.iloc[test_index, -1] = i
train_df.head()


# In[ ]:


train_df.fold.value_counts().plot.bar();


# ## DataBlock 
# 
# now that data is in dataframe and also folds are also defined for cross validation, we will build dataloaders, for which we will use datablock.
# 
# If you want to learn how fastai datablock see my blog series [Make code Simple with DataBlock api](https://kirankamath.netlify.app/blog/fastais-datablock-api/)

# we will create a function get_data to create dataloader.
# 
# get_data uses fold to split data to be used for cross validation using IndexSplitter. 
# for multiLabel problem compared to single only extra thing to be done is to add MultiCategoryBlock in blocks, this is how fastai makes it easy to work.

# In[ ]:


def get_data(fold=0, size=224,bs=32):
    return DataBlock(blocks=(ImageBlock,MultiCategoryBlock),
                       get_x=ColReader(0),
                       get_y=ColReader(1, label_delim=' '),
                       splitter=IndexSplitter(train_df[train_df.fold == fold].index),
                       item_tfms=[FlipItem(p=0.5),Resize(512,method='pad')],
                   batch_tfms=[*aug_transforms(size=size,do_flip=True, flip_vert=True, max_rotate=180.0, max_lighting=0.6,max_warp=0.1, p_affine=0.75, p_lighting=0.75,xtra_tfms=[RandomErasing(p=0.5,sh=0.1, min_aspect=0.2,max_count=2)]),Normalize],
                      ).dataloaders(train_df, bs=bs)


# ## metrics

# Since this is multi label problem normal accuracy function wont work, so we have accuracy_multi. fastai has this which we can directly use in metrics but I wanted to know how that works so took code of it.

# In[ ]:


def accuracy_multi(inp, targ, thresh=0.5, sigmoid=True):
    "Compute accuracy when `inp` and `targ` are the same size."
    if sigmoid: inp = inp.sigmoid()
    return ((inp>thresh)==targ.bool()).float().mean()


# F_score is way of evaluation for this competition so used this.

# In[ ]:


def F_score(output, label, threshold=0.2, beta=1):
    prob = output > threshold
    label = label > threshold

    TP = (prob & label).sum(1).float()
    TN = ((~prob) & (~label)).sum(1).float()
    FP = (prob & (~label)).sum(1).float()
    FN = ((~prob) & label).sum(1).float()

    precision = torch.mean(TP / (TP + FP + 1e-12))
    recall = torch.mean(TP / (TP + FN + 1e-12))
    F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
    return F2.mean(0)


# ## Gathering test set

# In[ ]:


test_df = pd.read_csv('../input/jovian-pytorch-z2g/submission.csv')
tstpng = test_df.copy()
tstpng['Image'] = tstpng['Image'].apply(str) + ".png"
tstpng['Image'] = "../input/jovian-pytorch-z2g/Human protein atlas/test/" + tstpng['Image']
tstpng.head()


# ## Training

# I have used technique called mixup, its a data augmentation technique. 
# 
# In fastai Mixup is callback, and
# this Callback is used to apply MixUp data augmentation to your training.
# to know more read [this](http://dev.fast.ai/callback.mixup)

# I have tried this first time, but this technique didnot improve my result in this problem. It usually improves accuracy after 80 epochs but I have trained for 20 epoches. so there was no difference in accuracy without it. so you can ignore this. 
# 
# But to know about how mixup works is good, I will separate blog on this, so follow my twitter for updates.

# In[ ]:


mixup = MixUp(0.3)


# gc is for garbage collection

# In[ ]:


import gc


# I have created 3 folds where I simply get the data from a particular fold, create a model, add metrics, I have used resnet34.
# And that's the whole training process. I just trained model on each fold and saved predictions for the test set.

# I have used a technique called progressive resizing. 
# 
# this is very simple: start training using small images, and end training using large images. Spending most of the epochs training with small images, helps training complete much faster. Completing training using large images makes the final accuracy much higher. this approach is called progressive resizing.
# 
# we should use the `fine_tune` method after we resize our images to get our model to learn to do something a little bit different from what it has learned to do before. 

# I have used `cbs=EarlyStoppingCallback(monitor='valid_loss')` so that model doesnot overfit.

# append all prediction to list so that we use it later.
# 
# I have run the model for less epochs to see code works and show result, or stopped model in between(it took so much time)
# 
# This method gave me F_score of `.77` and accuracy of `>91%` so you can try.
# 
# My Purpose here is to write blog and explain how to approach and how code works.

# If GPU is out of memory delete learner and empty cuda cache done in last line of code.

# In[ ]:


all_preds = []

for i in range(3):
    dls = get_data(i,256,64)
    learn = cnn_learner(dls, resnet34, metrics=[partial(accuracy_multi, thresh=0.2),partial(F_score, threshold=0.2)],cbs=mixup).to_fp16()
    learn.fit_one_cycle(10, cbs=EarlyStoppingCallback(monitor='valid_loss'))
    learn.dls = get_data(i,512,32)
    learn.fine_tune(10,cbs=EarlyStoppingCallback(monitor='valid_loss'))
    tst_dl = learn.dls.test_dl(tstpng)
    preds, _ = learn.get_preds(dl=tst_dl)
    all_preds.append(preds)
    del learn
    torch.cuda.empty_cache()
    gc.collect()


# stack all the prediction stored in list and average the values.

# In[ ]:


subm = pd.read_csv("../input/jovian-pytorch-z2g/submission.csv")
preds = np.mean(np.stack(all_preds), axis=0)


# You should have list of labels which we get using vocab.

# In[ ]:


k = dls.vocab


# In[ ]:


preds[0]


# I found threshold of 0.2 works good for my code.
# 
# then all the labels predicted above 0.2 are labels of that image using vocab. 

# In[ ]:


thresh=0.2
labelled_preds = [' '.join([k[i] for i,p in enumerate(pred) if p > thresh]) for pred in preds]


# put them in Labels column

# In[ ]:


test_df['Label']=labelled_preds


# this step is to submit result to kaggle.

# In[ ]:


test_df.to_csv('submission.csv',index=False)

