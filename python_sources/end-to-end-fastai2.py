#!/usr/bin/env python
# coding: utf-8

# # End to End Toxic Comments in fastai2

# In this notebook we're going to explore end-to-end training of the Toxic Comments multi-label dataset using fastai2 and the high-level DataBlock API. First let's install `fastai2`:

# In[ ]:


get_ipython().system('pip install fastai2 --quiet')


# Since our task is a text problem, let's grab the `fastai2` text sub-library:

# In[ ]:


from fastai2.text.all import *


# In[ ]:


path = Path('../input/jigsaw-toxic-comment-classification-challenge')


# To work with the data we need to unzip everything:

# In[ ]:


from zipfile import ZipFile

with ZipFile(path/'train.csv.zip', 'r') as zip_ref:
    zip_ref.extractall('../output/kaggle/working')
    
with ZipFile(path/'test.csv.zip', 'r') as zip_ref:
    zip_ref.extractall('../output/kaggle/working')
    
with ZipFile(path/'test_labels.csv.zip', 'r') as zip_ref:
    zip_ref.extractall('../output/kaggle/working')
    
with ZipFile(path/'sample_submission.csv.zip', 'r') as zip_ref:
    zip_ref.extractall('../output/kaggle/working')


# In[ ]:


path_w = Path('../output/kaggle/working')


# In[ ]:


path_w.ls()


# Now let's see our training data:

# In[ ]:


df = pd.read_csv(path_w/'train.csv')


# In[ ]:


df.head()


# So we have our `comment_text` and a one-hot-encoded `y-label` for multi-class classification. Let's look at how to build a `DataBlock` for this to use in the `fastai2` framework. First let's look at the `TextBlock`. This will be where we dictate what kind of tokenizer we use, if it's a language model, and so forth:

# In[ ]:


blocks = (TextBlock.from_df(text_cols='comment_text', is_lm=True, res_col_name='text'))


# Now before we actually do this, one trick is to train the langauge model on as much data as we possibly can, as it's unlabelled. We'll make another `DataFrame` for this specifically:

# In[ ]:


test_df = pd.read_csv(path_w/'test.csv')


# In[ ]:


test_df.head()


# In[ ]:


text_df = pd.Series.append(df['comment_text'], test_df['comment_text'])


# In[ ]:


text_df = pd.DataFrame(text_df)


# In[ ]:


text_df.head()


# Now you may notice there was a `res_col_name` parameter. This is where our *tokenized* text will be output to. Next we need to tell how to grab our `x` and how we want to split our data. We'll split by 10% randomly:

# In[ ]:


get_x = ColReader('text')
splitter = RandomSplitter(0.1, seed=42)


# Note that this `get_x` should be the same as our output column

# Now let's build the `DataBlock` Pipeline:

# In[ ]:


lm_dblock = DataBlock(blocks=blocks,
                     get_x=get_x,
                     splitter=splitter)


# And now the `DataLoaders`:

# In[ ]:


lm_dls = lm_dblock.dataloaders(text_df, bs=64)


# It will take abit to pre-process everything, `fastai` will tokenize for us beforehand (should take about 14 minutes). 
# 
# Next, as per the ULM-FiT technique, we'll want to train our langauge model. First let's build our `Learner`:

# In[ ]:


lm_learn = language_model_learner(lm_dls, AWD_LSTM, pretrained=True, metrics=[accuracy, Perplexity()])


# `fastai` has a nice in-house `fit` called `fine_tune`, which follows the freezing/unfreezing transfer-learning protocol. We can pass in the number of frozen epochs and unfrozen, however Jeremy and Sylvain found that one was enough, so we'll pass in the number of *unfrozen* epochs as well as a learning rate:

# In[ ]:


lm_learn.to_fp16()
lm_learn.fine_tune(10, 4e-3)


# In[ ]:


lm_learn.save_encoder('fine_tuned')


# Now that we have our pre-trained model, let's build our down-stream multi-label classification task:

# ## Toxic Comment Classification

# For our next part we'll want to make a `DataBlock` that uses the original vocab and sets us up for a multi-label classification problem:

# In[ ]:


ys = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult',
       'identity_hate']


# In[ ]:


blocks = (TextBlock.from_df('comment_text', seq_len=lm_dls.seq_len, vocab=lm_dls.vocab), 
          MultiCategoryBlock(encoded=True, vocab=ys))


# In[ ]:


toxic_clas = DataBlock(blocks=blocks,
                      get_x=ColReader('text'),
                      get_y=ColReader(ys),
                      splitter=RandomSplitter())


# We can test if it works by calling `toxic_clas.summary()`:

# In[ ]:


toxic_clas.summary(df.iloc[:100])


# Which it works just fine! So let's build our `DataLoaders`:

# In[ ]:


dls = toxic_clas.dataloaders(df)


# Next we'll make our `text_classification_learner`:

# Now this is where things get a bit tricky, as we want to know how to build our thresholds for `BCELossLogits` and `accuracy_multi` (as ideally we'd want them both to be the same). To make sure my model is very strong, I'll set their thresholds to activations > 0.8:

# In[ ]:


loss_func = BCEWithLogitsLossFlat(thresh=0.8)
metrics = [partial(accuracy_multi, thresh=0.8)]


# In[ ]:


learn = text_classifier_learner(dls, AWD_LSTM, metrics=metrics, loss_func=loss_func)


# And then we just find the learning rate and fit!

# In[ ]:


learn.lr_find()


# We'll use the unfreezing methodology for the ULM-FiT model to train and for its learning rate:

# In[ ]:


learn.load_encoder('fine_tuned');


# In[ ]:


learn.to_fp16()

lr = 1e-2
moms = (0.8,0.7, 0.8)
lr *= learn.dls.bs/128
learn.fit_one_cycle(1, lr, moms=moms, wd=0.1)


# In[ ]:


learn.freeze_to(-2)
lr/=2
learn.fit_one_cycle(1, slice(lr/(2.6**4), lr), moms=moms, wd=0.1)


# In[ ]:


learn.freeze_to(-3)
lr /=2
learn.fit_one_cycle(1, slice(lr/(2.6**4), lr), moms=moms, wd=0.1)


# In[ ]:


learn.unfreeze()
lr /= 5
learn.fit_one_cycle(2, slice(lr/(2.6**4),lr), moms=(0.8,0.7,0.8), wd=0.1)


# ## Submitting the Predictions:

# Now let's try to submit some predictions. To make a test set we should call `learn.dls.test_dl` and pass in our `DataFrame` with the text to use:

# In[ ]:


dl = learn.dls.test_dl(test_df['comment_text'])


# For getting the predictions, we call `learn.get_preds` and pass in this `DataLoader`:

# In[ ]:


preds = learn.get_preds(dl=dl)


# Now let's see how our sample submission wants it?

# In[ ]:


sub = pd.read_csv(path_w/'sample_submission.csv')


# In[ ]:


sub.head()


# Easy enough, let's push those predictions to it. They come in the same order we passed them in as:

# In[ ]:


preds[0][0].cpu().numpy()


# In[ ]:


sub[ys] = preds[0]


# In[ ]:


sub.head()


# And now we can submit it!

# In[ ]:


sub.to_csv('submission.csv', index=False)

