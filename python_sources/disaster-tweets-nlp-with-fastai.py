#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai.text import *


# # Importing data

# In[ ]:


path=Path('/kaggle/input/nlp-getting-started/')
train_df=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv',index_col='id')
train_df.head(5)


# # Prep data without target/ API Block

# In[ ]:


cols=['text']
data_bunch = (TextList.from_df(train_df, cols=cols)
                .split_by_rand_pct(0.2)
                .label_for_lm()  
                .databunch(bs=48))
data_bunch.show_batch()


# # Make learner and train on data

# In[ ]:


learn = language_model_learner(data_bunch,
                               AWD_LSTM,
                               pretrained_fnames=['/kaggle/input/wt103-fastai-nlp/lstm_fwd','/kaggle/input/wt103-fastai-nlp/itos_wt103'],
                               pretrained=True,
                               drop_mult=0.5)


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(8, 1e-2)


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(2, slice(1e-05, 1e-03))


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(2, slice(1e-06, 1e-03))


# In[ ]:


learn.freeze()


# In[ ]:


learn.fit_one_cycle(1, slice(1e-04, 1e-03))


# In[ ]:


learn.save_encoder('fine_tuned_enc')


# # Make data prep with API Block
# we are now using target to make a prediction on the tweets

# In[ ]:


train=train_df[:8000]
val=train_df[2000:]
test=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv',index_col='id')


# In[ ]:


target_cols=['target']
data_clas = TextClasDataBunch.from_df('.', train, val, test,
                  vocab=data_bunch.vocab,
                  text_cols=cols,
                  label_cols=target_cols,
                  bs=32)


# # Create classifier learner and train with target

# In[ ]:


learn_classifier = text_classifier_learner(data_clas, 
                                           AWD_LSTM,
                                           pretrained=False,
                                           drop_mult=0.8,
                                           metrics=[accuracy])


fnames = ['/kaggle/input/wt103-fastai-nlp/lstm_fwd.pth','/kaggle/input/wt103-fastai-nlp/itos_wt103.pkl']
learn_classifier.load_pretrained(*fnames, strict=False)



# load the trained model without target from encoder saved  
learn_classifier.load_encoder('fine_tuned_enc')


# In[ ]:


learn_classifier.lr_find()
learn_classifier.recorder.plot()


# In[ ]:


learn_classifier.fit_one_cycle(4, 1e-3)


# In[ ]:


learn_classifier.fit_one_cycle(2, 1e-05)


# In[ ]:


learn_classifier.fit_one_cycle(7, slice(1e-06, 1e-03))


# # Make prediction from test set

# In[ ]:


preds_test, target_test = learn_classifier.get_preds(DatasetType.Test, ordered=True)
y = torch.argmax(preds_test, dim=1)
y.numpy().shape


# # Submission 

# In[ ]:


submission=pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
print(submission.shape)


# In[ ]:


submission['target']=y.numpy()
submission.head()


# In[ ]:


submission['target'].value_counts()


# In[ ]:


submission.to_csv('submission.csv',index=False)
print('Model ready for submission!')

