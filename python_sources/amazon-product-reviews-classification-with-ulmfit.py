#!/usr/bin/env python
# coding: utf-8

# # <center> Amazon product reviews classification with ULMFiT. Starter
# 
# Here we mostly follow the training scheme described by Jeremy Howard in [fast.ai Lesson 4](https://course.fast.ai/videos/?lesson=4): taking a pretrained language model, fine-tuning it with unlabeled data, then fine-tuning classification head for our particular task.
# 
# This is just a starter. At each step, I also mention how you can do better.

# In[ ]:


from tqdm import tqdm_notebook
import torch
import fastai
from fastai.text import *
fastai.__version__


# # Preprocessing
# Here we write all news texts from train, validation and text files into `unlabeled_news.csv` - to train a language model.
# 
# Then, we write texts and labels into `train_val.csv` and texts only into `test.csv`.
# 
# **How to do better:** go for that big unlabeled set as well.

# In[ ]:


train = pd.read_csv('../input/train.csv').fillna(' ')
valid = pd.read_csv('../input/valid.csv').fillna(' ')
test = pd.read_csv('../input/test.csv').fillna(' ')


# In[ ]:


pd.concat([train['text'], valid['text'], test['text']]).to_csv(
    'unlabeled_news.csv', index=None, header=True)


# In[ ]:


pd.concat([train[['text', 'label']],valid[['text', 'label']]]).to_csv(
    'train_val.csv', index=None, header=True)
test[['text']].to_csv('test.csv', index=None, header=True)


# In[ ]:


folder = '.'
unlabeled_file = 'unlabeled_news.csv'


# # Reading unlabeled data to train ULMFiT language model

# In[ ]:


get_ipython().run_cell_magic('time', '', "data_lm = TextLMDataBunch.from_csv(folder, unlabeled_file, text_cols='text')")


# # LM training 
# 
# Here we resort to the training scheme described by Jeremy Howard, [fast.ai](https://course.fast.ai/):
#  - finding good initial learning rate
#  - training for one epoch
#  - unfreezing and more training
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', 'learn = language_model_learner(data_lm, drop_mult=0.3, arch=AWD_LSTM)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'learn.lr_find(start_lr = slice(10e-7, 10e-5), end_lr=slice(0.1, 10))')


# In[ ]:


learn.recorder.plot(skip_end=10, suggestion=True)


# In[ ]:


best_lm_lr = 3e-3 #learn.recorder.min_grad_lr
# best_lm_lr


# In[ ]:


get_ipython().run_cell_magic('time', '', 'learn.fit_one_cycle(1, best_lm_lr)')


# In[ ]:


learn.unfreeze()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'learn.fit_one_cycle(5, best_lm_lr)')


# # Generating some text
# 
# It's always interesting to see whether a LM is able to generate nice text. With LM training improvement (in terms of loss), at some point you'll notice some nice improvement in quality of the generated text.
# 
# No much sense, but at least some structure :) And now with GPT-2 we see that quantitative improvements can also lead to qualital improvements.

# In[ ]:


learn.predict('I really liked this cat food because', n_words=200)


# In[ ]:


learn.save_encoder('amazon_reviews_enc')


# # Training classification head
# 
# Here again we follow Jeremy Howard. 
# 
# **How to do better:** hyperparam tuning (though it's extremely annoying with such a heavy model), more epochs after unfreezing, check for some live examples of ULMFiT training, different learning rates for different layers etc.

# In[ ]:


train_file, test_file = 'train_val.csv', 'test.csv'


# In[ ]:


data_clas = TextClasDataBunch.from_csv(path=folder, 
                                        csv_name=train_file,
                                        test=test_file,
                                        vocab=data_lm.train_ds.vocab, 
                                        bs=64,
                                        text_cols='text', 
                                        label_cols='label')


# In[ ]:


data_clas.save('ulmfit_data_clas_amazon_reviews')


# In[ ]:


learn_clas = text_classifier_learner(data_clas, drop_mult=0.3, arch=AWD_LSTM)
learn_clas.load_encoder('amazon_reviews_enc')


# In[ ]:


learn_clas.lr_find(start_lr=slice(10e-7, 10e-5), end_lr=slice(0.1, 10))


# In[ ]:


learn_clas.recorder.plot(skip_end=10, suggestion=True)


# In[ ]:


best_clf_lr = 3e-3 #learn_clas.recorder.min_grad_lr
# best_clf_lr


# In[ ]:


learn_clas.fit_one_cycle(1, best_clf_lr)


# In[ ]:


learn_clas.freeze_to(-2)


# In[ ]:


learn_clas.fit_one_cycle(1, best_clf_lr)


# In[ ]:


# learn_clas.unfreeze()


# In[ ]:


learn_clas.fit_one_cycle(5, best_clf_lr)


# In[ ]:


learn_clas.show_results()


# # Predictions for the test set
# 

# In[ ]:


data_clas.add_test(test["text"])


# In[ ]:


test_preds, _ = learn_clas.get_preds(DatasetType.Test, ordered=True)


# # Forming a submission file

# In[ ]:


test_pred_df = pd.DataFrame(test_preds.data.cpu().numpy(),
                            columns=['birds', 'bunny rabbit central', 'cats', 'dogs', 'fish aquatic pets', 'small animals'])
ulmfit_preds = pd.Series(np.argmax(test_pred_df.values, axis=1),
                        name='label').map({0: 'birds', 1: 'bunny rabbit central', 2: 'cats', 3: 'dogs', 4: 'fish aquatic pets', 5: 'small animals'})


# In[ ]:


ulmfit_preds.head()


# In[ ]:


ulmfit_preds.to_csv('ulmfit_predictions.csv', index_label='id', header=True)

