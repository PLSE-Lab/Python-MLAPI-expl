#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[ ]:


from fastai import *
from fastai.text import *

from scipy.stats import spearmanr


# In[ ]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

SEED = 999
seed_everything(SEED)


# ### Loading data

# In[ ]:


data_dir = '../input/google-quest-challenge/'
get_ipython().system('ls {data_dir}')


# In[ ]:


train_raw = pd.read_csv(f'{data_dir}train.csv')
train_raw.head()


# In[ ]:


test_raw = pd.read_csv(f'{data_dir}test.csv')
test_raw.head()


# In[ ]:


train_raw.shape, test_raw.shape


# In[ ]:


# randomly shuffle our data
train_raw = train_raw.sample(frac=1, random_state=1).reset_index(drop=True)


# ### Train-validation split
# 
# The train data has only `6079` records. Let's use `1000` records for validation purpose and use rest of the data for training our model.

# In[ ]:


val_count = 1000
trn_count = train_raw.shape[0] - val_count

df_val = train_raw[:val_count]
df_trn = train_raw[val_count:val_count+trn_count]


# ### Pre-trained AWD_LSTM weights
# 
# In order for the "Submit to Competition" button to be active, internet must be turned off. So we will get AWD_LSTM pre-trained weights from kaggle dataset.
# 
# First let's create directory structure for the weight files -

# In[ ]:


get_ipython().system('mkdir -p ~/.fastai/models/wt103-fwd')


# Now let's copy the files to the above directory -

# In[ ]:


get_ipython().system('cp ../input/wt103-fastai-nlp/lstm_fwd.pth ~/.fastai/models/wt103-fwd/')
get_ipython().system('cp ../input/wt103-fastai-nlp/itos_wt103.pkl ~/.fastai/models/wt103-fwd/')


# ### Fine-tuning language model

# Our dataset contains 30 target labels -

# In[ ]:


target_cols = train_raw.columns.tolist()[-30:]


# We will use `question_body` and `answer` columns for training our model.
# 
# First we create a databunch object for our language model -

# In[ ]:


data_lm = TextLMDataBunch.from_df('.', df_trn, df_val, test_raw,
                  include_bos=False,
                  include_eos=False,
                  text_cols=['question_body', 'answer'],
                  label_cols=target_cols,
                  bs=32,
                  mark_fields=True,
                  collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),
             )


# In[ ]:


data_lm.show_batch()


# Now let's fine-tune the pre-trained language model -

# In[ ]:


learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)
learn.fit_one_cycle(5, 1e-2)


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(5, 1e-3)


# We save the encoder to be able to use it for classification taks.

# In[ ]:


learn.save_encoder('ft_enc')


# ### Building classifier

# In[ ]:


data_cls = TextClasDataBunch.from_df('.', df_trn, df_val, test_raw,
                  include_bos=False,
                  include_eos=False,
                  text_cols=['question_body', 'answer'],
                  label_cols=target_cols,
                  bs=32,
                  mark_fields=True,
                  collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),
             )


# In[ ]:


data_cls.show_batch()


# Now we create a text classifier learner and load the encoder we previously saved -

# In[ ]:


learn = text_classifier_learner(data_cls, AWD_LSTM, drop_mult=0.5)
learn.load_encoder('ft_enc');


# Finally we train our classification model -

# In[ ]:


learn.fit_one_cycle(7, 1e-2)


# In[ ]:


learn.freeze_to(-2)
learn.fit_one_cycle(3, 1e-3)


# In[ ]:


learn.freeze_to(-3)
learn.fit_one_cycle(5, 1e-3)


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(3, 1e-3)


# ### Validation score
# 
# I noticed the `learn.get_preds()` method was returning predictions in random order. So let's create a helper function to get the predictions in correct order.

# In[ ]:


def get_ordered_preds(learn, ds_type, preds):
  np.random.seed(42)
  sampler = [i for i in learn.data.dl(ds_type).sampler]
  reverse_sampler = np.argsort(sampler)
  preds = [p[reverse_sampler] for p in preds]
  return preds


# In[ ]:


val_raw_preds = learn.get_preds(ds_type=DatasetType.Valid)
val_preds = get_ordered_preds(learn, DatasetType.Valid, val_raw_preds)


# Submissions are evaluated on the mean column-wise Spearman's correlation coefficient. Let's see how our model performs on validation data -

# In[ ]:


score = 0
for i in range(30):
    score += np.nan_to_num(spearmanr(df_val[target_cols].values[:, i], val_preds[0][:, i]).correlation) / 30
score


# ### Test predictions

# In[ ]:


test_raw_preds = learn.get_preds(ds_type=DatasetType.Test)
test_preds = get_ordered_preds(learn, DatasetType.Test, test_raw_preds)


# ### Submission file

# In[ ]:


sample_submission = pd.read_csv(f'{data_dir}sample_submission.csv')
sample_submission.head()

sample_submission.iloc[:, 1:] = test_preds[0].numpy()
sample_submission.to_csv('submission.csv', index=False)
sample_submission.head()


# In[ ]:




