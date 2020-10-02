#!/usr/bin/env python
# coding: utf-8

# https://docs.fast.ai/text.html

# In[ ]:


from fastai import *
from fastai.text import * 
from fastai.gen_doc.nbdoc import *
from fastai.datasets import * 
from fastai.datasets import Config
from pathlib import Path
import pandas as pd


# In[ ]:


path = Path('../input/')


# In[ ]:


df_test = pd.read_csv(path/'test.csv')
df_test.head()


# In[ ]:


df = pd.read_csv(path/'train.csv')#.sample(frac=0.05)
df.head()


# In[ ]:


data_lm = (TextList.from_csv(path, 'train.csv', cols='question_text')
            .random_split_by_pct(.2)
            .label_for_lm()
            .add_test(TextList.from_csv(path, 'test.csv', cols='question_text'))
            .databunch())


# In[ ]:


data_clas = (TextList.from_csv(path, 'train.csv', cols='question_text', vocab=data_lm.vocab)
            #.split_from_df(col='target')
             .random_split_by_pct(.2)
            .label_from_df(cols='target')
            #.add_test(TextList.from_csv(path, 'test.csv', cols='question_text'))
            .databunch(bs=32))


# In[ ]:


data_clas.show_batch()


# In[ ]:


MODEL_PATH = "/tmp/model/"


# In[ ]:


learn = text_classifier_learner(data_clas,model_dir=MODEL_PATH)
#learn.load_encoder('mini_train_encoder')
learn.fit_one_cycle(2, slice(1e-3,1e-2))
#learn.save('mini_train_clas')


# In[ ]:


learn.show_results()


# In[ ]:


# Language model data
#data_lm = TextLMDataBunch.from_csv(path, 'train.csv', cols='question_text')
# Classifier model data
#data_clas = TextClasDataBunch.from_csv(path, 'train.csv', vocab=data_lm.train_ds.vocab, bs=4) #32

#data_lm = TextLMDataBunch.from_df(path, df)
#data_clas = TextClasDataBunch.from_df(path, df,  vocab=data_lm.train_ds.vocab)


# In[ ]:


data_lm.show_batch()


# In[ ]:


#??language_model_learner()


# In[ ]:


learn = language_model_learner(data_lm, drop_mult=0.5, model_dir=MODEL_PATH) #bptt=65, emb_sz=400, pretrained_model=URLs.WT103


# In[ ]:


acc_02 = partial(accuracy_thresh, thresh=0.2)
f_score = partial(fbeta, thresh=0.2)

learn.metrics = metrics=[accuracy,acc_02, f_score]


# In[ ]:


learn.fit_one_cycle(1, 1e-2)


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(1, 1e-3)


# In[ ]:


learn.predict("This is a review about", n_words=10)


# In[ ]:


learn.save_encoder('ft_enc')


# In[ ]:


learn = text_classifier_learner(data_clas, drop_mult=0.5)
learn.load_encoder('ft_enc')


# In[ ]:


data_clas.show_batch()


# In[ ]:


learn.fit_one_cycle(1, 1e-2)


# In[ ]:


learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(5e-3/2., 5e-3))


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(1, slice(2e-3/100, 2e-3))


# In[ ]:


learn.predict("This was a great movie!")


# In[ ]:


learn.show_results()


# In[ ]:


probs, _ = learn.get_preds(DatasetType.Test)


# In[ ]:


preds = np.argmax(probs, axis=1)


# In[ ]:


ids = df_test["qid"].copy()


# In[ ]:


submission = pd.DataFrame(data={
    "qid": ids,
    "prediction": preds
})
submission.to_csv("submission.csv", index=False)
submission.head(n=50)

