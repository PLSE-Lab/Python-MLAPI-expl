#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from fastai.text import *
from fastai.imports import *

np.random.seed(seed=1)


# In[ ]:


imdb_path = untar_data(URLs.IMDB)
imdb_path.ls()


# In[ ]:


imdb_lm = (TextList.from_folder(imdb_path)
                  .filter_by_folder(include=['test', 'train', 'unsup'])
                  .split_by_rand_pct(seed=1)
                  .label_for_lm()
                  .databunch(bs=32))

# imdb_lm.save(imdb_path/'imdb_databunch')


# In[ ]:


# imdb_lm = load_data(imdb_path, 'imdb_databunch', bs=32)
imdb_lm.show_batch()


# In[ ]:


len(imdb_lm.vocab.itos)


# In[ ]:


learn = language_model_learner(imdb_lm, AWD_LSTM)


# In[ ]:


learn.lr_find()
learn.recorder.plot(skip_end=20)


# In[ ]:


learn.fit_one_cycle(1, 1e-02, moms=(0.8, 0.6))


# In[ ]:


learn.recorder.plot_lr(show_moms=True)


# In[ ]:


learn.lr_find()
learn.recorder.plot(skip_end=20)


# In[ ]:


learn.fit_one_cycle(1, 1e-05, moms=(0.8, 0.6))


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot(skip_end=25)


# In[ ]:


learn.fit_one_cycle(1, 5e-03)


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(5, max_lr=slice(1e-05, 1e-03))


# In[ ]:


learn.save('fine_tuned')


# In[ ]:


learn.load('fine_tuned');


# In[ ]:


TEXT = "i liked this movie because"
N_WORDS = 40
N_SENTENCES = 3


# In[ ]:


print("\n\n".join(learn.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))


# In[ ]:


learn.save_encoder('fine_tuned_enc')


# In[ ]:


data_class = (TextList.from_folder(imdb_path, vocab=imdb_lm.vocab)
              .split_by_folder(valid='test')
              .label_from_folder(classes=['neg', 'pos'])
              .databunch(bs=32))


# In[ ]:


data_class.show_batch()


# In[ ]:


learn = text_classifier_learner(data_class, AWD_LSTM, drop_mult=0.3)
learn.load_encoder('fine_tuned_enc')
learn.freeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot(skip_end=10)


# In[ ]:


learn.fit_one_cycle(1, 1e-03)


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot(skip_end=20)


# In[ ]:


learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(1e-05, 1e-03), moms=(0.7, 0.5))


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(3, slice(1e-06, 1e-03))


# In[ ]:


learn.save('fine_tuned_classifier')


# In[ ]:


learn.load('fine_tuned_classifier')


# In[ ]:


learn.predict('I enjoyed the movie.')


# In[ ]:


learn.predict('The movie was worth watching')


# In[ ]:


learn.predict('The movie was waste')


# In[ ]:




