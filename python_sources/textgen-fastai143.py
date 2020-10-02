#!/usr/bin/env python
# coding: utf-8

# Note: this kernel is based on fastai 1.043dev0  to test the newly implemented beam search in fastai. Attempt here is to generate movie reviews. You can find more about IMDB dataset here: https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson3-imdb.ipynb 

# In[ ]:


get_ipython().system('pip uninstall -y fastai')


# In[ ]:


get_ipython().system('pip install git+https://github.com/fastai/fastai.git')


# In[ ]:


# show install info
#import fastai.utils.collect_env
#fastai.utils.collect_env.show_install()


# In[ ]:


from fastai.text import *


# In[ ]:


path = untar_data(URLs.IMDB)
path.ls()


# In[ ]:


bs=64
data_lm = (TextList.from_folder(path)
           #Inputs: all the text files in path
            .filter_by_folder(include=['train', 'test', 'unsup']) 
           #We may have other temp folders that contain text files so we only keep what's in train and test
            .random_split_by_pct(0.1)
           #We randomly split and keep 10% (10,000 reviews) for validation
            .label_for_lm()           
           #We want to do a language model so we label accordingly
            .databunch(bs=bs))
data_lm.save('tmp_lm')


# In[ ]:


data_lm = TextLMDataBunch.load(path, 'tmp_lm', bs=bs)
data_lm.show_batch()


# In[ ]:


# drop_mult is a parameter that controls the % of drop-out used
learn = language_model_learner(data_lm, AWD_LSTM, pretrained=True, drop_mult=0.30)


# In[ ]:


learn.lr_find()
learn.recorder.plot(skip_end=12)


# In[ ]:


learn.fit_one_cycle(1, 1e-2, moms=(0.8,0.7))


# In[ ]:


learn.save('tweet_head')
learn.load('tweet_head');


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(4, 1e-3, moms=(0.8,0.7))


# In[ ]:


learn.save('tweet_fine_tuned')
learn.load('tweet_fine_tuned');


# In[ ]:


TEXTS = ["xxbos","the","this","when","i really", "you can","if", "i was", "what"]
N_WORDS = 100 


# In[ ]:


print("\n".join(str(i+1) + ". " + learn.predict(TEXTS[i], N_WORDS,no_unk=True, temperature=0.85) for i in range(len(TEXTS))))


# In[ ]:


print("\n".join(str(i+1) + ". " + (learn.beam_search(TEXTS[i], N_WORDS, temperature=0.85, top_k=6,beam_sz=20)).replace('Xxunk','').replace('xxunk','') for i in range(len(TEXTS))))


# In[ ]:




