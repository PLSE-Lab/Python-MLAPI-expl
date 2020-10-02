#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai.text import *
import html
import json
from sklearn.model_selection import train_test_split


# In[ ]:


BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag

PATH=Path('/kaggle/input/bnwiki/lolol/lolol')


# In[ ]:


LM_PATH=Path('/temp')
LM_PATH.mkdir(exist_ok=True)


# In[ ]:


LANG_FILENAMES = [str(f) for f in PATH.rglob("*/*")]
print(len(LANG_FILENAMES))
LANG_FILENAMES[0:5]


# In[ ]:


LANG_TEXT = []
for i in LANG_FILENAMES:
    for line in open(i):
        LANG_TEXT.append(json.loads(line))
        
LANG_TEXT = pd.DataFrame(LANG_TEXT)


# In[ ]:


LANG_TEXT.head(10)


# In[ ]:


LANG_TEXT.to_csv(f"{LM_PATH}/wiki_bangla_corpus.csv", index=False)


# In[ ]:


LANG_TEXT = pd.read_csv(f"{LM_PATH}/wiki_bangla_corpus.csv")


# In[ ]:


data_lm = TextLMDataBunch.from_csv(LM_PATH,'wiki_bangla_corpus.csv')


# In[ ]:


learner=language_model_learner(data_lm,TransformerXL,pretrained=False,metrics=accuracy)


# In[ ]:


learner.load('/kaggle/input/lmtest/gen1')


# In[ ]:


learner.lr_find()


# In[ ]:


learner.recorder.plot()


# In[ ]:


lr=1e-2


# In[ ]:


learner.fit_one_cycle(15,lr)


# In[ ]:


learner.save('/kaggle/working/gen2',return_path=True)


# In[ ]:


learner.recorder.plot_losses()


# In[ ]:


learner.recorder.plot_metrics()

