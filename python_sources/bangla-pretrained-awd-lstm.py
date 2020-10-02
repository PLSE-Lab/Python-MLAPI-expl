#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai.text import *
import html
import json
from sklearn.model_selection import train_test_split

BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag

PATH=Path('/kaggle/input/lolol/lolol')


# In[ ]:


LM_PATH=Path('/temp')
LM_PATH.mkdir(exist_ok=True)

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


LANG_TEXT.to_csv(f"{LM_PATH}/wiki_bangla_corpus.csv", index=False)
LANG_TEXT = pd.read_csv(f"{LM_PATH}/wiki_bangla_corpus.csv")


# In[ ]:


data_lm = TextLMDataBunch.from_csv(LM_PATH,'wiki_bangla_corpus.csv',text_cols='text')


# In[ ]:


data_lm.show_batch()


# In[ ]:


learner=language_model_learner(data_lm,AWD_LSTM,pretrained=False,metrics=accuracy)


# In[ ]:


learner.lr_find()


# In[ ]:


learner.recorder.plot()


# In[ ]:


learner.fit_one_cycle(15,2e-2)


# In[ ]:


learner.recorder.plot_losses()


# In[ ]:


learner.recorder.plot_metrics()


# In[ ]:


learner.save('/kaggle/working/gen1')


# In[ ]:


learner.save_encoder('/kaggle/working/gen1enc')


# In[ ]:


data_lm.save('/kaggle/working/data.pkl')


# In[ ]:


torch.save(learner.model.state_dict(),'/kaggle/working/model_state.h5')

