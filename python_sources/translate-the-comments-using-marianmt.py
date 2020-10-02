#!/usr/bin/env python
# coding: utf-8

# # MarianMT

# This notebook is mostly for educational purpose. Transformers released a new function called [MarianMT](https://huggingface.co/transformers/model_doc/marian.html) which seems to be very powerful to translate text data.
# 
# In this notebook I used a small subset of the validation data because performing translations using this method takes a lot of time and probably doesn't give an edge compared to dataset transleted using tradtionnal methods.

# In[ ]:


get_ipython().system('pip install -U transformers')


# # Load data

# In[ ]:


import pandas as pd
from tqdm.notebook import tqdm


# In[ ]:


df = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
df = df.sample(100, random_state=12)
df.head(3)


# # Tokenize & translate

# In[ ]:


df['lang'].unique()


# In[ ]:


from transformers import MarianMTModel, MarianTokenizer


# In[ ]:


df['content_english'] = ''


# In[ ]:


for i, lang in tqdm(enumerate(['es', 'it', 'tr'])):
    if lang in ['es', 'it']:
        model_name = 'Helsinki-NLP/opus-mt-ROMANCE-en'
        df_lang = df.loc[df['lang']==lang, 'comment_text'].apply(lambda x: '>>{}<< '.format(lang) + x)
    else:
        model_name = 'Helsinki-NLP/opus-mt-{}-en'.format(lang)
        df_lang = df.loc[df['lang']==lang, 'comment_text']
    
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name, output_loading_info=False)
        
    batch = tokenizer.prepare_translation_batch(df_lang.values,
                                               max_length=192,
                                               pad_to_max_length=True)
    translated = model.generate(**batch)

    df.loc[df['lang']==lang, 'content_english'] = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]


# In[ ]:


df.head(3)


# In[ ]:


df.to_csv("df_translated.csv")


# MarianMT offers a wider range of things you can do, pleaste take a look at [the official documentation](https://huggingface.co/transformers/model_doc/marian.html).
