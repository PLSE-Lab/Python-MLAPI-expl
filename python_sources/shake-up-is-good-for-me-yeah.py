#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from transformers import *
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertForMaskedLM.from_pretrained('bert-base-uncased') 

# DEFINE SENTENCE
str = '[CLS] Shaking up of kaggle is [MASK] for [MASK] . [SEP]'
indices = tokenizer.encode(str, add_special_tokens=False, return_tensors='tf')

# PREDICT MISSING WORDS
pred = bert_model(indices)
masked_indices = np.where(indices==103)[1]

# DISPLAY MISSING WORDS
predicted_words = np.argmax( np.asarray(pred[0][0])[masked_indices,:] ,axis=1)
predicted_words_list = tokenizer.decode(predicted_words).split(' ')
str = str.split(' ')
str = [i for i in str if i not in ['[CLS]', '[SEP]']]
str = ' '.join(str)
print(str.replace('[MASK]', '{}').format(predicted_words_list[0], predicted_words_list[1]))

