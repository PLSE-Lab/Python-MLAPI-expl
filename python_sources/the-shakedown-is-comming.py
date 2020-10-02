#!/usr/bin/env python
# coding: utf-8

# Inspiration
# * https://www.kaggle.com/nxrprime/kaggle-is-chicken-as-proven-by-bert
# * https://www.kaggle.com/c/google-quest-challenge/discussion/129399
# 

# Push **Code** button to see the code.

# In[ ]:


from transformers import *
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertForMaskedLM.from_pretrained('bert-base-uncased') 

def bert_predict(str_mask):
    indices = tokenizer.encode(str_mask, add_special_tokens=False, return_tensors='tf')

    # PREDICT MISSING WORDS
    pred = bert_model(indices)
    masked_indices = np.where(indices==103)[1]

    # DISPLAY MISSING WORDS
    predicted_words = np.argmax( np.asarray(pred[0][0])[masked_indices,:] ,axis=1)
    predicted_words_decoded = tokenizer.decode(predicted_words)
    print(f"Original phrase: {str_mask}")
    str_out = str_mask.replace("[MASK]", "{}")
    str_out = str_out.replace("[CLS] ", "")
    str_out = str_out.replace("[SEP]", "")          
    print(f"Predicted words decoded: {predicted_words_decoded}")
    print(f"Resulted phrase: {str_out.format(predicted_words_decoded)}")    


# In[ ]:


str_mask = '[CLS] The leaderboard drastic change will be soon [MASK] on Kaggle. [SEP]'
bert_predict(str_mask)

