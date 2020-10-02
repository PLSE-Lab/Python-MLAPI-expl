#!/usr/bin/env python
# coding: utf-8

# # Hello!
# Here is a BERT toy to play with text. It uses a ready-made model and it can not be considered something serious. Nevertheless, sometimes interesting phrases are obtained.
# If hands reach, I will try to make a generator of texts with Markov Chains or improve this model. In the meantime, you can play around with the text below. Have a good time! 

# In[ ]:


from transformers import BertTokenizer, TFBertForMaskedLM
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertForMaskedLM.from_pretrained('bert-base-uncased') 


# In[ ]:


# text = input('Write your text here: ') # Just start and type your phrase # Uncomment this line
text = 'Some text' # Comment this line!!! It only for committing notebook


# In[ ]:


num_of_words = np.random.randint(3,8) # How many new words generate

all_text = [text]
sentence = ('[CLS] {} [MASK] . [SEP]'.format(text))
for i in range(num_of_words):
    
    indices = tokenizer.encode(sentence, add_special_tokens=False, return_tensors='tf')
    prediction = bert_model(indices)
    masked_indices = np.where(indices==103)[1]

    output = np.argmax( np.asarray(prediction[0][0])[masked_indices,:] ,axis=1)
    new_word = tokenizer.decode(output)
    all_text.append(' ' + new_word)
    new_text = ''.join(all_text)
    sentence = ('[CLS] {} [MASK] . [SEP]'.format(new_text))

print(''.join(all_text))


# ## Some generated phrases: "input -> generated"
# * ***My python skill is*** -> excellent too good too bad training skills
# * ***Kaggle is a very*** -> popular game show show host
# * ***This quarantine will*** -> work perfectly well now
# * ***I like big*** -> things too much anyway
# * ***Now say my name*** -> again please please please
# * ***ps4 is much*** -> faster however slower
# * ***Toss a coin to your*** -> left hand side up
# * ***Please upvote*** -> it again please do
# > But I think once will be enough ;)
