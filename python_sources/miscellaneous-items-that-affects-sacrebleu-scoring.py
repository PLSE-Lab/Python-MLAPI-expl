#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip3 install sacrebleu')
import sacrebleu


# # Note
# From the competition data page:
# * train_tcn.csv and train_en.csv are not parallel data
# * dev_tcn.csv and dev_en.csv are parallel data which you can check your model's sacrebleu score with __lowercase__ parameter (https://github.com/mjpost/sacrebleu).
# 
# This notebook will investigate the effects of various text effects on scarebleu scoring.  
# 
# <br> FunFact: https://en.wikipedia.org/wiki/Sacrebleu#:~:text=Sacrebleu%20or%20sacre%20bleu%20is,Lord%20thy%20God%20in%20vain.%22

# ### Read Dev Data

# In[ ]:


dev_en_csv = '/kaggle/input/shopee-product-title-translation-open/dev_en.csv'
dev_tcn_csv = '/kaggle/input/shopee-product-title-translation-open/dev_tcn.csv'
df = pd.concat([pd.read_csv(dev_tcn_csv), pd.read_csv(dev_en_csv)], axis=1).drop('split', axis=1)
df.head()


# ### Use sacreblue and demonstrate 100% match

# In[ ]:


refs = [df['text'], df['translation_output']]
sys = df['translation_output']
bleu = sacrebleu.corpus_bleu(sys, refs, lowercase=True)
print(bleu.score)


# ### Investigate effect of word order
# Please note that we swapped the order of "Phone Case" to "Case Phone" in this example below

# In[ ]:


refs = [[df.iloc[0]['text']], [df.iloc[0]['translation_output']]]
sys = ['Oppo A75 A75S A73 Case Phone Soft Rabbit Silicone Case']
print(f'refs: {refs}')
print(f'sys: {sys}')
bleu = sacrebleu.corpus_bleu(sys, refs, lowercase=True)
print(bleu.score)


# ### Investigate effect of additional word at the end
# Please note that we added "AddWord" at the end of the sentence

# In[ ]:


refs = [[df.iloc[0]['text']], [df.iloc[0]['translation_output']]]
sys = ['Oppo A75 A75S A73 Phone Case Soft Rabbit Silicone Case AddWord']
print(f'refs: {refs}')
print(f'sys: {sys}')
bleu = sacrebleu.corpus_bleu(sys, refs, lowercase=True)
print(bleu.score)


# ### Investigate effect of repeated word in the middle of the sentence
# Please note that we sandwiched additional "Phone" in the middle of "Phone" and "Case"

# In[ ]:


refs = [[df.iloc[0]['text']], [df.iloc[0]['translation_output']]]
sys = ['Oppo A75 A75S A73 Phone Phone Case Soft Rabbit Silicone Case AddWord']
print(f'refs: {refs}')
print(f'sys: {sys}')
bleu = sacrebleu.corpus_bleu(sys, refs, lowercase=True)
print(bleu.score)


# ### Investigate effect of Capitalization of word
# Please note that we change upper case of "Phone" to "phone"

# In[ ]:


refs = [[df.iloc[0]['text']], [df.iloc[0]['translation_output']]]
sys = ['Oppo A75 A75S A73 phone Case Soft Rabbit Silicone Case']
print(f'refs: {refs}')
print(f'sys: {sys}')
bleu = sacrebleu.corpus_bleu(sys, refs, lowercase=True)
print(bleu.score)


# ### Investigate effect of missing unique word
# Please note that we removed "Phone" from the sentence

# In[ ]:


refs = [[df.iloc[0]['text']], [df.iloc[0]['translation_output']]]
sys = ['Oppo A75 A75S A73 Case Soft Rabbit Silicone Case']
print(f'refs: {refs}')
print(f'sys: {sys}')
bleu = sacrebleu.corpus_bleu(sys, refs, lowercase=True)
print(bleu.score)


# ### Investigate effect of word removal from both ends
# Please note that we removed the first word "Oppo" and last word "Case" from the sentence

# In[ ]:


refs = [[df.iloc[0]['text']], [df.iloc[0]['translation_output']]]
sys = ['A75 A75S A73 Phone Case Soft Rabbit Silicone']
print(f'refs: {refs}')
print(f'sys: {sys}')
bleu = sacrebleu.corpus_bleu(sys, refs, lowercase=True)
print(bleu.score)


# In[ ]:




