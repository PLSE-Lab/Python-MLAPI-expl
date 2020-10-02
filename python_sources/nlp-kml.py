#!/usr/bin/env python
# coding: utf-8

# In[4]:


import nltk##01_accessing-corpora.py
import re
import os


# In[ ]:


from nltk.corpus import gutenberg as gt
print(gt.fileids())


# In[ ]:


shkspr_hmlt = gt.words('shakespeare-hamlet.txt')
print(shkspr_hmlt,end='\n\n\n')


# In[ ]:


print(len(shkspr_hmlt))


# In[ ]:


shkspr_hmlt = gt.raw('shakespeare-hamlet.txt')
print(shkspr_hmlt,end='\n\n\n')


# In[ ]:


for fileid in gt.fileids():
    #raw_data = gt.raw(fileid)
    num_words = len(gt.words(fileid))
    num_sents = len(gt.sents(fileid))
    #vocabulary = set([w.lower() for w in gt.words(fileid)])
    print('Data for File ID:', fileid)
    print('Number of words:', num_words, 'nNumber of sentences:', num_sents)
    #print('Vocabulary:\n', vocabulary, end='\n\n\n')
    print('Words:', gt.words(fileid), end='\n\n\n')


# In[ ]:


############


# In[ ]:


from nltk.corpus import PlaintextCorpusReader #02 Loading your own corpus.py


# In[ ]:


import os


# In[ ]:


corpus_root = os.getcwd() + '/'


# In[ ]:


corpus_root


# In[ ]:


file_ids = '.*.txt'


# In[ ]:


corpus = PlaintextCorpusReader(corpus_root, file_ids)


# In[ ]:


print(corpus.fileids())


# In[ ]:


#print(corpus.words('shakespeare-taming-of-the-shrew.txt'))


# In[ ]:


##############


# In[ ]:


from nltk.tokenize import word_tokenize, sent_tokenize #03 Tokenization


# In[ ]:


# INTRODUCTION


# In[ ]:


def read_file(filename):
    with open(filename,'r') as file:
        test = file.read()
    return text
words = word_tokenize(gt.raw('shakespeare-hamlet.txt'))


# In[ ]:


print('Size as a list: ', len(words))
print('Size as a set:',len(set(words)))
print(words[:100])


# In[ ]:


sentences = sent_tokenize(gt.raw('shakespeare-hamlet.txt'))


# In[ ]:


print('No. of sentences:', len(sentences), '\n')
for sentence in sentences[:5]:
    print(sentence.strip())
print('Size as a list:', len(sentences))
print('Size as a set:', len(set(sentences)), end='\n\n')
print(sentences[:10])


# In[ ]:


#################


# In[3]:


import re #04_regular-expression.py


# In[ ]:


# search function

if((re.search('^a','abc'))):
    print('Found it !')
re.search('^a','abc')
re.search('^a','Abc')


# In[ ]:


if((re.search('^a','Abc'))):
    print('Found it !')
else:
    print('Not found')


# In[ ]:


# get all words which ends with ed


# In[ ]:


wrds_ndng-wth_ed = [w for w in words if re.search('ed$', w)]
print(len(wrds_ndng_wth_ed))
print(len(set(wrds_ndng_wth_ed)))
print(wrds_ndng_wth_ed)


# In[ ]:


wrds_ndng_wth_er = [w for w in words if re.search('ner$', str(w).lower())]
print(len(wrds_ndng_wth_er))
print(len(set(wrds_ndng_wth_er)))
print(wrds_ndng_wth_er)


# In[ ]:


print(len(set([w for w in words if re.search('^a*',w)])))


# In[ ]:


print(set([w for w in words if re.search('^a*',w)]))


# In[ ]:



print(len(set([w for w in words if re.search('^a+', w)])))


# In[ ]:



print(set([w for w in words if re.search('^a+',w)]))


# In[ ]:



print(len(set([w for w in words if re.search('^a*',w)])))


# In[ ]:


print(set([w for w in words if re.search('^a*', w)]))


# In[ ]:



print(len(set([w for w in words if re.search('^a+',w)])))


# In[ ]:



print(set([w for w in words if re.search('^a+',w)]))


# In[ ]:



print([w for w in words if re.search('^a*', w)])


# In[ ]:


print([w for w in words if re.search('^a*', w)])


# In[ ]:


print([w for w in words if re.search('at*',w)])


# In[ ]:


####################


# In[16]:


from nltk.corpus import gutenberg, brown, nps_chat #05_application of regex.py


# In[ ]:


#import nltk

moby = nltk.Text(gutenberg.words('melville-moby_dick.txt'))
print(moby.findall(r'<a><.*><man>'))


# In[ ]:


#Creating a Text object
chat_obj = nltk.Text(nps_chat.words())
print(chat_obj.findall(r'<.*><.*><bro>'))


# In[ ]:


hobbies_learned = nltk.Text(brown.words(categories=['hobbies','learned']))
print(hobbies_learned.findall(r'<\w*><and><other><\w*s>'))


# In[ ]:


#passing your own list of words
text = 'Hello, I am an electrical engineer who is currently learning DataScience'


# In[ ]:


obj = nltk.Text(nltk.word_tokenize(text))


# In[ ]:


print(obj.findall(r'<.*ing>+'))


# In[ ]:


#######################


# In[ ]:


from nltk import PorterStemmer, LancasterStemmer #06 stemming.py


# In[ ]:


porter = PorterStemmer()


# In[ ]:


tokens = ['lying']
print(porter.stem(tokens[0]))
lancaster = LancasterStemmer()
print(lancaster.stem(tokens[0]))


# In[ ]:


print(tokens[0])


# In[ ]:


print([porter.stem(t) for t in tokens])


# In[ ]:


#######################


# In[ ]:


from nltk import WordNetLemmatizer #07 lemmatizaion.py
from nltk.corpus import brown


# In[ ]:


tokens = brown.words(categories=['religion'])


# In[ ]:


wnl = WordNetLemmatizer()


# In[ ]:


# print(set([wnl.lemmatize()]))


# In[ ]:


print(set([wnl.lemmatize(t) for t in tokens]))


# In[ ]:


tokens = nltk.word_tokenize('the women ar not lying')
print([wnl.lemmatize(t) for t in tokens])


# In[ ]:


#################


# In[5]:


text = 'A Linux server, like any other computer you may be familiar with, runs applications. To the computer, these are'        ' considered "processes" While Linux will handle the low-level, behind-the-scenes management in a process\'s '        'life-cycle, you will need a way of interacting with the operating system to manage it from a higher-level.'
print(re.split(' ', text))


# In[6]:


print(re.split(' ',text))


# In[7]:


print(re.split('\s+',text))


# In[ ]:


print(re.split('\W',text))


# In[8]:


print(re.findall('\w+|\S|\w*', text))


# In[9]:


print(re.findall("\w+[-']+\w+",text))


# In[1]:


####################


# In[10]:


#Introduction to tagger
# POS Tagging is used for analysing the context
#Then get all the words which are used in the same context
test = 'We are learning Natural Languages Processing'


# In[11]:


tokens=nltk.word_tokenize(text)


# In[12]:


print(nltk.pos_tag(tokens))


# In[13]:


print(nltk.help.upenn_tagset('PRP'))


# In[14]:


print(nltk.help.upenn_tagset('VBP'))


# In[15]:


test=nltk.Text(word.lower() for word in nltk.corups.brown.words())


# In[ ]:




