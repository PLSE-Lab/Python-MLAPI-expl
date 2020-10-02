#!/usr/bin/env python
# coding: utf-8

# The kernel is under updating. The exploration will contain three parts:
# 
# 1. Text summarization (Bart, Bert, GPT2, XLNet)-I did not run all the code. It seems some bugs existed at the moment. If you want to see how to run with Bert summarization, please check out my last version. This is the first time that I put all those summarizations in one kernel, not sure if there is any conflict among those modules.  
# 2. LDA
# 3. Spark NLP ?

# ## Introduction
# 
# Since the end of 2019, Coronavirus has been a hot topic that attracted huge attention all around the world. According to the [Economist](https://www.economist.com/graphic-detail/2020/03/20/coronavirus-research-is-being-published-at-a-furious-pace), in the first quarter of 2020 alone, there are more than 1000 scientific papers published contained the word "coronavirus."
# 
# 
# ![Coronavirus research is being published at a furious pace](https://www.economist.com/img/b/1280/757/85/sites/default/files/20200321_WOC751.png)

# Before we can take benefits from those massive amounts of data, we often have to face a challenge of how to grab the essential knowledge quickly. Fortunately, text summarization could be one solution to solve this problem. From its name, you may guess that text summarization is an approach that shortens long pieces of information into a shorter version. Generally speaking, there are two types of text summarization techniques. i.e., extractive and abstractive text summarization. Here I am not going to make a long discussion about text summarization itself. This kernel is still under the exploration stage. I may use a different approach to finish those tasks given by Kaggle. However, I will provide a quick demo of how to implement extractive text summarization to generate summaries for those articles that do not have an abstract. For simplification purposes, I will use the [bert-extractive-summarizer](https://pypi.org/project/bert-extractive-summarizer/) module to summarize the text. 
# 
# If you are interested in text summarization technique, please refer to this [blog](https://towardsdatascience.com/a-quick-introduction-to-text-summarization-in-machine-learning-3d27ccf18a9f).

# ## Data preprocessing-to be updated
# 
# I will add one flow chart when having more time, but here are the main steps:
# 
# 1. Importing metadata and converting it to a dataframe meta(45774)
# 2. Selecting four variables from meta dataset: "paper_id2, "title2, "abstract" and "publish_time", combining these four vairables
#    into a new dataset called meta_sm(45774)
# 2. Removing data w/o "publish_id" from metadata(left 31753)
# 3. Removing duplicates from metadata (left 31272)
# 4. Parsing data from json file ("paper_id", "text_body") and converting it to a dataframe df. (33375)
# 5. Merging meta_sm and df by "paper_id" (inner join left 29636)
# 6. Selecting papers based on two criteria: (left 1198)
# 
#     * Contains "COVID",etc. key words in the text_body 
#     * Published after "2019-11-01" 
#     
# 7. Removing papers that have less than 500 or more than 8000 words. (left 1082). Reason: Some papers have very short
#     "text_body", and those texts may contain only citations. 
# 8. Split dataset into train (has abstract  863 articles) and test (missing abstract 219 articles)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import glob
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from  collections import OrderedDict


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os


# In[ ]:


meta=pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")
meta.head()


# In[ ]:


meta.shape


# In[ ]:


meta=meta[((meta['has_pdf_parse']==True) |(meta['has_pmc_xml_parse']==True))]
meta_sm=meta[['cord_uid','sha','pmcid','title','abstract','publish_time','url']]
meta_sm.drop_duplicates(subset ="title", keep = False, inplace = True)
meta_sm.loc[meta_sm.publish_time=='2020-12-31'] = "2020-03-31"
meta_sm.head()


# In[ ]:


meta_sm.shape


# **parse json data**

# In[ ]:


sys.path.insert(0, "../")

root_path = '/kaggle/input/CORD-19-research-challenge/'
#inspired by this kernel. Thanks to the developer ref. https://www.kaggle.com/fmitchell259/create-corona-csv-file
# Just set up a quick blank dataframe to hold all these medical papers. 

df = {"paper_id": [], "text_body": []}
df = pd.DataFrame.from_dict(df)
df


# In[ ]:


collect_json = glob.glob(f'{root_path}/**/*.json', recursive=True)

for i,file_name in enumerate (collect_json):
    row = {"paper_id": None, "text_body": None}
    if i%2000==0:
        print ("====processed " + str(i)+ ' json files=====')
        print()

    with open(file_name) as json_data:
            
        data = json.load(json_data,object_pairs_hook=OrderedDict)
        
        row['paper_id']=data['paper_id']
        
        body_list = []
       
        for _ in range(len(data['body_text'])):
            try:
                body_list.append(data['body_text'][_]['text'])
            except:
                pass

        body = "\n ".join(body_list)
        
        row['text_body']=body 
        df = df.append(row, ignore_index=True)
  


# In[ ]:


df.shape


# In[ ]:


#merge metadata df with parsed json file based on sha_id
merge1=pd.merge(meta_sm, df, left_on='sha', right_on=['paper_id'])
merge1.head()


# In[ ]:


len(merge1)


# In[ ]:


#merge metadata set with parsed json file based on pcmid
merge2=pd.merge(meta_sm, df, left_on='pmcid', right_on=['paper_id'])
merge2.head()


# In[ ]:


len(merge2)


# In[ ]:


#combine merged sha_id and pcmid dataset, remove the duplicate values based on file name
merge_final= merge2.append(merge1, ignore_index=True)
merge_final.drop_duplicates(subset ="title", keep = False, inplace = True)
len(merge_final)


# In[ ]:


merge_final.head()


# In[ ]:


#remove articles that are not related to COVID-19 based on publish time
corona=merge_final[(merge_final['publish_time']>'2019-11-01') & (merge_final['text_body'].str.contains('nCoV|Cov|COVID|covid|SARS-CoV-2|sars-cov-2'))]
corona.shape


# In[ ]:


import re 
def clean_dataset(text):
    text=re.sub('[\[].*?[\]]', '', str(text))  #remove in-text citation
    text=re.sub(r'^https?:\/\/.*[\r\n]*', '',text, flags=re.MULTILINE)#remove hyperlink
    text=re.sub(r'\\b[A-Z a-z 0-9._ - ]*[@](.*?)[.]{1,3} \\b', '', text)#remove email
    text=re.sub(r'^a1111111111 a1111111111 a1111111111 a1111111111 a1111111111.*[\r\n]*',' ',text)#have no idea what is a11111.. is, but I remove it now
    text=re.sub(r'  +', ' ',text ) #remove extra space
    text=re.sub('[,\.!?]', '', text)
    text=re.sub(r's/ ( *)/\1/g','',text) 
    text=re.sub(r'[^\w\s]','',text) #strip punctions (recheck)
    return text


# In[ ]:


import warnings
warnings.filterwarnings('ignore')
corona['text_body'] =corona['text_body'].apply(clean_dataset)
corona['title'] =corona['title'].apply(clean_dataset)
corona['abstract'] =corona['abstract'].apply(clean_dataset)
corona['text_body'] = corona['text_body'].map(lambda x: x.lower())
coro=corona.reset_index(drop=True)
coro.head()


# In[ ]:


coro['count_abstract'] = coro['abstract'].str.split().map(len)
coro['count_abstract'].sort_values(ascending=True)


# In[ ]:


#check word count
y = np.array(coro['count_abstract'])

sns.distplot(y);


# In[ ]:


coro['count_text'] = coro['text_body'].str.split().map(len)
coro['count_text'].sort_values(ascending=True)


# In[ ]:


#check word count
import seaborn as sns
import matplotlib.pyplot as plt

y = np.array(coro['count_abstract'])

sns.distplot(y);


# In[ ]:


coro['count_text'] = coro['text_body'].str.split().map(len)
coro['count_text'].sort_values(ascending=True)


# In[ ]:


coro['count_text'].describe()


# In[ ]:


y = np.array(coro['count_text'])

sns.distplot(y);


# In[ ]:


coro2=coro[((coro['count_text']>500)&(coro['count_text']<4000))]
coro2.shape


# In[ ]:


coro2.to_csv("corona.csv",index=False)


# In[ ]:


#split articles w/o abstarct as the test dataset

test=coro2[coro2['count_abstract']<5]
test.head()


# In[ ]:


test.shape


# In[ ]:


train= coro2.drop(test.index)

train.head()


# In[ ]:


train.shape


# In[ ]:


train=train.reset_index(drop=True)
test=test.reset_index(drop=True)


# ## Text summarization

# **Bert Text Summarization**
# source:[HuggingFace](https://pypi.org/project/bert-extractive-summarizer/#description)

# In[ ]:


get_ipython().system('pip install bert-extractive-summarizer')


# In[ ]:


get_ipython().system('pip install spacy')
get_ipython().system('pip install transformers==2.6.0')
get_ipython().system('pip install neuralcoref')


# In[ ]:


# It seems there is something wrong with Bert Summarizer at the moment, if you want to see how it works, you can check out my last version


# In[ ]:



from summarizer import Summarizer
train['summary']=" "

for i in range(2):
    body=" "
    result=" " 
    full=" " 
    model = Summarizer()
    body=train['text_body'][i]
    result = model(body, min_length=200)
    full = ''.join(result)
    train['summary'][i]=full
     # print(i, train['summary'][i])
     # print("===next====")


# In[ ]:


#Bert does not work
# It seems there is something wrong with the environment at the moment, if you want to see how it works, you can check out my last version
train['summary'][0]


# In[ ]:


body=train['text_body'][0]


# In[ ]:


#GPT2
from summarizer import Summarizer,TransformerSummarizer
GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
full = ''.join(GPT2_model(body, min_length=200))
print(full)


# In[ ]:


model = TransformerSummarizer(transformer_type="XLNet",transformer_model_key="xlnet-base-cased")
full = ''.join(model(body, min_length=60))
print(full)


# **Bart Summarization**
# 
# source: Haggingface [Bart](https://huggingface.co/transformers/model_doc/bart.html)
# 
# Main features:
# * Bart uses a standard seq2seq/machine translation architecture with a bidirectional encoder (like BERT) and a left-to-right decoder (like GPT).
# 
# * The pretraining task involves randomly shuffling the order of the original sentences and a novel in-filling scheme, where spans of text are replaced with a single mask token.
# 
# * BART is particularly effective when fine tuned for text generation but also works well for comprehension tasks. It matches the performance of RoBERTa with comparable training resources on GLUE and SQuAD, achieves new state-of-the-art results on a range of abstractive dialogue, question answering, and summarization tasks, with gains of up to 6 ROUGE.

# Here is one example that I generated using the following code: 
# 
# [{'summary_text': ' coronavirus disease 2019 (covid-19) began in december 2019 in china leading to a public health emergency of international concern (pheic) clinical, laboratory, and imaging features have been partially characterized in some observational studies. We performed a systematic literature review with meta-analysis, using three databases to assess clinical, lab, imaging features, and outcomes of confirmed cases.'}] ref.[colab](https://gist.github.com/dizzySummer/0377bb6db284d3df45fdf75fe5394647#file-bart-summarization-ipynb)
# 

# In[ ]:


get_ipython().system('pip install spacy')
get_ipython().system('pip install transformers')


# In[ ]:


import transformers
import torch


# In[ ]:


from transformers import BartTokenizer, BartForConditionalGeneration
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[ ]:


model = BartForConditionalGeneration.from_pretrained('bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('bart-large-cnn')


# In[ ]:


from transformers import pipeline

# load BART summarizer
summarizer = pipeline(task="summarization")


# In[ ]:


print(train['text_body'][0])


# **LDA model**

# In[ ]:


#I will redo this part

#remove stop words
import gensim
from gensim.parsing.preprocessing import remove_stopwords

my_extra_stop_words = ['preprint','paper','copyright','case','also','moreover','use','from', 'subject', 're', 'edu', 'use','and','et','al','medrxiv','peerreviewed','peerreview','httpsdoiorg','license','authorfunder','grant','ccbyncnd','permission','grant','httpsdoiorg101101202002']

train['text_body']=train['text_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in (my_extra_stop_words) and word not in gensim.parsing.preprocessing.STOPWORDS and len(word)>3]))

coronaRe=train.reset_index(drop=True)


# In[ ]:


import spacy
nlp=spacy.load("en_core_web_sm",disable=['parser', 'ner'])

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    text_out=[]
    for word in texts:
      data=nlp(word)
      data=[word.lemma_ for word in data]
      text_out.append(data)
    return text_out
coronaRe['new_lem'] = lemmatization(coronaRe['text_body'],allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])


# In[ ]:


from gensim.corpora import Dictionary
docs = coronaRe['new_lem']
dictionary = Dictionary(docs)

# Filter out words that occur less than 10 documents, or more than 50% of the documents
dictionary.filter_extremes(no_below=10, no_above=0.5)

# Create Bag-of-words representation of the documents
corpus = [dictionary.doc2bow(doc) for doc in docs]

print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))


# In[ ]:


coronaRe.head()


# In[ ]:


import gensim.corpora as corpora
# Create Dictionary
dictionary = gensim.corpora.Dictionary(coronaRe['new_lem'])
count = 0
for k, v in dictionary.iteritems():
    #print(k, v)
    count += 1
#less than 15 documents (absolute number) or more than 0.5 documents (fraction of total corpus size, not absolute number).after the above two steps, keep only the first 4500 most frequent tokens.
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=4500)
# Create Corpus
bow_corpus = [dictionary.doc2bow(doc) for doc in coronaRe
              ['new_lem']]
bow_corpus_id=[ id for id in coronaRe['cord_uid']]
# View
print(bow_corpus[:1])


# In[ ]:


# Build LDA model
lda_model = gensim.models.LdaMulticore(corpus=bow_corpus,
                                       id2word=dictionary,
                                       num_topics=10, 
                                       random_state=100,
                                       chunksize=100,
                                       passes=10,
                                       per_word_topics=True)


# In[ ]:


from pprint import pprint
# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]


# In[ ]:


lda_df = lda_model.get_document_topics(bow_corpus,minimum_probability=0)
lda_df = pd.DataFrame(list(lda_df))

num_topics = lda_model.num_topics

lda_df.columns = ['Topic'+str(i) for i in range(num_topics)]
for i in range(len(lda_df.columns)):
    lda_df.iloc[:,i]=lda_df.iloc[:,i].apply(lambda x: x[1])
lda_df['Automated_topic_id'] =lda_df.apply(lambda x: np.argmax(x),axis=1)
lda_df.head()


# In[ ]:


#coherence score https://stackoverflow.com/questions/54762690/coherence-score-0-4-is-good-or-bad
from gensim.models import CoherenceModel
# Compute Coherence Score
from tqdm import tqdm
coherenceList_cv=[]
num_topics_list = np.arange(5,26)
for num_topics in tqdm(num_topics_list):
  lda_model = gensim.models.LdaModel(corpus=bow_corpus,
                                         id2word=dictionary,
                                         num_topics=num_topics,
                                         random_state=100,
                                         chunksize=100,
                                         passes=10,
                                         alpha='auto',
                                         per_word_topics=True)
  coherence_model_lda = CoherenceModel(model=lda_model, texts=coronaRe['new_lem'], coherence='c_v')
  coherence_lda = coherence_model_lda.get_coherence()
  coherenceList_cv.append(coherence_lda)
print('\nCoherence Score: ', coherence_lda)


# In[ ]:


#re-do (not correct)
plotData = pd.DataFrame({'Number of topics':num_topics_list,
                         'CoherenceScore_cv':coherenceList_cv})
f,ax = plt.subplots(figsize=(10,6))
sns.set_style("darkgrid")
sns.pointplot(x='Number of topics',y= 'CoherenceScore_cv',data=plotData)

plt.title('Topic coherence')


# In[ ]:


#final model

Lda = gensim.models.LdaMulticore
lda_final= Lda(corpus=bow_corpus, num_topics=17,id2word = dictionary, passes=10,chunksize=100,random_state=100)


# In[ ]:


from pprint import pprint
# Print the Keyword in the 11 topics
pprint(lda_final.print_topics())
doc_lda = lda_final[corpus]


# In[ ]:


lda_df = lda_final.get_document_topics(bow_corpus,minimum_probability=0)
lda_df = pd.DataFrame(list(lda_df))
lda_id=pd.DataFrame(list(bow_corpus_id))
num_topics = lda_final.num_topics

lda_df.columns = ['Topic'+str(i) for i in range(num_topics)]

for i in range(len(lda_df.columns)):
    lda_df.iloc[:,i]=lda_df.iloc[:,i].apply(lambda x: x[1])

lda_df['Automated_topic_id'] =lda_df.apply(lambda x: np.argmax(x),axis=1)

lda_df['cord_uid']= lda_id
lda_df[39:40]


# In[ ]:


topic=lda_df[['Automated_topic_id','cord_uid']]


# In[ ]:


plot_topics=lda_df.Automated_topic_id.value_counts().reset_index()
plot_topics.columns=["topic_id","quantity"]
plot_topics


# In[ ]:


ax = sns.barplot(x="topic_id", y="quantity",  data=plot_topics)


# In[ ]:


coronaRe['topic_id']= topic['Automated_topic_id']
coronaRe.head()


# **Named Entity Recognition (NER)**
# 
# I have also tested three different modules for name entity recognitions. Each of them has its pros and cons. 

# * Module A: Spacy

# In[ ]:


#https://medium.com/@manivannan_data/how-to-train-ner-with-custom-training-data-using-spacy-188e0e508c6
get_ipython().system('pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_ner_bionlp13cg_md-0.2.4.tar.gz')


# In[ ]:


import spacy
from spacy import displacy
from collections import Counter

import en_ner_bionlp13cg_md
nlp = en_ner_bionlp13cg_md.load()
text = train['abstract'][2]
doc = nlp(text)
print(list(doc.sents))


# In[ ]:


print(doc.ents)


# In[ ]:


from spacy import displacy
displacy.render(next(doc.sents), style='dep', jupyter=True,options = {'distance': 110})


# In[ ]:


displacy.render(doc, style='ent')


# In[ ]:


#!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz
#!pip install en_core_web_sm
#!pip install git+https://github.com/NLPatVCU/medaCy.git@development


# * Module B: [medacy](https://github.com/NLPatVCU/medaCy)

# In[ ]:


get_ipython().system('pip install git+https://github.com/NLPatVCU/medaCy.git@development')
get_ipython().system('pip install git+https://github.com/NLPatVCU/medaCy_model_clinical_notes.git')

