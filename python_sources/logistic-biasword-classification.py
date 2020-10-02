#!/usr/bin/env python
# coding: utf-8

# # Bias Word detection from Wikipedia edits.

# In[ ]:


get_ipython().system('pip install spacy')
get_ipython().system('python -m spacy download en_core_web_sm')
get_ipython().system('pip install afinn')
get_ipython().system('pip install xgboost')


# In[ ]:


import numpy as np # linear algebra
import pandas as pd
import nltk
import string
import re
import spacy
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize, TweetTokenizer 
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
nlp = spacy.load('en_core_web_sm')
from afinn import Afinn
af = Afinn()
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


# In[ ]:


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


# In[ ]:


path="../input/biased-text/"


# # Test file upload with Before and After edits

# In[ ]:


train_data = pd.read_csv(path+"biased.word.test",delimiter='\t',encoding='latin-1',header=None)


# In[ ]:


train_data=train_data.iloc[:,3:5]
train_data


# In[ ]:


train_data.columns=['before','after']


# In[ ]:


train_data


# # Text Precprocessing 

# In[ ]:


def preprocess(text):
    text = re.sub('[()!@#$.,;"]', '', text)
    text = re.sub('[(]', '', text)
    text = re.sub('[)]', '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = text.lower()
    text = text.split()
    text = ' '.join(text)
    text =text.strip()
    return text


# In[ ]:


X=[]
Y=[]


# In[ ]:


for i in range(len(train_data)):
    X.append(preprocess(train_data['before'][i]))
    Y.append(preprocess(train_data['after'][i]))


# In[ ]:


text_processed=pd.DataFrame(X,columns=["before"])
text_processed["after"]=Y


# In[ ]:


X=[]
Y=[]
train_data=[]


# # Bias Word extraction By Comparing Sentence.

# In[ ]:


biased=[]
count=-1
c=[]
for i in range(len(text_processed)):
    b=text_processed['before'][i].split()
    a=text_processed['after'][i].split()
    l1=len(b) if(len(b)<=len(a)) else len(a)
    for j in range(l1):
        if (b[j]!=a[j]):
            biased.append(b[j])
            count=i
            break
    if(count!=i):
        biased.append(b[len(b)-1])
            


# In[ ]:


truth=pd.DataFrame(biased)
truth.columns=['biased']


# In[ ]:


df = pd.DataFrame()
data=pd.DataFrame()


# In[ ]:


# Extract position of word in the sentence start/mid/end
position=[]
for i in range(len(text_processed)):
    t=text_processed['before'][i].split()
    l=len(t)
    text = ['start' if i<int(l/3) else 'end' if i>=int(2*l/3) else 'mid' for i in range(l)]
    text=' '.join(text)
    text =text.strip()
    position.append(text)
data['position']=position


# In[ ]:


# Extract grammatical relation of word in the sentence 

grammatical=[]
for i in range(len(text_processed)):
    t=nlp(text_processed['before'][i])
    text = [word.dep_ for word in t]
    text=' '.join(text)
    text =text.strip()
    grammatical.append(text)
data['grammatical']=grammatical


# In[ ]:


data


# In[ ]:


data2=[]
columns=['id','text','word_around','position','grammatical']
for index, row in text_processed.iterrows():
    text=row['before'].split()
    position=data['position'][index].split()
    grammer=data['grammatical'][index].split()
    for t in range(len(text)):
        value=[]
        value.append(index)
        value.append(text[t])
        around=["None" if((i<0)|(i>=len(text))) else text[i] for i in [x for x in range(t-2,t+3) if x != t] ]
        around=' '.join(around)
        value.append(around)
        value.append(position[t])
        value.append(grammer[t])
        zipped = zip(columns, value)
        a_dictionary = dict(zipped)
        data2.append(a_dictionary)
df = pd.DataFrame.from_dict(data2)


# In[ ]:


data=[]
columns=['POS-2','POS-1','POS+1','POS+2']
for index, row in df.iterrows():
    text=nltk.pos_tag(nltk.word_tokenize(row['word_around']))
    value=[]
    for i in text:
        if(i[0]=="None"):
            value.append("None")
        else:
            value.append(i[1])
    zipped = zip(columns, value)
    a_dictionary = dict(zipped)
    data.append(a_dictionary)
POS = pd.DataFrame.from_dict(data)


# In[ ]:


l=['positive','strong_subjectives','implicatives','npov_word','factives','assertives',
   'weak_subjectives','hedges','entailment','negative','report_verbs']


# In[ ]:


for k in l:
    data=pd.read_csv(path+k+'.csv')
    p=list(data.iloc[:,0])
    lexi=[]
    for i in range(len(df)):
        t=df['text'][i]
        text="True" if t in p else "False"
        lexi.append(text)   
    df[k]=lexi


# In[ ]:


columns=['positive', 'strong_subjectives', 'implicatives','factives', 'assertives',
         'weak_subjectives', 'hedges', 'entailment','negative', 'report_verbs']
col=[i+"_around" for i in columns]
for k in columns:
    data=pd.read_csv(path+k+'.csv')
    p=list(data.iloc[:,0])
    lexi=[]
    for index, row in df.iterrows():
        text=nltk.word_tokenize(row['word_around'])
        value=["True" if i in p else "False" for i in text]
        lexi.append("True" if "True" in value else "False")        
    df[k+"_around"]=lexi


# In[ ]:


data=[]
for index, row in df.iterrows():
    text=row['text']
    value = 1 if (truth['biased'][row['id']]==text) else 0      
    data.append(value)

label = pd.DataFrame.from_dict(data)
label.columns=['label']


# In[ ]:


pos_before=[]
for i in range(len(df)):
    t=nltk.word_tokenize(df['text'][i])
    for j in nltk.pos_tag(t):
        tagged = j[1] 
    pos_before.append(tagged)
df['POS']=pos_before


# In[ ]:


lemma=[]
for i in range(len(df)):
    t=df['text'][i]
    text = wordnet_lemmatizer.lemmatize(t) 
    lemma.append(text)
df['lemma']=lemma


# In[ ]:


polarity=[]
for i in range(len(df)):
    t=df['text'][i]
    sentiment_scores = af.score(t)
    text = 'negative' if sentiment_scores<0 else 'positive' if sentiment_scores>0 else 'neutral'
    polarity.append(text)
df['polarity']=polarity


# In[ ]:


df = pd.concat([df,POS,label], axis=1, sort=False)


# In[ ]:


df=df.drop('word_around',axis=1)
df


# In[ ]:


word=df['text']
lemma=df['lemma']
test= df.drop(['text','lemma'],axis=1)


# In[ ]:


col=['id','positive','strong_subjectives','implicatives','npov_word','factives','assertives','weak_subjectives',
     'hedges','entailment','negative','report_verbs','position','grammatical','polarity','POS-2','POS-1','POS',
     'POS+1','POS+2','positive_around','strong_subjectives_around','implicatives_around','factives_around',
     'assertives_around','weak_subjectives_around','hedges_around','entailment_around','negative_around','report_verbs_around','label']


# In[ ]:


test = test[col]


# In[ ]:


get_ipython().system('pip install gensim')


# In[ ]:


# import gensim.downloader as api

# wv = api.load('word2vec-google-news-300')
# model = wv


# In[ ]:


import gensim
path = "../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin"
model = gensim.models.KeyedVectors.load_word2vec_format(path,binary=True)


# In[ ]:


word_vec=[]
col=["word_vec_"+str(i) for i in range(300)]

for i in range(len(word)):
    
    if word[i] in model.vocab:
        w = model[word[i]]
    else:
        w=np.zeros(300)
    
    zipped = zip(col, w)
    a_dictionary = dict(zipped)
    word_vec.append(a_dictionary)


# In[ ]:


word_lemma=[]
col=["lemma_vec_"+str(i) for i in range(300)]

for i in range(len(lemma)):
    if lemma[i] in model.vocab:
        w = model[lemma[i]]
    else:
        w=np.zeros(300)

    zipped = zip(col, w)
    a_dictionary = dict(zipped)
    word_lemma.append(a_dictionary)


# In[ ]:


word_vec=pd.DataFrame(word_vec)
word_lemma=pd.DataFrame(word_lemma)
word=pd.DataFrame(word)


# In[ ]:


model=[]


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.classes_ = np.load('../input/biased-text/'+"classes.npy")
test_encoded = test.apply(le.fit_transform)


# In[ ]:


test=[]


# In[ ]:


testset =pd.concat([word,word_vec,word_lemma,test_encoded],axis=1)


# In[ ]:


word=testset['text']
y_test=testset['label']
ID=testset['id']


# In[ ]:


testset = testset.drop(['id','label','text'],axis=1)


# In[ ]:


col=ID.unique()


# In[ ]:


get_ipython().system('pip install scikit-learn --upgrade')


# In[ ]:


import pickle
filename = '../input/biased-text/'+'finalized_model_logistic_40000.sav'
model = pickle.load(open(filename, "rb"))


# In[ ]:


y_pred = pd.DataFrame(model.predict_proba(testset)[:,1],index=y_test.index,columns=['y_pred'])


# In[ ]:


analyze =pd.concat([ID,word,y_test,y_pred],axis=1)
analyze = analyze.set_index('id')
analyze


# In[ ]:


top1=[]
top2=[]
top3=[]
for i in col:
    l=list(analyze.loc[i].sort_values(by=['y_pred'], ascending=False)[:3]['text'])
    top1.append(l[0])
    top2.append(l[:2])
    top3.append(l)


# In[ ]:


result1=0
result2=0
result3=0

for i,row in truth.iterrows():
    result1+=1 if row['biased'] in top1[i] else 0
    result2+=1 if row['biased'] in top2[i] else 0
    result3+=1 if row['biased'] in top3[i] else 0


# In[ ]:


result1=result1/len(truth)
result2=result2/len(truth)
result3=result3/len(truth)


# In[ ]:


result1


# In[ ]:


result2


# In[ ]:


result3


# In[ ]:


top3=pd.DataFrame(top3)
result =pd.concat([text_processed,truth,top3],axis=1)
result.columns=['before','after','biased','top1','top2','top3']
result.to_csv('result.csv',index=None)


# In[ ]:




