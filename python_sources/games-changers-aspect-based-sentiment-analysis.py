#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import re


# In[ ]:


os.chdir('../input')
os.getcwd()


# # Aspect Based Sentiment Analysis on Car Reviews
# ## Taking Toyota Cars as an example

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        
data = pd.read_csv('/kaggle/input/hotel-reviews/Hilton_Hawaiian_Village_Waikiki_Beach_Resort-Honolulu_Oahu_Hawaii__en.csv',engine='python',index_col=False, nrows = 100)
data.head()


# ## **Combining the review title and review body for the text corpus****

# ## **Using spaCy for dependency parsing which forms the crux of aspect extraction**

# In[ ]:


import spacy
from tqdm import tqdm
nlp = spacy.load('en_core_web_lg', parse=True, tag=True, entity=True)


#  ## **Using spaCy's awesome displacy module to show the dependency relations**

# In[ ]:


nlp


# In[ ]:


data.head(3)


# In[ ]:


type(data)


# In[ ]:


data.shape


# In[ ]:


txt = 'This is a huge resort, capacity-wise (not necessarily in terms of the territory. It sits next to a commercial "village" of sorts (the usual array of overpriced restaurants and various shops catering to tourist clientele).There"s nothing special to it, I"d even call it "faceless" despite a stream of celebrities and dignitaries of all kinds that graced the hotel with their presence and even movies exposure - but, at the same time, there"s nothing to complain about. The hotel is being well-run and well-maintained. The rooms are rather standard for resorts of this kind. The beach is right there (not the best, though - large-grain sand and quite a bit of crushed shells). In the room, we were pleasantly surprised to find a PS-3 gaming console that can run both games and DVDs you may rent from automated kiosks (free if you have a status of certain level with Hilton).The hotel has a large, well-equipped gym (I counted 33 cardio machines plus HIIT/pulley rack, all typical main-muscle groups strength machines, a full rack of dumbbells running through 100 lbs, balls, batons, etc. There"s a nice spa, too. They also offer group classes in various sports.Breakfasts were ok, although I"d expect a bit higher variety of morning foods. In any case, there are options for dining within a walking distance from the hotel.All in all, a positive experience. By Hawaii standards, an excellent hotel'
doc = nlp(txt)
spacy.displacy.render(doc,style='dep',jupyter=True)


# **from https://nlp.stanford.edu/software/dependencies_manual.pdf**
# ### AMOD - adjectival modifier
# #### An adjectival modifier of a Noun is any adjectival phrase that serves to modify the meaning of the Noun
# ### ex - 'Great <--amod-- Car', 'Long <--amod-- range'

# In[ ]:


txt = 'We stayed at HHV every summer in the 1980"s. This is our first trip back since then. The Village is as amazing and as beautiful as almost 40 years ago. It is self-contained-ABC stores, jewelers, clothing stores, luggage stores, eateries of all stripes, etc. With 4000 rooms, the check-in desk is bustling 24-7. The beaches (public) are pristine and the water beautiful. There is even a Church on the Beach which has operated there for 49 years and is open to all faiths. We actually found the food competitive in taste and price. The only complaint I have is the soundproofing between rooms. Our neighbor played loud music night and day, beginning at 7.20 a.m.(perhaps they are having a staycation in their room). After security visited, they would lower the volume, but we could still hear the music and their voices. As soon as security left, they started up again. We finally asked to be moved. I feel bad for whoever ended with the room next. The next room was quiet, because we had nice neighbors. But because of the location and convenience, and customer service, I still rate them a 5.'
doc = nlp(txt)
spacy.displacy.render(doc,style='dep',jupyter=True)


# ### ADVMOD - adverb modifier
# #### An adverb modifier of a word is a (non-clausal) adverb or adverb-headed phrase that serves to modify
# #### the meaning of the word
# ### ex - 'Drives --advmod--> well'

# In[ ]:


txt =  "I love hilton, but for this hotel, I really cannot recommend for anyone. We booked the most expansive room of the hotel, but except bigger room, the whole room was old and feel like in a 3 stars hotel room. Not worthy for the price at all. Won't recommend for this hotel."
doc = nlp(txt)
spacy.displacy.render(doc,style='dep',jupyter=True)


# ### XCOMP -  open clausal complement
# #### An open clausal complement (xcomp) of a verb or an adjective is a predicative or clausal complement without its own subject
# ### ex - 'wonderful --xcomp--> drive'
# 

# In[ ]:


txt =  "Me and my father really enjoyed everything about this hotel. Would recommend it to anyone and hopefully we can soon return for another vacation.Beds were big and really comfy, also nice and big rooms."
doc = nlp(txt)
spacy.displacy.render(doc,style='dep',jupyter=True)


# ### NEG - self explanatory
# ### ex - not <--neg-- wonderful

# ### COMPOUND WORDS
# #### Generally from a review standpoint, compound words often do not offer us sentiments per se, hence my code looks for possible compound word pairs and then checks with the aspect words extracted if it can add more detail to the extracted aspects - ex Outstanding passenger van gives *more context* than Outstanding van (which is what my code would have extracted without the compound word search) while the compound word search will identify passenger van as a compound word

# In[ ]:


competitors = ['Chevy','chevy','Ford','ford','Nissan','nissan','Honda','honda','Chevrolet','chevrolet','Volkswagen','volkswagen','benz','Benz','Mercedes','mercedes','subaru','Subaru','VW']


# **Reason for using competitor name list is to remove potential misleading aspects-sentiments, since we are interested to acquire aspect info about Toyota and not any other brand. This is because a reviewer might be comparing a Benz saying it has superior handling when compared to the car the person is reviewing and this can lead to misclassifications******

# In[ ]:


data.head(2)


# In[ ]:


aspect_terms = []
comp_terms = []
easpect_terms = []
ecomp_terms = []
enemy = []
for x in tqdm(range(len(data['review_body']))):
    amod_pairs = []
    advmod_pairs = []
    compound_pairs = []
    xcomp_pairs = []
    neg_pairs = []
    eamod_pairs = []
    eadvmod_pairs = []
    ecompound_pairs = []
    eneg_pairs = []
    excomp_pairs = []
    enemlist = []
    if len(str(data['review_body'][x])) != 0:
        lines = str(data['review_body'][x]).replace('*',' ').replace('-',' ').replace('so ',' ').replace('be ',' ').replace('are ',' ').replace('just ',' ').replace('get ','').replace('were ',' ').replace('When ','').replace('when ','').replace('again ',' ').replace('where ','').replace('how ',' ').replace('has ',' ').replace('Here ',' ').replace('here ',' ').replace('now ',' ').replace('see ',' ').replace('why ',' ').split('.')       
        for line in lines:
            enem_list = []
            for eny in competitors:
                enem = re.search(eny,line)
                if enem is not None:
                    enem_list.append(enem.group())
            if len(enem_list)==0:
                doc = nlp(line)
                str1=''
                str2=''
                for token in doc:
                    if token.pos_ is 'NOUN':
                        for j in token.lefts:
                            if j.dep_ == 'compound':
                                compound_pairs.append((j.text+' '+token.text,token.text))
                            if j.dep_ is 'amod' and j.pos_ is 'ADJ': #primary condition
                                str1 = j.text+' '+token.text
                                amod_pairs.append(j.text+' '+token.text)
                                for k in j.lefts:
                                    if k.dep_ is 'advmod': #secondary condition to get adjective of adjectives
                                        str2 = k.text+' '+j.text+' '+token.text
                                        amod_pairs.append(k.text+' '+j.text+' '+token.text)
                                mtch = re.search(re.escape(str1),re.escape(str2))
                                if mtch is not None:
                                    amod_pairs.remove(str1)
                    if token.pos_ is 'VERB':
                        for j in token.lefts:
                            if j.dep_ is 'advmod' and j.pos_ is 'ADV':
                                advmod_pairs.append(j.text+' '+token.text)
                            if j.dep_ is 'neg' and j.pos_ is 'ADV':
                                neg_pairs.append(j.text+' '+token.text)
                        for j in token.rights:
                            if j.dep_ is 'advmod'and j.pos_ is 'ADV':
                                advmod_pairs.append(token.text+' '+j.text)
                    if token.pos_ is 'ADJ':
                        for j,h in zip(token.rights,token.lefts):
                            if j.dep_ is 'xcomp' and h.dep_ is not 'neg':
                                for k in j.lefts:
                                    if k.dep_ is 'aux':
                                        xcomp_pairs.append(token.text+' '+k.text+' '+j.text)
                            elif j.dep_ is 'xcomp' and h.dep_ is 'neg':
                                if k.dep_ is 'aux':
                                        neg_pairs.append(h.text +' '+token.text+' '+k.text+' '+j.text)
            
            else:
                enemlist.append(enem_list)
                doc = nlp(line)
                str1=''
                str2=''
                for token in doc:
                    if token.pos_ is 'NOUN':
                        for j in token.lefts:
                            if j.dep_ == 'compound':
                                ecompound_pairs.append((j.text+' '+token.text,token.text))
                            if j.dep_ is 'amod' and j.pos_ is 'ADJ': #primary condition
                                str1 = j.text+' '+token.text
                                eamod_pairs.append(j.text+' '+token.text)
                                for k in j.lefts:
                                    if k.dep_ is 'advmod': #secondary condition to get adjective of adjectives
                                        str2 = k.text+' '+j.text+' '+token.text
                                        eamod_pairs.append(k.text+' '+j.text+' '+token.text)
                                mtch = re.search(re.escape(str1),re.escape(str2))
                                if mtch is not None:
                                    eamod_pairs.remove(str1)
                    if token.pos_ is 'VERB':
                        for j in token.lefts:
                            if j.dep_ is 'advmod' and j.pos_ is 'ADV':
                                eadvmod_pairs.append(j.text+' '+token.text)
                            if j.dep_ is 'neg' and j.pos_ is 'ADV':
                                eneg_pairs.append(j.text+' '+token.text)
                        for j in token.rights:
                            if j.dep_ is 'advmod'and j.pos_ is 'ADV':
                                eadvmod_pairs.append(token.text+' '+j.text)
                    if token.pos_ is 'ADJ':
                        for j in token.rights:
                            if j.dep_ is 'xcomp':
                                for k in j.lefts:
                                    if k.dep_ is 'aux':
                                        excomp_pairs.append(token.text+' '+k.text+' '+j.text)
        pairs = list(set(amod_pairs+advmod_pairs+neg_pairs+xcomp_pairs))
        epairs = list(set(eamod_pairs+eadvmod_pairs+eneg_pairs+excomp_pairs))
        for i in range(len(pairs)):
            if len(compound_pairs)!=0:
                for comp in compound_pairs:
                    mtch = re.search(re.escape(comp[1]),re.escape(pairs[i]))
                    if mtch is not None:
                        pairs[i] = pairs[i].replace(mtch.group(),comp[0])
        for i in range(len(epairs)):
            if len(ecompound_pairs)!=0:
                for comp in ecompound_pairs:
                    mtch = re.search(re.escape(comp[1]),re.escape(epairs[i]))
                    if mtch is not None:
                        epairs[i] = epairs[i].replace(mtch.group(),comp[0])
            
    aspect_terms.append(pairs)
    comp_terms.append(compound_pairs)
    easpect_terms.append(epairs)
    ecomp_terms.append(ecompound_pairs)
    enemy.append(enemlist)
data['compound_nouns'] = comp_terms
data['aspect_keywords'] = aspect_terms
data['competition'] = enemy
data['competition_comp_nouns'] = ecomp_terms
data['competition_aspects'] = easpect_terms
data.head()


# In[ ]:


type(data)


# In[ ]:


data.aspect_keywords


# In[ ]:


data.head()


# In[ ]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
count = 0
global_sentiment = {}
for aspect_adj_tuple in data.aspect_keywords.iteritems():
    row_sentiment = {}
#     print(aspect_adj_tuple[0])
#     print(aspect_adj_tuple[1])
    for text in aspect_adj_tuple[1]:
        sentiment_array = []
        polarity = analyser.polarity_scores(text)
        sentiment_array.append(polarity['pos'])
        sentiment_array.append(polarity['neg'])
        row_sentiment[text] = sentiment_array
        
    global_sentiment[aspect_adj_tuple[0]] = row_sentiment
    
print(global_sentiment)


# In[ ]:


pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000
pd.options.display.max_colwidth = 1000


# In[ ]:


data[['review_body', 'aspect_keywords']].head(10)


# ## We use vaderSentiment for sentiment analysis because of it's speed and simplicity. It offers 3 types of polarity -  positive, negative and neutral. As a result we can filter all aspects which have high neutral scores hence minimizing errors caused due to wrong extraction of aspects and stopwords

# In[ ]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()


# In[ ]:


import operator
sentiment = []
for i in range(len(data)):
    score_dict={'pos':0,'neg':0,'neu':0}
    if len(data['aspect_keywords'][i])!=0: 
        for aspects in data['aspect_keywords'][i]:
            sent = analyser.polarity_scores(aspects)
            score_dict['neg'] += sent['neg']
            score_dict['pos'] += sent['pos']
        #score_dict['neu'] += sent['neu']
        sentiment.append(max(score_dict.items(), key=operator.itemgetter(1))[0])
    else:
        sentiment.append('NaN')
data['sentiment'] = sentiment
data.head()


# In[ ]:


int_sent = []
for sent in data['sentiment']:
    if sent is 'NaN':
        int_sent.append('NaN')
    elif sent is 'pos':
        int_sent.append('1')
    else:
        int_sent.append('0')
data['int_sent'] = int_sent
data.head()


# ### Here we have arbitarily taken ratings greater than 3 as positive and everything else as negative

# In[ ]:


d = {'sent':toy_rev['Positive Review'],'sent_pred':toy_rev['int_sent']}
metric_df = pd.DataFrame(data=d)
metric_df.head()


# In[ ]:


len(metric_df.sent)


# ## Removing NaN values in the sentiment predictions

# In[ ]:


metric_df = metric_df[metric_df.sent_pred != 'NaN']
len(metric_df.sent)


# In[ ]:


from sklearn.metrics import accuracy_score,auc,f1_score,recall_score,precision_score
print('accuracy')
print(accuracy_score(metric_df.sent, metric_df.sent_pred))
print('f1 score')
print(f1_score(metric_df.sent, metric_df.sent_pred,pos_label='1'))
print('recall')
print(recall_score(metric_df.sent, metric_df.sent_pred,pos_label='1'))
print('precision')
print(precision_score(metric_df.sent, metric_df.sent_pred,pos_label='1'))


# ## Possible improvements that can be made
# *  Tricky situation of removing stopwords to reduce unwanted extractions of non-aspects but this can also affect spaCy's dependency parsing. Same goes with noun chunk merging as well. If someone can think of a better way to remove stopwords and still retain spaCy's dependency goodness it can greatly improve the accuracy
# 
# * This is not a ML task per se since we do more of parsing than ML. Although Bi-Directional LSTM have been very good at ABSA tasks in the past, unlike semeval tasks we do not have a fixed topic for our aspects to fall into. If someone can use the parsing aspect of the code to implement BLSTM in this case, that would be great
# 
# * better alternatives to vaderSentiment if available (unsupervised/ semi-supervised methods might be better here I think)
# 
# * The very definition of aspects can be a bit vague at times hence we do not have a valid metric to measure the aspect extraction's accuracy
# 

# ## P.S This is my first Kaggle Kernel and I am fairly new to python programming as well, hence my non usage of list comprehensions and functions might be evident. I highly encourage everyone to fork my code and add your own twists to increase the accuracy of both aspect extractions and sentiment analysis.
