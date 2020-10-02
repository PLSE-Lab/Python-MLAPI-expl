#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from sklearn.feature_extraction.text import CountVectorizer
import re
from wordcloud import WordCloud
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis.sklearn


# In[ ]:


df=pd.read_csv('../input/democratic-debate-transcripts-2020/debate_transcripts_v3_2020-02-26.csv',encoding='ISO-8859-1')


# In[ ]:


df['date']=pd.to_datetime(df.date)
df['date']=df['date'].dt.date


# # Questions?

# ## Who have spoken more than 20 minutes in all debates?

# In[ ]:


speakers_debate=df.groupby(['speaker','date'])['speaking_time_seconds'].sum().reset_index()
speakers_debate.sort_values(by='date',inplace=True)
#Who have spoken more thank 20 minutes in every date?
top_speakerperdate=speakers_debate[speakers_debate['speaking_time_seconds'] >= 1200]
plt.figure(figsize=(20,7))
sns.barplot(x='date',y=top_speakerperdate['speaking_time_seconds']/60,hue='speaker',data=top_speakerperdate)
plt.ylabel('Speaking time in minute')
plt.title('Who have spoken more than 20 minutes in all debates?',fontsize=14)
plt.show()


# ## Who has talked the most?

# In[ ]:


plt.figure(figsize=(15,7))
mostlongspeech=df.groupby('speaker')['speaking_time_seconds'].sum()
top_speaker= mostlongspeech.sort_values(ascending=False)[:10]
sns.barplot(y=top_speaker.index,x=(top_speaker/60))
plt.title('Who has talked the most?',fontsize=14)
plt.ylabel('Speakers')
plt.xlabel('Speaking time in minute')
plt.show()


# # NLP processing

# In[ ]:


import re
from wordcloud import STOPWORDS
from nltk import FreqDist, word_tokenize
from nltk import bigrams, trigrams
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
# Make a set of stop words
additional_stopwords=('Well','well','will','make','us','we',
                      'I','make','got','need','want','think',
                      'going','go','one','thank','going',
                      'way','say','every','re','us','first',
                     'now','said','know','look','done','take',
                     'number','two','three','s','m',"t",
                      'let','don','tell','ve','put','maybe','whether','many', 'll','around','thing','Secondly','doesn','lot')
stopwords = set(STOPWORDS)
stopwords=stopwords.union(additional_stopwords)
d = '../input/masks-for-wordcloud/'
comments_mask = np.array(Image.open(d + 'united_states.png'))


# **Processing text:**

# In[ ]:


test_str = df.loc[11, 'speech']

def clean_text(text):
    text = re.sub(r'\n',' ', text) # Remove line breaks
    text = re.sub('\s+', ' ', text).strip() # Remove leading, trailing, and extra spaces
    text=  re.sub('\x92',' ', text )
    return text

def process_text(df):
    df['speech_clean'] = df['speech'].apply(lambda x: clean_text(x))
    return df

print("Original text: " + test_str)
print("Cleaned text: " + clean_text(test_str))
df = process_text(df)


# ## Top 20 common words

# In[ ]:


plt.figure(figsize=(10,10))
word_freq = FreqDist(w for w in word_tokenize(' '.join(df['speech_clean']).lower()) if 
                     (w not in stopwords) & (w.isalpha()))
df_word_freq = pd.DataFrame.from_dict(word_freq, orient='index', columns=['count'])
top20w = df_word_freq.sort_values('count',ascending=False).head(20)

sns.barplot(top20w['count'],top20w.index,color='blue')
plt.show()


# ## Top 20 bigrams and trigrams

# In[ ]:


fig,axes=plt.subplots(ncols=2,figsize=(20,10),dpi=100)
###bigrams
bigram = list(bigrams([w for w in word_tokenize(' '.join(df['speech_clean']).lower()) if 
              (w not in stopwords) & (w.isalpha())]))
fq = FreqDist(bg for bg in bigram)
bgdf = pd.DataFrame.from_dict(fq, orient='index', columns=['count'])
bgdf.index = bgdf.index.map(lambda x: ' '.join(x))
bgdf = bgdf.sort_values('count',ascending=False)

#trigrams
trigram = list(trigrams([w for w in word_tokenize(' '.join(df['speech_clean']).lower()) if 
              (w not in stopwords) & (w.isalpha())]))
tr_fq = FreqDist(bg for bg in trigram)
trdf = pd.DataFrame.from_dict(tr_fq, orient='index', columns=['count'])
trdf.index = trdf.index.map(lambda x: ' '.join(x))
trdf = trdf.sort_values('count',ascending=False)

sns.barplot(bgdf.head(20)['count'], bgdf.index[:20], ax=axes[1],color='green')
sns.barplot(trdf.head(20)['count'], trdf.index[:20],ax=axes[0], color='red')

axes[0].set_title('Top 20 Trigrams in Debates')
axes[1].set_title('Top 20 Bigrams in Debates')

plt.show()


# ##  America Wordcloud

# In[ ]:


long_string = ','.join(list(df['speech_clean'].values))
wordcloud = WordCloud(background_color='white',max_words=500, contour_width=5, contour_color='black',
                      width=800,height=400,stopwords=stopwords,mask=comments_mask)
wordcloud.generate(str(long_string))
wordcloud.to_image()


# # Topic modelling

# In[ ]:


vectorizer= CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
vectorized = vectorizer.fit_transform(df['speech'])
lda_model = LatentDirichletAllocation(n_components=10,learning_method='online',verbose=True,random_state=101)
lda_model_fitted = lda_model.fit_transform(vectorized)


# # Topic & Top 10 words for topic

# In[ ]:


for index,topic in enumerate(lda_model.components_):
    print(f'THE TOP 10 WORDS FOR TOPIC #{index}')
    print([vectorizer.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')


# In[ ]:


dictionary= {0:'American People',
            1:'Impeachment, Donald Trump, Elections',
            2:'National Security',
            3:'Education,Taxes,Health Insurance,Healt Care',
            4:'Global Diplomatic Issues',
            5:'Democratic Party Issues',
            6:'Gun Control',
            7:'Politics Issues',
            8:'International Relations and Conflicts',
            9:'Immigration and Criminal Justice'}


# In[ ]:


df['Topic'] = lda_model_fitted.argmax(axis=1)
df['Topic'].replace(dictionary,inplace=True)
topics=df[['speaker','speech_clean','Topic']]
top_candidate=topics.loc[topics.speaker.isin({'Bernie Sanders','Joe Biden',
                                              'Elizabeth Warren','Pete Buttigieg',
                                              'Amy Klobuchar'})]
topic_top_candidate=top_candidate.groupby(['speaker','Topic'])['speech_clean'].count().reset_index()


# In[ ]:


bernie_sanders=topic_top_candidate.loc[topic_top_candidate.speaker.isin({'Bernie Sanders'})]
joe_biden=topic_top_candidate.loc[topic_top_candidate.speaker.isin({'Joe Biden'})]
elizabeth_warren=topic_top_candidate.loc[topic_top_candidate.speaker.isin({'Elizabeth Warren'})]
pete_buttigieg=topic_top_candidate.loc[topic_top_candidate.speaker.isin({'Pete Buttigieg'})]
amy_klobuchar =topic_top_candidate.loc[topic_top_candidate.speaker.isin({'Amy Klobuchar'})]


labels_bernie = bernie_sanders['Topic'].values.tolist()
sizes_bernie = bernie_sanders['speech_clean'].values.tolist()

labels_biden = joe_biden['Topic'].values.tolist()
sizes_biden = joe_biden['speech_clean'].values.tolist()

labels_warren = elizabeth_warren['Topic'].values.tolist()
sizes_warren = elizabeth_warren['speech_clean'].values.tolist()

labels_buttigieg = pete_buttigieg['Topic'].values.tolist()
sizes_buttigieg = pete_buttigieg['speech_clean'].values.tolist()

labels_amy = amy_klobuchar['Topic'].values.tolist()
sizes_amy = amy_klobuchar['speech_clean'].values.tolist()


# ## Bernie Sanders 

# In[ ]:


explode = (0.1, 0 , 0, 0, 0, 0, 0, 0, 0, 0)  
fig, ax = plt.subplots(figsize=(20,10))

ax.pie(sizes_bernie, explode=explode,labels=labels_bernie, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax.set_title('Bernie Sanders',fontsize=20)
plt.tight_layout()
plt.show()


# ## Joe Biden

# In[ ]:


explode = (0.1, 0 , 0, 0, 0, 0, 0, 0, 0, 0)  
fig, ax = plt.subplots(figsize=(20,10))

ax.pie(sizes_biden, explode=explode,labels=labels_biden, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax.set_title('Joe Biden',fontsize=20)
plt.tight_layout()
plt.show()


# # Elizabeth Warren

# In[ ]:


explode = (0.1, 0 , 0, 0, 0, 0, 0, 0, 0, 0)  
fig, ax = plt.subplots(figsize=(20,10))

ax.pie(sizes_warren, explode=explode,labels=labels_warren, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax.set_title('Elizabeth Warren',fontsize=20)
plt.tight_layout()
plt.show()


# # Pete Buttigieg

# In[ ]:


explode = (0.1, 0 , 0, 0, 0, 0, 0, 0, 0, 0)  
fig, ax = plt.subplots(figsize=(20,10))

ax.pie(sizes_buttigieg, explode=explode,labels=labels_buttigieg, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax.set_title('Pete Buttigieg',fontsize=20)
plt.tight_layout()
plt.show()


# # Amy Klobuchar

# In[ ]:


explode = (0.1, 0 , 0, 0, 0, 0, 0, 0, 0, 0)  
fig, ax = plt.subplots(figsize=(20,10))

ax.pie(sizes_amy, explode=explode,labels=labels_amy, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax.set_title('Amy Klobuchar',fontsize=20)
plt.tight_layout()
plt.show()

