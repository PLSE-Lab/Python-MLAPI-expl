#!/usr/bin/env python
# coding: utf-8

# <h1><center>What India talks about</center></h1>
# <img src="http://blog.iefa.org/wp-content/uploads/2014/10/indian-woman-thinking-477531487.jpg">
# This is a great data set to practice some text mining and visualization skills. Although we all know what newspapers usually talk about, but its a lot of fun to go back to a dataset and uncover some interesting insights. For example if you are surprised to find out that BJP got mentioned almost 1.5 times more than Congress in the past 16, years of headlines in a leading national daily, then you should read on.....
#    
#    ## TOC
# 1. Persistent themes
#    *    Unigram, bigrams and trigrams
#    *    Common n-grams across years
#    *   Are we a country who loves to read about crime?
#    * Are suicides on a rise in India?
# 2. Indian Political Scene: BJP vs Congress
#    - Congress: Good, bad and ugly
#    - BJP: Good, bad and ugly
#    - NaMo vs RaGa
# 3. Why does india love Shah Rukh?
#    - Word frequency and co-occurence analysis
# 4. General trivia:
#    - Startups, when did the country catch the train? 
#    - Analytics, does mainstream media care?
#    - Kingfisher: How the decline was chronicled?
#   

# In[1]:


import numpy as np 
import pandas as pd 
import spacy
from wordcloud import WordCloud
data=pd.read_csv("../input/india-headlines-news-dataset/india-news-headlines.csv")
data=data[['publish_date','headline_text']].drop_duplicates()
data['publish_date']=pd.to_datetime(data['publish_date'],format="%Y%M%d")
data['year']=data['publish_date'].dt.year
nlp=spacy.load("en_core_web_lg")


#  <h1><center> Persistent Themes </center></h1>
#  To get a gist of what are the themes, that are being talked about, I followed a general approach of doing frequency counts, of unigrams, bigrams and trigrams on the whole dataset as well as for each year. Find below the code I used to create some simple visuals

# In[2]:


#The following code takes a really long time, so have created pickled versions of these objects and reading them loc
'''### Get imp words by year
import sklearn.feature_extraction.text as text
def get_imp(bow,mf,ngram):
    tfidf=text.CountVectorizer(bow,ngram_range=(ngram,ngram),max_features=mf,stop_words='english')
    matrix=tfidf.fit_transform(bow)
    return pd.Series(np.array(matrix.sum(axis=0))[0],index=tfidf.get_feature_names()).sort_values(ascending=False).head(100)
### Global trends
bow=data['headline_text'].tolist()
total_data=get_imp(bow,mf=5000,ngram=1)
total_data_bigram=get_imp(bow=bow,mf=5000,ngram=2)
total_data_trigram=get_imp(bow=bow,mf=5000,ngram=3)
### Yearly trends
imp_terms_unigram={}
for y in data['year'].unique():
    bow=data[data['year']==y]['headline_text'].tolist()
    imp_terms_unigram[y]=get_imp(bow,mf=5000,ngram=1)
imp_terms_bigram={}
for y in data['year'].unique():
    bow=data[data['year']==y]['headline_text'].tolist()
    imp_terms_bigram[y]=get_imp(bow,mf=5000,ngram=2)
imp_terms_trigram={}
for y in data['year'].unique():
    bow=data[data['year']==y]['headline_text'].tolist()
    imp_terms_trigram[y]=get_imp(bow,mf=5000,ngram=3)
'''
import pickle
total_data=pd.read_pickle('../input/total-datapkl/total_data.pkl')
total_data_bigram=pd.read_pickle("../input/total-datapkl/total_data_bigram.pkl")
total_data_trigram=pd.read_pickle("../input/total-data-trigrampkl/total_data_trigram.pkl")
f=open("../input/total-data-trigrampkl/imp_terms_unigram.pkl","rb")
d=f.read()
imp_terms_unigram=pickle.loads(d)
f.close()
f=open("../input/total-data-trigrampkl/imp_terms_biigram.pkl","rb")
d=f.read()
imp_terms_bigram=pickle.loads(d)
f.close()
f=open("../input/total-data-trigrampkl/imp_terms_triigram.pkl","rb")
d=f.read()
imp_terms_trigram=pickle.loads(d)
f.close()
### Common unigrams across all the years
common_unigram={}
for y in np.arange(2001,2017,1):
    if y==2001:       
        common_unigram[y]=set(imp_terms_unigram[y].index).intersection(set(imp_terms_unigram[y+1].index))
    else:
        common_unigram[y]=common_unigram[y-1].intersection(set(imp_terms_unigram[y+1].index))
### Common bigrams across all the years
common_bigram={}
for y in np.arange(2001,2017,1):
    if y==2001:
         common_bigram[y]=set(imp_terms_bigram[y].index).intersection(set(imp_terms_bigram[y+1].index))
    else:
        common_bigram[y]=common_bigram[y-1].intersection(set(imp_terms_bigram[y+1].index))
### Common trigrams, 1 year window
common_trigram_1yr={}
for y in np.arange(2001,2017,1):
    common_trigram_1yr[str(y)+"-"+str(y+1)]=set(imp_terms_trigram[y].index).intersection(set(imp_terms_trigram[y+1].index))
### Commin trigrams, 2 year window
common_trigram_2yr={}
for y in np.arange(2001,2015,3):
    if y==2001:
        common_trigram_2yr[str(y)+"-"+str(y+1)+"-"+str(y+2)]=set(imp_terms_trigram[y].index).intersection(set(imp_terms_trigram[y+1].index)).intersection(set(imp_terms_trigram[y+2].index))
    else:
        common_trigram_2yr[str(y)+"-"+str(y+1)+"-"+str(y+2)]=set(imp_terms_trigram[y].index).intersection(set(imp_terms_trigram[y+1].index)).intersection(set(imp_terms_trigram[y+2].index))


# <h2>Count of top 20 unigrams, bigrams and trigrams</h2>

# In[3]:


import matplotlib.pyplot as plt
plt.subplot(1,3,1)
total_data.head(20).plot(kind="bar",figsize=(25,10))
plt.title("Unigrams",fontsize=30)
plt.yticks([])
plt.xticks(size=20)
plt.subplot(1,3,2)
total_data_bigram.head(20).plot(kind="bar",figsize=(25,10))
plt.title("Bigrams",fontsize=30)
plt.yticks([])
plt.xticks(size=20)
plt.subplot(1,3,3)
total_data_trigram.head(20).plot(kind="bar",figsize=(25,10))
plt.title("Trigrams",fontsize=30)
plt.yticks([])
plt.xticks(size=20)


# Some observations that one can make here, it seems bollywood, particularly, Shah Rukh Khan is very famous (look at the trigrams). Also, you can notice that Narendra Modi, has had a fair share of headlines mentioning him in the past 16 years. Also, if you look at some of the bigrams and trigrams, you will find mention of **year old**, **year old girl**, **year old woman**. We will look at these tokens in more detail later. A final comment, if you look at unigrams, you will notice that **BJP** gets mentioned quite often. We will look at this also in detail, later

# <h2>Bigrams and Trigrams across years</h2>
# To get a sense of trend across years, I also plotted bigrams and trigrams across the years

# <h2> Top 5 Bigrams across years</h2>

# In[4]:


for i in range(1,18,1):
    plt.subplot(9,2,i)
    imp_terms_bigram[2000+i].head(5).plot(kind="barh",figsize=(20,25))
    plt.title(2000+i,fontsize=20)
    plt.xticks([])
    plt.yticks(size=20,rotation=5)


# <h2> Top 5 Trigrams across years</h2>

# In[5]:


for i in range(1,18,1):
    plt.subplot(9,2,i)
    imp_terms_trigram[2000+i].head(5).plot(kind="barh",figsize=(20,25))
    plt.title(2000+i,fontsize=20)
    plt.xticks([])
    plt.yticks(size=15,rotation=5)


# If you look at the trigrams and bigrams closely, you will realize, that reporting of crime, sports (cricket in particular) and Shah Rukh Khan is persistent!!!
# <img src="http://www.indiantelevision.com/sites/drupal7.indiantelevision.co.in/files/styles/smartcrop_800x800/public/images/tv-images/2017/05/31/SRK-KKR_0.jpg?itok=rIWW5rMD">
# 

# In[6]:


## Count of common tokens across the years
count_common_bi={}
for year in range(2001,2017,1):
    count_common_bi[year]=pd.Series()
    for word in common_bigram[year]:
        if year==2001:
            count_common_bi[year][word]=imp_terms_bigram[year][word]+imp_terms_bigram[year+1][word]
        else:
            count_common_bi[year][word]=count_common_bi[year-1][word]+imp_terms_bigram[year+1][word]


# <h2>Which bigrams have been conistently reported over years?</h2>
# The previous couple of plots, capture, what was reported the most overall in the past 16 years and what happened in each of these 16 years. The next question to ask is which stories were reported consistently every year in the past 16 years. The way I tackled this was by finding common bigrams for year 2001 to 2002, then for 2003 and 2001 and 2002 combined and so on. This was the result:
# <h3>Top 10 bigrams common across years<h3>

# In[7]:


for i in range(1,17,1):
    plt.subplot(9,2,i)
    count_common_bi[2000+i].sort_values(ascending=False).head(10).plot(kind="barh",figsize=(20,35))
    if (2000+i)==2001:
        plt.title(str(2000+i)+"-"+str(2000+i+1),fontsize=30)
    else:
        plt.title(":"+str(2000+i)+"-"+str(2000+i+1),fontsize=30)
    plt.xticks([])
    plt.yticks(size=20,rotation=5)


# <h2>Do we love to read about crime a lot?</h2>
# While looking at the plot created above, one thing that strikes you is that crime reporting is very persistent. All the tokens in the above figure are actually, telling you the common bigrams from one year to another. One thing that strikes you is the fact that token **year old** and **commits suicide** are very prominent across years.
# 
# <h3>Let's first fgure out the story behind the token <i>year old</i></h3>
# In order to figure out the **context** around *year old*,
# -  I found out which headlines contained this token
# - Extracted the noun and verbs that occur with this token
# 

# In[8]:


## Story of 'year old'
index=data['headline_text'].str.match(r'(?=.*\byear\b)(?=.*\bold\b).*$')
texts=data['headline_text'].loc[index].tolist()
noun=[]
verb=[]
for doc in nlp.pipe(texts,n_threads=16,batch_size=10000):
    try:
        for c in doc:
            if c.pos_=="NOUN":
                noun.append(c.text)
            elif c.pos_=="VERB":
                verb.append(c.text)            
    except:
        noun.append("")
        verb.append("")


# In[9]:


plt.subplot(1,2,1)
pd.Series(noun).value_counts().head(10).plot(kind="bar",figsize=(20,5))
plt.title("Top 10 Nouns in context of 'Year Old'",fontsize=30)
plt.xticks(size=20,rotation=80)
plt.yticks([])
plt.subplot(1,2,2)
pd.Series(verb).value_counts().head(10).plot(kind="bar",figsize=(20,5))

plt.title("Top 10 Verbs in context of 'Year Old'",fontsize=30)
plt.xticks(size=20,rotation=80)
plt.yticks([])


# You can clearly see that the context around **year old** is crime/violence (rape,kills,arrested) against women, mostly. Just to confim, our results, that we got via an NLP parse, let's look at some of the news items, where **year old** token actually occured
# 

# In[ ]:


data['headline_text'].loc[index].tolist()[0:20]


# <h2>Are sucides on a rise in India?</h2>
# Another, common bigram pattern that we had observed was *commits suicide*. For the records, the **suicide rate in India is lower than the world average**, source https://en.wikipedia.org/wiki/Suicide_in_India. However, it is surprising to note that instances of suicides, constantly make headlines in Indian daily. **Famer suicdes** are a big *political issue* in the country, is that the reason why instances of suicide get reported so much?
# 
# To get some sort of directional answer here, I again, reverted to an NLP based parse, extracting the nouns from the headlines which contained the token "commits suicide".
# 
# 

# In[ ]:


index_s=data['headline_text'].str.match(r'(?=.*\bcommits\b)(?=.*\bsuicide\b).*$')
text_s=data['headline_text'].loc[index].tolist()
noun_s=[]
for doc in nlp.pipe(text_s,n_threads=16,batch_size=1000):
    try:
        for c in doc:
            if c.pos_=='NOUN':
                noun_s.append(c.text)
    except:
        for c in doc:
            noun_s.append("") 


# In[ ]:


pd.Series(noun_s).value_counts().head(20).plot("bar",figsize=(15,5))
plt.xticks(fontsize=20)
plt.yticks([])
plt.ylabel("Frequency")
plt.title("Frequency of Nouns in the context of 'Commits Suicide'",fontsize=30)


# Its very surprising to not see "Famer" in the context of suicides. As a matter of fact, farm suicides are the top most reasons of sucides in India (https://data.gov.in/catalog/stateut-wise-professional-profile-suicide-victim). Farmer suicides were 37080 in 2014 compared to 24204 student suicides. Let's use regular expressions to find out, the trend in suicide reporting, we will:
# - First find out instances where "commits suicide" pattern occurs
# - Then figure out out of these, in how many instaces student and farmer ocucr respectively

# In[ ]:


index_s=data['headline_text'].str.match(r'(?=.*\bcommits\b)(?=.*\bsuicide\b).*$',case=False)
index_farmer=data.loc[index_s]['headline_text'].str.match(r'farmer',case=False)
index_stu=data.loc[index_s]['headline_text'].str.match(r'student',case=False)


# In[ ]:


print("Approximately {} percent of suicides reported were student related".format(round(np.sum(index_stu)/np.sum(index_s),2)*100))


# In[ ]:


print("Approximately {} percent of suicides reported were farmer related".format(round(np.sum(index_farmer)/np.sum(index_s),2)*100))


# Clearly, instances of farmer suicides are actually more in number than the ones reported by Times of India. Let's see, what are the keywords mentioned, when a "farmer" is mentioned in a headline?

# In[ ]:


ind_farmer=data['headline_text'].str.match(r'farmer|farmers',case=False)


# In[ ]:


text_f=data.loc[ind_farmer]['headline_text'].tolist()
noun_f=[]
verb_f=[]
for doc in nlp.pipe(text_f,n_threads=16,batch_size=1000):
    try:
        for c in doc:
            if c.pos_=='NOUN':
                noun_f.append(c.text)
            elif c.pos_=="VERB":
                verb_f.append(c.text)
    except:
        for c in doc:
            noun_f.append("") 
            verb_f.append("")


# In[ ]:


plt.subplot(1,2,1)
pd.Series(noun_f).value_counts()[2:].head(10).plot(kind="bar",figsize=(20,5))
plt.title("Top 10 Nouns in the context of 'Farmer(s)'",fontsize=25)
plt.xticks(size=20,rotation=80)
plt.yticks([])
plt.subplot(1,2,2)
pd.Series(verb_f).value_counts().head(10).plot(kind="bar",figsize=(20,5))
plt.title("Top 10 Verbs in the context of 'Farmer(s)'",fontsize=25)
plt.xticks(size=20,rotation=80)
plt.yticks([])


# <h1><center>Indian Political Scene: BJP vs Congress</center></h1>
# <img src="https://akm-img-a-in.tosshub.com/indiatoday/images/story/201703/congress-bjp_647_033117014707_0.jpg">
# <h2>Relative Frequency</h2>
# I calculated the frequency of how many times BJP and Congress occured in the corpus , here were the results

# In[ ]:


index_bjp=data['headline_text'].str.match(r"bjp.*$",case=False)
index_con=data['headline_text'].str.match(r"congress.*$",case=False)
print("BJP was mentioned {} times".format(np.sum(index_bjp)))
print("Congress was mentioned {} times".format(np.sum(index_con)))
print("BJP was mentioned {} times more than Congress".format(np.round(np.sum(index_bjp)/np.sum(index_con),2)))


# <h2>What were the headlines about BJP?</h2>

# In[ ]:


data_bjp=data.loc[index_bjp]
data_bjp['polarity']=data_bjp['headline_text'].map(lambda x: textblob.TextBlob(x).sentiment.polarity)
pos=" ".join(data_bjp.query("polarity>0")['headline_text'].tolist())
neg=" ".join(data_bjp.query("polarity<0")['headline_text'].tolist())
text=" ".join(data_bjp['headline_text'].tolist())


# In[ ]:


from wordcloud import WordCloud,STOPWORDS
import PIL
bjp_mask=np.array(PIL.Image.open("../input/image-masks/bjp.png"))
wc = WordCloud(max_words=500, mask=bjp_mask,width=5000,height=2500,background_color="white",stopwords=STOPWORDS).generate(text)
plt.figure( figsize=(30,15) )
plt.imshow(wc)
plt.yticks([])
plt.xticks([])
plt.axis("off")


# <h2>What did positive and negative headlines about BJP contain?</h2>

# In[ ]:


thumbs_up=np.array(PIL.Image.open("../input/image-masks/thumbsup.jpg"))
wc = WordCloud(max_words=500, mask=thumbs_up,width=5000,height=2500,background_color="white",stopwords=STOPWORDS).generate(pos)
thumbs_dn=np.array(PIL.Image.open("../input/image-masks/thumbsdown.jpg"))
wc1=WordCloud(max_words=500, mask=thumbs_dn,width=5000,height=2500,background_color="white",stopwords=STOPWORDS).generate(neg)
fig=plt.figure(figsize=(30,15))
ax=fig.add_subplot(1,2,1)
ax.imshow(wc)
ax.axis('off')
ax.set_title("Positive Headlines",fontdict={'fontsize':20})
ax=fig.add_subplot(1,2,2)
ax.imshow(wc1)
ax.axis('off')
ax.set_title("Negative Headlines",fontdict={'fontsize':20})


# <h2>What were the headlines about Congress?</h2>

# In[ ]:


data_con=data.loc[index_con]
data_con['polarity']=data_con['headline_text'].map(lambda x: textblob.TextBlob(x).sentiment.polarity)
pos=" ".join(data_con.query("polarity>0")['headline_text'].tolist())
neg=" ".join(data_con.query("polarity<0")['headline_text'].tolist())
text=" ".join(data_con['headline_text'].tolist())


# In[ ]:


con_mask=np.array(PIL.Image.open('../input/image-masks/congress.png'))
wc = WordCloud(max_words=500, mask=con_mask,width=5000,height=2500,background_color="white",stopwords=STOPWORDS).generate(text)
plt.figure( figsize=(30,15))
plt.imshow(wc)
plt.yticks([])
plt.xticks([])
plt.axis("off")


# <h2>What did positive and negative headlines about Congess contain?</h2>

# In[ ]:


wc = WordCloud(max_words=500, mask=thumbs_up,width=5000,height=2500,background_color="white",stopwords=STOPWORDS).generate(pos)
wc1=WordCloud(max_words=500, mask=thumbs_dn,width=5000,height=2500,background_color="white",stopwords=STOPWORDS).generate(neg)
fig=plt.figure(figsize=(30,15))
ax=fig.add_subplot(1,2,1)
ax.imshow(wc)
ax.axis('off')
ax.set_title("Positive Headlines",fontdict={'fontsize':20})
ax=fig.add_subplot(1,2,2)
ax.imshow(wc1)
ax.axis('off')
ax.set_title("Negative Headlines",fontdict={'fontsize':20})


# <h2>Modi vs Rahul Gandhi</h2>
# <h3>Relative Frequency</h3>
# Let's look at the number of times Modi and Rahul Gandhi occurs

# In[ ]:


index_nm=data['headline_text'].str.match("narendra modi",case=False)
index_rahul=data['headline_text'].str.match("rahul gandhi",case=False)
print("Modi has been mentioned {} times".format(np.sum(index_nm)))
print("Rahul Gandhi has been mentioned {} times".format(np.sum(index_rahul)))


# In[ ]:


data['year'].loc[index_nm].value_counts().reset_index().sort_values("index",ascending=True).plot('index','year',kind='bar',figsize=(20,8))
plt.title("Years vs Modi's mentions",fontsize=15)


# In[ ]:


data['year'].loc[index_rahul].value_counts().reset_index().sort_values("index",ascending=True).plot('index','year',kind='bar',figsize=(20,8))
plt.title("Years vs Rahul's mentions",fontsize=15)


# In[ ]:


text=" ".join(data.loc[index_rahul]['headline_text'].tolist())
rahul=np.array(PIL.Image.open("../input/image-masks/raga.jpg"))
wc = WordCloud(max_words=5000, mask=rahul,width=5000,height=2500,background_color="white",stopwords=STOPWORDS).generate(text)
plt.figure( figsize=(30,15))
plt.imshow(wc)
plt.yticks([])
plt.xticks([])
plt.title("WordCloud of Headlines about Rahul",fontsize=15)
plt.axis("off")


# This is still a work in progress. I wanted to do more, for example to look at some trivia and bollywood, but its already 3 in the morning. Will continue adding more to this kernel in coming days. Please comment below to let me know, your views and how can I improve this further, Please upvote the kernel if you enjoyed reading this.
