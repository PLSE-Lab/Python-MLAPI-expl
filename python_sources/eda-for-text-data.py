#!/usr/bin/env python
# coding: utf-8

# # 1. Importing packages

# In[ ]:


import pandas as pd
from textblob import TextBlob
from plotly.offline import iplot
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
import string
import re
from wordcloud import WordCloud , STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# # 2. Loading and Reading the data

# In[ ]:


import os
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data = pd.read_csv("../input/data.csv")


# # 3. Pre-processing

# In[ ]:


stop = set(stopwords.words('english'))
sno = nltk.stem.SnowballStemmer('english') #initialising the snowball stemmer

def cleanhtml(sentence): #function to clean the word of any html-tags
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', sentence)
    return cleantext

def cleanpunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    return cleaned


# In[ ]:


final_string=[]
for i, sent in enumerate(tqdm(data['raw_text'].values)):
    filtered_sentence=[]
    sent=cleanhtml(sent) # remove HTMl tags
    for w in sent.split():
        for cleaned_words in cleanpunc(w).split():
            if((cleaned_words.isalpha()) & (len(cleaned_words)>2)):
                if(cleaned_words.lower() not in stop):
                    s=(sno.stem(cleaned_words.lower())).encode('utf8')
                    filtered_sentence.append(s)

    str1 = b" ".join(filtered_sentence) #final string of cleaned words

    final_string.append(str1)
data['Cleaned_rawText']=final_string
data['Cleaned_rawText']=data['Cleaned_rawText'].str.decode("utf-8")


# ## Wordcloud:
# ### summarizing the context from data
# 

# In[ ]:


def plot_wordcloud(text,mask=None,max_words=500,max_font_size=100,figure_size=(24.0,16.0),title = None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='black',
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    width=800, 
                    height=400,
                    mask = mask)
    wordcloud.generate(str(text))
    
    plt.figure(figsize=figure_size)
    plt.imshow(wordcloud);
    plt.title(title, fontdict={'size': title_size, 'color': 'black', 
                              'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()      


# In[ ]:


plot_wordcloud(data['raw_text'], title="Word Cloud of raw_text")


# > <b>From wordcloud we can conclude that most of our raw_text summarize the context related to like <font color='red'>inspiring,
#     spirituality , masterpiece , mentally , simplistically. </font></b>

# <h1>4. Some Feature engineering:</h1>
# 
# <ol><b>Now let us create some meta features and then look at how they are distributed between the classes. The ones that we will create are</b>
#     <li> Number of words in the text</li>
#     <li> Number of unique words in the text</li>
#     <li> Number of characters in the text</li>
#     <li> Number of stopwords</li>
#     <li> Number of punctuations</li>
#     <li> Number of upper case words</li>
#     <li> Number of title case words</li>
#     <li> Average length of the words</li>
# </ol>

# In[ ]:


def features(data):
    #Number of words
    data['number_of_words']=data['raw_text'].apply(lambda x:len(str(x).split()))
    # Number of unique words in the text 
    data['number_unique_words']=data['raw_text'].apply(lambda x:len(set(str(x).split())))
    # Number of characters in the text
    data['num_char']=data['raw_text'].apply(lambda x:len(str(x)))
    # Number of stopwords in the text
    data['num_of_stopwords']=data['raw_text'].apply(lambda x:len([ w for w in str(x).lower().split() if w in STOPWORDS]))
    # Number of punctuation in the text
    data['num_punctuation'] = data['raw_text'].apply(lambda x:len([c for c in str(x) if c in string.punctuation]))
    # Number of upper case words
    data['num_words_upper']=data['raw_text'].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    return data


# In[ ]:


data = features(data)


# In[ ]:


# Average Word length
def avg_word(sentence):
    words = sentence.split(" ")
    return (sum(len(word) for word in words)/len(words))


# In[ ]:


data['raw_avg_word_length']=data['raw_text'].apply(lambda x: avg_word(x))


# In[ ]:


data.columns


# # 5. Checking the distribution of our feature

# ### 1. feature - num_char

# In[ ]:


data['num_char'].describe()


# In[ ]:


plt.figure(figsize=(10.0,8.0))
sns.boxplot(data['num_char'],orient='v')


# In[ ]:


plt.figure(figsize=(10.0,8.0))
sns.distplot(data['num_char'],kde=True)


# ### 2. feature - number_of_words

# In[ ]:


plt.figure(figsize=(10.0,8.0))
sns.distplot(data['number_of_words'])


# ### 3. feature - number_unique_words

# In[ ]:


plt.figure(figsize=(10.0,8.0))
sns.distplot(data['number_unique_words'])


# ### 4. feature - num_of_stopwords

# In[ ]:


plt.figure(figsize=(10.0,8.0))
sns.distplot(data['num_of_stopwords'])


# ### 5. feature - num_punctuation

# In[ ]:


plt.figure(figsize=(10.0,8.0))
sns.distplot(data['num_punctuation'])


# ### 6. feature - num_words_upper

# In[ ]:


plt.figure(figsize=(10.0,8.0))
sns.distplot(data['num_words_upper'])


# ### 7. feature - raw_avg_word_length

# In[ ]:


plt.figure(figsize=(10.0,8.0))
sns.distplot(data['raw_avg_word_length'])


# # 6. Polarity check for raw_text
# 
# ### Which raw_text is positive or negative

# In[ ]:


#By using textblob to get polarity of text
def get_polarity(text):
    try:
        pol = TextBlob(text).sentiment.polarity
    except:
        pol = 0.0
    return pol


# In[ ]:


data['raw_text_polarity'] = data['Cleaned_rawText'].apply(get_polarity)


# In[ ]:


# data.head()


# In[ ]:


# data[data['raw_text_polarity']==1].head(2)


# In[ ]:


# data[data['raw_text_polarity']==-1].head(2)


# In[ ]:


posdf = data[data['raw_text_polarity']==1]
negdf = data[data['raw_text_polarity']==-1]


# ## Wordcloud for positive and negative text

# In[ ]:


pos_text_data =" ".join(posdf.Cleaned_rawText)
neg_text_data = " ".join(negdf.Cleaned_rawText)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[20, 20])
wordcloud1 = WordCloud(background_color='white', height=400).generate(pos_text_data)
ax1.imshow(wordcloud1)
ax1.axis('off');
ax1.set_title('Positive text');

wordcloud2 = WordCloud(background_color='white', height=400).generate(neg_text_data)
ax2.imshow(wordcloud2)
ax2.axis('off');
ax2.set_title('Negative text');


# ><b><font color='red'>From the wordcloud we can observe that our raw_text features consist of positive and negative sentence like some of the follow:<br><font color='green'>Positive words: Best, Perfect ... <br> <font color='blue'>Negative words: Evil, Worst ...</font></b>

# In[ ]:




