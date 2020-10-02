#!/usr/bin/env python
# coding: utf-8

# ## Reading the data

# In[ ]:


import pandas as pd
AmazonData = pd.read_csv('C:/Users/sovon/Python ML/My Works/Final projects/Datasets/Text Mining/Amazon Reviews.csv')
print(AmazonData.shape)
AmazonData.head()


# In[ ]:


AmazonData.columns


# In[ ]:


FullData = pd.DataFrame(AmazonData['reviews.text'].values, columns=['Reviews'])


# In[ ]:


FullData.head()


# In[ ]:


ReviewsText = FullData['Reviews'].values
ReviewsText = str(ReviewsText)


# In[ ]:


ReviewsText


# In[ ]:


import re
CleanedReviewsText = re.sub(r'[^a-z A-Z]', r' ',ReviewsText)
CleanedReviewsText = re.sub(r'\b\w{1,3}\b', ' ', CleanedReviewsText)
CleanedReviewsText = CleanedReviewsText.lower()
CleanedReviewsText = re.sub(' +',' ', CleanedReviewsText)


# In[ ]:


CleanedReviewsText


# In[ ]:


from textblob import TextBlob
TextInBlob = TextBlob(CleanedReviewsText)


# In[ ]:


TextInBlob.sentiment


# In[ ]:


NounPhrases = TextInBlob.noun_phrases
NounPhrases
newNounPhrases = []
for i in NounPhrases:
    n = re.sub(' ', '_', i)
    newNounPhrases.append(n)


# In[ ]:


newNounPhrases


# In[ ]:


newNounPhrasesStr = ' '.join(newNounPhrases)
newNounPhrasesStr


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
wordcloudImage = WordCloud(max_words=50,
                           font_step=2 ,
                            max_font_size=500,
                            width=1000,
                            height=720
                          ).generate(newNounPhrasesStr)
plt.figure(figsize=(10,10))
plt.axis('off')
plt.imshow(wordcloudImage)
plt.show()


# In[ ]:


PositiveWords = pd.read_table('C:/Users/sovon/Python ML/My Works/Final projects/Datasets/Text Mining/Positive_words.txt', encoding='latin')
NegativeWords = pd.read_table('C:/Users/sovon/Python ML/My Works/Final projects/Datasets/Text Mining/Negative_words.txt', encoding='latin')


# In[ ]:


NegativeWords.head()


# In[ ]:


def ComputeSentiment(inpData):
    PositiveScore = 0
    NegativeScore = 0
    OverallSentiment = ''
    inpData = inpData.lower()
    wordList = inpData.split()
    for words in wordList:
        if (words in PositiveWords.values):
            PositiveScore+=1
        if(words in NegativeWords.values):
            NegativeScore+=1
    if((PositiveScore-NegativeScore)>0):
        OverallSentiment = 'Positive'
    else:
        OverallSentiment = 'Negative'
    return(OverallSentiment)


# In[ ]:


FullData['Sentiment'] = FullData['Reviews'].apply(ComputeSentiment)


# In[ ]:


FullData.head()


# In[ ]:


FullData['Sentiment'].unique()


# In[ ]:


FullData.groupby('Sentiment').size().plot.bar(color = ['grey', 'black'])


# In[ ]:


ReviewsText


# In[ ]:


import nltk
from nltk.tokenize import sent_tokenize, word_tokenize


# In[ ]:


sent_tokenize(ReviewsText)


# In[ ]:


from nltk import word_tokenize, pos_tag
words = word_tokenize(CleanedReviewsText)
wordsPOS = pos_tag(words)
OnlyAdjectives = (' ').join([POSTags[0] for POSTags in wordsPOS if POSTags[1] in ['JJ', 'JJR', 'JJS']])
print(OnlyAdjectives)


# In[ ]:


TextBlob(OnlyAdjectives).sentiment


# In[ ]:


patternsToFind = '''NP: {<JJ><VBG>}
                    NP:{<RB><JJ><NN>}'''
SampleText = word_tokenize(CleanedReviewsText)
SampleTextPOS = pos_tag(SampleText)
PatternParser = nltk.RegexpParser(patternsToFind)
ParsedResult = PatternParser.parse(SampleTextPOS)
for i in ParsedResult:
    if(type(i)==nltk.tree.Tree):
        print(i)

