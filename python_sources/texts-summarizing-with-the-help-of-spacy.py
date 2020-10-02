#!/usr/bin/env python
# coding: utf-8

# # Introduction
# > In this notebook, I will try to explore the 10% Stackoverflow Dataset. It contains thousands of questions and answer sessions in Stackoverflow. And at the end of this notebook, I will try to make simple text summarizer that will summarize given question. The summarized question can be used as a question title also.I will use Spacy as natural language processing library for handling this project. 

# In[ ]:


import numpy as np # linear algebra
import spacy
nlp = spacy.load('en_core_web_sm')
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from IPython.display import display
import base64
import string
import re
from collections import Counter
from time import time
# from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords
from nltk.corpus import stopwords
import nltk
import heapq
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger("lda").setLevel(logging.WARNING)

stopwords = stopwords.words('english')
sns.set_context('notebook')


# # 1. Import Dataset
# > In this section, I will load the desired dataset for this notebook. This dataset has huge number of question and answer. It will be hard to work with full dataset. So I will randomly sample the dataset into smaller chunks for easy purpose. 

# In[ ]:


df_questions = pd.read_csv("../input/Questions.csv", nrows=5000,usecols =['Score', 'Title', 'Body'],encoding='latin1')
df_questions = df_questions.dropna()
df_questions.head(15)


# There is some HTML tag and different code syntax. I need to remove those because those codes and wired punctuations are not beneficial for this project. I will be using python build in libraries for handling regex and spacy for NLP taskl 

# # 2. Text preprocessing
# > In this step, I will be using Spacy for preprocessing text, in others words I will clearing not useful features from question title like punctuation, stopwords. For this task,  there are two useful libraries available in Python. 1. NLTK 2. Spacy. In this notebook, I will be working with Spacy because it is very fast and has many useful features compared to NLTK. So without further do let's get started! 

# In[ ]:


get_ipython().system('python -m spacy download en_core_web_lg')
nlp = spacy.load('en_core_web_lg')
def normalize_text(text):
    tm1 = re.sub('<pre>.*?</pre>', '', text, flags=re.DOTALL)
    tm2 = re.sub('<code>.*?</code>', '', tm1, flags=re.DOTALL)
    tm3 = re.sub('<[^>]+>', '', tm1, flags=re.DOTALL)
    return tm3.replace("\n", "")


# In[ ]:


# in this step we are going to remove code syntax from text 
df_questions['Body_Cleaned_1'] = df_questions['Body'].apply(normalize_text)


# In[ ]:


print('Before normalizing text::::::::::\n')
print(df_questions['Body'][2])
print('\nAfter normalizing text:::::::::\n')
print(df_questions['Body_Cleaned_1'][2])


# We can see a huge difference after normalizing our text. Now we can see our text is more manageable. This will help us to explore the question and later making summarizer.  
# 
# We are also seeing that there are some punctuation and stopwords. We also don't need them. In the first place, I don't remove them because we are gonna need this in future when we will make summarizer. So let's make another column that will store our normalized text without punctuation and stopwords. 

# In[ ]:


# Clean text before feeding it to spaCy
punctuations = '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~'
# Define function to cleanup text by removing personal pronouns, stopwords, and puncuation
def cleanup_text(docs, logging=False):
    texts = []
    doc = nlp(docs, disable=['parser', 'ner'])
    tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
    tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations]
    tokens = ' '.join(tokens)
    texts.append(tokens)
    return pd.Series(texts)


df_questions['Body_Cleaned'] = df_questions['Body_Cleaned_1'].apply(lambda x: cleanup_text(x, False))


# In[ ]:


print('Question Body with punctuatin and stopwords:::::::::::\n')
print(df_questions['Body_Cleaned_1'][0])
print('\nQuestion Body after removing punctuation and stopwrods:::::::::::::\n')
print(df_questions['Body_Cleaned'][0])


# Wow! See! Now our text looks much readable and less messy! 

# # 3. Distribution of Score
# > In this section, I will try understand the distribution of score. Here score mean number of upvote the question got in stackoverflow. 

# In[ ]:


plt.subplot(1, 2, 1)
(df_questions['Score']).plot.hist(bins=30, figsize=(30,5), edgecolor='white',range=[0,250])
plt.xlabel('Number of upvotes', fontsize=17)
plt.ylabel('frequency', fontsize=17)
plt.tick_params(labelsize=15)
plt.title('Number of upvotes distribution', fontsize=17)
plt.show()


# The distribution of upvotes lies between 0 to 100 mostly. Majority of the question got upvote between 0 to 10. 

# # 4. Analyze question body
# > In this section, I will try to analyze question body. In StackOverflow, the question body plays a vital role. A good description can make your question stand out. It also helps get an answer faster. Lastly, It will help you get some upvote. Let's see what we can find in the question body. 

# In[ ]:


df_questions['Title_len'] = df_questions['Body_Cleaned'].str.split().str.len()
df = df_questions.groupby('Title_len')['Score'].mean().reset_index()
trace1 = go.Scatter(
    x = df['Title_len'],
    y = df['Score'],
    mode = 'lines+markers',
    name = 'lines+markers'
)
layout = dict(title= 'Average Upvote by Question Body Length',
              yaxis = dict(title='Average Upvote'),
              xaxis = dict(title='Question Body Length'))
fig=dict(data=[trace1], layout=layout)
py.iplot(fig)


# Hmm! We can see that questions got popular when the body length is not so long or not so short. 

# # 5. Question Summarizer
# > In this step, I will try to make a question summarizer. There is a huge amount of research going for text summarization. But I will try to do a simple technique for text summarization. The technique describes below. 

# ### 1. Convert Paragraphs to Sentences
# > We first need to convert the whole paragraph into sentences. The most common way of converting paragraphs to sentences is to split the paragraph whenever a period is encountered. 
# 
# ### 2. Text Preprocessing
# > After converting paragraph to sentences, we need to remove all the special characters, stop words and numbers from all the sentences. 
# 
# ### 3. Tokenizing the Sentences
# > We need to tokenize all the sentences to get all the words that exist in the sentences
# 
# ### 4. Find Weighted Frequency of Occurrence
# > Next we need to find the weighted frequency of occurrences of all the words. We can find the weighted frequency of each word by dividing its frequency by the frequency of the most occurring word.
# 
# ### 5. Replace Words by Weighted Frequency in Original Sentences
# > The final step is to plug the weighted frequency in place of the corresponding words in original sentences and finding their sum. It is important to mention that weighted frequency for the words removed during preprocessing (stop words, punctuation, digits etc.) will be zero and therefore is not required to be added
# 
# ### 6. Sort Sentences in Descending Order of Sum
# > The final step is to sort the sentences in inverse order of their sum. The sentences with highest frequencies summarize the text. 

# In[ ]:


# this is function for text summarization

def generate_summary(text_without_removing_dot, cleaned_text):
    sample_text = text_without_removing_dot
    doc = nlp(sample_text)
    sentence_list=[]
    for idx, sentence in enumerate(doc.sents): # we are using spacy for sentence tokenization
        sentence_list.append(re.sub(r'[^\w\s]','',str(sentence)))

    stopwords = nltk.corpus.stopwords.words('english')

    word_frequencies = {}  
    for word in nltk.word_tokenize(cleaned_text):  
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1


    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():  
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)


    sentence_scores = {}  
    for sent in sentence_list:  
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]


    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

    summary = ' '.join(summary_sentences)
    print("Original Text::::::::::::\n")
    print(text_without_removing_dot)
    print('\n\nSummarized text::::::::\n')
    print(summary)  


# Now we have written the function let's try to summarize some questions. 

# In[ ]:


generate_summary(df_questions['Body_Cleaned_1'][3], df_questions['Body_Cleaned'][3])


# In[ ]:


generate_summary(df_questions['Body_Cleaned_1'][5], df_questions['Body_Cleaned'][5])


# In[ ]:


generate_summary(df_questions['Body_Cleaned_1'][67], df_questions['Body_Cleaned'][68])


# That's awesome! We successfully made a simple question summarizer. 

# # 6. Conclusion
# > Thanks for reading this notebook. If you have any suggestion feel free to reach me in the comment. And don't forget to upvote. 
