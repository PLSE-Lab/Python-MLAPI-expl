#!/usr/bin/env python
# coding: utf-8

# Text Summarization with NLTK in Python

# Text summarization is a subdomain of Natural Language Processing (NLP) that deals with extracting summaries from huge chunks of texts. There are two main types of techniques used for text summarization: NLP-based techniques and deep learning-based techniques. In this article, we will see a simple NLP-based technique for text summarization. We will not use any machine learning library in this article. Rather we will simply use Python's NLTK library for summarizing Wikipedia articles.

# 
# 
#  
# 

# In[ ]:


pip install --upgrade pip


# **** * Fetching Articles from Wikipedia
#  Before we could summarize Wikipedia articles, we need to fetch them from the web. To do so we will use a couple of libraries. The first library that we need to download is the beautiful soup which is very useful Python utility for web scraping. Execute the following command at the command prompt to download the Beautiful Soup utility.**********

# In[ ]:


pip install beautifulsoup4


# In[ ]:


pip install lxml


# **Another important library that we need to parse XML and HTML is the lxml library. Execute the following command at command prompt to download lxml
# ******

# In[ ]:


pip install nltk


# **NLTK is a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers for industrial-strength NLP libraries, and an active discussion forum.**

# # Preprocessing
# The first preprocessing step is to remove references from the article. Wikipedia, references are enclosed in square brackets. The following script removes the square brackets and replaces the resulting multiple spaces by a single space. 
# 
# # Removing Square Brackets and Extra Spaces
# 
# The article_text object contains text without brackets. However, we do not want to remove anything else from the article since this is the original article. We will not remove other numbers, punctuation marks and special characters from this text since we will use this text to create summaries and weighted word frequencies will be replaced in this article.
# 
# To clean the text and calculate weighted frequences, we will create another object. 
# 
# # Removing special characters and digits
# 
# Now we have two objects article_text, which contains the original article and formatted_article_text which contains the formatted article. We will use formatted_article_text to create weighted frequency histograms for the words and will replace these weighted frequencies with the words in the article_text object.
# 
# # Converting Text To Sentences
# At this point we have preprocessed the data. Next, we need to tokenize the article into sentences. We will use thearticle_text object for tokenizing the article to sentence since it contains full stops. The formatted_article_text does not contain any punctuation and therefore cannot be converted into sentences using the full stop as a parameter.
# 
# # Find Weighted Frequency of Occurrence
# To find the frequency of occurrence of each word, we use the formatted_article_text variable. We used this variable to find the frequency of occurrence since it doesn't contain punctuation, digits, or other special characters.
# 
# In the script above, we first store all the English stop words from the nltk library into a stopwords variable. Next, we loop through all the sentences and then corresponding words to first check if they are stop words. If not, we proceed to check whether the words exist in word_frequency dictionary i.e. word_frequencies, or not. If the word is encountered for the first time, it is added to the dictionary as a key and its value is set to 1. Otherwise, if the word previously exists in the dictionary, its value is simply updated by 1.
# 
# Finally, to find the weighted frequency, we can simply divide the number of occurances of all the words by the frequency of the most occurring word.
# 
# # Calculating Sentence Scores
# We have now calculated the weighted frequencies for all the words. Now is the time to calculate the scores for each sentence by adding weighted frequencies of the words that occur in that particular sentence. 
# 
# n the script above, we first create an empty sentence_scores dictionary. The keys of this dictionary will be the sentences themselves and the values will be the corresponding scores of the sentences. Next, we loop through each sentence in the sentence_list and tokenize the sentence into words.
# 
# We then check if the word exists in the word_frequencies dictionary. This check is performed since we created the sentence_list list from the article_text object; on the other hand, the word frequencies were calculated using the formatted_article_text object, which doesn't contain any stop words, numbers, etc.
# 
# We do not want very long sentences in the summary, therefore, we calculate the score for only sentences with less than 30 words (although you can tweak this parameter for your own use-case). Next, we check whether the sentence exists in the sentence_scores dictionary or not. If the sentence doesn't exist, we add it to the sentence_scores dictionary as a key and assign it the weighted frequency of the first word in the sentence, as its value. On the contrary, if the sentence exists in the dictionary, we simply add the weighted frequency of the word to the existing value.
# 
# # Getting the Summary
# Now we have the sentence_scores dictionary that contains sentences with their corresponding score. To summarize the article, we can take top N sentences with the highest scores. The following script retrieves top 7 sentences and prints them on the screen.
# 
# In the script above, we use the heapq library and call its nlargest function to retrieve the top 7 sentences with the highest scores.

# In[ ]:


import bs4 as bs
import urllib.request
import re
import nltk

scraped_data = urllib.request.urlopen('https://en.wikipedia.org/wiki/Severe_acute_respiratory_syndrome_coronavirus_2')
article = scraped_data.read()

parsed_article = bs.BeautifulSoup(article,'lxml')

paragraphs = parsed_article.find_all('p')

article_text = ""

for p in paragraphs:
    article_text += p.text
# Removing Square Brackets and Extra Spaces
article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
article_text = re.sub(r'\s+', ' ', article_text)
# Removing special characters and digits
formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )
formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)
sentence_list = nltk.sent_tokenize(article_text)
stopwords = nltk.corpus.stopwords.words('english')

word_frequencies = {}
for word in nltk.word_tokenize(formatted_article_text):
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
import heapq
summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

summary = ' '.join(summary_sentences)
print(summary)

