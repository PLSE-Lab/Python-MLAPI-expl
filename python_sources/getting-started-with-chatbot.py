#!/usr/bin/env python
# coding: utf-8

# # Chatbot Tourist Guide (ToGu)
# 
# View the source code on [Github](https://github.com/manzoormahmood/Tourist-Guide-Chatbot)
# 
# Watch the working chatbot video on [Youtube](https://youtu.be/DQl79__3xKU)
# 
# Lets get connected on [Linkedin](https://www.linkedin.com/in/manzoor-bin-mahmood/)
# 
# ### Upvote my work if you like it.

# ## What is a chatbot and its application
# A chatbot is an artificial intelligence software that can simulate a conversation with a user in natural language. A chatbot is used for interactions between people and services to enhance customer experience.
# 
# One of the important use of chatbot is as a tourist guide. It will help in providing information about an unknown city to the user. Information like important monuments in the city, famous foods, travel mode, etc. can be found easily.

# ## Step for getting started with any kind of chatbot.
# 
# * Setting up Environment
# * Text Gathering
# * Text cleaning
# * Word Embedding
# * Generating answer
# * Conversation

# ## Setting up Environment
# 
# ### Installing lib not included in kaggle(Using Internet)
# * pyttsx3 is used for getting answer from chatbot as a speech.
# * SpeechRecognition is used for speech input from user.
# * pyspellchecker is used for spelling checking 

# In[ ]:


get_ipython().system('pip install pyttsx3 #used for speech output')
get_ipython().system('pip install SpeechRecognition #used for speech input')
get_ipython().system('pip install pyspellchecker #spelling checker')


# ### Importing lib

# In[ ]:


import nltk
from spellchecker import SpellChecker
import urllib
import bs4 as bs
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
import warnings
warnings.filterwarnings("ignore")
import random
from sklearn.metrics.pairwise import cosine_similarity
import random
import string 
import pandas as pd
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os
from pandas import DataFrame
import pyttsx3 
import speech_recognition as sr
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True) 


# ## Text Gathering
# The time-consuming part of a chatbot is making the text corpus. For making a chatbot, details regarding the city have to be obtained. Information like weather details, monuments, food, travel mode, etc may be needed as a text corpus.
# 
# For this chatbot, we have obtained weather information from timeanddate.com and other details are obtained from Wikipedia.

# ### Weather details about city
# Weather details have been obtained from website using BeautifulSoup as a scrapping tool. Alternatively, other websites can also be scrapped for getting more detailed information about the weather. 

# In[ ]:


page1=requests.get('https://www.timeanddate.com/weather/india') #switch on the Internet option from right side window


# In[ ]:




def temp(topic):
    
    page = page1
    soup = BeautifulSoup(page.content,'html.parser')

    data = soup.find(class_ = 'zebra fw tb-wt zebra va-m')

    tags = data('a')
    city = [tag.contents[0] for tag in tags]
    tags2 = data.find_all(class_ = 'rbi')
    temp = [tag.contents[0] for tag in tags2]

    indian_weather = pd.DataFrame(
    {
        'City':city,
        'Temperature':temp
    }
    )
    
    df = indian_weather[indian_weather['City'].str.contains(topic.title())] 
    
    return (df['Temperature'])


# ### Scrape wiki for city details
# 
# Scrape city details from wiki. Again using BeautifulSoup.

# In[ ]:


def wiki_data(topic):
    
    topic=topic.title()
    topic=topic.replace(' ', '_',1)
    url1="https://en.wikipedia.org/wiki/"
    url=url1+topic

    source = urllib.request.urlopen(url).read()

    # Parsing the data/ creating BeautifulSoup object
    soup = bs.BeautifulSoup(source,'lxml')

    # Fetching the data
    text = ""
    for paragraph in soup.find_all('p'):
        text += paragraph.text

    import re
    # Preprocessing the data
    text = re.sub(r'\[[0-9]*\]',' ',text)
    text = re.sub(r'\s+',' ',text)
    text = text.lower()
    text = re.sub(r'\d',' ',text)
    text = re.sub(r'\s+',' ',text)
    
    
    return (text)


# ## Text Cleaning
# Now comes the most important part of nlp i.e. text cleaning. Without this we can't get useful results.
# * Removing special char
# * Stemming
# * Lemmatization
# * Stop words
# * Part of speech (POS)
# * Name entity recognition (NER)
# * Sentiment Analysis
# * Spelling checker
# * Tokenization
# * Creating dictionary for city names

# ### Removing special char
# Using this function all the special char are removed.

# In[ ]:


def rem_special(text):
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    return(text.translate(remove_punct_dict))

sample_text="I am sorry! I don't understand you."
rem_special(sample_text)


# ### Stemming
# Strip suffixes from the end of the word. 
# 
# Eating --> Eat
# 
# Going --> Go

# In[ ]:


from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 

def stemmer(text):
    words = word_tokenize(text) 
    for w in words:
        text=text.replace(w,PorterStemmer().stem(w))
    return text

stemmer("He is Eating. He played yesterday. He will be going tomorrow.")


# ### Lemmatization
# Determine that the two words have same root.
# 
# Corpus and Corpora--> Corpus (root)

# In[ ]:


lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

sample_text="rocks corpora better" #default noun
LemTokens(nltk.word_tokenize(sample_text))


# ### Stop Words
# Stop words are very frequently appearing words like 'a' , 'the', 'is'.

# In[ ]:


from nltk.tokenize.toktok import ToktokTokenizer
tokenizer = ToktokTokenizer()

stopword_list = nltk.corpus.stopwords.words('english')

def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

remove_stopwords("This is a sample sentence, showing off the stop words filtration.")


# ### Part of Speech (POS)
# The part-of-speech tag signifies whether the word is a noun, adjective, verb, and so on. Refer [this](https://spacy.io/usage/linguistic-features) page for more details

# In[ ]:


import spacy 
spacy_df=[]
spacy_df1=[]
df_spacy_nltk=pd.DataFrame()
nlp = spacy.load('en_core_web_sm') 
  
# Process whole documents 
sample_text = ("The heavens are above. The moral code of conduct is above the civil code of conduct") 
doc = nlp(sample_text) 
  
# Token and Tag 
for token in doc:
    spacy_df.append(token.pos_)
    spacy_df1.append(token)


df_spacy_nltk['origional']=spacy_df1
df_spacy_nltk['spacy']=spacy_df
#df_spacy_nltk


# ### Name Entity Recognition (NER)
# spaCy supports the following entity types:
# 
# PERSON
# 
# NORP (nationalities, religious and political groups)
# 
# FAC (buildings, airports etc.)
# 
# ORG (organizations)
# 
# GPE (countries, cities etc.)
# 
# LOC (mountain ranges, water bodies etc.)
# 
# PRODUCT (products)
# 
# EVENT (event names)
# 
# WORK_OF_ART (books, song titles)
# 
# LAW (legal document titles)
# 
# LANGUAGE (named languages)
# 
# DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL and CARDINAL.
# 
# 

# In[ ]:


import spacy 
nlp = spacy.load('en_core_web_sm') 

def ner(sentence):
    doc = nlp(sentence) 
    for ent in doc.ents: 
        print(ent.text, ent.label_) 
    

sentence = "A gangster family epic set in 1919 Birmingham, England; centered on a gang who sew razor blades in the peaks of their caps, and their fierce boss Tommy Shelby."
ner(sentence) 


# ### Sentiment Analysis
# The sentiment returns polarity. The polarity score is a float within the range [-1.0, 1.0]. 

# In[ ]:


from textblob import TextBlob

def senti(text):
    testimonial = TextBlob(text)
    return(testimonial.polarity)

sample_text="This apple is good"
print("polarity",senti(sample_text))
sample_text="This apple is not good"
print("polarity",senti(sample_text))


# ### Spelling checker
# Check for any incorrect spelling in text and find the correct spelling.
# 

# In[ ]:



spell = SpellChecker()


def spelling(text):
    splits = sample_text.split()
    for split in splits:
        text=text.replace(split,spell.correction(split))
        
    return (text)
    
    
sample_text="hapenning elephnt texte luckno sweeto"
spelling(sample_text)


# ### Tokenization
# It is the process of breaking sentence or word into tokens.

# In[ ]:


#TOkenisation
print(nltk.sent_tokenize("Hey how are you? I am fine."))
print(nltk.word_tokenize("Hey how are you? I am fine."))


# ### Creating dictionary for cities
# For chatbot the user may input city name in many form. For example "bangalore" can be referred as "bangalore", "bengaluru" or "blr". The chatbot shoud be able to identify to which city the user is referring to.

# In[ ]:


city = {} 
city["bangalore"]=["bangalore","bengaluru","blr"]
city["lucknow"] = ["lucknow", "lko"]
city["delhi"]=["new delhi","ndls","delhi"]


# In[ ]:


def city_name(sentence):
    for word in sentence.split():
        for key, values in city.items():
            
            if word.lower() in values:
                return(key)
                
    
city_name("blr")


# ## Word Embedding
# Word embedding is the representation of word so that it can be feed as input the machine learning models. ML model take input only in form of numerical values, so words are converted to vector form.
# 
# There are two type of word embedding. 
# 1. Frequency based: Count vector, TF-IDF
# 2. Prediction based. Continous bag of words (CBOW), skip-gram.

# ### TF-IDF

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
documentA = 'This is about Messi'
documentB = 'This is about TFIDF'
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([documentA, documentB])
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)
df


# *Note: TF-IDF is basic for word embedding with less accuracy. Other methods like CBOW, skip-gram can be used for better accuracy.If this notebook gets good response then I will add frequency also.*

# ## Generating answer
# This section combines all the codes and generated the output.

# ### Pre-processing all

# In[ ]:


def LemNormalize(text):
    text=rem_special(text) #remove special char
    text=text.lower() # lower case
    text=remove_stopwords(text) # remove stop words
    
    return LemTokens(nltk.word_tokenize(text))


# ### Cosine similarity

# In[ ]:


#Generating answer
def response(user_input):
    
    ToGu_response=''
    sent_tokens.append(user_input)
    
    
    
    word_vectorizer = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')   
    all_word_vectors = word_vectorizer.fit_transform(sent_tokens)  
    
   
    similar_vector_values = cosine_similarity(all_word_vectors[-1], all_word_vectors) 
    idx=similar_vector_values.argsort()[0][-2]
    

    matched_vector = similar_vector_values.flatten()
    matched_vector.sort()
    vector_matched = matched_vector[-2]
    
    if(vector_matched==0):
        ToGu_response=ToGu_response+"I am sorry! I don't understand you."
        return ToGu_response
    else:
        ToGu_response = ToGu_response+sent_tokens[idx]
        return ToGu_response


# ### Get city name
# Take input from user. Then fetch wiki data and weather details.

# In[ ]:


#topic=str(input("Please enter the city name you want to ask queries for: ")) #Uncomment this line to take input from command prompt or jupyter notebook.
topic="bangalore" # sample city

topic=city_name(topic) # fetch city name in case of invalid input or any discrepancy in the city name

text=wiki_data(topic) # fetch wiki data about city

sent_tokens = nltk.sent_tokenize(text)# converts to list of sentences 
word_tokens = nltk.word_tokenize(text)# converts to list of words
weather_reading=(temp(topic)).iloc[0] #fetch weather


# ### Greeting
# Generate greeting when user give any input among the variable GREETING_INPUTS.

# In[ ]:


# greetings Keyword matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey")
GREETING_RESPONSES = ["hi", "hey", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# ### Places
# This piece of code extracts the city details from corpus using name entity recognition(NER). As told earlier "FAC" is used for airports, city. This feature from spcay is used to find monuments and important tourist places in city.

# In[ ]:


PLACES_INPUTS = ("places", "monuments", "buildings","places", "monument", "building")

import spacy 
nlp = spacy.load('en_core_web_sm') 

def ner(sentence):
    places_imp=""
    doc = nlp(sentence) 
    for ent in doc.ents: 
        if (ent.label_=="FAC"):
            #print(ent.text, ent.label_) 
            places_imp=places_imp+ent.text+","+" "
            
    return(places_imp)
    

places_imp=ner(text) 


s=places_imp
l = s.split() 
k = [] 
for i in l: 
  
    # If condition is used to store unique string  
    # in another list 'k'  
    if (s.count(i)>1 and (i not in k)or s.count(i)==1): 
        k.append(i) 

PLACES_RESPONSES = ' '.join(k)

def places(sentence):
    for word in sentence.split():
        if word.lower() in PLACES_INPUTS:
            return (PLACES_RESPONSES)


# ### Weather
# Getting weathe details when user inputs any input among variable WEATHER_INPUTS.

# In[ ]:


WEATHER_INPUTS = ("weather", "temp", "temperature")

WEATHER_RESPONSES =weather_reading

def weather(sentence):
    for word in sentence.split():
        if word.lower() in WEATHER_INPUTS:
            return (WEATHER_RESPONSES)


# ### Chat
# This is the main chat function. Sentiment score is for finding the sentiment of the user input. According to user sentiment we can reply to the user. 
# 
# speak() function is used for speech output from chatbot. (Run on your local system)
# 

# In[ ]:


continue_dialogue=True
print("ToGu: Hello")
#speak("Hello")

while(continue_dialogue==True):
    user_input = input("User:")
    user_input=user_input.lower()
    user_input=spelling(user_input) #spelling check
    print("Sentiment score=",senti(user_input)) #sentiment score
    
    if(user_input!='bye'):
        if(user_input=='thanks' or user_input=='thank you' ):
            print("ToGu: You are welcome..")
            #speak(" You are welcome")
            
        else:
            if(greeting(user_input)!=None):
                tmp=greeting(user_input)
                print("ToGu: "+tmp)
                #speak(tmp)
                
            elif(weather(user_input)!=None):
                tmp=weather(user_input)
                print("ToGu: "+tmp)
                #speak(tmp)
                
                
            elif(places(user_input)!=None):
                tmp=places(user_input)
                print("ToGu: Important places are "+tmp)
                #speak("Important places are")
                #speak(tmp)
                
            else:
                print("ToGu: ",end="")
                temp=response(user_input)
                print(temp) 
                #speak(temp)
                sent_tokens.remove(user_input)
                

    else:
        continue_dialogue=False
        print("ToGu: Goodbye.")
        #speak("goodbye")
        



# # Conclusion
# Thank you for going through full notebook. Feel free to write comments and suggestions. If you want any particular topic to be added in this notebook plz let me know.
# 
# # **Upvote my notebook.**
# 
# *Links*
# 
# [Github](https://github.com/manzoormahmood/Tourist-Guide-Chatbot)
# 
# [Youtube](https://youtu.be/DQl79__3xKU)
# 
# [Linkedin](https://www.linkedin.com/in/manzoor-bin-mahmood/)
