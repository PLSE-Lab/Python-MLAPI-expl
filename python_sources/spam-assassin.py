#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


from os import walk
from os.path import join 

import pandas as pd
import matplotlib.pyplot as plt


import nltk as nltk  

from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# constant
EXAMPLE_FILE = '/kaggle/input/spamdata/SpamData/01_Processing/practice_email.txt'

SPAM_1_PATH = '/kaggle/input/spamdata/SpamData/01_Processing/spam_assassin_corpus/spam_1/'
SPAM_2_PATH = '/kaggle/input/spamdata/SpamData/01_Processing/spam_assassin_corpus/spam_2/'
EASY_NONSPAM_1_PATH = '/kaggle/input/spamdata/SpamData/01_Processing/spam_assassin_corpus/easy_ham_1/'
EASY_NONSPAM_2_PATH = '/kaggle/input/spamdata/SpamData/01_Processing/spam_assassin_corpus/easy_ham_2/'

SPAM_CAT = 1
HAM_CAT = 0

DATA_JSON_FILE = '/kaggle/input/spamdata/SpamData/01_Processing/email-text-data.json'


# In[ ]:


# Reading files
stream = open(EXAMPLE_FILE, encoding='latin-1')
message = stream.read()
stream.close()
print(message)


# In[ ]:


import sys
sys.getfilesystemencoding()


# In[ ]:


print(type(message))


# In[ ]:


stream = open(EXAMPLE_FILE, encoding='latin-1')
is_body = False
lines = []

for line in stream:
    if is_body:
        lines.append(line)
    elif line == '\n':
        is_body = True
stream.close()

email_body = '\n'.join(lines)
print(email_body)


# ## Email body extraction

# In[ ]:


def email_body_generator(path):
    
    for root, dirnames, filenames in walk(path):
        for file_name in filenames:
            
            filepath = join(root, file_name)
            
            stream = open(filepath, encoding='latin-1')
            
            is_body = False
            lines = []

            for line in stream:
                if is_body:
                    lines.append(line)
                elif line == '\n':
                    is_body = True
            stream.close()

            email_body = '\n'.join(lines)
            #print(email_body)
            yield file_name , email_body


# In[ ]:


def df_from_directory(path, classification):
    rows = []
    row_names = []
    
    for file_name, email_body in email_body_generator(path):
        rows.append({'MESSAGE' : email_body, 'CATEGORY': classification})
        row_names.append(file_name)
    return pd.DataFrame(rows, index=row_names)
    


# In[ ]:


spam_emails = df_from_directory(SPAM_1_PATH, 1)
spam_emails = spam_emails.append(df_from_directory(SPAM_2_PATH, 1))


# In[ ]:


spam_emails.head()


# In[ ]:


spam_emails.shape


# In[ ]:


ham_emails = df_from_directory(EASY_NONSPAM_1_PATH, 0)
ham_emails = ham_emails.append(df_from_directory(EASY_NONSPAM_2_PATH, 0))
ham_emails.head()


# In[ ]:


ham_emails.shape


# In[ ]:


data = pd.concat([spam_emails, ham_emails])
print('Shape of entire dataframe is ', data.shape)
data.head()


# In[ ]:


data.tail()


# In[ ]:


data.sample(5)


# ## Data cleaning : Checking for missing value

# In[ ]:


data['MESSAGE'].isnull().sum()


# In[ ]:


data['MESSAGE'].isnull().value_counts()


# In[ ]:


data['MESSAGE'].isna().sum()


# In[ ]:


data['MESSAGE'].isna().value_counts()


# In[ ]:


data['MESSAGE'].isna().any()


# In[ ]:


# Check if there sre empy emails (string lenth zero)
(data.MESSAGE.str.len() == 0).any()


# In[ ]:



(data.MESSAGE.str.len() == 0).sum()


# In[ ]:


data[(data.MESSAGE.str.len() == 0)]


# In[ ]:


data.MESSAGE.isnull().sum()


# ## **Locate empty emails**

# In[ ]:


data[(data.MESSAGE.str.len() == 0)].index


# In[ ]:


data.index.get_loc('cmds')


# ## remove system file entries

# In[ ]:


data.drop(['cmds'], inplace=True)


# In[ ]:


data[(data.MESSAGE.str.len() == 0)].index


# In[ ]:


data.shape


# ## Add document IDs to track Emails in Dataset

# In[ ]:


document_ids = range(0, len(data.index))
document_ids


# In[ ]:


data['Doc_id'] = document_ids


# In[ ]:


data.head()


# In[ ]:


data['FILE_NAME'] = data.index


# In[ ]:


data.head()


# In[ ]:


data = data.set_index('Doc_id')


# In[ ]:


data.head()


# In[ ]:


data.sample(5)


# In[ ]:


data.sample(5)


# In[ ]:


#saving datafile
data.to_json('/kaggle/working/DATA_JSON_FILE')


# ## Data Visualisation

# In[ ]:


# NUmber of spam message
data['CATEGORY'].value_counts()


# In[ ]:


amount_of_spam = data.CATEGORY.value_counts()[1]
amount_of_ham = data.CATEGORY.value_counts()[0]


# In[ ]:


category_name = ['Spam', 'Legit Mail']
sizes = [amount_of_spam, amount_of_ham]

plt.figure(figsize=(6, 6), dpi=107)
plt.pie(sizes, labels=category_name, textprops={'fontsize': 16}, autopct='%1.1f%%', explode=[0, 0.1])
plt.show()


# In[ ]:


category_name = ['Spam', 'Legit Mail']
sizes = [amount_of_spam, amount_of_ham]

plt.figure(figsize=(6, 6), dpi=107)
plt.pie(sizes, labels=category_name, textprops={'fontsize': 12}, autopct='%1.0f%%',startangle=90, )


# draw circle

centre_circle = plt.Circle((0,0), radius=0.6, fc='white')
plt.gca().add_artist(centre_circle)
plt.show()


# In[ ]:


category_names = ['Spam', 'Legit Mail', 'Updates', 'Promotions']
sizes = [25, 43, 19, 22]
custom_colours = ['#ff7675', '#74b9ff', '#55efc4', '#ffeaa7']
offset = [0.05, 0.05, 0.05, 0.05]

plt.figure(figsize=(2, 2), dpi=227)
plt.pie(sizes, labels=category_names, textprops={'fontsize': 6}, startangle=90, 
       autopct='%1.0f%%', colors=custom_colours, pctdistance=0.8, explode=offset)

# draw circle
centre_circle = plt.Circle((0, 0), radius=0.6, fc='white')
plt.gca().add_artist(centre_circle)

plt.show()


# # Natural Language Processing

# ## Text Preprocessing

# In[ ]:


# Convert to llower case
msg = "All the person In this World are good by Heart."
msg.lower()


# In[ ]:


nltk.download('punkt')


# In[ ]:


nltk.download('stopwords')


# In[ ]:


nltk.download('gutenberg')


# In[ ]:


nltk.download('shakespeare')


# ## Tokenising

# In[ ]:


msg = 'All work and no play makes Jack a dull boy.'
word_tokenize(msg.lower())


# ## Remmoving stop word

# In[ ]:


stop_words = set(stopwords.words('english'))


# In[ ]:


type(stop_words)


# In[ ]:


if 'this'  in stop_words: print('Found it ')


# In[ ]:


# Challenge: print out 'Nope. Not in here' if the word "hello" is not contained in stop_words
if 'hello' not in stop_words: print('Nope!!, Its not there')


# In[ ]:


msg = 'All work and no play makes Jack a dull boy. To be or not to be.'
words = word_tokenize(msg.lower())

filtered_words = []
for word in words:
    if word not in stop_words:
        filtered_words.append(word)
        
print(filtered_words)


#  **Word stems**

# In[ ]:


msg = 'All work and no play makes Jack a dull boy. To be or not to be. \ Nobody expects the Spanish Inquisition!'
words = word_tokenize(msg.lower())

stemmer = PorterStemmer()

filtered_words = []
for word in words:
    if word not in stop_words:
        stemmed_word = stemmer.stem(word)
        filtered_words.append(stemmed_word)
        
print(filtered_words)


# In[ ]:


msg = 'All work and no play makes Jack a dull boy. To be or not to be. \ Nobody expects the Spanish Inquisition!'
words = word_tokenize(msg.lower())

stemmer = SnowballStemmer('english')

filtered_words = []
for word in words:
    if word not in stop_words:
        stemmed_word = stemmer.stem(word)
        filtered_words.append(stemmed_word)
        
print(filtered_words)


# **Removing Panctuations**

# In[ ]:


msg = 'All work and no play makes Jack a dull boy. To be or not to be. \ Nobody expects the Spanish Inquisition!'

words = word_tokenize(msg.lower())
stemmer = SnowballStemmer('english')
filtered_words = []

for word in words:
    if word not in stop_words:
        if word.isalpha():
            stemmed_word = stemmer.stem(word)
            filtered_words.append(stemmed_word)
        
print(filtered_words)


# In[ ]:


msg = 'All work and no play makes Jack a dull boy. To be or not to be. \ Nobody expects the Spanish Inquisition!'

words = word_tokenize(msg.lower())
stemmer = SnowballStemmer('english')
filtered_words = []

for word in words:
    if word not in stop_words and word.isalpha():
        stemmed_word = stemmer.stem(word)
        filtered_words.append(stemmed_word)
        
print(filtered_words)


# In[ ]:


data.at[2, 'MESSAGE']


# In[ ]:


data.head()


# In[ ]:


from bs4 import BeautifulSoup


# In[ ]:


soup = BeautifulSoup(data.at[0, 'MESSAGE'], 'html.parser')


# In[ ]:


print(soup.prettify())


# In[ ]:


soup.get_text()


# ## Function for email processing

# In[ ]:




