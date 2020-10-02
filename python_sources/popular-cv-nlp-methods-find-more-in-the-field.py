#!/usr/bin/env python
# coding: utf-8

# **Understanding the popularity of Computer Vision and Natural Language Processing application among the Kaggle Survey takers**
# 
# Computer Vision and Natural Language Processing are two application fields of Machine Learning and Data Science which is finding increasing applications in building reallife application products. Ranging from building chatbots for customer support to document analysis NLP has found place in many user friendly products. On the other hand Self-Driving cars, surgical robots are all applications of Computer Vision field along with Robotics. 
# 
# **How is NLP and CV relevant for Kaggle?**
# Most of the popular courses offered on CV and NLP through platforms like Udacity and Coursera encourages the students to work on realtime application data available in Kaggle for developing skills in this field. Being the biggest hub for open source data, Kaggle plays an important role for fetching openly available data as well as using the notebook and hardware ( both GPU and TPU) service provided. This obiviously makes us believe that the recent Kaggle survey can tell us more about the popularity of CV and NLP among the users and their advanced methods. 
# 
# **Goal of this analysis:** Understand the frequency of users availing the CV and NLP resources available across the internet, the popular methods they are employing for building meaningful products. With this analysis we also look forward to find channels, groups etc to target for more CV and NLP users in the future surveys of Kaggle.
# 
# ****Approach:**** The survey had questionaire about the popular NLP methods and CV methods where the responses were typed in as text replies by the users. From the typed sentence we are trying to tokenize the words using NLTK Python library and extract the methods terms. With these method terms we try to build a WordCloud to spot the methods in the response and a frequency chart to understand their popularity. 

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


#Reading the CSV files into respective dataframes
questions = pd.read_csv('/kaggle/input/kaggle-survey-2019/questions_only.csv')
mcr = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')
otr = pd.read_csv('/kaggle/input/kaggle-survey-2019/other_text_responses.csv')
schema = pd.read_csv('/kaggle/input/kaggle-survey-2019/survey_schema.csv')


# From the description of data we understand the fields `Q26_OTHER_TEXT` and `Q27_OTHER_TEXT` of `other_text_responses.csv` gives the responses for the methods used for CV and NLP. So our primary analysis will be done on these fields. 

# In[ ]:


#Converting the string characters to lower case to bring in uniformity for the field Q26_OTHER_TEXT
text_lower_cv = pd.DataFrame(otr['Q26_OTHER_TEXT'].str.lower())
text_cv = text_lower_cv.Q26_OTHER_TEXT.unique()


# In[ ]:


# Installing NLTK to analyse the textual content of the field Q27_OTHER_TEXT
get_ipython().system('pip install nltk')


# In[ ]:


#Load nltk
import nltk


# In[ ]:


#Creating a list with all the responses
sentence_cv = ['']
i = 0
for i in range(len(text_cv)):
    sentence_cv.append(text_cv[i])
sentence_cv


# In[ ]:


#Deleting some of the responses that doesn't give any relevant methods
del sentence_cv[0:3]
sentence_cv.remove('am learning this')
del sentence_cv[11]
del sentence_cv[12]



# In[ ]:


#Replacing spaces with hyphen to get full names
sentence_cv[0] = 'time-based, lstm, i3d'
sentence_cv[3] ='video-analysis'
sentence_cv[10] ='glcm-wavelet'
sentence_cv[12] ='triplet-loss'
sentence_cv[13] ='triplet-loss'
sentence_cv[18] ='pose-estimation'
sentence_cv[19] ='pose-estimation'
sentence_cv[23] ='triplet-loss'
sentence_cv[29] ='i3d'
sentence_cv


# In[ ]:


#Creating onse single sentence with the unique value for further tokenization purpose
one_sentence_cv = "" 
for index, value in enumerate(sentence_cv):
    one_sentence_cv += (str(value)+",")
print(one_sentence_cv)


# In[ ]:


#text.dropna()
#Implement Word Tokenization
from nltk.tokenize import word_tokenize
tokenized_word_cv = word_tokenize(one_sentence_cv)

#Stopwords
from nltk.corpus import stopwords
stop_words_cv =  stopwords.words('english')
newStopWords_cv = [',','.','(',')','?','-']
stop_words_cv.extend(newStopWords_cv)

#Removing StopWords
filtered_sent_cv = []
for w in tokenized_word_cv:
    if w not in stop_words_cv:
        filtered_sent_cv.append(w)
        
#Frequency Distribution
from nltk.probability import FreqDist
fdist_cv = FreqDist(filtered_sent_cv)
print(fdist_cv)

fdist_cv.most_common(65)
    


# In[ ]:


#Creating a frequency distribution dataframe of the most used CV Methods
freq_words_cv = pd.DataFrame(filtered_sent_cv)
freq_words_cv.columns =['words']
freq_words_cv.head()
type(freq_words_cv)


# In[ ]:


#install wordcloud package using pip
get_ipython().system(' pip install wordcloud')


# In[ ]:


#Importing Matplotlib for plotting purpose
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#import sub features
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS


# In[ ]:


#WordCloud figure parameters
mpl.rcParams['figure.figsize']=(10.0,14.0)    #(6.0,4.0)
mpl.rcParams['font.size']=24              #10 
mpl.rcParams['savefig.dpi']=250           #72 
mpl.rcParams['figure.subplot.bottom']=.1


# In[ ]:


#Generating the worldcloud with the website name
wordcloud_cv = WordCloud(
                          background_color='white',
                          max_words=100,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(freq_words_cv['words']))


# In[ ]:


#Printing the wordcloud and storing in a png file
print(wordcloud_cv)
fig = plt.figure(1)
plt.imshow(wordcloud_cv)
plt.axis('off')
plt.show()


# Among the terms tokenized from the user responses we can see some of the terms like `i3d`,`lstm`,`cnn` etc are commonly known CV methods. To understand all the methods and frequency we can plot a frequency plot using these terms. 

# In[ ]:


#Frequency Distribution plot
import matplotlib.pyplot as plt
plt.tick_params(axis='x', which='major', labelsize=10)
plt.title('Frequently used CV Methods')
fdist_cv.plot(60,cumulative = False)
plt.show()


# **Observation:** The most popular methods used for CV are 'CNN','triplet-loss' with four and three members responding respectively followed by 'i3d','openCV','classification'. 
# However we see that the frequency of users using these methods confine to single digit.
# So, lets also understand the usage of NLP Methods. 

# In[ ]:


#Converting the string characters to lower case to bring in uniformity for the field Q27_OTHER_TEXT
text_lower = pd.DataFrame(otr['Q27_OTHER_TEXT'].str.lower())
text = text_lower.Q27_OTHER_TEXT.unique()


# In[ ]:


#Creating a list with all the response
sentence_nlp = ['']
i = 0
for i in range(len(text)):
    sentence_nlp.append(text[i])
sentence_nlp
 


# In[ ]:


#Deleting the sentences that are noise and doesn't give any relevant methods
del sentence_nlp[0:3]
del sentence_nlp[3]
sentence_nlp.remove('am learning this ')
sentence_nlp


# In[ ]:


#Replacing few spaces with hyphens to preserve the full form of the methods
sentence_nlp[23] = 'topic-modeling'
sentence_nlp[25] = 'stopwords, lemmatization, tfidf, bow'
sentence_nlp[27] = 'text-mining by r and python libraries only'
sentence_nlp


# In[ ]:


#Converting the unque responses into one sentence for further tokenization purpose
one_sentence_nlp = "" 
for index, value in enumerate(sentence_nlp):
    one_sentence_nlp += (str(value)+",")
print(one_sentence_nlp)


# In[ ]:


#text.dropna()
#Implement Word Tokenization
from nltk.tokenize import word_tokenize
tokenized_word_nlp = word_tokenize(one_sentence_nlp)

#Stopwords
from nltk.corpus import stopwords
stop_words_nlp =  stopwords.words('english')
newStopWords_nlp = [',','.','(',')','?','-']
stop_words_nlp.extend(newStopWords_nlp)

#Removing StopWords
filtered_sent_nlp = []
for w in tokenized_word_nlp:
    if w not in stop_words_nlp:
        filtered_sent_nlp.append(w)
        
#Frequency Distribution
from nltk.probability import FreqDist
fdist_nlp = FreqDist(filtered_sent_nlp)
print(fdist_nlp)

fdist_nlp.most_common(65)
    


# In[ ]:


#Creating a Dataframe with the frequency of each word detected
freq_words_nlp = pd.DataFrame(filtered_sent_nlp)
freq_words_nlp.columns =['words']
freq_words_nlp.head()
type(freq_words_nlp)


# In[ ]:


#Generating the worldcloud with the website name
wordcloud_nlp = WordCloud(
                          background_color='white',
                          max_words=100,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(freq_words_nlp['words']))


# In[ ]:


#Printing the wordcloud and storing in a png file
print(wordcloud_nlp)
fig = plt.figure(1)
plt.imshow(wordcloud_nlp)
plt.axis('off')
plt.show()


# **Observation:** The common methods we can identify from this WordCloud are 'lemmatization','tfidf','scdv','ulmfit','svm','clustering','lstm' etc. To understand them better let us try to understand them in a frequency plot. 

# In[ ]:


#Frequency Distribution plot
import matplotlib.pyplot as plt
plt.tick_params(axis='x', which='major', labelsize=14)
plt.title('Frequently used NLP Methods')
fdist_nlp.plot(25,cumulative = False)
plt.show()


# **Observation:** The most popular method is 'tfidf' and 'word2vec' which is given as a response by two users. Rest methods of NLP such as spacy, sparse, scdv, flair, lstm,ocr,spss,ulmfit are all responded only by one user per method.
# 
# This makes us our reliability on the data very frail. This calls for fresh methods for survey to reach out to the community who are actively involved in handling NLP for their work or research. 

# We know that the survey was taken from about 19717 survey takers. However the unique responses for the methods used for CV and NLP seems to be very less compared to the total number of survey takers. This calls for us to explore more on the reason for this less responses for these questions as the response can give us an idea about:
# * popularity of CV and NLP among Kaggle users
# * understand scope of Kaggle usage among these users

# In[ ]:


#Identify number of NON NULL responses for CV and NLP Question
otr.count(axis = 0)


# **Observation:** Out of 19717 surveytakers only 38 users responded for the methods they use for NLP and 34 responded for the methods they use for CV. 
# Based only on this data we can say the NLP and CV methods are very unpopular and there aren't much users working on these fields. 
# However, with the huge number of competitions being held on Kaggle itself based on the fields of NLP and CV we are quite assured that there are immensely large number of users across Kaggle itself who work in these fields, but the probability is that we have missed out that population in this survey. 
# 
# Our goal is to target these missed out population in the future surveys. To understand this we can analyze the popular hosted notebooks among the survey takers and analyzing the profession of the survey takers can tell us whom whom we missed. 
# 
# Our Hypotheisis is that: With the profession we can understand whether they are NLP or CV engineers, students or professors who work in the field of NLP and CV. In the case of university students and professors the chances for them to respond for the methods based on NLP or CV are extremely high.
# With the hosted notebook medium we can understand which are the other popular forums where the survey links needs to be shares through emails, notifications or ads to reach to more users in this field. Combining these twoinformation for the future survey we can identify which professionals or students to target in future to get accurate information on their application on CV and NLP.

# In[ ]:


# initialize list of lists 
data_count = [['Kaggle',mcr['Q17_Part_1'].count()]
               ,['Colab',mcr['Q17_Part_2'].count()]
               ,['GCloud',mcr['Q17_Part_3'].count()]
               ,['MAzure',mcr['Q17_Part_4'].count()]
               ,['Paperspace',mcr['Q17_Part_5'].count()]
               ,['FloydHub',mcr['Q17_Part_6'].count()]
               ,['Bynder',mcr['Q17_Part_7'].count()]
               ,['IBMWatson',mcr['Q17_Part_8'].count()]
               ,['CodeOcean',mcr['Q17_Part_9'].count()]
               ,['AWS',mcr['Q17_Part_10'].count()]
               ,['None',mcr['Q17_Part_11'].count()]
               ,['Other',mcr['Q17_Part_12'].count()]]
notebook_type_count = pd.DataFrame(data_count, columns = ['Notebook', 'Users_Count'])        
notebook_type_count.head()


# In[ ]:


# x-coordinates of left sides of bars  
left = notebook_type_count['Notebook']
  
# heights of bars 
height = notebook_type_count['Users_Count']
  
# labels for bars 
plt.tick_params(axis='x', which='major', labelsize=8)
  
# plotting a bar chart 
plt.bar(left, height, 
        width = 0.8) 
  
# naming the x-axis 
plt.xlabel('Notebook') 
# naming the y-axis 
plt.ylabel('Height') 
# plot title 
plt.title('Popularity of Hosted Notebooks') 
  
# function to show the plot 
plt.show() 


# **Observation:** Followed by Kaggle is Google Colab and some of them who are not using any hosted Notebook.
# 
# **Possible community to be targetted:** As we see a lot users using Google Colab, various google forums can be platforms for sharing the survey links in future so that we can get more responses from people working in CV and NLP. 

# In[ ]:


import seaborn as sns
pt1 = mcr[['Q5']]
pt1 = pt1.rename(columns={"Q5": "Title"})
pt1.drop(0, axis=0, inplace=True)

# plotting to create pie chart 
plt.figure(figsize=(38,36))
plt.subplot(221)
pt1["Title"].value_counts().plot.pie(autopct = "%1.0f%%",colors = sns.color_palette("prism",5),startangle = 60,wedgeprops={"linewidth":2,"edgecolor":"k"},shadow =True)
plt.title("Title Distribution")


# **Observation:** Here we find 21% of student users contributing to this survey. However this community can probably consist of undergraduates pursuing any degree or not specifically doing research on NLP or CV. Similarly we do not see Professors or trainers who are guiding students. 
# 
# **Possible community to be targetted:** Sending survey links to universities doing research and projects on NLP and CV can help in increasing more people to respond with greater frequency and even newer methods. 

# **Conclusion:** With a number of applications increasing in CV and NLP Kaggle is a promising platform for developement and research in the field of CV and NLP. On reaching out to more Kaggle community members in future surveys it gives the budding NLP and CV scientists to find newer methods and popular methods to build promising application
