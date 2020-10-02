#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import warnings; 
warnings.simplefilter('ignore')

from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[ ]:


df_review = pd.read_csv("../input/employee-reviews/employee_reviews.csv", sep=',', error_bad_lines=False)


# ## Data set description
# <ul>
# <li><b>company</b>: Name of the company which is being reviewed
# <li><b>location</b> : This dataset is global. This column shows the location of the company for which review has been given
# <li><b>dates</b>: Date on which review has been posted
# <li><b>job-title</b>: The designation of the employee who is posting review. Also includes whether the reviewer is a 'Current' or 'Former' Employee at the time of the review
# <li><b>Summary</b>: Short summary of employee review
# <li><b>pros</b>: Positive reviews of the company
# <li><b>cons</b>: Negative reviews of the company
# <li><b>overall ratings</b>: Overall rating from 1-5 that the employee has given to the company
# <li><b>work-balance-stars</b>: Rating from 1 -5 given to the work life balance within the company
# <li><b>Culture and Values Rating</b>: Rating from 1 -5 given to the culture and values of the company
# <li><b>Career Opportunities Rating</b>:  Rating from 1 -5 given for the career opportunities offered offered by the company 
# <li><b>Comp & Benefits Rating</b>: Rating from 1 -5 given for the compensation and benefits offered by the company
# <li><b>Senior Management Rating</b>: Rating from 1 -5 given to the senior managemet of the company
# <li><b>Helpful Review Count</b>: A count of how many people found the review to be helpful
# <li><b>Link to Review</b> : direct link to the page that contains the review

# ## Data Cleaning

# In[ ]:


df_review.replace(to_replace = 'none', value = np.nan, inplace = True)


# In[ ]:


df_review.rename(columns = {'dates':'date'}, inplace = True)


# In[ ]:


df_copy = df_review.copy()


# In[ ]:


df = df_copy.dropna()


# In[ ]:


df.head(1)


# In[ ]:


df.info()


# In[ ]:


df['date'] = df['date'].astype(dtype=np.datetime64, inplace=True)


# In[ ]:


df['overallratings'] = df['overall-ratings'].astype(dtype=np.float64)
df['work-balance-stars'] = df['work-balance-stars'].astype(dtype=np.float64)
df['culture-values-stars'] = df['culture-values-stars'].astype(dtype=np.float64)
df['carrer-opportunities-stars'] = df['carrer-opportunities-stars'].astype(dtype=np.float64)
df['comp-benefit-stars'] = df['comp-benefit-stars'].astype(dtype=np.float64)
df['senior-mangemnet-stars'] = df['senior-mangemnet-stars'].astype(dtype=np.float64)


# In[ ]:


df['is_current_employee'] = df['job-title'].apply(lambda x: 1 if 'Current' in x else 0)
df['is_high_Overall'] = df['overall-ratings'].apply(lambda x: 1 if x>3 else 0)
df['is_high_worbalance']= df['work-balance-stars'].apply(lambda x: 1 if x >3 else 0)
df['is_high_culturevalue']= df['culture-values-stars'].apply(lambda x: 1 if x >3 else 0)
df['is_high_careeropp']= df['carrer-opportunities-stars'].apply(lambda x: 1 if x >3 else 0)
df['is_high_compbenefit']= df['comp-benefit-stars'].apply(lambda x: 1 if x >3 else 0)
df['is_high_srmngmt']= df['senior-mangemnet-stars'].apply(lambda x: 1 if x >3 else 0)


# In[ ]:


sns.factorplot(x = 'overall-ratings', y = 'company',hue= 'is_current_employee', data = df, kind ='box',                aspect =2)


# #### Conclusion: The graph shows that the overall rating is best for Google & Facebook and worst for Netflix. The ex-employees as well as the former employees have given good ratings for both these companies. If we consider the reviews given by the current employees of these two companies, almost all of them fall under 3 to 5 and maximum out of them fall under 4 to 5. Like the dots show, very few of the employees have given 1 or 2.

# In[ ]:


sns.factorplot(x = 'work-balance-stars', y = 'company',hue= 'is_current_employee', data = df, kind ='box',                aspect =2)


# #### Conclusion: The graph shows that the work life balance at Google is the best and at Amazon is the worst so those candidates who are looking out for jobs and prefer good work life balance, should consider other options apart from Amazon

# In[ ]:


sns.factorplot(x = 'culture-values-stars', y = 'company', hue= 'is_current_employee', data = df, kind ='box',                aspect =2)


# #### Conclusion: The graph shows that the cultural values are the best at Google and Facebook and worst at Amazon and Netflix so those candidates who are looking out for jobs and prefer good cultural values at workplace, should consider other options apart from Amazon & Netflix

# In[ ]:


sns.factorplot(x = 'carrer-opportunities-stars', y = 'company', hue= 'is_current_employee', data = df, kind ='box',                aspect =2)


# #### Conclusion: The graph shows that the career opportunities are the best at Facebook & Microsoft and worst at Netflix so those candidates who are looking out for jobs and want great career opportunities, should consider other options apart from Netflix

# In[ ]:


sns.factorplot(x = 'comp-benefit-stars', y = 'company', hue= 'is_current_employee', data = df, kind ='box',                aspect =2)


# #### Conclusion: The graph shows that the compensation benefits are the best at Google & Facebook and worst for Apple.

# In[ ]:


sns.factorplot(x = 'senior-mangemnet-stars', y = 'company', hue= 'is_current_employee', data = df, kind ='box',                aspect =2)


# #### Conclusion: The graph shows that the senior management is the best at Facebook and worst at Amazon followed by Netflix.

# ## Apart from worklife balance, culture values of organization, career opportunities, senior management, compensation benefits, what is it that the employees find appealing?

# In[ ]:


import re
# Natural Language Tool Kit 
import nltk  
nltk.download('stopwords') 
# nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')
# to remove stopword 
from nltk.corpus import stopwords 

# for Stemming propose  
from nltk.stem.porter import PorterStemmer 
from nltk.stem.snowball import SnowballStemmer
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


# In[ ]:


df_review["review"] = df_review["pros"] + ' ' + df_review["cons"] + ' ' + df_review["advice-to-mgmt"]


# In[ ]:


df_review.dropna(how='any',subset=['review'],inplace = True)


# ###### Defining method to remove non alpha words, changing it to lowercase and removing stopwords

# In[ ]:


sw = stopwords.words('english')
def clean(text):
    
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [t for t in text if len(t) > 0]
    text = [t for t in text if t not in sw]
    text = ' '.join(text)
    return text


# ###### Defining method to get wordnet for a pos_tag

# In[ ]:


def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# ###### Defining method to lemmatize text

# In[ ]:


# ps = PorterStemmer()
sw = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
def lemmatize(text):
    text = nltk.word_tokenize(text)
    pos_tags = pos_tag(text)
    #     text = [ps.stem(word) for word in text if not word in set(sw)]
    text = [lemmatizer.lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    text = ' '.join(text)
    return text


# ###### Sample clean and lemmatize review of first row

# In[ ]:


clean(df_review.iloc[0].review)


# In[ ]:


lemmatize(df_review.iloc[0].review)


# ###### Cleaning and lemmatizing review column of dataframe 

# In[ ]:


df_review['review_clean'] = df_review['review'].apply(lambda x: clean(x))


# In[ ]:


df_review['review_lemmatize'] = df_review['review_clean'].apply(lambda x: lemmatize(x))


# In[ ]:


df_review.info()


# #### Importing Wordcloud package to draw wordcloud

# In[ ]:


from wordcloud import WordCloud, STOPWORDS


# In[ ]:


stopwords = set(STOPWORDS)
extras = ["great","team","work","company","place","good","people","employee","none","make","one","go",         "day","call","new","come","think","happen","within","look","store","retail","feel",         "life","sometime","environment","move","keep","still","review","group","year","role",         "want","try","office","create","look","even","level","many","thing","much","even",         "hour","year","always","every","things","project","product","need","time","give",          "take","never"]
stopwords.update(extras)
companies = list(df_review.company.unique())
for company in companies:
    stopwords.add(company)


# ###### defining method to generate wordclouds for each company

# In[ ]:


def wordclouds(df_review,companies):
    for company in companies:
        temp = df_review.loc[df_review["company"]==company]
        text = " ".join(str(review) for review in temp.review_clean)
        # Create and generate a word cloud image:
        wordcloud = WordCloud(stopwords = stopwords, collocations = False).generate(text)
        # Display the generated image:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(company.upper())
        plt.show()


# ###### calling wordclouds method. Prints wordcloud for each company 

# In[ ]:


wordclouds(df_review,companies)


# ##### Conclusion: 
# The word cloud shows the most used words by the employees per company. As can be seen from the graphs above, Google & Facebook have received the best overall ratings and when dug deep to know what other factors apart from work-life balance, culture & values, compensation benefits, senior management and career opportunities have been spoken about by the employees, it was found that 'food' is also one of the most popular words used by the employees of these two companies. So, food can also have an impact on the ratings. 

# In[ ]:




