#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Below are all of the packages I imported for this notebook
#My main goal was to classify the main research goals for Task 4 - Vaccines and Therapeutics using NLTK natural language processing
#I focused my analysis on the Title and Abstract only, as I wanted to capture what the researcher was doing specifically without the noise of other citations etc in the body of a paper

import pandas as pd 
import numpy as np
from collections import Counter
import nltk
import string
from collections import Counter
from nltk.probability import FreqDist
from io import StringIO
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


# In[ ]:


covid = pd.read_csv('../input/CORD-19-research-challenge/metadata.csv', sep=',')
print(type(covid))
covid.head(2)


# In[ ]:


#here I wanted to obtain specifically the First Author-main researcher and the Last Author- Professor so the user can see which labs are most active for a given topic 
covid['first_author'] = covid['authors'].str.split(';').str[0]
covid['last_author'] = covid['authors'].str.split(';').str[-1]


# In[ ]:


#the goal here was to create a new column called Text Analysis which would be used to train the language processor and perform searches
covid['text_analysis'] = covid['title'] + covid['abstract'] + covid['first_author'] + covid['last_author']
covid['text_analysis'] = covid['text_analysis'].str.lower()


# In[ ]:


#the purpose of below was to simplify the date format and also to tokenize the Text Analysis comment before applying stopwords
covid['date_format'] =  pd.to_datetime(covid['publish_time'])
covid['month_year'] = covid['date_format'].dt.to_period('M')
covid['text_analysis'] = covid['text_analysis'] .astype(str)
covid['text_tokenize'] = covid['text_analysis'].apply(nltk.word_tokenize)


# In[ ]:


string.punctuation
useless_words = nltk.corpus.stopwords.words("english") + list(string.punctuation)
covid['clean_tokenize'] = covid['text_tokenize'].apply(lambda x: [item for item in x if item not in useless_words])
covid['clean_tokenize'] = covid['clean_tokenize'] .astype(str)


# In[ ]:


print('total papers', covid['title'].nunique())
print('unique first authors', covid['first_author'].nunique())
print('unique last authors', covid['last_author'].nunique())
print('unique PMCID', covid['pmcid'].nunique())


# **Defining the questions**
# 
# Because the natural language processor is trained using a text sample, it is important to ensure what is being fed during training is as targeted as possible. Currently there aren't too many papers specifically targeted at COVID-19, however I expect this number exponentially grow with time, and the goal is to prevent the processor from capturing too many articles as being within scope. Below are the text filteres I used to filter journals for training.
# 
# Task 4- 
# **Question 1- Effectiveness of drugs being developed and tried to treat COVID-19 patients.
# **Clinical and bench trials to investigate less common viral inhibitors against COVID-19 such as naproxen, clarithromycin, and minocyclinethat that may exert effects on viral replication.
# 
# 'corona','inhibit','replication'
# 
# **Question 2- Methods evaluating potential complication of Antibody-Dependent Enhancement (ADE) in vaccine recipients
# 
# 'antibody','vaccine','corona'
# 
# **Question 3- Exploration of use of best animal models and their predictive value for a human vaccine
# 
# 'trial','predict'
# 
# **Question 4- Capabilities to discover a therapeutic (not vaccine) for the disease, and clinical effectiveness studies to discover therapeutics, to include antiviral agents.
# 
# 'therapeutics','antiviral','covid-19'
# 
# **Unrelated- Papers which didn't include any of the above text filter, or anything related to COVID-19
# 
# **Doesn't contain** 
# 'covid19','inhibit','replication','antibody','vaccine','corona','therapeutics','antiviral','animal','predictive','virus','sars','airborne','respitory','mers','nan', and papers for 2018 (to reduce the number)

# In[ ]:


covid['question1'] = covid['text_analysis'].str.contains('corona')&covid['text_analysis'].str.contains('inhibit')&covid['text_analysis'].str.contains('replication')
covid['question2'] = covid['text_analysis'].str.contains('antibody')&covid['text_analysis'].str.contains('vaccine')& covid['text_analysis'].str.contains('corona')
covid['question3'] = covid['text_analysis'].str.contains('trial')&covid['text_analysis'].str.contains('predict')
covid['question4'] = covid['text_analysis'].str.contains('therapeutics')&covid['text_analysis'].str.contains('antiviral')&covid['text_analysis'].str.contains('covid-19')
covid['unrelated'] = ~covid['text_analysis'].str.contains('covid-19') & ~covid['text_analysis'].str.contains('inhibit')& ~covid['text_analysis'].str.contains('replication') & ~covid['text_analysis'].str.contains('antibody') & ~covid['text_analysis'].str.contains('vaccine') & ~covid['text_analysis'].str.contains('corona')&~covid['text_analysis'].str.contains('therapeutics')&~covid['text_analysis'].str.contains('antiviral')&~covid['text_analysis'].str.contains('animal') & ~covid['text_analysis'].str.contains('predictive')&~covid['text_analysis'].str.contains('virus')& ~covid['text_analysis'].str.contains('sars')& ~covid['text_analysis'].str.contains('airborne')& ~covid['text_analysis'].str.contains('respitory')& ~covid['text_analysis'].str.contains('mers')& covid['publish_time'].str.contains('2018')& ~covid['text_analysis'].str.contains('nan')


# In[ ]:


#Below changes resopnse from True/False
covid['question1'] = np.where(covid['question1'], 'Question1', 'N')
covid['question2'] = np.where(covid['question2'], 'Question2', 'N')
covid['question3'] = np.where(covid['question3'], 'Question3', 'N')
covid['question4'] = np.where(covid['question4'], 'Question4', 'N')
covid['unrelated'] = np.where(covid['unrelated'], 'Unrelated', 'N')


# In[ ]:


#The goal here is to factorize the questions into different categories for the NLTK
covid['not_question'] = covid['question1'].str.contains('N')& covid['question2'].str.contains('N') & covid['question3'].str.contains('N') & covid['question4'].str.contains('N') & covid['unrelated'].str.contains('N')
covid['category_id'] = covid[['question1','question2','question3','question4','unrelated']].max(axis=1)
covid['category_id_num'] = covid['category_id'].factorize()[0]


# In[ ]:


#The final step is to create my training datafram
train_df=covid[~covid['category_id'].str.contains('N')]
category_id_df = train_df[['category_id', 'category_id_num']].drop_duplicates().sort_values('category_id_num')
category_to_id = dict(category_id_df.values)
train_df.head(2)


# In[ ]:


#Below is using a Scikit Learn to calculate a vector for each of the narratives
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
train_df_features = tfidf.fit_transform(train_df['clean_tokenize']).toarray()
labels = train_df['category_id_num']
train_df_features.shape


# In[ ]:


#Below obtains the most correlated unigrams and bigrams
N = 2
for Product, category_id_num in sorted(category_to_id.items()):
  features_chi2 = chi2(train_df_features, labels == category_id_num)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
 


# In[ ]:


#The final step is to train the Naive Bayes Classifier
X_train, X_test, y_train, y_test = train_test_split(train_df['title'], train_df['category_id'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)


# In[ ]:


#Paste title in the square brackets
print(clf.predict(count_vect.transform(['Long-Term Persistence of Robust Antibody and Cytotoxic T Cell Responses in Recovered Patients Infected with SARS Coronavirus'])))


# In[ ]:


print(clf.predict(count_vect.transform(['Practical fluid therapy and treatment modalities for field conditions for horses and foals with gastrointestinal problems'])))


# In[ ]:




