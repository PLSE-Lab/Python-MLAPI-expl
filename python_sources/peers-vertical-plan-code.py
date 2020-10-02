#!/usr/bin/env python
# coding: utf-8

# # Background
# ## Vertical PEER Plan
# 1. The VPs and AVPs of each department will ask their mentees/deputies to answer an evaluation form every month:
#     * [CC](https://docs.google.com/forms/d/e/1FAIpQLSevrr_vKTqF27UJJcc-7oAxJpU7yyCGPG6wY2mLAnX6jr_6qw/viewform)
#     * [TnD](https://docs.google.com/forms/d/e/1FAIpQLSdv3hM7xXazGXT2yz153zQ30Zg8oy2PlZbynVMicklysPrfsg/viewform)
#     * [PMD](https://docs.google.com/forms/d/e/1FAIpQLScf6ubggeijQVgo0Y4e8EutylqdZOAGA7fLypOvHAC6-zp_wg/viewform)
#     * [MF](https://docs.google.com/forms/d/e/1FAIpQLSe5h8jlMD1x-J7WyusK40XabZpW7Q441_2ULSZOC9L91gTqgg/viewform)
#     * [MCR](https://docs.google.com/forms/d/e/1FAIpQLSdZNwrjIsyyJ4pQT6FSLUSWTexUBW7kMFJexOAID53u2pXb_g/viewform)
#     * [Fin](https://docs.google.com/forms/d/e/1FAIpQLSddmU7efzZn-icdYpgHVRpEvk1piHA4BOYjR1WF-Xupj-uezQ/viewform)
# 2. The results will then be analysed by this code in order to gain insights on how the department can improve its deputy development program
# 
# ## Kamustahan
# 1. The members of Ateneo PEERS will be asked to do Kamustahans with their partner (who is of the same level as them i.e. a member will be partnered with another member) where they will record vital information from their conversation [here](https://docs.google.com/forms/d/e/1FAIpQLSeUOZDKqBI4QsLnsna2F93ifdRjTlL8zxLM0_LkOWn6NY1qZw/viewform)
# 2. The results will then be analysed by this code in order to gain insights on what activities may be given to that members in order for him/her to be better able to reach his/her ideal self through Ateneo PEERS
# 

# # Ateneo PEERS Vertical PEER Plan
# This is a test run for handling PEERS member data by the EVP during AY 2019-2020. Please pay attention to how the data is analysed rather than the data itself as this is just random data. 

# ## Context
# In Ateneo PEERS, there is a lack of of evaluative system for the status of each department on how they promote the advocacy of the org and thrusts of the President

# ## Business Objective
# In what areas do the department heads improve on in order to better help their deputies become their ideal selves?

# ## Data: [dummy test dataset](http://docs.google.com/spreadsheets/d/1TZ3UCq_Q8Sfzsmfxr6X7KCEZvGHsIMWmg9iaFXKPxkc/edit#gid=2028921453)
# ### Independent Variables
# * **D Mentor effort**: grade (ranging from 1 being the lowest to 5) given by deputies on how much effort the VPs and AVPs gave in developing them over the past month
# * **D Deputy Reward Satisfaction**: grade (ranging from 1 being the lowest to 5) given by deputies on how satisfied they are of the rewards they have earned over the past month
# * **D "Department Skill" Application Score**: grade (ranging from 1 being the lowest to 5) given by the deputies on how much they have been able to practice the fundamental skills of their department over the past month
# * **D Month's Cooperation Score**: total score of the departments' actions aimed at cooperation with other departments over the past month
# * **D Month's Sustainability Score**: 's total score of the departments' actions aimed at making their activities sustainable over generations over the past month
# * **"month"**: the month being graded by an input
#     * example: June, July
# * **lag1 D "variable"**: the average of the variable's values last month
#     * example: lag1 D Time Management Improvement Score
# * **lag2 D "variable"**: the average of the variable's values last last month
#     * example: lag2 D Time Management Improvement Score
# 
# ### Independent Variables
# * **D "Department Skill" Improvement Score**: grade (ranging from 1 being the lowest to 5) given by the deputies on how much they have improved on the fundamental skills of their department over the past month
# 
# ### Note: 
# * "D" stands for deseasonalised which means that it has been stripped of its seasonal patterns
# * there are several "lag variables" in order to determine how variables from the farther past affects present variables
# * to condense the data and make it easier to analyse, for all of the scores, if there are grades that were given in the same date, they are averaged into one input

# ## Methodology
# * **data gathering from deputies**: every month, deputies grade the "mentor effort", "deputy reward satisfaction", "department skill application", "department skill improvement" through a form like [this](http://docs.google.com/forms/d/e/1FAIpQLScf6ubggeijQVgo0Y4e8EutylqdZOAGA7fLypOvHAC6-zp_wg/viewform)
# * **data gathering from CB**: departments are required to document activities aimed at cooperating with other departments and making their activities sustainable and then report these in this [form](https://docs.google.com/forms/d/18wCCvd1wHInOibWMvYKb0r0Mdvv9uWPa9pmKk3PVs6k/edit) to be analysed and graded by the EB
# * **data analysis**: 
#     * the performance ("D Mentor effort", "D Deputy Reward Satisfaction", "D Month's Cooperation Score", "D Month's Sustainability Score") of the departments are compared through simple charts
#     * then time series analysis is done in order to know opportunities to improve the departments' member development

# ## Code Prototype

# In[ ]:


get_ipython().system('wget http://tinyurl.com/dec130-helperfunc')
get_ipython().system('mv dec130-helperfunc default-functions.py')
get_ipython().run_line_magic('run', 'default-functions.py')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import rcParams
import scipy.stats as stats
import os
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ### This loads the datasets and then fills missing data through [ffill method](https://www.geeksforgeeks.org/python-pandas-dataframe-ffill/) by filling missing values with the one above it 

# In[ ]:


# loading the datasets
## for comparisons
Reward_Satisfaction_Comparison_df = pd.read_csv("../input/deputy reward satisfaction comparison.csv", index_col=0)     
Mentor_Effort_Comparison_df = pd.read_csv("../input/mentor effort comparison.csv", index_col = 0)
Cooperation_Score_Comparison_df = pd.read_csv("../input/months cooperation score comparison.csv", index_col = 0)
Sustainability_Score_Comparison_df = pd.read_csv("../input/months sustainability score comparison.csv", index_col = 0)     

## for department evaluation
PMD_reg_df = pd.read_csv("../input/PMD final.csv", index_col = 0)

## for kamustahan
Kamustahan_df = pd.read_csv("../input/Kamustahan.csv", index_col = 0)

# this fills in the missing data
Reward_Satisfaction_Comparison_df = Reward_Satisfaction_Comparison_df.fillna(method='ffill')
Mentor_Effort_Comparison_df = Mentor_Effort_Comparison_df.fillna(method='ffill')
Cooperation_Score_Comparison_df = Cooperation_Score_Comparison_df.fillna(method='ffill')
Sustainability_Score_Comparison_df = Sustainability_Score_Comparison_df.fillna(method='ffill')
PMD_reg_df = PMD_reg_df.fillna(method='ffill')
Kamustahan_df = Kamustahan_df.fillna(method='ffill')


# ### Previews of Each Dataset

# In[ ]:


Reward_Satisfaction_Comparison_df.head()


# In[ ]:


Mentor_Effort_Comparison_df.head()


# In[ ]:


Cooperation_Score_Comparison_df.head()


# In[ ]:


Sustainability_Score_Comparison_df.head()


# In[ ]:


PMD_reg_df.head()


# In[ ]:


Kamustahan_df.head()


# ### Comparing Departments
# Here is where I will see if your department is underperforming or overperforming compared to other departments, which I will use to advice you.

# In[ ]:


Reward_Satisfaction_Comparison_df.plot.line()


# In[ ]:


Mentor_Effort_Comparison_df.plot.line()


# In[ ]:


Cooperation_Score_Comparison_df.plot.line()


# In[ ]:


Sustainability_Score_Comparison_df.plot.line()


# ### Time Analysis for PMD (and will be done for each skill for each department)
# This will be one of the supporting information I will use in order to advice you.

# #### [Reference](https://docs.google.com/presentation/d/1r4-1AoO0kD7GSdgfeuGBNrPdPgSDAzaU7jB-UjiSKJk/edit?usp=sharing)
# Click on this if you want to view a short lesson on the theory behind time analysis

# #### Checking for Multicollinearity
# * "get_top_abs_correlations" get the top n variables that are correlated, positive or negative .95 correlation is considered high
#     * when the correlation between 2 IVs is high, it is hard to know which is truly affecting the DV
#     * ex: "number of old refrigerators used" and having a high "electricity cost" are highly correlated variables that can predict chances of bankruptcy, then obviously the "number of old refrigerators used" as the "electricity cost" is the true factor affecting the DV
# * "get_redundant_pairs" is a function made to remove duplicates in the correlation list (in other words, so that "variable 1 x variable 2" and "variable 2 x variable" correlations are not in the list)

# In[ ]:


def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]


# In[ ]:


get_redundant_pairs(PMD_reg_df)
get_top_abs_correlations(PMD_reg_df, n=10)


# In[ ]:


import statsmodels.api as sm
PMD_TimeManagement_model = sm.OLS(PMD_reg_df["D Time Management Improvement Score"], sm.add_constant(PMD_reg_df.drop(['D Time Management Improvement Score', 'D Problem-Solving Improvement Score','D Planning Improvement Score','D Management Improvement Score'], axis = 1))).fit()  
PMD_TimeManagement_model.summary()


# In[ ]:


PMD_TimeManagement_model = sm.OLS(PMD_reg_df["D Problem-Solving Improvement Score"], sm.add_constant(PMD_reg_df.drop(['D Time Management Improvement Score', 'D Problem-Solving Improvement Score','D Planning Improvement Score','D Management Improvement Score'], axis = 1))).fit()  
PMD_TimeManagement_model.summary()


# In[ ]:


PMD_TimeManagement_model = sm.OLS(PMD_reg_df["D Planning Improvement Score"], sm.add_constant(PMD_reg_df.drop(['D Time Management Improvement Score', 'D Problem-Solving Improvement Score','D Planning Improvement Score','D Management Improvement Score'], axis = 1))).fit()  
PMD_TimeManagement_model.summary()


# In[ ]:


PMD_TimeManagement_model = sm.OLS(PMD_reg_df["D Management Improvement Score"], sm.add_constant(PMD_reg_df.drop(['D Time Management Improvement Score', 'D Problem-Solving Improvement Score','D Planning Improvement Score','D Management Improvement Score'], axis = 1))).fit()  
PMD_TimeManagement_model.summary()


# # Ateneo PEERS Kamustahan
# This is a test run for handling PEERS Kamustahan data by the EVP during AY 2019-2020. Please pay attention to how the messages are analysed rather than the messages themselves as these are just random messages. 

# ## Context
# According to both active and inactive PEERS AY 2018-19 members, members reaching out to fellow members is one of the main reasons for someone to be active in Ateneo PEERS.
# 

# ## Business Objective
# What kinds of activities can be given to certain members based on what they talk about during the monthly Kamustahans?

# ## Data: [dummy test dataset](http://docs.google.com/spreadsheets/d/1TZ3UCq_Q8Sfzsmfxr6X7KCEZvGHsIMWmg9iaFXKPxkc/edit#gid=2028921453)
# 
# ### Variables
# * **Your Student Number**: the student number of the PEERS member taking notes of his/her partner's answers to the [Kamustahan form](https://docs.google.com/forms/d/e/1FAIpQLSeUOZDKqBI4QsLnsna2F93ifdRjTlL8zxLM0_LkOWn6NY1qZw/viewform)
# * **Partners_Student_Number**: the student number of the PEERS member who said the things written on the Kamustahan form
# * **(Academic, PEERS, Others) Busy Score**: grade (ranging from 1 being the lowest to 5) given by the partner PEERS member on how busy s/he is with academics, Ateneo PEERS, and other extra-curriculars accordingly
# * **(Academic, PEERS, Others) Goal**: a short sentence given by the partner PEERS member on his/her goals with academics, Ateneo PEERS, and other extra-curriculars accordingly
# 

# ## Methodology
# * **data gathering from members**: every month, deputies check up on their partner's current short-term personal goals and fill up the [Kamustahan Form]
# * **data analysis**: 
#     * members with an average score of 3 on their busy scores will be labelled as members who can do more PEERS activities
#     * these members' Kamustahan answers will be analysed using text analytics in order to determine what kind of PEERS activities are suited for them to be invited to

# ## Code Prototype

# ### Text Analysis for Kamustahan
# This will be one of the supporting information I will use in order to advice you.

# ### [Reference](https://colab.research.google.com/drive/16Ph8V7oHnKhwV7sLTWoMreo38YM0_h-W)
# Click on this if you want to view a short lesson on the theory behind text analysis

# ### Preview of Dataset

# In[ ]:


Kamustahan_df.head()


# ### Cleaning the data
# 
# For text specifically, we do mainly three key steps to clean the text:
# - **Tokenizing**: converting a document to its atomic elements.
# - **Stopping**: removing meaningless words.
# - **Stemming**: merging words that are equivalent in meaning, i.e. eat, ate eaten, eating 

# ### Tokenizing

# In[ ]:


# breaking down the text to single words
# example: "I love Spongebob" becomes "I" "love" "Spongebob"
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

words_academic = [ tokenizer.tokenize(x.lower())  for x in Kamustahan_df.Academic_Goal]
words_PEERS = [ tokenizer.tokenize(x.lower())  for x in Kamustahan_df.PEERS_Goal]
words_other = [ tokenizer.tokenize(x.lower())  for x in Kamustahan_df.Others_Goal]

Kamustahan_df['Academic_Goal_words']= words_academic
Kamustahan_df['PEERS_Goal_words']= words_PEERS
Kamustahan_df['Others_Goal_words']= words_other
Kamustahan_df.head()


# ### Stopping

# In[ ]:


# importing data of stopwords AKA useless words like "the" "he" "she"
from nltk.corpus import stopwords
from IPython.display import clear_output
import nltk
nltk.download('stopwords')
stopwords.words('english')
clear_output()
stp = stopwords.words('english')


# In[ ]:


# adding my own stop words to the list
stp = stopwords.words('english') + ['ii','iii','read full','full article','read full article','silk','looking', 'information','read','full','article', 'glynn', 'wsj', 'com', 'jamesglynnwsj', 'james',
                              'james glynn wsj com', 'jamesglynnwsj', 'james glynn wsj com', 'james glynn wsj', 'james glynn', 'com jamesglynnwsj','glynn wsj com jamesglynnwsj', 'q3','3q','corresponding','graph', 'olga', 'cotaga','olgacotaga','olga', 'olga cotaga','jamesglynn','m2','2017','pm','daily shot','daily','shot']
                               


# In[ ]:


# Removing Stop Words
Kamustahan_df['Academic_Goal_words']=[[y for y in x if y not in stp] for x in Kamustahan_df['Academic_Goal']]
Kamustahan_df['PEERS_Goal_words']=[[y for y in x if y not in stp] for x in Kamustahan_df['PEERS_Goal']]
Kamustahan_df['Others_Goal_words']=[[y for y in x if y not in stp] for x in Kamustahan_df['Others_Goal']]


# ### Stemming

# In[ ]:


# stemming words to turn words like "Star" and "Wars" that have no meaning separately to "Star Wars"
from nltk.stem.porter import PorterStemmer
p_stemmer = PorterStemmer()
Kamustahan_df['Academic_Goal_words'] = [[p_stemmer.stem(y) for y in x] for x in Kamustahan_df['Academic_Goal']]
Kamustahan_df['PEERS_Goal_words'] = [[p_stemmer.stem(y) for y in x] for x in Kamustahan_df['PEERS_Goal']]
Kamustahan_df['Others_Goal_words'] = [[p_stemmer.stem(y) for y in x] for x in Kamustahan_df['Others_Goal']]


# ### Now that the data has been cleaned, the number of times each word appears is counted

# In[ ]:


# this function makes columns that counts the number of times a word appears in each message
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
def vectorize(dataset, nFeat, stp, how = 'CountVectorizer'):
    no_features = nFeat
    
    if how == 'CountVectorizer':
        tf_vectorizer = CountVectorizer(max_features=no_features, min_df = 3, max_df = 70, stop_words=stp, ngram_range = (1, 10))
    else:
        tf_vectorizer = TfidfVectorizer(max_features=no_features, min_df = 3, max_df = 70, stop_words=stp, ngram_range = (1, 10), norm=None)     
        
    tf = tf_vectorizer.fit_transform(dataset)
    tf_feature_names = tf_vectorizer.get_feature_names()
        
    return tf_vectorizer, tf, tf_feature_names


# ### Analysing the text
# Just because a word appears rarely does not mean it is insignificant.
# 
# For example, I say the words "Data Science" to few people but it is very important to me.
# 
# Then we determine the significance of each word using [Inverse Document Frequency](https://nlp.stanford.edu/IR-book/html/htmledition/inverse-document-frequency-1.html)  in determining the "weight" of words.

# In[ ]:


# doing inverse document frequency for academic goal messages
vect_idf1, corp_idf1, tf_names_idf1 = vectorize(Kamustahan_df['Academic_Goal'], 500, stp, how= TfidfVectorizer)   
tf_idf_doc1 = pd.DataFrame(corp_idf1.toarray())
tf_idf_doc1.columns = tf_names_idf1
tf_idf_doc1.head()


# In[ ]:


# doing inverse document frequency for PEERS goal messages
vect_idf2, corp_idf2, tf_names_idf2 = vectorize(Kamustahan_df['PEERS_Goal'], 500, stp, how= TfidfVectorizer)   
tf_idf_doc2 = pd.DataFrame(corp_idf2.toarray())
tf_idf_doc2.columns = tf_names_idf2
tf_idf_doc2.head()


# In[ ]:


# doing inverse document frequency for others goal messages
vect_idf3, corp_idf3, tf_names_idf3 = vectorize(Kamustahan_df['PEERS_Goal'], 500, stp, how= TfidfVectorizer)   
tf_idf_doc3 = pd.DataFrame(corp_idf2.toarray())
tf_idf_doc3.columns = tf_names_idf3
tf_idf_doc3.head()


# ### Using LDA, I find the top 10 most prominent topics from my tf-idf matrix and the 10 words that best describe them. 
# 
# One of the most popular topic modelling techniques is what we call **Latent Dirchlet Allocation**.
# 
# This algorithm assumes that each document is a mixture of different topics, and that each topic is a mixture of words. 
# ![](https://storage.googleapis.com/imageforlectures/Icons/lda.png)
# 
# However, the only things we observe are the documents and words, and what we are interested in is actually the topic.
# 
# How the algorithm solves this is by, for each document, finding the groups of words that maximize the probability that each document comes from these groups. The groups that maximize this probability are the **topics**
# 

# In[ ]:


from sklearn.decomposition import NMF, LatentDirichletAllocation
def LDArun(corpus, tf_feature_names, ntopics, nwords, print_yes):
    
    model = LatentDirichletAllocation(n_components=ntopics, max_iter=500,learning_method='online', random_state=10).fit(corpus)
    
    
    normprob = model.components_ / model.components_.sum(axis=1)[:, np.newaxis]
    
    topics = {}
    
    
    for topic_idx, topic in enumerate(normprob):
        kw = {}
        k = 1
        
        for i in topic.argsort()[:-nwords -1:-1]:
                kw[k] = tf_feature_names[i]
                k+=1
                
        topics[topic_idx] = kw
        
        if print_yes:
            print('topic', topic_idx, "|",[tf_feature_names[i]  for i in topic.argsort()[:-nwords - 1:-1]])
    

    return (model, pd.DataFrame(topics).transpose())

def getTopic(x):
    if max(x) == 1/len(x):
        return ('N/A')
    else:
        return np.argmax(list(x))


# ### From the LDA the following can be done
# 1. 5 topics from academic, PEERS, and others goals can be determined
# 2. the 10 most significant words in each topic can be determined
# 3. based on the 10 words, the topic may be labelled

# In[ ]:


# getting 5 topics of academic goals and the top 10 words in that topic to know what to name the topic
lda1, topics1 = LDArun(corp_idf1, tf_names_idf1,5, 10, True)


# In[ ]:


# naming the topics in the academic goals messages
topic_dict1 = {0:'Honors', 1:'Passing', 2:'Bawi', 3:'Latin Honors', 4:'Minor'}


# In[ ]:


# getting 5 topics of PEERS goals and the top 10 words in that topic to know what to name the topic
lda2, topics2 = LDArun(corp_idf2, tf_names_idf2,5, 10, True)


# In[ ]:


# naming the topics in the PEERS goals messages
topic_dict2 = {0:'Officer', 1:'Experiment', 2:'Mental Health', 3:'Crush', 4:'Help'}


# In[ ]:


lda3, topics3 = LDArun(corp_idf3, tf_names_idf3,5, 10, True)


# In[ ]:


# naming the topics in the others goals messages
topic_dict3 = {0:'Family', 1:'Friends', 2:'Church', 3:'Basketball', 4:'Spongebob'}


# ### Now that the topics are known, the topics most talked about by members can be determined.
# 
# ### Then, their interests can be known in order to know which kinds of PEERS activities would be suited for them.
# 

# In[ ]:


import numpy as np
topic_df1 = pd.DataFrame(lda1.transform(corp_idf1))
topic_df1["Partners_Student_Number"] = Kamustahan_df.Partners_Student_Number.values
topic_df1 = topic_df1.set_index("Partners_Student_Number")
topic_df1.columns = topic_dict1.values()
topic_df1['Final Topic'] = [topic_dict1[np.argmax([v, w, x, y, z])] for v, w, x, y, z in zip(topic_df1.iloc[:, 0], topic_df1.iloc[:, 1], topic_df1.iloc[:, 2], topic_df1.iloc[:, 3], topic_df1.iloc[:, 4])]


# In[ ]:


# Final Topic shows what they mostly talk about when talking about academic goals
topic_df1.head()


# In[ ]:


import numpy as np
topic_df2 = pd.DataFrame(lda2.transform(corp_idf2))
topic_df2["Partners_Student_Number"] = Kamustahan_df.Partners_Student_Number.values
topic_df2 = topic_df2.set_index("Partners_Student_Number")
topic_df2.columns = topic_dict2.values()
topic_df2['Final Topic'] = [topic_dict2[np.argmax([v, w, x, y, z])] for v, w, x, y, z in zip(topic_df2.iloc[:, 0], topic_df2.iloc[:, 2], topic_df2.iloc[:, 2], topic_df2.iloc[:, 3], topic_df2.iloc[:, 4])]


# In[ ]:


# Final Topic shows what they mostly talk about when talking about PEERS goals
topic_df2.head()


# In[ ]:


import numpy as np
topic_df3 = pd.DataFrame(lda3.transform(corp_idf3))
topic_df3["Partners_Student_Number"] = Kamustahan_df.Partners_Student_Number.values
topic_df3 = topic_df3.set_index("Partners_Student_Number")
topic_df3.columns = topic_dict3.values()
topic_df3['Final Topic'] = [topic_dict3[np.argmax([v, w, x, y, z])] for v, w, x, y, z in zip(topic_df3.iloc[:, 0], topic_df3.iloc[:, 3], topic_df3.iloc[:, 3], topic_df3.iloc[:, 3], topic_df3.iloc[:, 4])]


# In[ ]:


# Final Topic shows what they mostly talk about when talking about other goals
topic_df3.head()


# ### Whose Academic goals over time do you want to know?

# In[ ]:


cols = 'Final Topic'
specific_acads = topic_df1.loc[18234, cols]
specific_acads


# ### ^ This means that Student Number 18234 has always been talking about getting Honors and is probably consistently aiming for Dean's List

# ### Whose PEERS goals over time do you want to know?

# In[ ]:


specific_PEERS = topic_df2.loc[18234, cols]
specific_PEERS


# ### Whose Others goals over time do you want to know?

# In[ ]:


specific_others = topic_df3.loc[18234, cols]
specific_others

