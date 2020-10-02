#!/usr/bin/env python
# coding: utf-8

# ## Objective
# 
# * To predict the salary of a job based on the data posted on the company's website or on any job portal
# * To compare how addition of text(job description) into the models increases the prediction power

# ## About data
# 
# I have used the job salary prediction dataset from Kaggle to perform the analysis and achieve the objective that i have mentioned above.
# Following is the link for the data description [Data Description](https://www.kaggle.com/c/job-salary-prediction/data)

# ## Approach
# 
# Running a classfier on the data is never the first step of any analysis be it on numerical data or text data. We will have to first understand the data, clean it, perform exploratory data analysis following which we will perform feature extraction and only then think of training a classifier on the training data.
# 
# So as a logical first step, i started with importing the data and performing exploratory data analysis on the data. For easy understanding of the entire analysis, i have listed all the steps that i have followed below and divide my entire analysis based in the same chronological order.
# 
# 1. Exploratory Data analysis
# 2. Exploratory Data analysis - Job descriptions
# 3. Models without text variables
# 4. Models with text variables
# 5. Conclusion

# ### 1.Exploratory data analysis - general

# In[ ]:


# Import all required modules for the analysis(make sure that you installed all these modules prior to importing)
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import math
from IPython.display import display,HTML
from patsy import dmatrices
import seaborn as sns; sns.set()

import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('pylab', 'inline')


# In[ ]:


# Reading the train data
train_df = pd.read_csv('../input/Train_rev1.csv')


# In[ ]:


# Checking the data types in the train data
print (train_df.info())

# Let's look at the unique values present in the data frame to have a general understanding of the data
names = train_df.columns.values
uniq_vals = {}
for name in names:
    uniq_vals[name] = train_df.loc[:,name].unique()
    print("Count of %s : %d" %(name,uniq_vals[name].shape[0]))


# In[ ]:


# Distribution of salaries based on the train data
pylab.rcParams['figure.figsize'] = (20,10)
plt.hist(train_df['SalaryNormalized'], bins='auto')
plt.xlabel('Salaries')
plt.ylabel('Number of postings')
plt.title('Histogram of Salaries')


# We can observe that the job descriptions are skewed mostly towards the lower end(mostly < 50,000) showing that most of the jobs are on the lower end of the job salary spectrum. This distribution might be useful as we move further into the analysis, as this might help us detect if there is any bias in our final analysis.

# **Technical limitation**:  
# As you can see, there are about ~240 k rows in the dataset. As i am currently running the analysis on my personal system with 8 GB, i will randomly take a small number of rows to train my classifier and incrementally add data to train untill its feasible.

# In[ ]:


# Randomly selecting 2500 rows to train the classifier
import random
random.seed(1)
indices = list(train_df.index.values)
random_2500 = random.sample(indices,2500)

# Subsetting the train data based on the random indices
train_df1 = train_df.loc[random_2500].reset_index()


# Now let's see the salary distribution in this data and compare it with the original data

# In[ ]:


pylab.rcParams['figure.figsize'] = (20,10)
plt.hist(train_df1['SalaryNormalized'], bins='auto')
plt.xlabel('Salaries')
plt.ylabel('Number of postings')
plt.title('Histogram of Salaries')


# The distribution seemingly holds good and is comparably similar to the orginial distribution. With this, let's move forward with the actual analysis of cleaning the data and creating the features for the analysis.

# ## 2. Exploratory Data analysis - Job descriptions
# 
# Let's look into the job descriptions and try to answer some of the questions to have more clarity
# 1. What are the top 5 parts of speech in the job descriptions? How frequently do they appear?
# 2. How do these numbers change if you exclude stopwords?
# 3. What are the 10 most common words after removing stopwords and lemmatization?

# **1. What are the top 5 parts of speech in the job descriptions? How frequently do they appear?** 

# In[ ]:


import re
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize 
stop_words = set(stopwords.words('english')) 
from string import punctuation
import collections


# While looking at the data, you can observe that the numbers are masked as *** and they turn out to be of no value for us in the analysis. In addition to that, there are a few data cleaning steps that i have performed in the below code
# 1. Remove website links from the data
# 2. Remove punctuations
# 3. Removing numbers
# 
# By running these steps, we can achieve a higher accuracy as the data becomes more cleaned and the predictive power of the algorithm increases because of that.

# In[ ]:


# To obtain the full width of a cell in a dataframe
pd.set_option('display.max_colwidth', -1)
desc = train_df1.loc[1,'FullDescription']

# Creating a list of words from all the job descriptions in train_df1 data
all_desc = []
for i in range(0,train_df1.shape[0]):
    desc = train_df1.loc[i,'FullDescription']
    desc1 = desc.lower()
    # Removing numbers, *** and www links from the data
    desc2 = re.sub('[0-9]+\S+|\s\d+\s|\w+[0-9]+|\w+[\*]+.*|\s[\*]+\s|www\.[^\s]+','',desc1)
    # Removing punctuation
    for p in punctuation:
        desc2 = desc2.replace(p,'')
    all_desc.append(desc2)


# Now next step after cleaning the descriptions is to tokenize them. Here for simplicity purpose, i have just considered word tokenize. We can also tokenize the descriptions by sentences but that is not suitable for our problem here.
# [Refer this link](https://textminingonline.com/dive-into-nltk-part-ii-sentence-tokenize-and-word-tokenize) to know more about word tokenizer and sentence tokenizer

# In[ ]:


# Creating word tokens for all the descriptions
final_list = []
for desc in all_desc:
    word_list = word_tokenize(desc)
    final_list.extend(word_list)


# Now before finding the most common parts of speech in the job description, we first need to tag each word with the relevant parts of speech. I have used pos_tag from nltk library to tag the parts of speech. This uses the pos tags from treebank pos repository.   
# After creating the pos tags, the next would be to find the frequent occurence of each parts of speech and find the most common parts of speech

# In[ ]:


# 3. Tagging parts of speech
pos_tagged = nltk.pos_tag(final_list)

# 4. Identifying the most common parts of speech
tag_fd = nltk.FreqDist(tag for (word, tag) in pos_tagged)
tag_fd.most_common()[:5]


# You can see that Noun(NN) , adjective(JJ), preposition(IN), determiner(DT), plural nouns(NNS)  are the most common parts of speech from the job descriptions.
# [Refer this link](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html) to look at the descriptions of the parts of speech for the above result

# **2.How do these numbers change if you exclude stopwords?**  
# In english,for that matter in any language, there will be a lot of stop words that repeat a lot of times in the sentence but do not actually carry a lot of information.  
# For examples: the, a , an ,and etc.  
# nltk library has a repository for stop words. Let's remove those stopwords and check how the above results will change.

# In[ ]:


# Excluding stopwords from the analysis
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) 

list_wo_stopwords = []
for w in final_list:
    if w not in stop_words:
        list_wo_stopwords.append(w)
        
# 3. Tagging parts of speech
pos_tagged_wo_sw = nltk.pos_tag(list_wo_stopwords)

# 4. Identifying the most common parts of speech
tag_fd_wo_sw = nltk.FreqDist(tag for (word, tag) in pos_tagged_wo_sw)
tag_fd_wo_sw.most_common()[:5]


# After removing stopwords, there are two important observations in comparison to the previous result
# 1. Prepositions and determiners disappeared from the top 5 set as most of these are present in the stopwords imported from NLTK 
# 2. The counts of nouns and plural nouns have decreased and the adjectives have increased.
# 3. Verb, gerund or present participle(VBG) and Verb, non-3rd person singular present(VBP) moved to the top 5 list

# **3. What are the 10 most common words after removing stopwords and lemmatization?**
# 
# Here is some backgrond information about [lemmatization](https://textminingonline.com/dive-into-nltk-part-iv-stemming-and-lemmatization)
# 
# As we have already removed stopwords and create a dataframe list_wo_stopwords earlier, our first step here would be to perform lemmatization and then identify the 10 most common words.  
# I have also plotted the wordcloud of all the words to visualize these words.  

# In[ ]:


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Lemmatization without specifying parts of speech
list_lemmatized = []
for word in list_wo_stopwords:
    list_lemmatized.append(lemmatizer.lemmatize(word))

word_freq_lem = dict(collections.Counter(list_lemmatized))
keys = list(word_freq_lem.keys())
values = list(word_freq_lem.values())
df_lem = pd.DataFrame({'words':keys,'freq':values})
display(df_lem.sort_values(by = 'freq',ascending = False)[:10])

from wordcloud import WordCloud
from collections import Counter
word_could_dict=Counter(word_freq_lem)
wordcloud = WordCloud(width = 1000, height = 500).generate_from_frequencies(word_could_dict)

plt.figure(figsize=(15,8))
plt.imshow(wordcloud)


# ## 3. Model without text variables
# 
# As i have told, we will perform the analysis with 2 types of variables
# 1. Non-text variables
# 2. Text variables
# 
# Just to revisit, our objective is to predict the salaries based on the information posted on the website including variables such as location, company and job description.
# 
# With this in mind, let's proceed with the analysis
# 
# For this analysis, let's define the target variable based on the salary normalized. This converts the problem into a classfication problem and reduces the complexity. Further, based on the requirement we can perform a regression analysis to predict a number for the salary

# **Creating the target column by splitting the salary normalized into high(above 75th percentile) and low(below 75th percentile)**

# In[ ]:


p_75 = np.percentile(train_df1['SalaryNormalized'], 75)
train_df1['target'] = train_df1['SalaryNormalized'].apply(lambda x: 1 if x>=p_75 else 0)


# ### Creation of features
# **Creating a proxy variable for location**
# Here as there are so many locations, creating dummy variable will bloat the dataset a lot. So, we will create a proxy variable
# for the location by taking the cities with high cost of living under one group(1) and the others in a separate group(0).
# 
# We have considered 17 cities as cities with high cost of living [Source](https://www.thisismoney.co.uk/money/mortgageshome/article-5283699/The-cheapest-expensive-cities-live-in.html)

# In[ ]:


costly_cities = ['London','Brighton','Edinburgh','Bristol','Southampton','Portsmouth','Exeter','Cardiff','Manchester',
                 'Birmingham','Leeds','Aberdeen','Glasgow','Newcastle','Sheffield','Liverpool']
costly_cities_lower = [x.lower() for x in costly_cities]

# More robust if lower() is applied
train_df1['location_flag'] = train_df1['LocationNormalized'].apply(lambda x: 1 if x in costly_cities else 0)


# **Creating dummy variables for all the columns except for job descriptions and splitting the data into train and validation set**  

# In[ ]:


# Dropping job description column from the dataset
train_x = train_df1.drop(['FullDescription','index','Id','LocationRaw','Title','Company','LocationNormalized','SalaryRaw','SalaryNormalized',
                    'target'],axis=1)

train_x1 = pd.get_dummies(train_x,drop_first=True)
X_n = np.array(train_x1)
y_n = np.array(train_df1['target'])

from sklearn.model_selection import train_test_split
X_train_num, X_val_num, y_train_num, y_val_num = train_test_split(X_n, y_n, test_size=0.3, random_state=1)


# **Let's run a Bernoulli Naive Bayes algorithm to predict the salary of a job using the non-text variables. I have selected this algorithm because most often Naive bayes gives good results in simplistic scenarios.  
# It also acts as a good starting point to compare the effect of non-text and text variables on prediction.**

# In[ ]:


# Bernoulli
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
clf.fit(X_train_num, y_train_num)

from sklearn import metrics
prediction_train = clf.predict(X_val_num)
mat_n = metrics.confusion_matrix(y_val_num, prediction_train)
mat_n
print (metrics.accuracy_score(y_val_num, prediction_train))


# **Using test data to predict the salary**

# In[ ]:


# Baseline accuracy
1-(sum(y_val_num)/len(y_val_num))
# sum(prediction_train)


# You can see that we did not very well with just numeric values in the model as the accuracy is almost equal to the baseline accuracy. Now let's look at the model with just the job descriptions and see if we can predict the salaries with some higher accuracies

# ## 4. Models using text as variables
# 
# Now let's run the Naive Bayes using the words in the job description as variables. You will notice that text in itself has a higher prediction power than the non-text variables
# 
# For this model, we will run both Bernoulli and Multinomial Naive Bayes([click to know more](https://syncedreview.com/2017/07/17/applying-multinomial-naive-bayes-to-nlp-problems-a-practical-explanation/)), to check the performance of both the models
# 
# We will run both the models in multiple steps
# 1. Without removing stopwords from the data
# 2. After removing stopwords from the data
# 3. After lemmatizing the data
# 
# This will help us understand the effect of each step on the accuracy of the result
# 
# Here i have created a function to make it easy to perform the above 3 steps

# In[ ]:


def models(l):
    # Counting the occurence of each word in the corpus
    from sklearn.feature_extraction.text import CountVectorizer
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(l)
    count_vect.get_feature_names()
    X_matrix= X_train_counts.todense()

    y = np.array(train_df1['target'])

    # Creating the train and test split
    from sklearn.model_selection import train_test_split
    X_train_m, X_val_m, y_train_m, y_val_m = train_test_split(X_train_counts, y, test_size=0.3, random_state=1)

    #Multinomial

    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB().fit(X_train_m, y_train_m)
    labels_m = clf.predict(X_val_m)

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    mat_m = confusion_matrix(y_val_m, labels_m)

    # Bernoulli
    # Changing the data to binary to input BernoulliNB
    x_train_b1 = X_train_counts.todense()
    X_train_counts_ber = np.where(x_train_b1 >=1 ,1,0)

    # Creating the train and test split for bernoulli
    from sklearn.model_selection import train_test_split
    X_train_b, X_val_b, y_train_b, y_val_b = train_test_split(X_train_counts_ber, y, test_size=0.3, random_state=1)

    from sklearn.naive_bayes import BernoulliNB
    clf = BernoulliNB().fit(X_train_b, y_train_b)
    labels_b = clf.predict(X_val_b)

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    mat_b = confusion_matrix(y_val_b, labels_b)
    print ('Confusion matrix:',mat_b)
    print ('Accuracy using BernoulliNB:',accuracy_score(y_val_b, labels_b))


    print ('Confusion matrix:',mat_m)
    print ('Accuracy using MultinomialNB:',accuracy_score(y_val_m, labels_m))


# **1. Without removing stopwords**

# In[ ]:


models(all_desc)


# **2. After removing stopwords from the data**

# In[ ]:


# Removing stopwords
def remove_stopwords(s):
    big_regex = re.compile(r'\b%s\b' % r'\b|\b'.join(map(re.escape, stop_words)))
    return big_regex.sub('',s)

all_desc_wo_sw = [remove_stopwords(s) for s in all_desc]
models(all_desc_wo_sw)


# **3. After lemmatizing the data**
# 
# Lemmatization usually refers to doing things properly with the use of a vocabulary and morphological analysis of words, normally aiming to remove inflectional endings only and to return the base or dictionary form of a word, which is known as the lemma.([Click to know more](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html))
# 
# We will lemmatize and see if there is any improvement in the result

# In[ ]:


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

from nltk.corpus import wordnet

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None 

# Lemmatizing the data
all_desc_lemm = []
for i in range(0,len(all_desc_wo_sw)):
    desc = all_desc_wo_sw[i]
    desc2 = re.sub('[0-9]+\S+|\s\d+\s|\w+[0-9]+|\w+[\*]+.*|\s[\*]+\s|www\.[^\s]+','',desc)
    for p in punctuation:
        desc2 = desc2.replace(p,'')
    tagged = nltk.pos_tag(word_tokenize(desc2))
    list_lemmatized = []
    for word, tag in tagged:
        wntag = get_wordnet_pos(tag)
        if wntag is None:# not supply tag in case of None
            list_lemmatized.append(lemmatizer.lemmatize(word)) 
        else:
            list_lemmatized.append(lemmatizer.lemmatize(word, pos=wntag))
    k = ' '.join(list_lemmatized)   
    all_desc_lemm.append(k)

models(all_desc_lemm)


# ## 5. Conclusion

# You can see the accuracy has increased from 80% in the first step to 82% in the lemmatization while using text data. There is a still a lot of scope for this analysis. We can use SVM algorithm to predict the salary based on the text data. In the next version of this analysis, i will try more models and compare them with the results from the current analysis. Hope you enjoyed the analysis. Please share your thoughts.