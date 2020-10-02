#!/usr/bin/env python
# coding: utf-8

# # A tutorial in text analytics
# ### Salary Prediction using job descriptions
# 
# The data comes from an old competition posted by Adzuna to predict salaries for jobs in the UK based on a variety of attributes, the most important of which is the job description (as we will see shortly). This is a good example to show that even unstructured data such as text can be used to make pretty solid predictions - something that goes against intuition somehow!

# In[ ]:


# import libraries
import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import word_tokenize

import re

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import math

import warnings
warnings.filterwarnings('ignore')


# NLTK is the key library here! It has a host of functions that help us quickly make sense of a large amount of text data! Some of the basic terminology associated with the NLTK library is as follows - 
#  
# 1. **Corpus** - This is just the text string. It could be one line long or a 100 lines long.
# 2. **Tokens** - Tokens are just the words appearing in a text string. For instance, in the string "I like New York" we have 4 tokens - "I", "like", "New" and "York". Although this can be done using a simple split, NLTK has functions that tokenize rapidly especially if we have a lot of text in a huge dataframe!
# 3. **Vocabulary** - This is the entire collection of all tokens found in a corpus!
# 4. **Parts of Speech** - Going back to basic English classes, every sentence has words that belong to some part of speech. There are verbs, nouns, proper nouns, pronouns, etc. NLTK has the `pos_tagger` function that allows us to quickly tag parts of speech to a string. You might be thinking if its any good at it and I can tell you that it is!
# 5. **Stop Words** - Stop words are english words such as "i", "me", "myself" which although add sense to a sentence, contribute nothing towards making it special enough to be of significance in a prediction problem.

# In[ ]:


# Read in the train_rev1 datafile downloaded from kaggle
df = pd.read_csv('../input/Train_rev1.csv')
df.head(2)


# In[ ]:


print("Data Shape:", df.shape)


# The data that we have has around 240k rows and 12 columns. For the sake of simplicity and faster processing, lets just pick a random sample of 10000 rows from it and proceed further.

# In[ ]:


# randomly sample 10000 rows from the data
import random
random.seed(1)
indices = df.index.values.tolist()

random_10000 = random.sample(indices, 10000)

random_10000[:5]


# In[ ]:


# subset the imported data on the selected 2500 indices
train = df.loc[random_10000, :]
train = train.reset_index(drop = True)
train.head(2)


# So, the train data now has 10000 rows and some columns. Lets see what these columns are

# In[ ]:


train.columns.values


# **Id** - A unique identifier for each job ad
# 
# **Title** - A freetext field supplied by the job advertiser as the Title of the job ad.  Normally this is a summary of the job title or role.
# 
# **FullDescription** - The full text of the job ad as provided by the job advertiser.  Whenever we see ***s, these are values stripped from the description in order to ensure that no salary information appears within the descriptions. 
# 
# **LocationRaw** - The freetext location as provided by the job advertiser.
# 
# **LocationNormalized** - Normalized location of the job location.
# 
# **ContractType** - full_time or part_time.
# 
# **ContractTime** - permanent or contract.
# 
# **Company** - the name of the employer as supplied by the job advertiser.
# 
# **Category** - which of 30 standard job categories this ad fits into.
# 
# **SalaryRaw** - the freetext salary field in the job advert from the advertiser.
# 
# **SalaryNormalised** - the annualised salary interpreted by Adzuna from the raw salary. We convert this value to a categorical variable denoting 'High salary' or 'Low Salary' and try to predict those.
# 
# **SourceName** - the name of the website or advertiser where the job advert is posted. 

# In[ ]:


# some problems with the way FullDescription has been encoded
def convert_utf8(s):
    return str(s)

train['FullDescription'] = train['FullDescription'].map(convert_utf8)


# ### Job Descriptions
# Let's look at one job description to see how these are - 

# In[ ]:


train.loc[2, 'FullDescription']


# ### Cleaning up the descriptions
# A look at the description above shows us that these descriptions contain - numbers, urls and certain strings as '*' which I believe are either phone numbers or salary figures that have been removed so that these do not affect our predictions! We will have to remove these strings before we try out any analytics!
# 
# Approach - We will use the substitute feature to find and substitute these anomalous strings in our job descriptions

# In[ ]:


# Remove the urls first - Anything that has .com, .co.uk or www. is a url!
def remove_urls(s):
    s = re.sub('[^\s]*.com[^\s]*', "", s)
    s = re.sub('[^\s]*www.[^\s]*', "", s)
    s = re.sub('[^\s]*.co.uk[^\s]*', "", s)
    return s

train['Clean_Full_Descriptions'] = train['FullDescription'].map(remove_urls)


# In[ ]:


# Remove the star_words
def remove_star_words(s):
    return re.sub('[^\s]*[\*]+[^\s]*', "", s)

train['Clean_Full_Descriptions'] = train['Clean_Full_Descriptions'].map(remove_star_words)


# In[ ]:


def remove_nums(s):
    return re.sub('[^\s]*[0-9]+[^\s]*', "", s)

train['Clean_Full_Descriptions'] = train['Clean_Full_Descriptions'].map(remove_nums)


# In[ ]:


# Remove the punctuations
from string import punctuation

def remove_punctuation(s):
    global punctuation
    for p in punctuation:
        s = s.replace(p, '')
    return s

train['Clean_Full_Descriptions'] = train['Clean_Full_Descriptions'].map(remove_punctuation)


# In[ ]:


# Convert to lower case
train['Clean_Full_Descriptions'] = train['Clean_Full_Descriptions'].map(lambda x: x.lower())


# The `Clean_Full_Descriptions` now has the full descriptions without punctuations, numbers, star words or urls!

# ## What are the top 5 parts of speech in the job description? How frequently do they appear? 
# NLTK has a parts of speech tagger that looks at a sentence and assigns parts of speech to different words in it such as - nouns, plural nouns, determiners, pronouns, verbs, adverbs, etc. This tagging is done based on the word (token) itself.
# 
# For example, if I have a sentence such as "I love riding my motorcycle", NLTK assigns the following parts of speech to it - 
# 

# In[ ]:


print(nltk.pos_tag(word_tokenize("I love riding my motorcycle")))


# where **PRP** - Preposition, **VBP** - Verb present participle, **VBG**- Verb/Gerund, **PRP$** - possessive pronoun and **NN** - Noun.

# To get the top 5 parts of speech in the job descriptions, we will - 
# Approach - 
# 1. Tokenize each description under the clean_full_description column.
# 2. NLTK pos_tag function returns the word followed by its part of speech tagging in a tuple. We need to extract the second element from these tuples.
# 3. For each description get the parts of speech tagging for each string in the full description

# In[ ]:


# define a function for parts of speech tagging
# make a corpus of all the words in the job description
corpus = " ".join(train['Clean_Full_Descriptions'].tolist())

# This is the NLTK function that breaks a string down to its tokens
tokens = word_tokenize(corpus)

# Get the parts of speech tag for all words
answer = nltk.pos_tag(tokens)
answer_pos = [a[1] for a in answer]

# print a value count for the parts of speech
all_pos = pd.Series(answer_pos)
all_pos.value_counts().head()


# As expected - **Nouns (NN), plural nouns (NNS), adjectives (JJ), Pronouns (IN) and Determiners (DT)** are the top 5 parts of speech tagged in the corpus. Mostly these will be the top 5 parts of speech for any corpus!

# ## How does the frequency change if we exclude stopwords?
# **Stopwords** are words such as* I, me, myself, they, and*, etc. that are useful for meaningful sentence structure but **do not really add value to the prediction task at hand. This is because they appear with equal probability in both high salary and low salary job descriptions.** To make computations faster, we remove these stopwords before we go into the prediction algorithm!
# 
# Approach -
# 1. Make a list of stopwords from the nltk library.
# 2. Remove stopwords from each of the descriptions.
# 3. Apply the pos_tagger again.

# In[ ]:


# store english stopwords in a list
from nltk.corpus import stopwords
en_stopwords = stopwords.words('english')

# define a function to remove stopwords from descriptions
def remove_stopwords(s):
    global en_stopwords
    s = word_tokenize(s)
    s = " ".join([w for w in s if w not in en_stopwords])
    return s

# Create a new column of descriptions with no stopwords
train['Clean_Full_Descriptions_no_stop'] = train['Clean_Full_Descriptions'].map(remove_stopwords)

# make a corpus of all the words in the job description
corpus = " ".join(train['Clean_Full_Descriptions_no_stop'].tolist())

# This is the NLTK function that breaks a string down to its tokens
tokens = word_tokenize(corpus)

answer = nltk.pos_tag(tokens)
answer_pos = [a[1] for a in answer]

all_pos = pd.Series(answer_pos)
all_pos.value_counts().head()


# What do we find? Removing stopwords totally removed **Pronouns and Determiners** from the top 5. In their place we now have **Verb/Gerund (VBG) and Verb/Present Participle (VBP)**.

# ## Does this data support Zipf's law?
# [**Zipf's law**](https://en.wikipedia.org/wiki/Zipf's_law) states that for any word (including stopwords) in a given corpus of natural language utterances, its frequency of occurence is inversely proportional to its rank in the frequency table! This can also be stated as 'Frequency of the word in a corpus times its rank is a constant'! Mathematically, this can be expressed as  - 
# 
# $$R*x_{r} = c$$
# 
# $$R = \frac{c}{x_{r}}$$
# 
# $$\log(R) = -1\log(\frac{x_{r}}{c})$$
# 
# Since $R*x_{r}$ is a constant, we approximate it by $N*x_{n}$ where $N$ is the largest rank and $x_{n}$ is the frequency associated with the largest rank. Thereby, we get - 
# $$\log(R) = -1\log(\frac{x_{r}}{n*x_{n}})$$
# 
# where $R$ is the rank of the word and $x_{r}$ is the frequency of occurence!
# 
# This last equation can be modelled as a linear regression and we can check if Zipf's law holds if the coefficient of the variable comes out to be -1!
# 
# Approach - 
# 1. Create a corpus of the cleaned descriptions.
# 2. Tokenize the words in the corpus.
# 3. Take a count of these words and associate ranks with these words! Take the top 100 words
# 4. Create a linear regression model as given in the last equation above and check the value of the coefficient!

# In[ ]:


# prepare corpus from the descriptions that still have stopwords
corpus = " ".join(train['Clean_Full_Descriptions'].tolist())

#tokenize words
tokenized_corpus = nltk.word_tokenize(corpus)
fd = nltk.FreqDist(tokenized_corpus)

# get the top words
top_words = []
for key, value in fd.items():
    top_words.append((key, value))

# sort the list by the top frequencies
top_words = sorted(top_words, key = lambda x:x[1], reverse = True)

# keep top 100 words only
top_words = top_words[:100]

# Keep the frequencies only from the top word series
top_word_series = pd.Series([w for (v,w) in top_words])
top_word_series[:5]

# get actual ranks of these words - wherever we see same frequencies, we give same rank
word_ranks = top_word_series.rank(method = 'min', ascending = False)


# In[ ]:


# Get the value of the denominator n*x_n
denominator = max(word_ranks)*min(top_word_series)

# Y variable is the log of word ranks and X is the word frequency divided by the denominator
# above
Y = np.array(np.log(word_ranks))
X = np.array(np.log(top_word_series/denominator))

# fit a linear regression to these, we dont need the intercept!
from sklearn import linear_model
reg_model = linear_model.LinearRegression(fit_intercept = False)
reg_model.fit(Y.reshape(-1,1), X)
print("The value of theta obtained is:",reg_model.coef_)

# make a plot of actual rank obtained vs theoretical rank expected
plt.figure(figsize = (8,5))
plt.scatter(Y, X, label = "Actual Rank vs Frequency")
plt.title('Log(Rank) vs Log(Frequency/nx(n))')
plt.xlabel('Log Rank')
plt.ylabel('Log(Frequency/nx(n))')

plt.plot(reg_model.predict(X.reshape(-1,1)), X, color = 'red', label = "Zipf's law")
plt.legend()


# Based on the result of the regression coefficient and the supporting graph, we find that the top 100 words in the data do support Zipf's law.

# ## What are the 10 most common words in the job description data?
# We calculated the frequent words list above, but this also had stopwords in it and most probably stopwords appear very frequently in the top 100 words. **Let's see what words (other than stopwords) appear in this corpus of job descriptions!**
# 
# ### Concept 1 - Lemmatization
# 'Lemma' is a latin word meaning *root*. What lemmatization does is to **take similar words - 'experience', 'experiences', 'experiencing' and reduces them to their root word *experience*.** This is helpful for us since it brings out the most important words in the corpus and doesnt treat each form of the word separately! 
# 
# ### Concept 2 - Stemming
# Stemming is a heuristic process that trims the ends of similar looking words to end up with the same words - thereby reducing the vocabulary of words we have! **This is not that intuitive however, since sometimes it would take 'operate', 'operates' and 'operating' and trim them all to 'oper' which is not even a word!** So while this can be useful for purely prediction purposes, this is not meaningful as an output! Check out this [link](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html) for an excellent explanation on Stemming and Lemmatization!

# In[ ]:


# import the necessary functions from the nltk library
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()

# prepare corpus from the descriptions that dont have stopwords
corpus = " ".join(train['Clean_Full_Descriptions_no_stop'].tolist())

#tokenize words
tokenized_corpus = nltk.word_tokenize(corpus)

# lemmatize these tokens
lemmatized_tokens = [lmtzr.lemmatize(token) for token in tokenized_corpus]

# word frequencies for the lemmatized tokens
fd = nltk.FreqDist(lemmatized_tokens)

# get the top words
top_words = []
for key, value in fd.items():
    top_words.append((key, value))

# sort the list by the top frequencies
top_words = sorted(top_words, key = lambda x:x[1], reverse = True)

# keep top 10 words only
top_words = top_words[:10]

top_words


# The top 10 words make a lot of intuitive sense - words like **experience, role, work, business and skill** are essential parts of any job description!

# # On to Predictions - 
# Having understood the basics of text analytics, let's move on to predicting salary using the data that we have and see if using job descriptions gives us a good model to use to predict job salaries! To see the power of text data, we will compare two models - 
# 1. Using job descriptions (without the stopwords) as the only predictor
# 2. Using all the other variables as the predictors except for Job ID.
# 
# We will convert this to a classification problem - <br>
# *Predict high (>75th percentile) or low salary (<75th percentile) using the data provided*. 

# In[ ]:


# get the 75th percentile value of salary!
sal_perc_75 = np.percentile(train['SalaryNormalized'], 75)

# make a new target variable that captures whether salary is high (1) or low (0)
train['Salary_Target'] = np.where(train['SalaryNormalized'] >= sal_perc_75, 1, 0)


# Let's do the customary checks - data types of variables and missing values in the data.

# In[ ]:


train.dtypes.value_counts()


# In[ ]:


train.isnull().sum()[train.isnull().sum()>0]


# **Observations** -
# 1. Most values in our dataframe are of the 'Object' or 'String' data type. This means that we will have to convert these to dummy variables to proceed! 
# 2. There are missing values in the variables as shown above! These are all 'character' variables so when we create dummies, there will be a new column for the 'NA' values.
# 
# Before we create dummies, lets see which columns are categorical in nature!

# In[ ]:


# this gives us the categorical variables and the number of unique entries in them!
train.select_dtypes('object').nunique(dropna = False)


# *LocationNormalized* has the job location information! This has a large number of cities which would give us a huge number of columns - we DONT want that! Lets instead make a list of expensive cities and make an indicator variable denoting whether a city is expensive or not!

# #### Get a list of expensive cities in England

# In[ ]:


exp_cities = ['London', 'Oxford', 'Brighton', 'Cambridge', 'Bristol', 'Portsmouth', 
              'Reading', 'Edinburgh', 'Leicester', 'York', 'Exeter']


# In[ ]:


def check_city(s):
    '''Given a Normalized Location this tells us if it is an expensive city'''
    global exp_cities
    answer = pd.Series(exp_cities).map(lambda x: x in s)
    answer = min(np.sum(answer),1)
    return answer

# add the indicator as a column in the dataframe
train['Exp_Location'] = train['LocationNormalized'].map(check_city)


# ### What's the baseline accuracy for this problem?
# Baseline accuracy is defined as the accuracy that we would get if we dont fit a model, but just keep predicting the majority class irrespective of the data. How are our two classes represented in this data?

# In[ ]:


train['Salary_Target'].value_counts()/len(train)


# For our problem, since we split salaries using the 75th percentile as the threshold, **we have high salaries (1) 25% of the time and low salaries (0) 75% of the time!** This gives us an imbalanced class problem. Therefore, in addition to reporting accuracy we would also report the [AUC-ROC score](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)!

# ## Introducing the Naive Bayes Algorithm
# For prediction problems using text data, Naive Bayes algorithm is one of the better algorithms out there.  They have been demonstrated to be fast, reliable and accurate in a number of applications of NLP. The basic fundamentals for this algorithm derive from the Naive Bayes Theorem which states (mathematically) - 
# $$P(A|B) = \frac{P(B|A)*P(A)}{P(B)}$$
# 
# For our problem, we have to predict a class given a vector of predictors. The above formula becomes - 
# 
# $$P(Class|Vector) = \frac{P(Vector|Class)*P(Class)}{P(Vector)}$$
# 
# The final choice of the class between two classes (in our case, high or low salary), is based on which probability is higher - 
# 
# $$P(Class = High Salary|Vector)$$  vs $$P(Class = Low Salary|Vector)$$
# 
# Since the probability of the Vector $P(Vector)$ remains the same for both calculations, we ignore it. So, in essence we have to find the RHS of the equation -  
# $$P(Class|Vector) \propto P(Vector|Class)*P(Class)$$
# 
# The $P(Class)$ is just the probability that a class appears, given by - 
# $$\frac{Number of rows with that Class}{Total Rows}$$
# 
# Sometimes, because the product $$P(Vector|Class)*P(Class)$$ can get really small, we take the log of the expression to calculate - 
# $$log(P(Vector|Class)*P(Class))$$
# 
# and then compare these values for both classes for a document (or job description)

# ## Two types of Naive Bayes Algorithms
# We understood that essentially a naive bayes problem boils down to calculating the following probability - 
# $$P(Vector|Class)$$
# 
# There are two approaches to get here and this gives rise to bernoulli naive bayes and multinomial naive bayes. The difference is subtle and has to do with how we represent our data. 
# 
# ### Bernoulli Naive Bayes -
# For Bernoulli Naive Bayes each column in the dataset has to be a 1 or 0 type variable. If we speak about only job description data, we can say that for each term in a job description (where each term is a unique column of my dataframe), I will have either a 0 or a 1 depending on whether a word is present in my document or not. Lets say I have the following training data, categorising documents as belonging to either the 'Sport' class or the 'Informatics' class.
# 
# | Document ID | goal | tutor | variance | speed | drink | defence | performance | field | Class       |
# |-------------|------|-------|----------|-------|-------|---------|-------------|-------|-------------|
# | 1           | 1    | 0     | 0        | 0     | 1     | 1       | 1           | 1     | Sport       |
# | 2           | 0    | 0     | 1        | 0     | 1     | 1       | 0           | 0     | Sport       |
# | 3           | 0    | 1     | 0        | 1     | 0     | 1       | 1           | 0     | Sport       |
# | 4           | 1    | 0     | 0        | 1     | 0     | 1       | 0           | 1     | Sport       |
# | 5           | 1    | 0     | 0        | 0     | 1     | 0       | 1           | 1     | Sport       |
# | 6           | 0    | 0     | 1        | 1     | 0     | 0       | 1           | 1     | Sport       |
# | 7           | 0    | 1     | 1        | 0     | 0     | 0       | 1           | 0     | Informatics |
# | 8           | 1    | 1     | 0        | 1     | 0     | 0       | 1           | 1     | Informatics |
# | 9           | 0    | 1     | 1        | 0     | 0     | 1       | 0           | 0     | Informatics |
# | 10          | 0    | 0     | 0        | 0     | 0     | 0       | 0           | 0     | Informatics |
# | 11          | 0    | 0     | 1        | 0     | 1     | 0       | 1           | 0     | Informatics |
# 
# and I want to classify these two documents - 
# 
# | Document ID | goal | tutor | variance | speed | drink | defence | performance | field | Class |
# |-------------|------|-------|----------|-------|-------|---------|-------------|-------|-------|
# | 12          | 1    | 0     | 0        | 1     | 1     | 1       | 0           | 1     | ?     |
# | 13          | 0    | 1     | 1        | 0     | 1     | 0       | 1           | 0     | ?     |
# <br> 
# 
# 
# For the above example we get - 
# $P(Sport) = \frac{6}{11}$, $P(Informatics) = \frac{5}{11}$. The probabilities for the words for each class are - 
# 
# $P(Goal|Sport) = \frac{1}{2}$, $P(tutor|Sport) = \frac{1}{6}$, $P(Goal|Informatics) = \frac{1}{5}$, etc.
# 
# **Naive bayes is called naive bayes because of the assumption that all words occur independently**, which means - 
# 
# $$P(Vector|Class) = \Pi_{t = 1}^{V}(d_{jt}*P(w_{t}|Class) + (1 - d_{jt})*(1 - P(w_{t}|Class))$$
# 
# where $w_{t}$ are the words in the dataframe and $d_{jt}$ are the elements in $j$ row and $t$ column.
# 
# Essentially, we multiply the probability of occurence or non-occurence of a word in a class times an indicator (1 for occurence and 0 for non-occurence). Intuitively, this says that a description can be of a particular class both by what words it has and also by what words it doesnt have! So for document 12 the probability that it belongs to class Sport is - 
# $$P(Sport|Doc12) = \frac{6}{11}*(\frac{1}{2}*(1 - \frac{1}{6})*(1 - \frac{1}{3})*\frac{1}{2}*\frac{1}{2}*\frac{2}{3}*(1 - \frac{2}{3})*\frac{2}{3}) = 5.6 * 10^{-3}$$

# ### Multinomial Naive Bayes - 
# For multinomial naive bayes, our representation of the data changes and now we use the frequency of appearance of a term rather than just a 1 or 0 depending on the occurence. For this algorithm, the 
# $$P(Vector|Class) = P(Class) * \Pi_{t=1}^{V}P(w_{t}|Class)^{d_{jt}}$$
# 
# where $w_{t}$ are the words in the vocabulary and $d_{jt}$ are the elements in row j and column t. In this case, the  bag-of-words representation changes slightly and we each element $d_{jt}$ is the frequency of occurence of a term in the document. Suppose that we have the following documents - 
# 
# | Text                         | Class      |
# |------------------------------|------------|
# | A great game                 | Sports     |
# | The election was over        | Not Sports |
# | Very clean match             | Sports     |
# | A clean but forgettable game | Sports     |
# | It was a close election      | Not Sports |
# 
# The bag-of-words representation of this data is as follows - 
# 
# | Text                         | a | great | game | the | election | was | over | very | clean | match | but | forgettable | it | close | Class      |
# |------------------------------|---|-------|------|-----|----------|-----|------|------|-------|-------|-----|-------------|----|-------|------------|
# | A great game                 | 1 | 1     | 1    | 0   | 0        | 0   | 0    | 0    | 0     | 0     | 0   | 0           | 0  | 0     | Sports     |
# | The election was over        | 0 | 0     | 0    | 1   | 1        | 1   | 1    | 0    | 0     | 0     | 0   | 0           | 0  | 0     | Not Sports |
# | Very clean match             | 0 | 0     | 0    | 0   | 0        | 0   | 0    | 1    | 1     | 1     | 0   | 0           | 0  | 0     | Sports     |
# | A clean but forgettable game | 1 | 0     | 1    | 0   | 0        | 0   | 0    | 0    | 1     | 0     | 1   | 1           | 0  | 0     | Sports     |
# | It was a close election      | 1 | 0     | 0    | 0   | 1        | 1   | 0    | 0    | 0     | 0     | 0   | 0           | 1  | 1     | Not Sports |
# 
# 
# For multinomial Naive Bayes, we look at the probability of a word appearing in each class $P(w_{t}|Class)$, which for this example is given by the sum of frequencies in all documents of a class where the word appears (sum of column values for that word) divided by the total number of times any word appears (sum of all terms in the bag-of-words representation). Therefore we get (for some words) - 
# 
# | Word  | P(Word,Sports) | P(Word,Not Sports) |
# |-------|----------------|--------------------|
# | a     | (2/11)         | (1/9)              |
# | close | (0/11)         | (1/9)              |
# | game  | (2/11)         | (0/9)              |
# | very | (1/11)         | (0/9)             |
# 
# Note - I use comma in the column names above because the table gives me a weird error. It actually means $P(Word|Sports)$ and $P(Word|Not Sports)$. 

# We see a problem in the above table - For words where the frequency of appearance in training data is 0 (for example for the word 'close' in Sports documents), we would also get a 0 probability if that word appears in the testing data even though that test document might belong to 'Sports'. To avoid this error, we add a very small frequency to the probability of appearance of each term in the corpus and the formula becomes - 
# $$P(w_{t}|Class) = \frac{(1 + \text{sum of column where word appears})}{(\text{count of vocabulary} + \text{sum of all terms in the bag of words representation})}$$.
# 
# So our table of probabilities changes to - 
# 
# | Word  | P(Word,Sports)  | P(Word,Not Sports) |
# |-------|-----------------|--------------------|
# | a     | (1 + 2/11 + 14) | (1 + 1/9 + 14)     |
# | close | (1 + 0/11 + 14) | (1 + 1/9 + 14)     |
# | game  | (1 + 2/11 + 14) | (1 + 0/9 + 14)     |
# | very | (1 + 1/11 + 14) | (1 + 0/9  + 14)    |
# 
# For a test document - "A very close game", we get the probabilities as - 
# $$P(document|Sports) = \frac{3}{25}^{1} * \frac{2}{25}^{1} * \frac{1}{25}^{1} * \frac{3}{25}^{1} * \frac{3}{5}$$
# $$P(document|Sports) = 4.61 * 10^{-5}$$
# 
# $$P(document|Not Sports) = \frac{2}{23}^{1} * \frac{1}{23}^{1} * \frac{2}{23}^{1} * \frac{1}{23}^{1} * \frac{2}{5}$$
# $$P(document|Not Sports) = 1.43 * 10^{-5}$$
# 
# So, this document belongs to the Sports class!

# ### Bernoulli Naive Bayes using everything but descriptions
# Now that we have the basics in place, lets apply these models. Just as we showed for document-vocabulary dataframe above, bernoulli naive bayes can be applied to any data which has variables with the domain 0 or 1. So, for our data, since we have all categorical variables, we can get dummy variables for those and then apply bernoulli naive bayes!

# In[ ]:


# Subset the columns required
columns_required = ['ContractType', 'ContractTime', 'Company', 'Category', 'SourceName', 'Exp_Location', 'Salary_Target']
train_b1 = train.loc[:, columns_required]

# Convert the categorical variables to dummy variables
train_b1 = pd.get_dummies(train_b1)

# Lets separate the predictors from the target variable
columns_selected = train_b1.columns.values.tolist()
target_variable = ['Salary_Target']

# predictors are all variables except for the target variable
predictors = list(set(columns_selected) - set(target_variable))

# setup the model
from sklearn.naive_bayes import BernoulliNB

X = np.array(train_b1.loc[:,predictors])
y = np.array(train_b1.loc[:,target_variable[0]])

# create test train splits 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)

model = BernoulliNB()

# Fit the model and predict the output on the test data
model.fit(X_train, y_train)

# Predicted output
predicted = model.predict(X_test)

# Accuracy
from sklearn import metrics

print("Model Accuracy is:", metrics.accuracy_score(y_test, predicted))
print("Area under the ROC curve:", metrics.roc_auc_score(y_test, predicted))
print("Confusion Matrix:\n",metrics.confusion_matrix(y_test, predicted))


# 1. The prediction accuracy achieved is **78.12%** using just the numerical variables!
# 2. The area under the ROC curve (AUC-ROC score) is **66.6%**. 
# 
# 3. **The confusion matrix has the actual y labels as the rows and predicted labels as the columns.** 
#     So the first cell in the matrix is read as - *The number of times the actual salary was low (0) and our model also predicted it as low (0)*

# ### Multinomial Naive Bayes using job descriptions

# In[ ]:


# Lets lemmatize the job descriptions before we run the model
def text_lemmatizer(s):
    '''Given a description, this lemmatizes it'''
    tokenized_corpus = nltk.word_tokenize(s)
    
    # lemmatize
    s = " ".join([lmtzr.lemmatize(token) for token in tokenized_corpus])
    return s

# lemmatize the descriptions
train['Clean_Full_Descriptions_no_stop_lemm'] = train['Clean_Full_Descriptions_no_stop'].map(text_lemmatizer)

# make the X and y matrices for model fitting
X = np.array(train.loc[:, 'Clean_Full_Descriptions_no_stop_lemm'])
y = np.array(train.loc[:, 'Salary_Target'])

# split into test and train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)

# Convert the arrays into a presence/absence matrix
from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer()
X_train_counts = count_vectorizer.fit_transform(X_train)
X_test_counts = count_vectorizer.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
nb_mult_model = MultinomialNB().fit(X_train_counts, y_train)
predicted = nb_mult_model.predict(X_test_counts)

print("Model Accuracy:", metrics.accuracy_score(y_test, predicted))
print("Area under the ROC curve:", metrics.roc_auc_score(y_test, predicted))
print("Model Confusion Matrix:\n", metrics.confusion_matrix(y_test, predicted))


# ### Bernoulli Naive Bayes using Job Descriptions

# In[ ]:


# Calculate the frequencies of words using the TfidfTransformer
X_train_bern = np.where(X_train_counts.todense() > 0 , 1, 0)
X_test_bern = np.where(X_test_counts.todense() > 0, 1, 0)

# Fit the model
from sklearn.naive_bayes import BernoulliNB
nb_bern_model = BernoulliNB().fit(X_train_bern, y_train)
predicted = nb_bern_model.predict(X_test_bern)

# print the accuracies
print("Model Accuracy:", metrics.accuracy_score(y_test, predicted))
print("Area under the ROC curve:", metrics.roc_auc_score(y_test, predicted))
print("Model Confusion Matrix:\n", metrics.confusion_matrix(y_test, predicted))


# ### Concept - Mutual Information
# Just like the concept of feature importance in predictive models such as regression or random forests, we have the concept of mutual information while performing text analytics. For our example, mutual Information would tell us which words are the most indicative of high or low salary. The formula for mutual information looks like - 
# $$MI(x,y) = p(x, y) * log_2(p(x,y)/(p(x) * p(y)))$$
# 
# where $x$ is the word in the vocabulary and $y$ is a class. The values in the expression are found as follows - 
# $p(x) = \frac{n_{x}}{N}$, $p(y) = \frac{n_{y}}{N}$ and $p(x,y) = \frac{n_{x,y}}{N}$,
# 
# where $n_{x}$ is count of documents with word 'x', $n_{y}$ is count of documents with class 'y', $n_{x,y}$ is count of documents with class 'y' and word 'x' and 'N' is the total documents.
# 
# The interpretation of this is simple, large positive values of MI mean that the presence of a certain words are strongly indicative that the document belongs to the class. Negative values means that the presence is negatively associated with the class. 
# 
# The term inside the log can be thought of as a 'Lift' - How much do we expect this word and class to appear together. If the word and class are independent $p(x,y)$ equals $p(x)*p(y)$ and MI goes to 0.

# ### Words that indicate high/low salary

# In[ ]:


# extract the column names for the columns in our training dataset.
column_names = [x for (x,y) in sorted(count_vectorizer.vocabulary_.items(), key = lambda x:x[1])]

# probability of high salary
p_1 = np.mean(y_train)

# probability of low salary
p_0 = 1 - p_1

# create an array of feature vectors
feature_vectors = np.array(X_train_bern)

# probability of word appearance
word_probabilities = np.mean(feature_vectors, axis = 0)

# probability of seeing these words for class= 1 and class = 0 respectively
p_x_1 = np.mean(feature_vectors[y_train==1, :], axis = 0)
p_x_0 = np.mean(feature_vectors[y_train==0, :], axis = 0)

# words that are good indicators of high salary (class = 1)
high_indicators = p_x_1 * (np.log2(p_x_1) - np.log2(word_probabilities) - np.log2(p_1))

high_indicators_series = pd.Series(high_indicators, index = column_names)

# words that are good indicators of low salary (class = 0)
low_indicators = p_x_0 * (np.log2(p_x_0) - np.log2(word_probabilities) - np.log2(p_0))

low_indicators_series = pd.Series(low_indicators, index = column_names)


# ### Get words indicative of low salary
# The numbers against the terms show the mutual information of these words with the low salary output

# In[ ]:


low_indicators_series[[i for i in low_indicators_series.index if i not in en_stopwords]].sort_values(ascending = False)[:10].index


# ### Get words indicative of high salary
# The numbers against the terms show the mutual information of these words with the low salary output

# In[ ]:


high_indicators_series[[i for i in high_indicators_series.index if i not in en_stopwords]].sort_values(ascending = False)[:10].index


# ## Conclusion - 
# We saw that using just the job descriptions gives us an accuracy comparable and slightly better as compared to using the other predictors in this case. There is a lot more that can be done with text and I have just scratched the surface here for text analytics. I hope you liked this. Share your thoughts! :)
