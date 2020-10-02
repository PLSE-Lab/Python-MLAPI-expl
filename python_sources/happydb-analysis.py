#!/usr/bin/env python
# coding: utf-8

# # Introduction
# In this notebook, I'm trying to do some preliminary analysis on the HappyDB dataset, in the meanwhile introducing some basic and useful analytical tools/libraries.
# 
# This notebook consists of three parts:
# * **Data preparation and basic analyses.** The main tasks are loading the data, doing a word count and drawing a word cloud.
# * **Entity extraction.** Here I first do a straighforward entity extraction of seasons with Python, followed by a more sophisticated entity extraction of purchased products. For the second task, I'll introduce a convenient yet powerful entity extraction system called **[Koko](http://pykoko.readthedocs.io/en/latest/)**.  
# * **Classification of genders.** I make this task a binary classification problem, and shows how to use the logistic regression model of **[scikit-learn](http://scikit-learn.org/stable/)** to finish the task.
# 
# Let's get started!  
# We first need to import a few necessary packages:

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, ImageColorGenerator


# # 1. Data preparation and basic analyses
# HappyDB consists of statement of happy moments by people around the world.  
# We can load the data and take a quick look inside.  
# ## 1.1 Load HappyDB

# In[ ]:


hm_data = pd.read_csv('../input/happydb/cleaned_hm.csv')
hm_data.head()


# The dataset is composed of happy moments from two reflection_period, 24 hours and 3 months.  
# Let's take a quick look at the data with reflection_period of 3 months.

# In[ ]:


hm_data.loc[hm_data["reflection_period"]=='3m'].head()


# Of all the columns, "cleaned_hm" is of particular interest as it contains happy moments with proper cleaning (e.g., typo correction).  
# The rest of the analysis will be mostly based on "cleaned_hm".  
# ## 1.2 Word count
# To get a general idea of the cleaned happy moments, we can perform a statistical analysis based on the number of words.

# In[ ]:


df_hm = hm_data[hm_data['cleaned_hm'].notnull()]
len_count = df_hm['cleaned_hm'].apply(lambda x: len(x.split()))
len_count.describe()


# Looks like most of the happy moments are short sentences, as expected!  
# Some happy moments even have only two words.

# In[ ]:


df_hm[df_hm['cleaned_hm'].apply(lambda x: len(x.split()))==2].head()


# So what's the distribution of the word count?

# In[ ]:


length_order = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39",                 "40-44", "45-49", ">=50"]
length_category = len_count.apply(lambda x: length_order[min(10, int(x/5))])
length_counts = pd.DataFrame(length_category.value_counts()).reset_index()
length_counts.columns = ['word numbers', '# of moments']

sns.barplot(x='word numbers', y='# of moments', data=length_counts, order=length_order)


# Most of the happy moments are between five words and twenty words.  
# ## 1.3 Word frequency
# I'm also curious about what words people mention most in their happy moments.  
# A good tool that could help us in this case is a word cloud.

# In[ ]:


text = ' '.join(df_hm['cleaned_hm'].tolist())
text = text.lower()
wordcloud = WordCloud(background_color="white", height=2700, width=3600).generate(text)
plt.figure( figsize=(14,8) )
plt.imshow(wordcloud.recolor(colormap=plt.get_cmap('Set2')), interpolation='bilinear')
plt.axis("off")


# It seems that some words appear more frequently, such as "friend", "family" and "daughter".  
# There are also noise words that are not very informative, such as "happy", "yesterday" and "today".  
# Let's clean the word cloud by removing these noise.

# In[ ]:


LIMIT_WORDS = ['happy', 'day', 'got', 'went', 'today', 'made', 'one', 'two', 'time', 'last', 'first', 'going', 'getting', 'took', 'found', 'lot', 'really', 'saw', 'see', 'month', 'week', 'day', 'yesterday', 'year', 'ago', 'now', 'still', 'since', 'something', 'great', 'good', 'long', 'thing', 'toi', 'without', 'yesteri', '2s', 'toand', 'ing']

text = ' '.join(df_hm['cleaned_hm'].tolist())
text = text.lower()
for w in LIMIT_WORDS:
    text = text.replace(' ' + w, '')
    text = text.replace(w + ' ', '')
wordcloud = WordCloud(background_color="white", height=2700, width=3600).generate(text)
plt.figure( figsize=(14,8) )
plt.imshow(wordcloud.recolor(colormap=plt.get_cmap('Set2')), interpolation='bilinear')
plt.axis("off")


# Great. Now it's quite clear that "work", "friend" and "son" are the most frequent words in the dataset.
# # 2. Entity Extraction on HappyDB
# Now let's try to dig deeper into the dataset by focusing on sub-domains of daily life.  
# For example, what seasons make people happiest? What purchased products make people happiest?
# ## 2.1 Happiest seasons

# In[ ]:


seasons = ['Spring', 'Summer', 'Fall', 'Winter']

# Check each moment, and increase the count for the mentioned season
season_dic = dict((x,0) for x in seasons)
tokens_hm = df_hm['cleaned_hm'].apply(lambda x: x.split())
for _, value in tokens_hm.iteritems():
    for word in value:
        if word in seasons:
            season_dic[word] += 1
            
season_dic


# 'Spring' is clearly the winner, as it's mentioned significantly more frequently than other seasons.  
# I also tried to search for 'Autumn', but there are zero mentions. Any interested reader could try yourself as well.
# ## 2.2 Products that make people happy
# It's interesting to understand what products that people buy make them happy.  
# We can model the task as an entity extraction problem.  
# And I'll use an entity extraction system called [Koko](http://pykoko.readthedocs.io/en/latest/) for this task.
# ### 2.2.1 Retrieval of happy moments
# Koko takes texts as input. Let's first retrieve the cleaned happy moments (i.e., 'cleaned_hm' column) from HappyDB and put them into one text file.  
# For efficiency, I'll only use 1/8 of the happy moments for entity extraction.

# In[ ]:


# Read the happyDB sample file
with open('happydb.txt', 'w') as ofile:
    len = int(df_hm.shape[0] / 8)
    for i in range(0, len - 1):
        ofile.write(df_hm['cleaned_hm'].iloc[i] + '\n')
        
print("Happy moments are retrieved!")


# ### 2.2.2 Extraction of purchased products
# Now let's write a Koko query for product extraction as follows:

# In[ ]:


with open('../input/kokosamples/purchase_v1.koko', 'r') as file:
    print(file.read())


# This query tells Koko to extract noun phrases 'x' from HappyDB if "x" is preceded by either "buy" or "purchase".  
# 
# The weight in each "if" condition (e.g., {0.1} for ("buy" x)) represents the importance of the pattern specified in the condition.  
# Any appearance of an entity in happy moments that matches the pattern is considered a piece of evidence.  
# And each such piece of evidence would increment the entity's score by the condition's weight.
# 
# For example, if there's a happy moment "I buy a car", this moment is considered as evidence for "a car" based on the first condition, and 0.1 is added to "a car"'s score.  
# In Koko, the score of an entity is at most 1.
# 
# Finally, we can specify threshold in Koko queries.  
# Only entities scoring higher than the thresold would be returned in the results.  
# For simplicity, I put zero as thresold here, which shows all entities that have at least one piece of evidence in happy moments.  
# 
# **Let's run the Koko query now to see the results.**  
# Here I use [spaCy](https://spacy.io/) as the nlp processor for happy moments.
# Koko could leverage spaCy's APIs for entity extraction.  
# The extracted entities could be further matched against the conditions in the Koko query to get scored, ranked and filtered.
# 
# SpaCy is not the only option. We can also use Koko's default parser or [Google NLP API](https://cloud.google.com/natural-language/) as well.

# In[ ]:


import koko
import spacy

koko.run('../input/kokosamples/purchase_v1.koko', doc_parser='spacy')


# On one hand, the results give us useful information.  
# People are in general happy when they purchase a car, followed by costumes, tickets and phones etc.  
# On the other hand, there's much noise in the results.  We have pronouns such as "me", "it" and "my husband" as well.  
# This's not surprising though. We often say "buy somebody something" in daily life.  
# 
# **Fortunately, Koko allows us to specify exclusion rules to get rid of the unwanted noise.**  
# Here's an updated query excluding pronouns.  
# Also, to make results more concise, *I reset the thresold to 0.2 in all subsequent queries*.

# In[ ]:


with open('../input/kokosamples/purchase_v2.koko', 'r') as file:
    print(file.read())


# Let's run it again.

# In[ ]:


koko.run('../input/kokosamples/purchase_v2.koko', doc_parser='spacy')


# The results look much cleaner this time.  
# If we take a closer look at the query, we might wonder if the two conditions listed cover all the purchasing behavior?  
# The answer may be no.

# In[ ]:


# Select all happy moments used for entity extraction that contain 'purchased'.
df_hmee = df_hm[:int(df_hm.shape[0]/8)]
df_purchase = df_hmee[df_hmee['cleaned_hm'].apply(lambda x: x.find('purchased') != -1)]

print("Number of happy moments containing 'purchased': {}".format(df_purchase.shape[0]))


# Well, looks like we missed quite a few happy moments that contain 'purchased'.
# However, enumerating all purchase-related keywords we can think of in a Koko query is quite tedious.  
# 
# Koko provides a handy feature for solving this problem, which is called a descriptor.  
# To use a descriptor, we only need to write *one* condition, and put a tilde "~" between the keyword in the condition and the entity we're trying to extract.  

# In[ ]:


with open('../input/kokosamples/purchase_v3.koko', 'r') as file:
    print(file.read())


# During execution, Koko would automatically expand the keyword to all related words it can find in a word embedding file.   
# Of course, we need to supply the embedding file ourselves.

# In[ ]:


embedding_doc = "../input/glove840b/glove.840B.300d.txt"
koko.run('../input/kokosamples/purchase_v3.koko', doc_parser='spacy', embedding_file=embedding_doc)


# Now the results capture more purchasing experience in the dataset.  
# For example, now we have 12 instances of "a new car" compared to 4 instances in the previous example. 
# 
# From the given results, we may conclude that purchasing of expensive products, such as "car", "smartphone" or "bike", tend to make people happier.
# ## 3. A logistic regression classifier for gender
# The HappyDB dataset comes with demographic info with author's information.  
# In this part, let's try to train a classifier to identify the gender of each happy moment's author.  
# For simplicity, I make this problem a binary classification problem. And I'll use logistic regression to approach the task.
# ### 3.1 Data preparation
# First we load the demographic data.

# In[ ]:


demo_data = pd.read_csv('../input/happydb/demographic.csv')

demo_data.head()


# We then join the happy moments and the demographic data -- based on the "wid" column -- 
# to identify the author's gender for each happy moment.

# In[ ]:


merge_data = pd.merge(hm_data, demo_data, on='wid')
gender_data = merge_data[['cleaned_hm', 'gender']]

gender_data.head()


# We can take a quick look at the frequency distribution for gender.

# In[ ]:


gender_data.gender.value_counts().plot(kind='bar')


# Since this's a binary classification problem, we only consider gender of male or female.  
# Let's clean the data set to retain happy moments whose gender is either male or female.

# In[ ]:


gender_bin_data = gender_data[(gender_data['gender'] == 'm') | (gender_data['gender'] == 'f')]

print("Happy moments written by male/female: {}".format(gender_bin_data['cleaned_hm'].size))


# To prepare the data for classification task, we need to convert the representation of male and female into numbers.

# In[ ]:


gender_bin_data = gender_bin_data.assign(gender_bin=(np.where(gender_bin_data['gender']=='m', 1, 0)))

gender_bin_data.head()


# We use the first 70% happy moments as the training data, with the rest 30% as test data.

# In[ ]:


hm_size = gender_bin_data['cleaned_hm'].size
num_train_hm = int(0.7 * gender_bin_data['cleaned_hm'].size)

train_hm = gender_bin_data.iloc[0:num_train_hm]
test_hm = gender_bin_data.iloc[num_train_hm:hm_size]
test_hm = test_hm.reset_index(drop=True)

test_hm.head()


# We further clean up the texts, to remove numbers and punctuation.

# In[ ]:


def clean_up_texts(hm_data):
    prepro_hm = []
    stops = set(stopwords.words("english"))
    for i in range(0, hm_data['cleaned_hm'].size):
        # Remove non-english words, including punctuations and numbers
        letters = re.sub("[^a-zA-Z]", " ", hm_data.iloc[i]['cleaned_hm'])

        # Convert all words to lower case
        lower_words = letters.lower()

        # Tokenize the sentences
        tokens = lower_words.split()

        # Reconstruct the processed tokens into a string
        prepro_string = " ".join(tokens)

        prepro_hm.append(prepro_string)
        
    return prepro_hm
    
prepro_train = clean_up_texts(train_hm)
prepro_test = clean_up_texts(test_hm)
print("Texts cleaned up! \n")


# Let's take a peek at the cleaned data:

# In[ ]:


prepro_train[:10]


# ### 3.2 Feature selection
# The next step is to select proper features for training and testing.  
# 
# Here I start with the simpliest model: *bag-of-words model*.  
# The bag-of-words model tries to create a dictionary based on the input strings.  
# With the dictionary, each sentence can then be modeled as a vector representing the frequency of each word.
# I use scikit-learn here to build features for bag-of-words model.

# In[ ]:


vectorizer = CountVectorizer()
features_train_hm = vectorizer.fit_transform(prepro_train)
train_array_hm = features_train_hm.toarray()

print("Dimension of the training data: {}".format(train_array_hm.shape))


# There are 20737 distinct words in the dataset!  
# A quick look at the features (i.e., words) we use:

# In[ ]:


vocab = vectorizer.get_feature_names()

vocab[:20]


# Now we can train a logistic regression model with the extracted features.  
# ### 3.3 Training of the logistic regression classifier
# Let's train the logistic regression model:

# In[ ]:


from sklearn.linear_model import LogisticRegression

logi_model = LogisticRegression()
logi_model.fit(train_array_hm, train_hm['gender_bin'])

logi_model.score(train_array_hm, train_hm['gender_bin'])


# The training accuracy is 76%, a reasonable result.  
# We can see which words are the most influential by looking at the coefficients of the logistic model.

# In[ ]:


feature_names = vocab
coefficients = logi_model.coef_.tolist()[0]
weight_df = pd.DataFrame({'Word': feature_names,
                          'Coeff': coefficients})
weight_df = weight_df.sort_values(['Coeff', 'Word'], ascending=[0, 1])
weight_df.head(n=10)


# These are the words with the biggest positive coefficients.  
# In other words, these are the words in the dataset that strongly suggest male.  
# 
# Well, I can understand that mentioning of "wife", "gf" or "smoking" suggests a male author.  
# But why "seattle" and "stone"? This is definitely worth more investigation (maybe in another notebook).  
# 
# Let's also take a look at the least influential words:

# In[ ]:


weight_df.tail(n=10)


# These are the words that tend to make classifier predict 0, which means female.  
# Most of the words make sense to me, but again, the appearance of "mth" is kind of surprising.
# ### 3.4 Evaluation of the classifier
# To evaluate the performance of the classifier, let's use the trained model to make prediction on the test data.

# In[ ]:


features_test_hm = vectorizer.transform(prepro_test)
test_array_hm = features_test_hm.toarray()

print("Dimension of the test data: {}".format(test_array_hm.shape))


# In[ ]:


predictions = logi_model.predict(test_array_hm)


# In[ ]:


from sklearn import metrics

print(metrics.accuracy_score(test_hm['gender_bin'], predictions))


# We have 64% precision on the test data. Not bad!  
# For anyone who wants to learn more about bag-of-words model, I recommend this [tutorial](https://www.kaggle.com/c/word2vec-nlp-tutorial#part-1-for-beginners-bag-of-words).
# # 4. Conclusion
# This notebook is just a preliminary exploration of the dataset.  I'm sure there's much more to dig there.
# 
# Some of the future directions I can think of now is:
# - Extraction of locations that make people feel happy.
# - More sophisticated features with logistic regression model (e.g., ngram model).
# - More sophisticated models (e.g., a neural network).
# 

# In[ ]:




