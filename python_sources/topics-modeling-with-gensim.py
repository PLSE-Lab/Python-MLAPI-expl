#!/usr/bin/env python
# coding: utf-8

# # Quora insincere questions

# ### Objective of the competition : predict if a question is insincere (1) or not (0) to help the moderation system of Quora. 
# **Parameters to evaluate the insincerity:**
# * non-neutral tone (exagerated or rhetorical)
# * disparaging or inflamatory questions (suggest discriminatory idea, hateful, based on prejudices)
# * not grounded in reality (absurd or false)
# * sexual content for shock value + illicite sexual content

# I'm new to machine learning so i began by doing a topic modeling with gensim to see if i can identify recurent topics differencing sincere and insincere question.
# 
# The machine learning part is a more concise cause i wanted to focus on topic modeling. 

# # I. Imports and loading the data

# In[ ]:


import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#from vivadata.datasets.common import get_path_for_dataset
#base_path = get_path_for_dataset('quora')
X_train_filepath = os.path.join('..', 'input', 'train.csv')
X_test_filepath = os.path.join('..', 'input', 'test.csv')
sample_filepath = os.path.join('..', 'input', 'sample_submission.csv')
X_train_filepath, X_test_filepath, sample_filepath


# In[ ]:


df_train = pd.read_csv(X_train_filepath)
df_test = pd.read_csv(X_test_filepath)
sample = pd.read_csv(sample_filepath)
df_train.shape, df_test.shape, sample.shape


# Now that the data is loaded, let's take a quick look. 
# ![](https://media.giphy.com/media/94hqi5hBHu5W/giphy.gif)

# #  II.EDA

# In[ ]:


df_train.sample(10)


# - **qid** = unique question identifier
# - **question_text** = text of the question
# - **target** = 1: insincere question, 0:not insincere question

# In[ ]:


df_test.sample(10)


# **qid** = unique question identifier
# **question_text** = text of the question

# In[ ]:


df_train.info()


# In[ ]:


df_train.describe()


# In[ ]:


ax, fig = plt.subplots(figsize=(10, 7))
sns.countplot(x='target', data=df_train)
plt.title('Reparition of question by insincerity');


# There is 80 810 insincere questions on 1 306 122, so around 6% (0.0618). This would be a problem for the machin learning process cause it's an important disparity with 1 225 312 sincere ones. 

# **_Now, we need to transform the data from text to something the computer could understand in order to study it more precisely._**

# # III.Topic modelisation

# I'm gonna prepare the data,  then use gensim and lda modeling to create topics for sincere and insincere questions. 

# In[ ]:


import gensim
import nltk
from nltk.corpus import stopwords


# In[ ]:


# Define the target and the variable.
y_train = df_train.loc[:, 'target']
X_train = df_train.loc[:, 'question_text']
X_train.shape, y_train.shape


# In[ ]:


X_train.head()


# In[ ]:


# Create a variable with all the sincere questions.
X_train_sincere = X_train[df_train['target'] == 0]
X_train_sincere[:5]


# In[ ]:


# Create a variable with all insincere questions.
X_train_insincere = X_train[df_train['target'] == 1]
X_train_insincere[:5]


# ## III-A. Topic modeling with 10 topics

# ### III-A. 1. Tokenisation

# We need to transform the text data in token to be able to compute it. 

# In[ ]:


# First i load the list of stopwords, the words which aren't useful to understand the meaning.
stop_words = stopwords.words('english')
stop_words[:10]


# In[ ]:


# I transform the sincere questions in a list of words.
sincere_prepro_questions = [gensim.utils.simple_preprocess(question) 
                            for question in X_train_sincere]
sincere_prepro_questions[:2]


# In[ ]:


# I transform the insincere questions in a list of words.
insincere_prepro_questions = [gensim.utils.simple_preprocess(question) 
                              for question in X_train_insincere]
insincere_prepro_questions[:2]


# In[ ]:


# Verifying the length of the both lists.
len(sincere_prepro_questions), len(insincere_prepro_questions)


# In[ ]:


# I remove the stopwords from the sincere questions list of words.
clear_sincere_questions = [[word for word in question if word not in stop_words] 
                             for question in sincere_prepro_questions]
clear_sincere_questions[:2]


# In[ ]:


# I do the same for insincere one.
clear_insincere_questions = [[word for word in question if word not in stop_words] 
                             for question in insincere_prepro_questions]
clear_insincere_questions[:2]


# ### III-A.2. Creation of gensim dictionnaries

# In[ ]:


# Creation of a dictionnary of the words and their id for sincere questions
sincere_questions_dictionary = gensim.corpora.Dictionary(clear_sincere_questions)
sincere_token = sincere_questions_dictionary.token2id


# In[ ]:


# Association of the dictonary words and their frequency for sincere questions
sincere_dict_frequency = {sincere_questions_dictionary[k]: v for k,v in sincere_questions_dictionary.dfs.items()}


# In[ ]:


# Creation of a dictionnary of the words and their id for insincere questions
insincere_questions_dictionary = gensim.corpora.Dictionary(clear_insincere_questions)
insincere_token = insincere_questions_dictionary.token2id


# In[ ]:


# Association of the dictonary words and their frequency for insincere questions
insincere_dict_frequency = {sincere_questions_dictionary[k]: v for k,v in sincere_questions_dictionary.dfs.items()}


# ### III-A.3. Vectorization

# In[ ]:


# I transform the elements of the sincere dictionary in vectors. 
sincere_corpus = [sincere_questions_dictionary.doc2bow(question) 
                  for question in clear_sincere_questions]
sincere_corpus[:2]


# In[ ]:


# I do the same for the insincere questions. 
insincere_corpus = [insincere_questions_dictionary.doc2bow(question) 
                    for question in clear_insincere_questions]
insincere_corpus[:2]


# ### III-A.4. Lda Modeling

# Lda modeling is a statistical modeling which discovers abstract "topics" in documents. Here Lda modeling will help me to see if sincere and insincere questions have some majors topics. 
# 
# This topics could be used as a variable later, for the machine learning.

# In[ ]:


# Creating a lda model, which create 10 topics from the sincere questions
lda_model_sincere = gensim.models.ldamodel.LdaModel(
    corpus=sincere_corpus, num_topics=10, id2word=sincere_questions_dictionary,
    random_state=25, passes=5)


# In[ ]:


# Creating a lda model, which create 10 topics from the insincere questions
lda_model_insincere = gensim.models.ldamodel.LdaModel(
    corpus=insincere_corpus, num_topics=10, id2word=insincere_questions_dictionary,
    random_state=25, passes=5)


# In[ ]:


from pprint import pprint


# In[ ]:


# Printing the result of the modeling for sincere questions.
pprint(lda_model_sincere.print_topics()[0])


# In[ ]:


# Same for insincere ones. 
pprint(lda_model_insincere.print_topics()[0])


# ### III-A.5. Visualization
# 
# I will now measure how good the topic modeling is and create a visual of the topics. 
# 
# Measurement of how good a topic model is. 
# * Coherence score : 
#     * intrinsic measure: compare a word with the preceding & sucseeding ones. Calculate a log probability
#     * extrinsic measure: every single word is paired with every others. Use pointwise mutual information. 

# In[ ]:


import pyLDAvis.gensim
pyLDAvis.enable_notebook()


# In[ ]:


pyLDAvis.gensim.prepare(lda_model_sincere, sincere_corpus, sincere_questions_dictionary)


# Some topics overlapt each other (1 &4, 9&7, 2&3). The topic are less easily recognizable than for insincere questions. We can 
# recognize word showing interogation, some vocabulary about computer... but the vocabulary is larger and more miscellaneous. 

# In[ ]:


pyLDAvis.gensim.prepare(lda_model_insincere, insincere_corpus, 
                        insincere_questions_dictionary)


# The number of topics are not optimal, we can see that some topics (1, 3, 7 and 2, 5) overlapt. Maybe those subjects could
# could have be reunited in a same topics. 
# 
# The 1, 3, 7 seems to be about politic:
#     * the first about Donald Trump;
#     * the third about international politic;
#     * the seventh about political parties. 
#  We also see that the 30 most relevant words are sometimes repeated in the 3 topics ('trump', 'liberals', 'president', 'supporters'). 
# 
# The 2, 5 topics seems to be about racial and religious subject:
#     * the second refers to race ('white', 'black', 'chinese');
#     * and the second seems to be about Islam ('muslims', 'muslim', 'pakistant'). 
#    
# Topics 8 and 6, even if they are not close to the 1st cluster contain a lot of similar words and seems to also be about politics:
#        * the 8th about Donald Trump's election also ('trump', 'president', 'obama')
#        * the 6th about interior politic and social subjects ('gun', 'clinton', 'rape')
#        
# The 9th topic is about religion ('jews', 'christians', 'atheist')
# 
# The 10th and 4th seems to be about sex and gender:
#     * the 4th more about sex ('girl', 'sex', 'sexual');
#     * the 10th is less easy to classify but seems to be more about gender ('girl', 'gender', 'bible')
# 
# All those subjects has in common to be really polemics and suitable to trolls. This criteria would be easily used by a human 
# control but we need to find how to train your model to try to recognize it. 
# 
# The insincere questions seems easier to distinct than the sincere ones, cause they are centred on a less various spectrum of words. 

# ---

# **Now we do the same process with bigrams!**!

# ## III-B. Topics modeling with bigrams (10 topics)

# ### III-B.1. Creation of the bigrams (from preprocessed data)

# In[ ]:


from gensim.models import Phrases
from gensim.models.phrases import Phraser


# In[ ]:


# Create the bigram for sincere questions.
sincere_bigram_model = Phrases(sincere_prepro_questions, min_count=1, threshold=2)


# In[ ]:


# Same for insincere ones. 
insincere_bigram_model = Phrases(insincere_prepro_questions, min_count=1, threshold=2)


# In[ ]:


# Creation of the phrasers, which is needed to use the bigrams. First for sincere questions...
sincere_bigram_phraser = Phraser(sincere_bigram_model)


# In[ ]:


# Then for the insincere. 
insincere_bigram_phraser = Phraser(sincere_bigram_model)


# ### III-B.2.Creation of the bigram dictionary

# In[ ]:


sincere_bigrams = [sincere_bigram_model[question] for question in sincere_prepro_questions]
sincere_bigrams[:2]


# In[ ]:


insincere_bigrams = [insincere_bigram_model[question] for question in insincere_prepro_questions]
insincere_bigrams[:2]


# In[ ]:


# Creation of a dictionary with the words and their id for sincere bigrams
sincere_bigrams_dictionary = gensim.corpora.Dictionary(sincere_bigrams)
sincere_bigrams_token = sincere_bigrams_dictionary.token2id


# In[ ]:


# Association of the dictonary words and their frequency for sincere bigrams
sincere_bigrams_frequency = {sincere_bigrams_dictionary[k]: 
                             v for k,v in sincere_bigrams_dictionary.dfs.items()}


# In[ ]:


# Creation of a dictionary with the words and their id for insincere bigrams
insincere_bigrams_dictionary = gensim.corpora.Dictionary(insincere_bigrams)
insincere_bigrams_token = insincere_bigrams_dictionary.token2id


# In[ ]:


# Association of the dictonary words and their frequency for insincere bigrams
insincere_bigrams_frequency = {insincere_bigrams_dictionary[k]: 
                               v for k,v in insincere_bigrams_dictionary.dfs.items()}


# ### III-B.3. Vectorization

# In[ ]:


sincere_bigrams_corpus = [sincere_bigrams_dictionary.doc2bow(question) 
                  for question in clear_sincere_questions]
sincere_bigrams_corpus[:2]


# In[ ]:


insincere_bigrams_corpus = [insincere_bigrams_dictionary.doc2bow(question) 
                  for question in clear_insincere_questions]
insincere_bigrams_corpus[:2]


#  ### III-B.4. Lda Modeling

# In[ ]:


# Creating a lda model, which create 10 topics from the sincere bigrams
lda_model_sincere_bigrams = gensim.models.ldamodel.LdaModel(
    corpus=sincere_bigrams_corpus, num_topics=10, id2word=sincere_bigrams_dictionary,
    random_state=25, passes=5)


# In[ ]:


# Creating a lda model, which create 10 topics from the insincere bigrams
import warnings
warnings.filterwarnings('ignore')

lda_model_insincere_bigrams = gensim.models.ldamodel.LdaModel(
    corpus=insincere_bigrams_corpus, num_topics=10, id2word=insincere_bigrams_dictionary,
    random_state=25, passes=5)


# In[ ]:


pprint(lda_model_sincere_bigrams.print_topics()[0])


# In[ ]:


pprint(lda_model_insincere_bigrams.print_topics()[0])


# ### III-B.5. Visualization

# In[ ]:


pyLDAvis.gensim.prepare(lda_model_sincere_bigrams, sincere_bigrams_corpus, 
                        sincere_bigrams_dictionary)


# Even if the graph shows unique word, the circles take into account the bigrams. 
# 
# The repartition of the topics is quite different and some emerge more clearly. 
# 
# The first topic, which is one of the 2 biggest, seems to be about education and job. 
# 
# The 2nd one, which is also quite big, gather word indicating interrogation. It seems to be less a topic and more markers of real question (opposed to affirmation disguise in question)
# 
# The others topics are more mixed and once again some of them overlap each other, indicating that 10 topics is probably too much. Some secondary subjects appears but they're not as recognizable as insincere questions. 
# 

# In[ ]:


pyLDAvis.gensim.prepare(lda_model_insincere_bigrams, insincere_bigrams_corpus, 
                        insincere_bigrams_dictionary)


# The result are quite similar to the first one. Even if some subjects are analogous we can recognize most of the topics by reading the 
# most relevant words. 
# * Topics 1, 4 and 8 are about politics.
# *  Topics 2, 3 and  6, are about race and religions.
# * Topics 5, 7 and 9 are sex questions
# *  Topic 10 is more contrasted and we couldn't recognize a clear subject, it seems to gather different hot social subjects (gender, guns, criminals)

# _**The clear separation in 4 group of subject let think we should try a lda with 4 topics. **_

# ## III-C. Recreate new Lda with 4 topics

# We already have preprocessed data, wo we will just recreate the Lda in itself. 

# ### III-C.1. Creating lda model

# In[ ]:


# Creating a lda model, which create 4 topics from the sincere questions
import warnings
warnings.filterwarnings('ignore')
lda_model_sincere_4 = gensim.models.ldamodel.LdaModel(
    corpus=sincere_corpus, num_topics=4, id2word=sincere_questions_dictionary,
    random_state=25, passes=5)


# In[ ]:


# Creating a lda model, which create 4 topics from the insincere questions
lda_model_insincere_4 = gensim.models.ldamodel.LdaModel(
    corpus=insincere_corpus, num_topics=4, id2word=insincere_questions_dictionary,
    random_state=25, passes=5)


# In[ ]:


# Printing the result of the modeling for sincere questions.
pprint(lda_model_sincere_4.print_topics()[0])


# In[ ]:


# Printing the result of the modeling for insincere questions.
pprint(lda_model_insincere_4.print_topics()[0])


# ### III-C.2. Visualization

# In[ ]:


# Display the topic modeling for sincere questions.
pyLDAvis.gensim.prepare(lda_model_sincere_4, sincere_corpus, sincere_questions_dictionary)


# The 4 topics created by the lda model are very distinct, this is better than the test with 10 topics. 
# We still can't differenciate clearly the subjects of the topics. Some words come back a lot "best", "guess", "world", "people".
# This tend to show that the sincere question have various subject and there is no common theme around them. 

# In[ ]:


# # Display the topic modeling for insincere questions.
pyLDAvis.gensim.prepare(lda_model_insincere_4, insincere_corpus, insincere_questions_dictionary)


# We could see 4 distincts topics: 
# - topic 1 is about politic with top-3 keyword being "Trump", "liberals", "presidents". It's also the bigger topic;
# - topic 2 is about race and nationality with keywords "white", "black", "people";
# - topic 3 is about gender and sex, "women", "men", "gay";
# - topic 4 is about religion it seems, with keywords "christians", "muslims".
# 
# This topics could be use cause they rejoign the subjects given by Quora as sensible. This topic could be used as a variable 
# to characteristic insincere question and be used for machine learning. 

# ---

# # IV. Quick machine learning model

# This kernel is mostly center around the topic modeling but let's try a quick and simple model just to see how it works. This model doesn't use the topic modeling, which maybe could help to improve the final score. 

# In[ ]:


# I import all the functions from sklearn.
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# Train_test_split is not the most precise method of validation but I just want to quickly test the model as a first try. 

# In[ ]:


# I split the date into train and test, variable and target. 
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train)
X_train.shape, y_train.shape, X_test.shape, y_test.shape


# Now i have my training and testing sample for train my model. I will do a pipeline with a transformer, to turn the text value to vectors and an estimator, to predict. 

# In[ ]:


# I create of the transformer and the estimator and insertion in a pipeline.
tfidv = TfidfVectorizer(lowercase=True, stop_words='english')
multinomialnb = MultinomialNB()
pipe = make_pipeline(tfidv, multinomialnb)
pipe


# In[ ]:


# I fit of the pipeline on X_train and y_train,
# then predict on the y_test and estimate of the predictions. 
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred))


# **As expected for a naive model and without any ponderation on a disbalance datasets, the F1 score is quite good for 0 and really bad for 1.**
# 
# **Enginere featuring and ponderation could let us have a better score. **

# In[ ]:




