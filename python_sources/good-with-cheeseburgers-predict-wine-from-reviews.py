#!/usr/bin/env python
# coding: utf-8

# # Can we predict which variety of wine a review is describing?
# 
# Zack posted a cool dataset of 150k reviews from winemag.com, proposing we try to predict a wine's variety using the words in the review text. Given my sister says she'd make up words when selling wine, I thought it'd be fun to try out my first document classifier (and kaggle!) to see if there are any real patterns behind wine descriptions.
# 
# In this notebook I try out the following:
# 
#  - A quick look & clean of the data
#  - Creating term frequency inverse document frequency (Tfidf) vectors of the review descriptions
#  - Running Logistic Regression Classifier on our bag-of-words vectors
#  - Hyperparameter tuning with grid search
#  - Lastly, a check for any interesting topics with Latent Dirichlet Allocation 
# 
# 
# I was a little surprised by some of the results, feedback is most welcome!

# In[ ]:


import pandas as pd
df = pd.read_csv('./../input/winemag-data_first150k.csv')
df.head()


# In[ ]:


df.drop(df.columns[[0]], axis=1, inplace=True) # ditch that unnamed row numbers column
df.describe(include='all')


# So, we care about description and variety. Hm, description... uniques and counts don't add up. if there are ~90k unique descriptions and 150k total, we have some duplicates. Let's take a look at one just to be sure they're actually just duplicates:

# In[ ]:


dups = df[df.duplicated('description')]
dups.sort_values('description', ascending=False).iloc[3:5]


# Well good. I was a little worried reviewers were re-using their descriptions on different wines.... Let's drop those dups and checkout variety

# In[ ]:


dedupped_df = df.drop_duplicates(subset='description')
print('Total unique reviews:', len(dedupped_df))
print('\nVariety description \n', dedupped_df['variety'].describe())


# Variety will be our class label, and 632 labels seems like a lot for <100k documents. Well maybe they're evenly balanced classes?!

# In[ ]:


varieties = dedupped_df['variety'].value_counts()
varieties


# In[ ]:


varieties.describe()


# No cigar. You'd be right 10% of the time if your model just labeled everything "Pinot Noir", and the majority of labels have less than 5 reviews each. On the bright side, unbalanced labels seems to be the norm in many datasets, and the distribution on the top wines isn't as bad as it could be.
# 
# So what to do?
# 
# Sometime I'd love to try the "shove in similar, computer generated fake data" technique to balance the labels, but for now I'm going to try something quicker. Let's just look at the top 20 varieties. That may not be great for a production classifier of every winemag.com review in history, but I kinda doubt my sister's restaurant carried over 600 varieties of wine (ever order a "Teroldego Rotaliano"?)

# In[ ]:


top_wines_df = dedupped_df.loc[dedupped_df['variety'].isin(varieties.axes[0][:20])]
top_wines_df['variety'].describe()


# Now we've got ~70k reviews, 20 labels, and an unbalanced but much more manageable distribution of labels across our documents. Let's make some vectors & do some classifying!

# In[ ]:


# our labels, as numbers. 

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(top_wines_df['variety'])


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
vect = TfidfVectorizer()
x = vect.fit_transform(top_wines_df['description'])
x


# Now we've got a sparse matrix of our ~70k docs with ~26k columns. Let's have a go at learning, using a basic train test split, accuracy_score, and a Logistic Regression classifier.

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=10)


# In[ ]:


import time
start = time.time()
clf = LogisticRegression()
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
print(time.time() - start)


# In[ ]:


accuracy_score(y_test, pred)


# ~72% accuracy? Not bad! However, probably a good call to be skeptical and think through why this might be a little too optimistic. In terms of model performance, I've read that using a holdout method with train_test_split isn't as trusty as using a cross validation assessment. 
# 
# But in terms of our subject material, is there anything obvious I missed? One thing that came to mind was the names of the varieties: what if they're in the actual review? Our classifier would probably be pretty pleased with that set of features, and while it would be a reality of the data and therefore "fair game", it feels a bit like cheating the spirit of my original objective (... imagine, reviewer: "This lovely pinot noir tastes like berries!" classifier: "it's pinot noir!" reviewer: "you're so smart!" )
# 
# Let's try again, with any variety names removed:

# In[ ]:


wine_stop_words = []
for variety in top_wines_df['variety'].unique():
    for word in variety.split(' '):
        wine_stop_words.append(word.lower())
wine_stop_words = pd.Series(data=wine_stop_words).unique()
wine_stop_words


# In[ ]:


vect2 = TfidfVectorizer(stop_words=list(wine_stop_words))
x2 = vect2.fit_transform(top_wines_df['description'])
x2


# Double checking that I understand things here: before removing our variety name-like list, our matrix had 26587 columns. Now it's at 26565, and the difference is 22, the length of our stop words. Lovely. That means our reviews did include every one of those words. 
# 
# Note that it's likely a better move to check if each variety name mentioned in reviews is always equal with the review's actual label classifier, but I'll save that one for another day. Let's classify our new stuff:

# In[ ]:


x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y, test_size=0.25, random_state=10)
clf = LogisticRegression()
clf.fit(x_train2, y_train2)
pred = clf.predict(x_test2)
accuracy_score(y_test, pred)


# Huzzah. We've dropped our accuracy nearly 10%, meaning we were right about the importance of those variety presences in the reviews. 
# 
# Let's see how much we can get back up with some hyperparameter tuning. 

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

x_train3, x_test3, y_train3, y_test3 = train_test_split(
    top_wines_df['description'].values, y, test_size=0.25, random_state=10)

pipe = Pipeline([
    ('vect', TfidfVectorizer(stop_words=list(wine_stop_words))), 
    ('clf', LogisticRegression(random_state=0))
])

param_grid = [
  {
    'vect__ngram_range': [(1, 2)],
    'clf__penalty': ['l1', 'l2'],
    'clf__C': [1.0, 10.0, 100.0]
  },
  {
    'vect__ngram_range': [(1, 2)],
    'vect__use_idf':[False],
    'clf__penalty': ['l1', 'l2'],
    'clf__C': [1.0, 10.0, 100.0]
    },
]

grid = GridSearchCV(pipe, param_grid, scoring='accuracy', cv=3, verbose=1, n_jobs=-1)


# We did another train test split, as this time we'll be feeding the raw reviews as our "X" into our pipeline. I'm looking to see if IDF is helping or hurting, and what happens when we up our "grams" (number of words per column) from 1 to 2. With unigram we see "smokey" and "finish" as unrelated, but bigram we'll look for the whole phrase "smokey finish" in reviews.
# 
# We're also checking if L1 or L2 penalties could help us ignore some unnecessary word columns. Let's try this out!

#     
# 
#     grid.fit(x_train3, y_train3)
# 
#     Fitting 5 folds for each of 12 candidates, totalling 60 fits
#     [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed: 18.8min
#     [Parallel(n_jobs=-1)]: Done  60 out of  60 | elapsed: 32.9min finished
#     
#     print(grid.best_params_, grid.best_score_)
#     {'clf__C': 10.0, 'clf__penalty': 'l2', 'vect__ngram_range': (1, 2)} 0.63607423082
# 
#     grid.best_estimator_.score(x_test3, y_test3)
#     0.6463681815628336
# 
#  

# (Copied & pasted in above the results from my home computer)
# 
# Alright, we brought the score back up a bit! Usint tfidf was in the best params (assuming they don't show the defaults, or it could be because I didn't list it explicitly), L2 penalty may have helped and putting bigrams into play also probably helped. ~65% accuracy isn't half bad, and it could probably be brought up more were we to dig into those hyperparams.
# 

# Lastly, let's see if there are any interesting topics across reviews from Latent Dirichlet Allocation decomposition. For this one I'm going to replace our previous stop list of wine varieties and instead use sklearn's basic english set, so our topics aren't mucked up. 

# In[ ]:


from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction import text
stop_words = text.ENGLISH_STOP_WORDS.union(wine_stop_words)
vect = TfidfVectorizer(stop_words=stop_words)
x = vect.fit_transform(top_wines_df['description'])
lda = LDA(learning_method='batch')
topics = lda.fit_transform(x)


# There doesn't seem to be a great way to print topics built in, so we'll borrow this [handy print function][1] found in Muller & Guido's intro to ML with Python:
# 
# 
#   [1]: https://github.com/amueller/mglearn/blob/master/mglearn/tools.py

# In[ ]:


import numpy as np
def print_topics(topics, feature_names, sorting, topics_per_chunk=5,
                 n_words=10):
    for i in range(0, len(topics), topics_per_chunk):
        # for each chunk:
        these_topics = topics[i: i + topics_per_chunk]
        # maybe we have less than topics_per_chunk left
        len_this_chunk = len(these_topics)
        # print topic headers
        print(("topic {!s:<8}" * len_this_chunk).format(*these_topics))
        print(("-------- {0:<5}" * len_this_chunk).format(""))
        # print top n_words frequent words
        for i in range(n_words):
            try:
                print(("{!s:<14}" * len_this_chunk).format(
                    *feature_names[sorting[these_topics, i]]))
            except:
                pass
        print("\n")
        

sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
feature_names = np.array(vect.get_feature_names())
        
print_topics(topics=range(10), feature_names=feature_names, sorting=sorting)


# Nothing crazy jumps out at me, but thinking about topic 0: "sweet, peach, & honey", maybe these would be white wines or roses? Topic 1 & 4 I could see being topics about reds: "wood, spice, firm, tannins, black..." 
# 
# I would love to add a topics column next to each document for the topic the document is most likely a member of, no doubt some interesting patterns could up. I couldn't find any great examples of how to do that online, so if anyone knows of one please let me know!
# 
# One last interesting thing: when I first ran LDA topics on the original .json file of reviews, "cheeseburger" popped up as topic term...

# In[ ]:


top_wines_df[top_wines_df['description'].str.contains('cheeseburger')].values


# In[ ]:


dedupped_df[dedupped_df['description'].str.contains('cheeseburger')]['points'].describe()


# In[ ]:


dedupped_df['points'].describe()


# Some 50 reviews use the word **cheeseburger** ("it will happily wash down simple fare, like cheeseburgers"), and on average their review rating points are ~84, about a full standard deviation lower than the overall average. 
# 
# I guess bad wine goes with good cheeseburgers?
