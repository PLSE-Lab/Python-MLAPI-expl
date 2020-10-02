#!/usr/bin/env python
# coding: utf-8

# ![](https://image-cdn.essentiallysports.com/wp-content/uploads/20200508154130/Animal-Crossing-New-Horizons.jpeg)

# # Introduction

# Animal Crossing is a social simulation video game developed and published by Nintendo. In Animal Crossing, the player character is a human who lives in a village inhabited by various anthropomorphic animals. The series is notable for its open-ended gameplay and extensive use of the video game console's internal clock and calendar to simulate real passage of time.
# 
# In the game, the player can carry out activities such as fishing, bug catching, mining etc. The game is much of a real-life simulator, players can build house, design clothes and even sell/buy turnip (mock stock market) to make money. It is undoubly one of the most popular game these days. Some people find it very obsessive, but others find it stressful and time-consuming (like me, lol). Here, we aim to analyze the reviews of the users, and build a NLP pipeline to understand what is the key reasons people like/dislike animal crossing.  
# 
# If you find this notebook interesting, please UPVOTE!

# ## ETL

# In[ ]:


# load packages and data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import lightgbm as lgb
from sklearn.pipeline import Pipeline, FeatureUnion
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, auc
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('darkgrid')


# In[ ]:


df = pd.read_csv('../input/animal-crossing/user_reviews.csv')


# In[ ]:


df.head()


# In[ ]:


# check null values
df.info()


# In[ ]:


# check the distribution of grade
plt.figure(figsize=(8,6))
plt.bar(df.grade.value_counts().index, df.grade.value_counts().values)
plt.xlabel('Review Grade')
plt.ylabel('Count')
plt.title('Distribution of Review Grade');


# We can see that the grade for animal crossing from users are bimodal. Many users gave fairly low score (0, 1) and fairly high score (9, 10). Relatively fewer users gave average score like 5, 6.
# 
# Since the review scores are bimodally disgtributed, I will group users with 0-4 as lower (0), and users with 5-10 as high (1) for later analysis. 

# In[ ]:


df['target'] = pd.cut(df.grade, 2, labels=[0, 1])
df.target.value_counts()


# ## Popular Words

# ### Word Counter

# In[ ]:


def tokenize(text):
    """Tokenize each review text
    Args: text
    Return: token lists after normalization and lemmatization
    """
    # remove punctuation and change to lowercase
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text).lower()
    # tokenize the word into words
    tokens = word_tokenize(text)

    # remove stopwords
    stop_words  = set(stopwords.words('english'))
    
    tokens = [word for word in tokens if word not in stop_words]

    # lemmatize the word
    lemmatizer = WordNetLemmatizer()
    clean_token = []
    for token in tokens:
        clean_token.append(lemmatizer.lemmatize(token, pos='v').lower().strip())
    return clean_token


# In[ ]:


# concatenate all comments in low grade group 
dict_low = df[df.target == 0].to_dict(orient='list')
low_grade = dict_low['text']

dict_high = df[df.target == 1].to_dict(orient='list')
high_grade = dict_high['text']


# In[ ]:


# tokenize text for Counter
low_grade = ' '.join(low_grade)
low_tokens = tokenize(low_grade)

high_grade = ' '.join(high_grade)
high_tokens = tokenize(high_grade)


# In[ ]:


# Count the most popular words 
low_counter = Counter(low_tokens)
low_top20 = low_counter.most_common(20)

high_counter = Counter(high_tokens)
high_top20 = high_counter.most_common(20)


# In[ ]:


plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
plt.barh(range(len(low_top20)), [val[1] for val in low_top20], align='center')
plt.yticks(range(len(low_top20)), [val[0] for val in low_top20])
plt.xlabel('Count')
plt.ylabel('Top 20 Common Words')
plt.title('Most 20 common words in low grading group')

plt.subplot(1, 2, 2)
plt.barh(range(len(high_top20)), [val[1] for val in high_top20], align='center')
plt.yticks(range(len(high_top20)), [val[0] for val in high_top20])
plt.xlabel('Count')
plt.ylabel('Top 20 Common Words')
plt.title('Most 20 common words in high grading group');


# There is a lot overlapping for the most common words. Some infomation we can extract is that in the high score group, users put **fun**, word **great**, **really** and **10** more often in their reviews.
# 
# One thing maybe interesting to see is to look at the most common adjective to see how people decribe their exprience. 

# In[ ]:


# use pos_tag in NLP to select adjectives
ad_tokens_low = []
for word, tag in pos_tag(low_tokens):
    if tag in ('JJ', 'JJR', 'JJS'):
        ad_tokens_low.append(word)

ad_tokens_high = []
for word, tag in pos_tag(high_tokens):
    if tag in ('JJ', 'JJR', 'JJS'):
        ad_tokens_high.append(word)


# In[ ]:


# Count the most popular adjective/adverb 
ad_low_counter = Counter(ad_tokens_low)
ad_low_top20 = ad_low_counter.most_common(20)

ad_high_counter = Counter(ad_tokens_high)
ad_high_top20 = ad_high_counter.most_common(20)


# In[ ]:


plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
plt.barh(range(len(ad_low_top20)), [val[1] for val in ad_low_top20], align='center')
plt.yticks(range(len(ad_low_top20)), [val[0] for val in ad_low_top20])
plt.xlabel('Count')
plt.ylabel('Top 20 Common Adjective')
plt.title('Most 20 common adjective in low grading group')

plt.subplot(1, 2, 2)
plt.barh(range(len(ad_high_top20)), [val[1] for val in ad_high_top20], align='center')
plt.yticks(range(len(ad_high_top20)), [val[0] for val in ad_high_top20])
plt.xlabel('Count')
plt.ylabel('Top 20 Common adjective')
plt.title('Most 20 common adjective in high grading group');


# The user gave bad grade to the game generally think it's **bad**, **terrible**, **ridiculous**, while the people like the game think it's **amazing**, **fantastic**, **great**, **perfect**. Also, we can see that in generally people feel bad about the game describe it more objectively(?).

# ### WordCloud

# In[ ]:


# setup stop words
stop_words  = set(stopwords.words('english'))


# In[ ]:


# update stop words
stop_words.update(['this', 'game', 'the', 'play'])


# In[ ]:


# generate word cloud for low score group
wordcloud = WordCloud(background_color='white', max_words=1000, contour_width=3,contour_color='firebrick', 
                      stopwords = stop_words)

wordcloud.generate(re.sub(r'[^a-zA-Z0-9]', ' ', low_grade).lower())
plt.figure(figsize=(12,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:


# generate word cloud for high score group
wordcloud = WordCloud(background_color='white', max_words=1000, contour_width=3,contour_color='firebrick', 
                      stopwords = stop_words)

wordcloud.generate(re.sub(r'[^a-zA-Z0-9]', ' ', high_grade).lower())
plt.figure(figsize=(12,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# Again, the low grade group gave more description about the game, with some info may refer why they didn't like the game - per console? per switch? may indicate they want to play the game with multiple person. The high grade group have more positive words in general, like great, new, the best, good etc.

# ## Model Build (NLP pipeline)

# In[ ]:


class AdCount(BaseEstimator, TransformerMixin):
    """ Custom transformer to count the number of adj and adv in text
    Args: text
    Return: Adjective and Adverb counts in the text
    """
    def Ad_count(self, text):
        text = re.sub(r'[^a-zA-Z0-9]', ' ', text).lower()
        # tokenize the word into words
        tokens = word_tokenize(text)
        # remove stopwords
        stop_words  = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        count = 0
        for word, tag in pos_tag(tokens):
            if tag in ('RB', 'RBR', 'RBS', 'JJ', 'JJR', 'JJS'):
                count+=1
        return count

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        counts = pd.Series(X).apply(self.Ad_count)
        return pd.DataFrame(counts)


# In[ ]:


def build_model():
    """
    A Adaboost classifier machine learning pipeline for
    natural language processing with tdidf, adcount, and gridsearch for optimization.
    Args: X_train, y_train 
    Returns:
        Fitted model
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
             ('ad_count', AdCount())
             ])),
        ('clf', lgb.LGBMClassifier(objective='binary', random_state=0))
    ])
    parameters = {
        #'clf__n_estimators': [100],
        'clf__learning_rate': [0.01, 1],
        'clf__num_leaves': [31, 62]
        #'clf__min_samples_split': [5]
        #'clf__estimator__C': [1, 10],
        #'clf__estimator__max_iter': [1000, 100000]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


# In[ ]:


def evaluate_model(cv, X_test, y_test):
    """Draw ROC curve for the model
    Args:
        Classification Model
        X_test, y_test, Array-like
    return: ROC curve and model pickles
    """
    y_pred = cv.predict_proba(X_test)[:,1]
    print('\nBest Parameters:', cv.best_params_)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr) # compute area under the curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % (roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")


# In[ ]:


# split train and test 
X = df.text
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# ### RandomForester

# In[ ]:


# build model (randomforests)
model = build_model()
model.fit(X_train, y_train)


# In[ ]:


# evaluate
pred = model.predict(X_test)
print('Accuracy_score is: {}'.format(accuracy_score(pred, y_test)))
evaluate_model(model, X_test, y_test)


# In[ ]:


f = open('rf'+'.pkl', 'wb')
pickle.dump(model, f)


# ### Adaboost

# In[ ]:


# build model Adaboost
model_ad = build_model()
model_ad.fit(X_train, y_train)


# In[ ]:


# evaluate
pred_ad = model_ad.predict(X_test)
print('Accuracy_score is: {}'.format(accuracy_score(pred_ad, y_test)))
evaluate_model(model_ad, X_test, y_test)


# In[ ]:


f = open('ada'+'.pkl', 'wb')
pickle.dump(model, f)


# ### Light LGB

# In[ ]:


# build model Adaboost
model_lgb = build_model()
model_lgb.fit(X_train, y_train)


# In[ ]:


# evaluate
pred_lgb = model_ad.predict(X_test)
print('Accuracy_score is: {}'.format(accuracy_score(pred_lgb, y_test)))
print('Best parameter is: {}'.format(model_lgb.best_params_))
evaluate_model(model_lgb, X_test, y_test)


# In[ ]:


f = open('lgb'+'.pkl', 'wb')
pickle.dump(model_lgb, f)


# In[ ]:


# fun test
test = ["it is amazing. I'm totally adicted"]
print(model.predict(test))
print(model_ad.predict(test))
print(model_lgb.predict(test))


# In[ ]:


# fun test
ntest = ["This game sucks, make me stressful"]
print(model.predict(ntest))
print(model_ad.predict(ntest))
print(model_lgb.predict(ntest))


# Overall all models perform fairly well. Lgb is slighly faster than randomforest.

# ## Conclusion

# Animal Crossing New Horizon is one of the most 'hit' game these days, but the comments for this game seems varies. Here we analyze some of the reviews and grades put by users. There are some interesting pattern: 1. people don't like this game seems put more objective comments. 2. The wordcloud may indicate that people don't like this game is because limited consoles/villager for one switch. We also tried few techniques to build a model to predict the grade given a comment in the future. We can improve the model by tuning parameters more, or updating the stopwords (for instance nintendo, game, play etc are pretty repetitive. Also, make more custom features would be interesting. 
Here is an image from my own village few weeks ago. LOL
# ![](https://i.ibb.co/qmb9wxs/100785669-1595363370613145-5432053625553682432-o.jpg)

# ![](http://https://scontent-lax3-1.xx.fbcdn.net/v/t1.0-9/100785669_1595363370613145_5432053625553682432_o.jpg?_nc_cat=110&_nc_sid=dd7718&_nc_oc=AQmDREUGTSCH8rjCa6A6ZLs2f5LpmXItY8V65uqXpmQ3XfTJbw7Q6q81PAh-xG8jGJQ&_nc_ht=scontent-lax3-1.xx&oh=c4d7eb45d69ffccdf358fd427b40c4b0&oe=5EF376B5)
