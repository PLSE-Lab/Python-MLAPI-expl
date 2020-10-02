#!/usr/bin/env python
# coding: utf-8

# # Amazon Fine Foods Review - An EDA and Modelling 

# The dataset provided to us presents the review listing of fine foods from Amazon from Oct 1999 to Oct 2012 (13 Years) . From the overview of the dataset we understand that  there are 568,454 reviews given by 256,059 users on 74,258 products with 260 users given more than  50 reviews.The dataset presents a great opportunity to explore in various dimensions like - users who always provide positive reviews , negative reviews , topic modelling , sentiment analysis etc . This kernel is an attempt in those lines .

# ### Import necessary libraries 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import warnings 
import string
import re
import itertools
from bs4 import BeautifulSoup
from collections import Counter
from wordcloud import WordCloud
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

## Modelling :
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,auc,roc_curve,confusion_matrix,make_scorer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import stopwords
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer


# ### Reading the dataset

# In[ ]:


### Read the dataset:
Kaggle=1
if Kaggle==0:
    review=pd.read_csv("Reviews.csv",parse_dates=["Time"])
else:
    review=pd.read_csv("../input/Reviews.csv",parse_dates=["Time"])


# ### Overview of the dataset

# In[ ]:


review.head()


# The explanations for the variables are as follows:
# 
# * IdRow - Unique Identifier for rows
# * ProductId - Unique identifier for the product
# * UserId - Unqiue identifier for the user
# * ProfileName - Profile name of the user
# * HelpfulnessNumerator - Number of users who found the review helpful
# * HelpfulnessDenominator - Number of users who indicated whether they found the review helpful or not
# * Score - Rating between 1 and 5
# * Time - Timestamp for the review
# * Summary - Brief summary of the review
# * Text - Text of the review

# In[ ]:


review.describe()


# From the summary statistics, we see that on an average 2 people found the review helpful and the average rating of the food has been 4.18.

# ### Some EDA :

# In[ ]:


print("There are {} unique product IDs and there are {} uniques users who have submitted their reviews.".format(review['ProductId'].nunique(),review['UserId'].nunique()))


# Lets check the rating distribution . 

# In[ ]:


plt.figure(figsize=(8,8))
ax=sns.countplot(review['Score'],color='skyblue')
ax.set_xlabel("Score")
ax.set_ylabel('Count')
ax.set_title("Distribution of Review Score")


# The number of reviews for rating 5 is on the higher side compared to other scores . Lets check if the rating is correlated with the helpfulness .

# In[ ]:


## Borrowed from https://www.kaggle.com/neilash/team-ndl-algorithms-and-illnesses

plt.scatter(review.Score, review.HelpfulnessDenominator, c=review.Score.values, cmap='tab10')
plt.title('Useful Count vs Rating')
plt.xlabel('Rating')
plt.ylabel('Useful Count')
plt.xticks([i for i in range(1,6)]);


# In[ ]:


### Borrowed from https://www.kaggle.com/neilash/team-ndl-algorithms-and-illnesses

# Create a list (cast into an array) containing the average usefulness for given ratings
use_ls = []

for i in range(1, 6):
    use_ls.append([i, np.sum(review[review.Score == i].HelpfulnessDenominator) / np.sum([review.Score == i])])
    
use_arr = np.asarray(use_ls)


# In[ ]:


plt.scatter(use_arr[:, 0], use_arr[:, 1], c=use_arr[:, 0], cmap='tab10', s=200)
plt.title('Average Useful Count vs Rating')
plt.xlabel('Rating')
plt.ylabel('Average Useful Count')
plt.xticks([i for i in range(1, 6)]);


# It is interesting to note that there are many users who found review score 1 more helpful . I think that such reviews might have helped them to avoid a particular food.Lets check few samples of such reviews.

# In[ ]:


useful_rating=review.sort_values('HelpfulnessDenominator',ascending=False)


# In[ ]:


# Print most helpful reviews:
for i in useful_rating.Text.iloc[:3]:
    print(i,'\n')


# In[ ]:


## Print least helpful reviews :
for i in useful_rating.Text.iloc[-3:]:
    print(i,'\n')


# Clearly , the most helpful reviews is on the negative side which the reviewers have provided whereas the least helpful is on the neutal side -atleast from the samples part.

# Lets check top 5 persons whose reviews people have found most reviews.

# In[ ]:


useful=review.groupby('ProfileName')['HelpfulnessDenominator'].mean().reset_index().sort_values('HelpfulnessDenominator',ascending=False)


# In[ ]:


useful.head()


# In[ ]:


plt.figure(figsize=(8,8))
ax=sns.barplot(x='ProfileName',y='HelpfulnessDenominator',data=useful[:5],palette=sns.color_palette(palette="viridis_r"))
ax.set_title("Average Usefulness Rating by Profile-Top 5 Users")
ax.set_xlabel("Profile Name")
ax.set_ylabel("Average People")
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)


# On an average 923 people have found BabbChuck "BabbChuck" review helpful and 878 people have found reviews by P. Schmidt review's helpful.

# Lets check out who has consistently provided most positive scores .

# In[ ]:


scores=review.groupby('ProfileName')['Score'].mean().reset_index().sort_values(by='Score',ascending=False)
scores.head()


# In[ ]:


plt.figure(figsize=(8,8))
ax=sns.barplot(x='ProfileName',y='Score',data=scores[:10],palette=sns.color_palette(palette="viridis_r"))
ax.set_title("Users with most positive scores-Top 10 Users")
ax.set_xlabel("Profile Name")
ax.set_ylabel("Average Score")
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)


# The plot tells us that on an average most of the users have left positive reviews in the blog . 
# 
# Lets see who has written most lengthy reviews on an average.

# In[ ]:


review['review_length']=review['Text'].str.len()


# In[ ]:


length=review.groupby('ProfileName')['review_length'].mean().reset_index().sort_values(by='review_length',ascending=False)
length.head()


# In[ ]:


plt.figure(figsize=(8,8))
ax=sns.barplot(x='ProfileName',y='review_length',data=length[:10],palette=sns.color_palette(palette="viridis_r"))
ax.set_title("Average Length of the review-Top 10 Users")
ax.set_xlabel("Profile Name")
ax.set_ylabel("Average Length")
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)


# ### Text Classification:

# We have our scores for each of the reviews along with the review text.A model can be developed to predict the scores given the review . We first split the data to train and test .(80-20 ratio) .Considering huge size of the dataset , we sample 2L of the dataset and model the data.

# In[ ]:


review_sample=review.sample(10000)


# In[ ]:


y=review_sample['Score']
x=review_sample['Text']
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100,stratify=y)


# In[ ]:


print('Dimensions of train:{}'.format(X_train.shape),'\n','Dimensions of test:{}'.format(X_test.shape))


# In[ ]:


y_train.value_counts()


# In[ ]:


y_test.value_counts()


# We see that the scores are skewed towards the high scores therefore we can see its an imbalanced dataset.

# In[ ]:


y_train.describe()


# In[ ]:


### Some preprocessing exercise in train dataset: - Inspired  from https://www.kaggle.com/c/word2vec-nlp-tutorial#part-1-for-beginners-bag-of-words
def review_to_words(raw_review):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw review), and 
    # the output is a single string (a preprocessed review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    text_words = letters_only.lower()                          
    #              
    # 
    #4.Remove stopwords and Tokenize the text
    tokens = nltk.word_tokenize(text_words)
    tokens_text = [word for word in tokens if word not in set(nltk.corpus.stopwords.words('english'))]
    #
    #5.Lemmantize using wordnetLemmantiser:
    lemmantizer=WordNetLemmatizer()
    lemma_text = [lemmantizer.lemmatize(tokens) for tokens in tokens_text]
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( lemma_text ))   


# Thus we have remove the HTML tags using BeautifulSoup , removed the punctuations , converted the text to lower case ,tokenized the text and used wordnet Lemmatizer after which the review text looks clean .Lets apply them all over the train data .

# In[ ]:


# Get the number of reviews based on the dataframe column size


# Initialize an empty list to hold the clean reviews
X_train_clean = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list 
for text in tqdm(X_train):
    # Call our function for each one, and add the result to the list of
    # clean reviewsb
    X_train_clean.append(review_to_words(text))


# For modelling we use both Countvectorizer and TF-IDF models and compare the accuracy between the two.

# In[ ]:


#https://www.kaggle.com/nilanml/imdb-review-deep-model-94-89-accuracy
##Creating bag of words model :

vectorizer=CountVectorizer(ngram_range=(1,1)) 
train_feature=vectorizer.fit_transform(X_train_clean)

tfidf_transformer=TfidfVectorizer(ngram_range=(1,1))
train_feature_tfidf=tfidf_transformer.fit_transform(X_train_clean)


# In[ ]:


# Get the number of reviews based on the dataframe column size


# Initialize an empty list to hold the clean reviews
X_test_clean = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list 
for text in tqdm(X_test):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    X_test_clean.append(review_to_words(text))


# In[ ]:


test_feature=vectorizer.transform(X_test_clean)
test_feature_tfidf=tfidf_transformer.transform(X_test_clean)


# In[ ]:


prediction=dict()


# ### Naive Bayes Model

# In[ ]:


nb=MultinomialNB()
nb.fit(train_feature, y_train)


# In[ ]:


prediction['Naive']=nb.predict(test_feature)


# In[ ]:


print(accuracy_score(y_test,prediction['Naive']))


# The accuracy of the model is 66 % .Lets plot the confusion matrix.

# In[ ]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:


class_names = set(review['Score'])
cnf_matrix = confusion_matrix(y_test, prediction['Naive'])
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, Naive Bayes Model')


# From the confusion matrix it is seen that reviews having scores 1,2,3,4 are also predicted as 5 . This means that the model is not good enough to differentiate between the scores with the review words.

# Lets check the prediction with TF-IDF model.

# In[ ]:


nb.fit(train_feature_tfidf,y_train)
prediction['Naive_TFIDF']=nb.predict(test_feature_tfidf)


# In[ ]:


print(accuracy_score(y_test,prediction['Naive_TFIDF']))


# The accuracy of the model has not improved.Infact,it has decreased.

# ### Logistic Regression Model

# In[ ]:


lr=LogisticRegression()


# In[ ]:


lr.fit(train_feature,y_train)


# In[ ]:


prediction['Logit']=lr.predict(test_feature)


# In[ ]:


print(accuracy_score(y_test,prediction['Logit']))


# In[ ]:


lr.fit(train_feature_tfidf,y_train)


# In[ ]:


prediction['Logit_TFIDF']=lr.predict(test_feature_tfidf)


# In[ ]:


print(accuracy_score(y_test,prediction['Logit_TFIDF']))


# Both the accuracy of the model with logistic is around 67 % .

# ### Grid Search:

# The model has been trained on an imbalanced data .Therefore lets apply oversampling methods to the dataset and retrain the model for improving the score . Also ,Lets try to improve the model with grid search.Lets create a scoring criteria first .We define accuracy score as the metric.

# In[ ]:


sm=SMOTE(random_state=100)


# In[ ]:


X_sm,y_sm=sm.fit_sample(train_feature_tfidf,y_train)


# In[ ]:


Counter(y_sm)


# Thus by using SMOTE we have oversampled the dataset to match the scores with the top score.

# In[ ]:


scorer=make_scorer(accuracy_score)


# Lets initiate a grid search with 5 fold cross validation.

# In[ ]:


# parameter grid
naive=MultinomialNB()
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

# Initialize Grid Search Model
model = GridSearchCV(estimator=naive, param_grid=param_grid, scoring=scorer,
                                 verbose=10, n_jobs=-1, iid=True, refit=True, cv=5)


# In[ ]:


# Fit Grid Search Model
model.fit(X_sm, y_sm)  # Using the TF-IDF model for training .
print("Best score: %0.3f" % model.best_score_)
print("Best parameters set:")
best_parameters = model.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))


# Thus by using SMOTE and cross validation we have improved the accuracy to 93 %.But since this is an imbalanced dataset , it will be wise to use metrics like confusion matrix,f1-score,precision-recall instead of accuracy score since this will be misleading. Lets apply the model to the test dataset and check the confusion matrix.

# In[ ]:


prediction['Naive_SMOTE']=model.predict(test_feature_tfidf)


# In[ ]:


class_names = set(review['Score'])
cnf_matrix = confusion_matrix(y_test, prediction['Naive_SMOTE'])
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, Naive Bayes Model(After SMOTE and Grid Search)')


# **Work in progress**
