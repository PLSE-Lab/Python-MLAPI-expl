#!/usr/bin/env python
# coding: utf-8

# # Understanding Amazon Alexa Reviews 

# ### Introduction

# The dataset provided gives us information on various variants of amazon Alexa devices and their reviews . This is a great dataset for natural language processing task like sentiment analysis , word2vec, topic modelling etc . Lets explore in those lines and produce some interesting results.

# ### Import necessary libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import string
import itertools

import nltk
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc,accuracy_score
from nltk.stem.porter import PorterStemmer
warnings.filterwarnings("ignore")
#nltk.download('punkt')


# ### Reading the dataset

# In[ ]:


Kaggle=1
if Kaggle==0:
    reviews =pd.read_csv("amazon_alexa.tsv",sep="\t")
    
else:
    reviews = pd.read_csv("../input/amazon_alexa.tsv",sep="\t")
    


# ### Data Exploration

# In[ ]:


reviews.head()


# There are 5 columns:
# 
# 1.Rating - Provides the rating for each of the variants of the amazon alexa device.
# 
# 2.Date - Date when the review , rating was posted.
# 
# 3.Variation - Amazon Alexa variant.
# 
# 4.Verified Reviews - Detailed review of the device.
# 
# 5.Feedback - Numeric number 0 or 1 .I am not sure what this is . Going by the report I guess it should be positive and negative feedback based on the rating.
# 

# In[ ]:


reviews.describe()


# From the summary of the data we see that there are no null values and the data is very clean .The average rating is 4.46 which is close to the maximum value.It is seen that most of the users have left positive reviews and are happy with the device . Lets plot the countplot of the ratings. 

# In[ ]:


plt.figure(figsize=(8,8))
ax=sns.countplot(reviews['rating'],palette=sns.color_palette(palette="viridis_r"))
ax.set_title("Distribution of the Amazon Alexa Rating")
ax.set_xlabel("Rating")
ax.set_ylabel("Count")


# From the countplot it is seen that there are more 5 ratings followed by 4 ratings .Lets check the average ratings by variant.

# In[ ]:


variant_rating=reviews.groupby('variation')['rating'].mean().reset_index()
variant_rating.head()


# In[ ]:


plt.figure(figsize=(8,8))
ax=sns.barplot(x='variation',y='rating',data=variant_rating,palette=sns.color_palette(palette="viridis_r"))
ax.set_title("Average Rating based on Alexa Variant")
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_xlabel("Rating")
ax.set_ylabel("Count")


# The average rating of the alexa devices based on different variant shows that the average rating is more than 4 for each devices .**Charcoal Fabric**,**Walnut Finish** have highest average ratings.

# ### Reviews

# Lets check the top 3 most positive and top 3 most negative reviews.

# In[ ]:


rating_review=reviews.sort_values(by='rating',ascending=False)
rating_review.head()


# In[ ]:


for i in rating_review['verified_reviews'].iloc[:6]:
    print(i, '\n')


# In[ ]:


for i in rating_review['verified_reviews'].iloc[-6:]:
    print(i, '\n')


# In[ ]:


rating_review['date'] = pd.to_datetime(rating_review['date'], errors='coerce')
month_count = rating_review['date'].dt.month.value_counts()
month_count = month_count.sort_index()
plt.figure(figsize=(9,6))
sns.barplot(month_count.index, month_count.values,color='green',alpha=0.4)
plt.xticks(rotation='vertical')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title("Reviews per Month")
plt.show()


# It is seen that the dataset has more reviews from July . Is there any pattern in the number of reviews posted by weekday ?

# In[ ]:


weekday_count = rating_review['date'].dt.weekday_name.value_counts()
weekday_count = weekday_count.sort_index()
plt.figure(figsize=(9,6))
sns.barplot(weekday_count.index, weekday_count.values,color='green',alpha=0.4,order=['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'])
plt.xticks(rotation='vertical')
plt.xlabel('Weekday', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title("Reviews by Weekday")
plt.show()


# The number of reviews written are more on Mondays than any other days of the week .Let us see the if the average rating differs by the day of the week.

# In[ ]:


rating_review['weekday']=rating_review['date'].dt.weekday_name
avg_weekday=rating_review.groupby('weekday')['rating'].mean()
plt.figure(figsize=(9,6))
sns.barplot(avg_weekday.index, avg_weekday.values,color='green',alpha=0.4,order=['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'])
plt.xticks(rotation='vertical')
plt.xlabel('Weekday', fontsize=12)
plt.ylabel('Avg Rating', fontsize=12)
plt.title("Average Rating over the day")
plt.show()


# From the barplot it is understood that there is no significant difference between the ratings over the day.On all days,the rating has been more than 4 .There has been a slight dip in the ratings provided on thursdays and fridays .

# ### Predicting the sentiment of the review :

# We have been provided with the rating score for each reviews . Using this I create a sentiment score - either positive or negative . I am interested to create a model that would predict these sentiment given the review text . I split the dataset into training and test . Given the small number of samples , I split the data 80-20 ratio.Before that I create the sentiment type variable using the score.

# In[ ]:


def sentiment(x):
    if x > 3:
        return 'positive'
    else:
        return 'negative'
        


# In[ ]:


rating = rating_review['rating']
rating=rating.map(sentiment)
review=rating_review['verified_reviews']


# In[ ]:


rating.describe()


# We see that there are 2741 positive reviews .Now lets split the dataset into train and test.

# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(review,rating,test_size=0.2,stratify=rating,random_state=100)


# In[ ]:


print("Shape of train is {} and shape of test is {}".format(X_train.shape,X_test.shape));


# In[ ]:


### Borrowed from https://www.kaggle.com/gpayen/building-a-prediction-model

stemmer = PorterStemmer()

## Stemming :

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

## Tokenisation:

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in nltk.corpus.stopwords.words('english')]
    stems = stem_tokens(tokens, stemmer)
    return ' '.join(stems)

## Removing the punctuation:

intab = string.punctuation
outtab = "                                "
trantab = str.maketrans(intab, outtab)  ### Remove the punctuations 

#--- Training set

corpus = []
for text in X_train:
    text = text.lower()
    text = text.translate(trantab)
    text=tokenize(text)
    corpus.append(text)


# In[ ]:


corpus[:5]


# In[ ]:


count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(corpus)        
        
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


# In[ ]:


#--- Test set

test_set = []
for text in X_test:
    text = text.lower()
    text = text.translate(trantab)
    text=tokenize(text)
    test_set.append(text)

X_new_counts = count_vect.transform(test_set)
X_test_tfidf = tfidf_transformer.transform(X_new_counts)

prediction = dict()


# ### Applying Naive Bayes Model:

# In[ ]:


model =MultinomialNB()
model.fit(X_train_tfidf,y_train)
prediction['Naive Bayes']=model.predict(X_test_tfidf)


# In[ ]:


print(accuracy_score(y_test,prediction['Naive Bayes']))


# The accuracy of the model is 86 % . Lets plot the confusion matrix for naive bayes.

# In[ ]:


df_confusion = pd.crosstab(y_test,prediction['Naive Bayes'], rownames=['Actual'], colnames=['Predicted'], margins=True)
df_confusion


# From the confusion matrix , it is understood that the model is good at predicting positive reviews more strongly than the negative reviews.All the negative reviews are predicted as positive .This is because the dataset is imbalanced with most of the reviews being positive .Sampling methods need to be applied to improve the confusion matrix .

# ### Applying Logistic Regression:

# In[ ]:


model = LogisticRegression()
model.fit(X_train_tfidf,y_train)
prediction['Logit']=model.predict(X_test_tfidf)


# In[ ]:


print(accuracy_score(y_test,prediction['Logit']))


# The accuracy from the logit is slightly higher - 87 %.

# In[ ]:


df_confusion = pd.crosstab(y_test,prediction['Logit'], rownames=['Actual'], colnames=['Predicted'], margins=True)
df_confusion


# The confusion matrix shows the general trend observed in NB method . The model is good at predicting positive reviews whereas no so good in predicting negative reviews.

# ### The metric Trap

# Though the accuracy rates are good ,it will be too early to conclude about the model since this is an unbalanced dataset with most of the reviews being positive . Therefore using the accuracy score as a metric will always give a high value .Therefore another robust metric like confusion matrix or AUC curve will provide a more realistic picture.Lets check the AUC curve for the two methods.

# ### AUC Curve

# In[ ]:


#Borrowed from https://www.kaggle.com/gpayen/building-a-prediction-model
def formatt(x):
    if x == 'negative':
        return 0
    return 1
vfunc = np.vectorize(formatt)

cmp = 0
colors = ['b', 'g', 'y', 'm', 'k']
for model, predicted in prediction.items():
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test.map(formatt), vfunc(predicted))
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.plot(false_positive_rate, true_positive_rate, colors[cmp], label='%s: AUC %0.2f'% (model,roc_auc))
    cmp += 1

plt.title('Classifiers comparaison with ROC')
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# From the AUC curve,it is seen that the AUC of logistic regression is slightly higher .

# ### Dealing with Imbalanced Datasets - SMOTE

# Inorder to impove the model performance ,lets train the model with balanced dataset . For this we use the **imblearn** library and use SMOTE(Synthetic Minority Oversampling) technique .

# In[ ]:


# Inspired from https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets

from imblearn.over_sampling import SMOTE


# In[ ]:


smote=SMOTE(random_state=100)
X_sm,y_sm=smote.fit_sample(X_train_tfidf,y_train)  ### Oversampling the training dataset.


# In[ ]:


## Applying the Naive Bayes and Logit again on the model:
model =MultinomialNB()
model.fit(X_sm,y_sm)
prediction['Naive Bayes_SMOTE']=model.predict(X_test_tfidf)


df_confusion_SMOTE = pd.crosstab(y_test,prediction['Naive Bayes_SMOTE'], rownames=['Actual'], colnames=['Predicted'], margins=True)
df_confusion_SMOTE


# In[ ]:


model = LogisticRegression()
model.fit(X_sm,y_sm)
prediction['Logit_SMOTE']=model.predict(X_test_tfidf)


df_confusion = pd.crosstab(y_test,prediction['Logit_SMOTE'], rownames=['Actual'], colnames=['Predicted'], margins=True)
df_confusion


# Comparing the confusion matrix before and after over-sampling , it is seen that the model prediction has improved after sampling .Lets plot the AUC metric and compare the scores for all the four models.

# ### Model Comparison

# In[ ]:


def formatt(x):
    if x == 'negative':
        return 0
    return 1
vfunc = np.vectorize(formatt)

cmp = 0
colors = ['b', 'g', 'y', 'm', 'k']
for model, predicted in prediction.items():
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test.map(formatt), vfunc(predicted))
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.plot(false_positive_rate, true_positive_rate, colors[cmp], label='%s: AUC %0.2f'% (model,roc_auc))
    cmp += 1

plt.title('Classifiers comparaison with ROC')
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# Thus from the AUC graph , it is seen that after applying the SMOTE the score has improved from 0.50 . Both the model's performance is at the same level . 

# ### Conclusion 

# To summarise,
# 
# * We first did some initial data analysis to understand about various variables present in the dataset.It was seen that the average score for the reviews was 4.46. The number of reviews by weekday and average rating was plotted to understand if there was any significant difference in the reviews by the day . 
# 
# * Models were created to predict the sentiment of the review given the review statement . Text cleaning steps like removing punctuation , stemming and creating the TF-IDF matrix . The dataset was split into train and test by 80-20 ratio . Given the highly imbalanced dataset , the AUC was around 0.5 for both Naive Bayes and Logit models.
# 
# * SMOTE techniques were applied and the model was rebuilt which improved the AUC scores to 0.8 for both the models.
# 
# Thanks for reading my notebook .Any feedback in the form of comments/upvotes are appreciated.
