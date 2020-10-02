#!/usr/bin/env python
# coding: utf-8

# <h1 align ='center'> Bank Review/Complaint Analysis </h1>

# <h4><i>Central banks collecting information about customer satisfaction with the services provided by different banks. Also collects the information about the complaints.</i></h4>
# <ul>
# <li><i>Bank users give ratings and write reviews about the services on central bank websites. These reviews and ratings help banks evaluate services provided and take necessary action to improve customer service. While ratings are useful to convey the overall experience, they do not convey the context which led a reviewer to that experience.</i></li>
# <li><i>If we look at only the rating, it is difficult to guess why the user rated the service as 4 stars. However, after reading the review, it is not difficult to identify that the review talks about good 'service' and 'experience'.</i></li></ul>

# <h2>The objetive of the case study is to analyze customer reviews and predict customer satisfaction with the reviews.
# </h2>

# ## Import necesssary libraries

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import string
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import nltk
from nltk.corpus import wordnet


# ### Import the data set

# In[ ]:


customer = pd.read_csv('../input/bank-reviewcomplaint-analysis/BankReviews.csv', encoding='windows-1252' )


# In[ ]:


customer.head()


# ## Data Audit

# In[ ]:


customer.info()


# In[ ]:


customer.shape


# In[ ]:


customer.isnull().sum()


# In[ ]:


customer['Stars'].value_counts()


# In[ ]:


plt.figure(figsize=(8,6))
sns.countplot(customer.Stars)
plt.show()


# ## Sentiment Analysis to find positive and negative reviews

# In[ ]:


X = customer['Reviews']
Y = customer['Stars']


# In[ ]:


X.head()


# In[ ]:


# UDF to find sentiment polarity of the reviews
def sentiment_review(text):
    analysis = TextBlob(text)
    polarity_text = analysis.sentiment.polarity
    if polarity_text > 0:
        return 'Positive'
    elif polarity_text == 0:
        return 'Neutral'
    else:
        return 'Negative'  


# In[ ]:


# creating dictionary which will contain both the review and the sentiment of the review
final_dictionary = []
for text in X:
    dictionary_sentiment = {}
    dictionary_sentiment['Review'] = text
    dictionary_sentiment['Sentiment'] = sentiment_review(text)
    final_dictionary.append(dictionary_sentiment)
print(final_dictionary[:5])


# In[ ]:


# Finding positive reviews
positive_reviews = []
for review in final_dictionary:
    if review['Sentiment'] =='Positive':
        positive_reviews.append(review)
print(positive_reviews[:5])
    


# In[ ]:


# Finding neutral reviews
neutral_reviews = []
for review in final_dictionary:
    if review['Sentiment'] =='Neutral':
        neutral_reviews.append(review)
print(neutral_reviews[:5])


# In[ ]:


# Finding negative reviews
negative_reviews = []
for review in final_dictionary:
    if review['Sentiment'] =='Negative':
        negative_reviews.append(review)
print(negative_reviews[:5])


# In[ ]:


# counting number of positive,neutral and negative reviews
reviews_count = pd.DataFrame([len(positive_reviews),len(neutral_reviews),len(negative_reviews)],index=['Positive','Neutral','Negative'])


# In[ ]:


reviews_count


# In[ ]:


reviews_count.plot(kind='bar')
plt.ylabel('Reviews Count')   
plt.show()


# In[ ]:


# printing first five positive reviews
i=1
for review in positive_reviews[:5]:
        print(i)
        print(review['Review'])
        print('******************************************************')
        i+=1


# In[ ]:


# printing first five negative reviews
i=1
for review in negative_reviews[:5]:
        print(i)
        print(review['Review'])
        print('******************************************************')
        i+=1


# ## Finding most frequently used Positive/ Negative words

# ### Data Preprocessing

# In[ ]:


# UDF to clean the reviews
def clean_text(text):
    text = text.lower()
    text = text.strip()
    text = "".join([char for char in text if char not in string.punctuation])
    return text


# In[ ]:


# X = customer['Reviews']
X.head()


# In[ ]:


# applying clean_text function defined above to remove punctuation, strip extra spaces and convert each word to lowercase
X = X.apply(lambda y: clean_text(y))


# In[ ]:


X.head()


# ### Coverting reviews to tokens

# In[ ]:


tokens_vect = CountVectorizer(stop_words='english')


# In[ ]:


token_dtm = tokens_vect.fit_transform(X)


# In[ ]:


tokens_vect.get_feature_names()


# In[ ]:


token_dtm.toarray()


# In[ ]:


token_dtm.toarray().shape


# In[ ]:


len(tokens_vect.get_feature_names())


# In[ ]:


pd.DataFrame(token_dtm.toarray(),columns = tokens_vect.get_feature_names())


# In[ ]:


print(token_dtm)


# In[ ]:


# creating a dataframe which shows the count of how many times a word is coming in the corpus
count_dtm_dataframe = pd.DataFrame(np.sum(token_dtm.toarray(),axis=0),tokens_vect.get_feature_names()).reset_index()
count_dtm_dataframe.columns =['Word','Count']


# In[ ]:


count_dtm_dataframe.head()


# In[ ]:


#adding sentiment column which shows sentiment polarity of each word
sentiment_word = []
for word in count_dtm_dataframe['Word']:
    sentiment_word.append(sentiment_review(word))
count_dtm_dataframe['Sentiment'] = sentiment_word


# In[ ]:


count_dtm_dataframe.head()


# In[ ]:


# separating positive words
positive_words_df= count_dtm_dataframe.loc[count_dtm_dataframe['Sentiment']=='Positive',:].sort_values('Count',ascending=False)


# In[ ]:


positive_words_df.head(20)


# In[ ]:


# plotting word cloud of 10 most frequently used positive words
wordcloud = WordCloud(width = 1000, height = 500).generate(' '.join(positive_words_df.iloc[0:11,0]))
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show()


# In[ ]:


# separating negative words
negative_words_df= count_dtm_dataframe.loc[count_dtm_dataframe['Sentiment']=='Negative',:].sort_values('Count',ascending=False)


# In[ ]:


negative_words_df.head(10)


# In[ ]:


# plotting word cloud of 10 most frequently used positive words
wordcloud = WordCloud(width = 1000, height = 500).generate(' '.join(negative_words_df.iloc[0:11,0]))
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show()


# ## Topic Modelling

# ### Splitting the data into train and test

# In[ ]:


train_X,test_X,train_Y,test_Y = train_test_split(X,Y,random_state = 123, test_size = 0.2)  


# In[ ]:


print('No.of observations in train_X: ',len(train_X), '| No.of observations in test_X: ',len(test_X))
print('No.of observations in train_Y: ',len(train_Y), '| No.of observations in test_Y: ',len(test_Y))


# 
# # Feature Generation using DTM and TDM

# ### Feature generation using DTM

# In[ ]:


vect = CountVectorizer(strip_accents='unicode', stop_words='english', ngram_range=(1,1),min_df=0.001,max_df=0.95)


# In[ ]:


train_X_fit = vect.fit(train_X)
train_X_dtm = vect.transform(train_X)
test_X_dtm = vect.transform(test_X)


# In[ ]:


print(train_X_dtm)


# In[ ]:


print(test_X_dtm)


# In[ ]:


vect.get_feature_names()


# In[ ]:


print('No.of features for are',len(vect.get_feature_names()))


# In[ ]:


train_X_dtm_df = pd.DataFrame(train_X_dtm.toarray(),columns=vect.get_feature_names())


# In[ ]:


train_X_dtm_df.head()


# In[ ]:


# Finding how many times a tem is used in corpus
train_dtm_freq = np.sum(train_X_dtm_df,axis=0)


# In[ ]:


train_dtm_freq.head(20)


# ### Feature generation using TDM

# In[ ]:


vect_tdm = TfidfVectorizer(strip_accents='unicode', stop_words='english', ngram_range=(1,1),min_df=0.001,max_df=0.95)


# In[ ]:


train_X_tdm = vect_tdm.fit_transform(train_X)
test_X_tdm = vect.transform(test_X)


# In[ ]:


print(train_X_tdm)


# In[ ]:


print(test_X_tdm)


# In[ ]:


vect_tdm.get_feature_names()


# In[ ]:


print('No.of features for are',len(vect_tdm.get_feature_names()))


# In[ ]:


# creating dataframe to to see which features are present in the documents
train_X_tdm_df = pd.DataFrame(train_X_tdm.toarray(),columns=vect_tdm.get_feature_names())


# In[ ]:


train_X_tdm_df.head()


# In[ ]:


test_X_tdm_df = pd.DataFrame(test_X_tdm.toarray(),columns=vect_tdm.get_feature_names())


# In[ ]:


test_X_tdm_df.head()


# In[ ]:


# Finding how many times a term is used in test corpus
test_tdm_freq = np.sum(test_X_tdm_df,axis=0)


# In[ ]:


test_tdm_freq.head(20)


# In[ ]:


# train a LDA Model
lda_model = LatentDirichletAllocation(n_components=20, learning_method='online', max_iter=50)
X_topics = lda_model.fit_transform(train_X_tdm)
topic_word = lda_model.components_ 
vocab = vect.get_feature_names()


# In[ ]:


# view the topic models
top_words = 10
topic_summaries = []
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(top_words+1):-1]
    topic_summaries.append(' '.join(topic_words))
    print(topic_words)


# In[ ]:


# view the topic models
top_words = 10
topic_summaries = []
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(top_words+1):-1]
    topic_summaries.append(' '.join(topic_words))
topic_summaries


# # Building Model

# ### Building Model on DTM

# In[ ]:


# building naive bayes model on DTM
naive_model = MultinomialNB()
naive_model.fit(train_X_dtm,train_Y)


# In[ ]:


predict_train = naive_model.predict(train_X_dtm)
predict_test = naive_model.predict(test_X_dtm)


# In[ ]:


len(predict_test)


# In[ ]:


print('Accuracy on train: ',metrics.accuracy_score(train_Y,predict_train))
print('Accuracy on test: ',metrics.accuracy_score(test_Y,predict_test))


# In[ ]:


# predict probabilities on train and test
predict_prob_train = naive_model.predict_proba(train_X_dtm)[:,1]
predict_prob_test = naive_model.predict_proba(test_X_dtm)[:,1]


# In[ ]:


print('ROC_AUC score on train: ',metrics.roc_auc_score(train_Y,predict_prob_train))
print('ROC_AUC score on test: ',metrics.roc_auc_score(test_Y,predict_prob_test))


# In[ ]:


# confusion matrix on test 
cm_test = metrics.confusion_matrix(test_Y,predict_test,[5,1])


# In[ ]:


cm_test


# In[ ]:


import seaborn as sns
sns.heatmap(cm_test,annot=True,xticklabels=[5,1],yticklabels=[5,1])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# ### Building Model on TDM

# In[ ]:


# building naive bayes model on DTM
naive_model = MultinomialNB()
naive_model.fit(train_X_tdm,train_Y)


# In[ ]:


predict_train = naive_model.predict(train_X_tdm)
predict_test = naive_model.predict(test_X_tdm)


# In[ ]:


len(predict_test)


# In[ ]:


print('Accuracy on train: ',metrics.accuracy_score(train_Y,predict_train))
print('Accuracy on test: ',metrics.accuracy_score(test_Y,predict_test))


# In[ ]:


# predict probabilities on train and test
predict_prob_train = naive_model.predict_proba(train_X_tdm)[:,1]
predict_prob_test = naive_model.predict_proba(test_X_tdm)[:,1]


# In[ ]:


print('ROC_AUC score on train: ',metrics.roc_auc_score(train_Y,predict_prob_train))
print('ROC_AUC score on test: ',metrics.roc_auc_score(test_Y,predict_prob_test))


# In[ ]:


# confusion matrix on test 
cm_test = metrics.confusion_matrix(test_Y,predict_test,[5,1])


# In[ ]:


cm_test


# In[ ]:


import seaborn as sns
sns.heatmap(cm_test,annot=True,xticklabels=[5,1],yticklabels=[5,1])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# <h3> Model showed better results using DTM values and using unigrams.</h3>

# ## We were asked that we can ignore intent analysis as that is covered in topic modelling. Hence skipping that part.
