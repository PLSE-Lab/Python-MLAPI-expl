#!/usr/bin/env python
# coding: utf-8

# **This invlolves the Sentiment prediction for a number of movies reviews obtained from Internet Movie data (IMDB).This dataset containes 50,000 movie reviews.Here we are predicting the sentiment of 20,000 labeled movie reviews and using remaining 30,000 reviews for training our models.**

# > **Import required libraries**

# In[ ]:


from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import nltk  # For test pre-processing
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn import preprocessing
import scikitplot as skplt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import roc_curve,auc
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import Sequential
from keras.layers import Embedding,LSTM,Dense
import warnings
warnings.filterwarnings('ignore')
import os
print(os.listdir("../input"))


# > Load the data

# In[ ]:


Movie_df=pd.read_csv('../input/IMDB Dataset.csv')
print('Shape of dataset::',Movie_df.shape)
Movie_df.head(10)


# Stats of our data

# In[ ]:


print("General stats::")
print(Movie_df.info())
print("Summary stats::\n")
print(Movie_df.describe())


# > Number of poitive & negative reviews

# In[ ]:


Movie_df.sentiment.value_counts()


# In[ ]:


reviews=Movie_df['review']
sentiment=Movie_df['sentiment']


# In[ ]:


#Summarize no. of classes
print('Classes::\n',np.unique(sentiment))


# > Split the data into train & test datasets

# In[ ]:


train_reviews=reviews[:30000]
train_sentiment=sentiment[:30000]
test_reviews=reviews[30000:]
test_sentiment=sentiment[30000:]
#Shape of train & test dataset
print('Shape of train dataset::',train_reviews.shape,train_sentiment.shape)
print('Shape of test dataset::',test_reviews.shape,test_sentiment.shape)


# > Encode our target labels

# In[ ]:


lb=preprocessing.LabelBinarizer()
#Encode 1 for positive label & 0 for Negative label
train_sentiment=lb.fit_transform(train_sentiment)
test_sentiment=lb.transform(test_sentiment)
#Reshape the array
train_sentiment=train_sentiment.ravel()  
test_sentiment=test_sentiment.ravel()
#Convert categoricals to numeric ones
train_sentiment=train_sentiment.astype('int64')
test_sentiment=test_sentiment.astype('int64')


# 
# Let's explore our data before normalization

# In[ ]:


train_reviews[0]


# In[ ]:


test_reviews[30001]


# In above paragraphs, we can observe stopwords,html tags,special charcters & numbers, which are not required for sentiment analysis.So we need to remove those by normalizing the review data to reduce dimensionality & noise in the data.

# In[ ]:


train_sentiment[0:10]


# In[ ]:


test_sentiment[0:10]


# > Data Pre-processing

# Let's normalize our data to remove stopwords, html tags and so on.

# In[ ]:



ps=PorterStemmer()
stopwords=set(stopwords.words('english'))
# Define function for data mining
def normalize_reviews(review):
    #Excluding html tags
    data_tags=re.sub(r'<[^<>]+>'," ",review)
    #Remove special characters/whitespaces
    data_special=re.sub(r'[^a-zA-Z0-9\s]','',data_tags)
    #converting to lower case
    data_lowercase=data_special.lower()
    #tokenize review data
    data_split=data_lowercase.split()
    #Removing stop words
    meaningful_words=[w for w in data_split if not w in stopwords]
    #Appply stemming
    text= ' '.join([ps.stem(word) for word in meaningful_words])
    return text


# > Normalize the train & test data

# In[ ]:


norm_train_reviews=train_reviews.apply(normalize_reviews)
norm_test_reviews=test_reviews.apply(normalize_reviews)


# 
# Let's look at our normalized data

# In[ ]:



norm_train_reviews[0]


# In[ ]:


norm_test_reviews[30001]


# > Let's create features using bag of words model

# In[ ]:


cv=CountVectorizer(ngram_range=(1,2))
train_cv=cv.fit_transform(norm_train_reviews)
test_cv =cv.transform(norm_test_reviews)
print('Shape of train_cv::',train_cv.shape)
print('Shape of test_cv::',test_cv.shape)


# Our train & test dataset contains 1929440 attributes each.

# > Let's build our traditional ML models

# > Random Forest model

# In[ ]:


get_ipython().run_cell_magic('time', '', "#Training the classifier\nrfc=RandomForestClassifier(n_estimators=20,random_state=42)\nrfc=rfc.fit(train_cv,train_sentiment)\nscore=rfc.score(train_cv,train_sentiment)\nprint('Accuracy of trained model is ::',score)")


# In[ ]:


get_ipython().run_cell_magic('time', '', '#Making predicitions\nrfc_predict=rfc.predict(test_cv)')


# In[ ]:


#How accuate our model is?
cm=confusion_matrix(test_sentiment,rfc_predict)
#plot our confusion matrix
skplt.metrics.plot_confusion_matrix(test_sentiment,rfc_predict,normalize=False,figsize=(12,8))
plt.show()


# > 0-Negative class,
# > 1-Positive class

# From the confusion matrix plot, it is concluded that, the Random Forest classifier with 20 decision trees classified the 81% of the reviews (16183 reviews) correctly & remaining 19% of reviews (3817 reviews) are misclassified.

# In[ ]:


#print classification report for performance metrics
cr=classification_report(test_sentiment,rfc_predict)
print('Classification report is::\n',cr)


# In[ ]:


# ROC curve for Random Forest Classifier
fpr_rf,tpr_rf,threshold_rf=roc_curve(test_sentiment,rfc_predict)
#Area under curve (AUC) score, fpr-False Positive rate, tpr-True Positive rate
auc_rf=auc(fpr_rf,tpr_rf)
print('AUC score for Random Forest classifier::',np.round(auc_rf,3))


# ** Let's build our deep learning model**

# 
# > Recurrent neural network (RNN) with LSTM (Long Short Term Memory) model

# In[ ]:


#Train dataset
X_train=train_cv
X_train=[str(x[0]) for x in X_train]
y_train=train_sentiment
# Test dataset
X_test=test_cv
X_test=[str(x[0]) for x in X_test]
y_test=test_sentiment


# In[ ]:


# Tokenize the train & test dataset
Max_Review_length=500
tokenizer=Tokenizer(num_words=Max_Review_length,lower=False)
tokenizer.fit_on_texts(X_train)
#tokenizig train data
X_train_token=tokenizer.texts_to_sequences(X_train)
#tokenizing test data
X_test_token=tokenizer.texts_to_sequences(X_test)

#Truncate or pad the dataset for a length of 500 words for each review
X_train=pad_sequences(X_train_token,maxlen=Max_Review_length)
X_test=pad_sequences(X_test_token,maxlen=Max_Review_length)


# In[ ]:


print('Shape of X_train datset after padding:',X_train.shape)
print('Shape of X_test dataset after padding:',X_test.shape)


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Most poplar words found in the dataset\nvocabulary_size=5000 \nembedding_size=64\nmodel=Sequential()\nmodel.add(Embedding(vocabulary_size,embedding_size,input_length=Max_Review_length))\nmodel.add(LSTM(30))\nmodel.add(Dense(1,activation='sigmoid',kernel_initializer='random_uniform'))\nmodel.summary()")


# In[ ]:


#Complile our model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[ ]:


get_ipython().run_cell_magic('time', '', '#Train our model\nbatch_size=128\nnum_epochs=6\nX_valid,y_valid=X_train[:batch_size],train_sentiment[:batch_size]\nX_train1,y_train1=X_train[batch_size:],train_sentiment[batch_size:]\n# Fit the model\nmodel.fit(X_train1,y_train1,validation_data=(X_valid,y_valid),validation_split=0.2,\n          batch_size=batch_size,epochs=num_epochs, verbose=1,shuffle=True)')


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Predictions\ny_predict_rnn=model.predict(X_test)\n#Changing the shape of y_predict to 1-Dimensional\ny_predict_rnn1=y_predict_rnn.ravel()\ny_predict_rnn1=(y_predict_rnn1>0.5)\ny_predict_rnn1[0:10]')


# In[ ]:


#Confusion matrix for RNN with LSTM
cm_rnn=confusion_matrix(y_test,y_predict_rnn1)
#plot our confusion matrix
skplt.metrics.plot_confusion_matrix(y_test,y_predict_rnn1,normalize=False,figsize=(12,8))
plt.show()


# > 0-Negative class,
# > 1-Positive class

# The confusion matrix plot states that the RNN with LSTM model classified 79% of reviews (15866 reviews) correctly & remaining 21% of reviews (4134 reviews) are misclassified.

# In[ ]:


#Classification report for performance metrics
cr_rnn=classification_report(y_test,y_predict_rnn1)
print('The Classification report is::\n',cr_rnn)


# In[ ]:


#ROC curve for RNN with LSTM
fpr_rnn,tpr_rnn,thresold_rnn=roc_curve(y_test,y_predict_rnn)
#AUC score for RNN
auc_rnn=auc(fpr_rnn,tpr_rnn)
print('AUC score for RNN with LSTM ::',np.round(auc_rnn,3))


# > **Receiver Operating Characterstic (ROC) Curve for Model Evaluation**

# > Now, let's plot the ROC for both Random Forest Classifier &  RNN with LSTM

# In[ ]:


get_ipython().run_cell_magic('time', '', "plt.figure(1)\nplt.plot([0,1],[0,1],'k--')\nplt.plot(fpr_rnn,tpr_rnn,label='RNN(area={:.3f})'.format(auc_rnn))\nplt.plot(fpr_rf,tpr_rf,label='Random Forest (area={:.3f})'.format(auc_rf))\nplt.xlabel('False Positive rate')\nplt.ylabel('True Positive rate')\nplt.title('ROC curve')\nplt.legend(loc='best')\nplt.show()")


# In[ ]:


#Model Evaluation on unseen dataset
Model_evaluation=pd.DataFrame({'Model':['Random Forest Classifier','RNN with LSTM'],
                              'f1_score':[0.81,0.79],
                              'roc_auc_score':[0.809,0.879]})
Model_evaluation


# The f1_score for Random forest classier is higher than for RNN with LSTM model & the roc_auc score for Random forest classifier is lower than for RNN with LSTM model. From the above scores, it is good to consider Random forest classifier than RNN with LSTM because it is comparatively less computationally expensive & works well on small & large amount of datasets.
