#!/usr/bin/env python
# coding: utf-8

# The techniques used in this kernal are:
# 
# * Bag of words
# * Tfidf 
# * Average Word2vec
# * Tfidf word2vec
# 
# We used Logistic regression for prediction.Inaddition to this we will also see how we can get the top features that influence positive class and negative class.Inassition to this we will see how we can check multicollinearity of features using **Pertubation test**.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sqlite3
import re
from bs4 import BeautifulSoup
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.model_selection import GridSearchCV
import pickle
import gc
from sklearn.metrics import roc_auc_score, auc, roc_curve, confusion_matrix
from sklearn.linear_model import LogisticRegression
# Converting to CSR_Matrix..
from scipy.sparse import csr_matrix


# ## Loading the data
# 
# The dataset is available in two forms
# 1. .csv file
# 2. SQLite Database
# 
# In order to load the data, We have used the SQLITE dataset as it is easier to query the data and visualise the data efficiently.
# <br> 
# 
# Here as we only want to get the global sentiment of the recommendations (positive or negative), we will purposefully ignore all Scores equal to 3. If the score is above 3, then the recommendation will be set to "positive". Otherwise, it will be set to "negative".

# In[ ]:


db = '/kaggle/input/amazon-fine-food-reviews/database.sqlite'
connection = sqlite3.connect(db)


df_filtered = pd.read_sql_query(""" SELECT * FROM Reviews WHERE Score != 3 """,connection)


print("Number of data points in our data", df_filtered.shape)
# df_filtered = df_filtered.head(3000)


# ## Exploratory Data Analysis

# We will make all reviews with score greater than 3 as 1(positive) and less than 3 as 0(negative)

# In[ ]:


df_filtered['Score'] = df_filtered['Score'].apply(lambda x: 1 if x>3 else 0)
df_filtered['Score'].head(3)


# It is observed (as shown in the table below) that the reviews data had many duplicate entries. Hence it was necessary to remove duplicates in order to get unbiased results for the analysis of the data.  Following is an example:

# In[ ]:


#Sorting data according to ProductId in ascending order
df_sorted=df_filtered.sort_values('ProductId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
#Deduplication of entries
df = df_sorted.drop_duplicates(subset={"UserId","ProfileName","Time","Text"}, keep='first', inplace=False)
print(df.shape)
df.head(3)


# <b>Observation:-</b> It was also seen that in two rows given below the value of HelpfulnessNumerator is greater than HelpfulnessDenominator which is not practically possible hence these two rows too are removed from calcualtions<br>
# 
# * Helpfulness Numerator: Number of users who found the review helpful <br>
# * Helpfulness Denominator: Number of users who indicated whether they found the review helpful or not

# In[ ]:


df = df[df['HelpfulnessNumerator'] <= df['HelpfulnessDenominator']]
df.shape


# In[ ]:


#checking how much data still remains

print(f'{round((df.shape[0]/df_filtered.shape[0])*100,2)}%')


# Now we will analyse target values

# In[ ]:


print(df['Score'].value_counts())
values = df['Score'].value_counts().values
sns.barplot(x=['Positive','Negative'],y=values)
plt.show()


# Clearly we have an imbalanced dataset. So It's better to use metrics like f1 score, AUC as perfomance metric

# ### Preprocessing Review Text
# 
# Now that we have finished deduplication our data requires some preprocessing before we go on further with analysis and making the prediction model.
# 
# Hence in the Preprocessing phase we do the following in the order below:-
# 
# 1. Begin by removing the html tags
# 2. Remove any punctuations or limited set of special characters like , or . or # etc.
# 3. Check if the word is made up of english letters and is not alpha-numeric
# 4. Check to see if the length of the word is greater than 2 (as it was researched that there is no adjective in 2-letters)
# 5. Convert the word to lowercase
# 6. Remove Stopwords
# 7. Finally Snowball Stemming the word (it was obsereved to be better than Porter Stemming)<br>
# 
# After which we collect the words used to describe positive and negative reviews

# In[ ]:


# replacing some phrases like won't with will not

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',             'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',             'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',             'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',             'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very',             's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',             've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",             'won', "won't", 'wouldn', "wouldn't"])


# In[ ]:


preprocessed_reviews = []
# tqdm is for printing the status bar
for sentance in tqdm(df['Text'].values):
    sentance = re.sub(r"http\S+", "", sentance)
    # removing html tags
    sentance = BeautifulSoup(sentance, 'lxml').get_text()
    sentance = decontracted(sentance)
    # removing extra spaces and numbers
    sentance = re.sub("\S*\d\S*", "", sentance).strip()
    # removing non alphabels
    sentance = re.sub('[^A-Za-z]+', ' ', sentance)
    # https://gist.github.com/sebleier/554280
    sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)
    preprocessed_reviews.append(sentance.strip())


# In[ ]:


#combining required columns
df['clean_text'] = preprocessed_reviews
df = df[['Time','clean_text','Score']]
#reseting index
df = df.reset_index(drop=True)


# For the ease of computation we will sample just 100k points

# In[ ]:


#sampling 100k points 
df_100k = df.sample(100000)
#sorting 100kpoints based on time
df_100k['Time'] = pd.to_datetime(df_100k['Time'],unit='s')
df_100k = df_100k.sort_values('Time')
#reseting index
df_100k = df_100k.reset_index(drop=True)


# In[ ]:


df_100k['Score'].value_counts()


# Observation: We have clearly an imbalenced dataset
# 

# ## Train-test split

# In[ ]:


#splitting data to train.cv and test
from sklearn.model_selection import train_test_split
x = df_100k['clean_text']
y = df_100k['Score']
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,stratify=y)
X_tr,X_cv,y_tr,y_cv = train_test_split(X_train,y_train,test_size=0.3,stratify=y_train)


# ## Bag Of Words

# In[ ]:


bow = CountVectorizer()
bow.fit(X_tr)
X_train_bow = bow.transform(X_tr)
X_cv_bow = bow.transform(X_cv)
X_test_bow = bow.transform(X_test)

print('shape of X_train_bow is {}'.format(X_train_bow.get_shape()))
print('shape of X_cv_bow is {}'.format(X_cv_bow.get_shape()))
print('shape of X_test_bow is {}'.format(X_test_bow.get_shape()))


# In[ ]:



C = [0.001,0.01,0.1,1,10,100]
train_auc = []
cv_auc = []

for c in C:
    model = LogisticRegression(penalty='l2',C=c,solver='liblinear')
    model.fit(X_train_bow,y_tr)
    y_tr_pred = model.predict(X_train_bow)
    y_cv_pred = model.predict(X_cv_bow)
    train_auc.append(roc_auc_score(y_tr,y_tr_pred))
    cv_auc.append(roc_auc_score(y_cv,y_cv_pred))
    


# In[ ]:


plt.grid(True)
plt.plot(np.log(C),train_auc,label='Train AUC')
plt.plot(np.log(C),cv_auc,label='CV AUC')
plt.scatter(np.log(C),train_auc)
plt.scatter(np.log(C),cv_auc)
plt.legend()
plt.xlabel("C: hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.show()


# In[ ]:


model = LogisticRegression(penalty='l2',C=0.1,solver='liblinear')
model.fit(X_train_bow,y_tr)
y_tr_pred = model.predict(X_train_bow)
y_cv_pred = model.predict(X_cv_bow)
train_fpr, train_tpr, thresholds = roc_curve(y_tr, model.predict_proba(X_train_bow)[:,1])
test_fpr, test_tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test_bow)[:,1])

plt.grid(True)
plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("fpr")
plt.ylabel("tpr")
plt.title("ROC CURVE FOR OPTIMAL K")
plt.show()

#Area under ROC curve
print('Area under train roc {}'.format(auc(train_fpr, train_tpr)))
print('Area under test roc {}'.format(auc(test_fpr, test_tpr)))


# In[ ]:


#Grid search
# params = {'C' : [0.001,0.01,0.1,1,10,100],
#           'penalty' : ['l1', 'l2']}

# log_clf = LogisticRegression(class_weight='balanced',solver='liblinear') 
# clf = GridSearchCV(log_clf, params, cv = 5, verbose=True, n_jobs=-1)
# best_clf = clf.fit(X_train_bow,y_tr)


# In[ ]:


cm = confusion_matrix(y_tr,model.predict(X_train_bow))
class_label = ["negative", "positive"]
df_cm = pd.DataFrame(cm, index = class_label, columns = class_label)
sns.heatmap(df_cm, annot = True, fmt = "d")
plt.title("Confusiion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# In[ ]:


cm = confusion_matrix(y_test,model.predict(X_test_bow))
class_label = ["negative", "positive"]
df_cm = pd.DataFrame(cm, index = class_label, columns = class_label)
sns.heatmap(df_cm, annot = True, fmt = "d")
plt.title("Confusiion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# ## Top features

# In[ ]:


feats = bow.get_feature_names()
coefs = model.coef_.reshape(-1,1)
dff = pd.DataFrame(coefs,columns=['coef'],index=feats)
top_neg = dff.sort_values(ascending=True,by='coef').head(10)
top_pos = dff.sort_values(ascending=False,by='coef').head(10)
print('Top 10 Positive features')
print(top_pos)
print('-'*50)
print('Top 10 Negative features')
print(top_neg)


# ## Pertubation test

# step 1 : Get weights W after we fit our model with data X.

# In[ ]:


W = model.coef_


# Step 2: Add noise to X and get new data X' ie X'= X + e 

# In[ ]:


#noise
epsilon = 0.00005
# adding noise X_ = X + epsilon
X_ = X_train_bow.data + epsilon 
print(X_.shape)
X_train_bow_dash = csr_matrix((X_, X_train_bow.indices, X_train_bow.indptr), shape=X_train_bow.shape)
print(X_train_bow_dash.shape)


# Step 3: Fit the model again in data X_ and get new weights W_

# In[ ]:


model = LogisticRegression(penalty='l2',C=0.1,solver='liblinear')
model.fit(X_train_bow_dash,y_tr)
W_ = model.coef_


# Step 4: Add a small eps value(to eliminate the divisible by zero error) to W and W_

# In[ ]:



epsilon2 = 0.000006
W = W + epsilon2
W_ = W_ + epsilon2


# Step 5:   find the % change between W and W_ (| (W-W_) / (W) |)*100)

# In[ ]:


change = abs((W - W_)/(W))
percentage_change = change*100
percentage_change = percentage_change[0]


# Printing Percentiles :
for i in range(10, 101, 10):
    print("{}th Percentile value : {}".format(i, np.percentile(percentage_change, i)))
    
print('--'*50)

for i in range(90, 101):
    print("{}th Percentile value : {}".format(i, np.percentile(percentage_change, i)))

print('--'*50)

for i in range(1, 11):
    print("{}th Percentile value : {}".format((i*1.0/10 + 99), np.percentile(percentage_change, i*1.0/10 + 99)))


# We found that after 99.9th percentile there is significiant rise in weight difference value. It shows existance of multicollinearity. If we remove those weights it will be better

# Step6: Remove those features showing high change

# In[ ]:


feats = bow.get_feature_names()
change_ = change.reshape(-1,1)
pertub_df = pd.DataFrame(change_,columns=['change'],index=feats)
print(pertub_df.shape)
pertub_df = pertub_df.sort_values(ascending=False,by=['change'])
pertub_df.head(3)


# In[ ]:


#picking features with high change (> 99.9th percentile value)
pertub_df = pertub_df[pertub_df['change'] < 0.4715049866279103]
print(pertub_df.shape)
pertub_df.head(3)


# Now we can fit a logistic regression model using these features and so there exists no(little) multicollinearity

# # TFIDF

# In[ ]:


#applying bow on x_train and x_test
vectorizer = TfidfVectorizer()
vectorizer.fit(X_tr)
# we use the fitted CountVectorizer to convert the text to vector
X_train_tfidf = vectorizer.transform(X_tr)
X_cv_tfidf = vectorizer.transform(X_cv)
X_test_tfidf = vectorizer.transform(X_test)


# In[ ]:


C = [0.001,0.01,0.1,1,10,100]
train_auc = []
cv_auc = []

for c in C:
    model = LogisticRegression(penalty='l1',C=c,solver='liblinear')
    model.fit(X_train_tfidf,y_tr)
    y_tr_pred = model.predict(X_train_tfidf)
    y_cv_pred = model.predict(X_cv_tfidf)
    train_auc.append(roc_auc_score(y_tr,y_tr_pred))
    cv_auc.append(roc_auc_score(y_cv,y_cv_pred))
    
plt.grid(True)
plt.plot(np.log(C),train_auc,label='Train AUC')
plt.plot(np.log(C),cv_auc,label='CV AUC')
plt.scatter(np.log(C),train_auc)
plt.scatter(np.log(C),cv_auc)
plt.legend()
plt.xlabel("C: hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.show()


# In[ ]:


model = LogisticRegression(penalty='l1',C=1,solver='liblinear')
model.fit(X_train_tfidf,y_tr)
y_tr_pred = model.predict(X_train_tfidf)
y_cv_pred = model.predict(X_cv_tfidf)
train_fpr, train_tpr, thresholds = roc_curve(y_tr, model.predict_proba(X_train_tfidf)[:,1])
test_fpr, test_tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test_tfidf)[:,1])

plt.grid(True)
plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("fpr")
plt.ylabel("tpr")
plt.title("ROC CURVE FOR OPTIMAL K")
plt.show()

#Area under ROC curve
print('Area under train roc {}'.format(auc(train_fpr, train_tpr)))
print('Area under test roc {}'.format(auc(test_fpr, test_tpr)))


# In[ ]:


cm = confusion_matrix(y_tr,model.predict(X_train_bow))
class_label = ["negative", "positive"]
df_cm = pd.DataFrame(cm, index = class_label, columns = class_label)
sns.heatmap(df_cm, annot = True, fmt = "d")
plt.title("Confusiion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# In[ ]:


cm = confusion_matrix(y_test,model.predict(X_test_bow))
class_label = ["negative", "positive"]
df_cm = pd.DataFrame(cm, index = class_label, columns = class_label)
sns.heatmap(df_cm, annot = True, fmt = "d")
plt.title("Confusiion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# # Avg Word2Vec

# In[ ]:


gc.collect()


# In[ ]:


preprocessed_reviews = X_tr.values
train_sentence = [rev.split() for rev in preprocessed_reviews]
# min_count = 5 considers only words that occured atleast 5 times
# size = length of vector
w2v_model_train = Word2Vec(train_sentence,min_count=5,size=50, workers=4)
w2v_words = list(w2v_model_train.wv.vocab)


# In[ ]:


train_reviews = X_tr.values
train_sentence = [rev.split() for rev in train_reviews]

sent_vectors_train = []
for sent in tqdm(train_sentence):
    sent_vec = np.zeros(50)
    cnt_words = 0
    for word in sent:
        if word in w2v_words:
            vector = w2v_model_train.wv[word]
            sent_vec += vector
            cnt_words += 1
    if cnt_words != 0:
        sent_vec /= cnt_words
    sent_vectors_train.append(sent_vec)

print(len(sent_vectors_train))
print(len(sent_vectors_train[0]))

######################################################################

cv_reviews = X_cv.values
cv_sentence = [rev.split() for rev in cv_reviews]

sent_vectors_cv = []
for sent in tqdm(cv_sentence):
    count = 0
    sent_vec = np.zeros(50)
    for word in sent:
        if word in w2v_words:
            vector = w2v_model_train.wv[word]
            sent_vec += vector
            count += 1
            
    if count != 0:
        sent_vec /= count
    sent_vectors_cv.append(sent_vec)

print(len(sent_vectors_cv))
print(len(sent_vectors_cv[0]))

########################################################
test_reviews = X_test.values
test_sentence = [rev.split() for rev in test_reviews]

sent_vectors_test = []
for sent in tqdm(test_sentence):
    count = 0
    sent_vec = np.zeros(50)
    for word in sent:
        if word in w2v_words:
            vector = w2v_model_train.wv[word]
            sent_vec += vector
            count += 1
            
    if count != 0:
        sent_vec /= count
    sent_vectors_test.append(sent_vec)

print(len(sent_vectors_test))
print(len(sent_vectors_test[0]))

    


# In[ ]:


X_train_avgw2v = sent_vectors_train
X_cv_avgw2v = sent_vectors_cv
X_test_avgw2v = sent_vectors_test


# In[ ]:


C = [0.001,0.01,0.1,1,10,100]
train_auc = []
cv_auc = []

for c in C:
    model = LogisticRegression(penalty='l2',C=c,solver='liblinear')
    model.fit(X_train_avgw2v,y_tr)
    y_tr_pred = model.predict(X_train_avgw2v)
    y_cv_pred = model.predict(X_cv_avgw2v)
    train_auc.append(roc_auc_score(y_tr,y_tr_pred))
    cv_auc.append(roc_auc_score(y_cv,y_cv_pred))
    
plt.grid(True)
plt.plot(np.log(C),train_auc,label='Train AUC')
plt.plot(np.log(C),cv_auc,label='CV AUC')
plt.scatter(np.log(C),train_auc)
plt.scatter(np.log(C),cv_auc)
plt.legend()
plt.xlabel("C: hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.show()


# In[ ]:


model = LogisticRegression(penalty='l2',C=0.1,solver='liblinear')
model.fit(X_train_avgw2v,y_tr)
y_tr_pred = model.predict(X_train_avgw2v)
y_cv_pred = model.predict(X_cv_avgw2v)
train_fpr, train_tpr, thresholds = roc_curve(y_tr, model.predict_proba(X_train_avgw2v)[:,1])
test_fpr, test_tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test_avgw2v)[:,1])

plt.grid(True)
plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("fpr")
plt.ylabel("tpr")
plt.title("ROC CURVE FOR OPTIMAL K")
plt.show()

#Area under ROC curve
print('Area under train roc {}'.format(auc(train_fpr, train_tpr)))
print('Area under test roc {}'.format(auc(test_fpr, test_tpr)))


# In[ ]:


#confusion matrix for train data
cm = confusion_matrix(y_tr,model.predict(X_train_avgw2v))
class_label = ["negative", "positive"]
df_cm = pd.DataFrame(cm, index = class_label, columns = class_label)
sns.heatmap(df_cm, annot = True, fmt = "d")
plt.title("Confusiion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# #confusion matrix for test data
# cm = confusion_matrix(y_test,.predict(X_test_avgw2v))
# class_label = ["negative", "positive"]
# df_cm = pd.DataFrame(cm, index = class_label, columns = class_label)
# sns.heatmap(df_cm, annot = True, fmt = "d")
# plt.title("Confusiion Matrix")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.show()

# # TFIDF Word2vec 

# In[ ]:


tfidf = TfidfVectorizer()
tfidf.fit(X_tr)
dictionary = dict(zip(tfidf.get_feature_names(), list(tfidf.idf_)))
tfidf_feats = tfidf.get_feature_names()


# In[ ]:




train_reviews = X_tr.values
train_sentence = [rev.split() for rev in train_reviews]
sent_vectors_train = []
for sent in tqdm(train_sentence):
    sent_vec = np.zeros(50)
    weight_sum = 0
    for word in sent:
        if word in tfidf_feats and word in w2v_words:
            vec = w2v_model_train.wv[word]
            # tf * idf
            tfidf_value = (sent.count(word)/len(sent)) * dictionary[word] 
            vec = vec * tfidf_value
            sent_vec += vec
            weight_sum += tfidf_value
    if weight_sum != 0:
        sent_vec /= weight_sum
    sent_vectors_train.append(sent_vec)
print(len(sent_vectors_train))
print(sent_vectors_train[0])

##########################################

cv_reviews = X_cv.values
cv_sentence = [rev.split() for rev in cv_reviews]
sent_vectors_cv = []
for sent in tqdm(cv_sentence):
    sent_vec = np.zeros(50)
    weight_sum = 0
    for word in sent:
        if word in tfidf_feats and word in w2v_words:
            vec = w2v_model_train.wv[word]
            # tf * idf
            tfidf_value = (sent.count(word)/len(sent)) * dictionary[word] 
            vec = vec * tfidf_value
            sent_vec += vec
            weight_sum += tfidf_value
    if weight_sum != 0:
        sent_vec /= weight_sum
    sent_vectors_cv.append(sent_vec)
print(len(sent_vectors_cv))
print(sent_vectors_cv[0])   

###############################################

test_reviews = X_test.values
test_sentence = [rev.split() for rev in test_reviews]
sent_vectors_test = []
for sent in tqdm(test_sentence):
    sent_vec = np.zeros(50)
    weight_sum = 0
    for word in sent:
        if word in tfidf_feats and word in w2v_words:
            vec = w2v_model_train.wv[word]
            # tf * idf
            tfidf_value = (sent.count(word)/len(sent)) * dictionary[word] 
            vec = vec * tfidf_value
            sent_vec += vec
            weight_sum += tfidf_value
    if weight_sum != 0:
        sent_vec /= weight_sum
    sent_vectors_test.append(sent_vec)
print(len(sent_vectors_test))
print(sent_vectors_test[0])   
        


# In[ ]:


X_train_tfw2v = sent_vectors_train
X_cv_tfw2v = sent_vectors_cv
X_test_tfw2v = sent_vectors_test


# In[ ]:


C = [0.001,0.01,0.1,1,10,100]
train_auc = []
cv_auc = []

for c in C:
    model = LogisticRegression(penalty='l2',C=c,solver='liblinear')
    model.fit(X_train_tfw2v,y_tr)
    y_tr_pred = model.predict(X_train_tfw2v)
    y_cv_pred = model.predict(X_cv_tfw2v)
    train_auc.append(roc_auc_score(y_tr,y_tr_pred))
    cv_auc.append(roc_auc_score(y_cv,y_cv_pred))
    
plt.grid(True)
plt.plot(np.log(C),train_auc,label='Train AUC')
plt.plot(np.log(C),cv_auc,label='CV AUC')
plt.scatter(np.log(C),train_auc)
plt.scatter(np.log(C),cv_auc)
plt.legend()
plt.xlabel("C: hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.show()


# In[ ]:


optimal_C = 0.1
clf_opt = LogisticRegression(C=optimal_C,penalty='l2',solver='liblinear')
clf_opt.fit(X_train_tfw2v, y_tr)
# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
# not the predicted outputs

train_fpr, train_tpr, thresholds = roc_curve(y_tr, clf_opt.predict_proba(X_train_tfw2v)[:,1])
test_fpr, test_tpr, thresholds = roc_curve(y_test, clf_opt.predict_proba(X_test_tfw2v)[:,1])

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("C: hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.show()

#Area under ROC curve
print('Area under train roc {}'.format(auc(train_fpr, train_tpr)))
print('Area under test roc {}'.format(auc(test_fpr, test_tpr)))


# In[ ]:


#confusion matrix for train data
cm = confusion_matrix(y_tr,model.predict(X_train_tfw2v))
class_label = ["negative", "positive"]
df_cm = pd.DataFrame(cm, index = class_label, columns = class_label)
sns.heatmap(df_cm, annot = True, fmt = "d")
plt.title("Confusiion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# In[ ]:


#confusion matrix for test data
cm = confusion_matrix(y_test,model.predict(X_test_tfw2v))
class_label = ["negative", "positive"]
df_cm = pd.DataFrame(cm, index = class_label, columns = class_label)
sns.heatmap(df_cm, annot = True, fmt = "d")
plt.title("Confusiion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

