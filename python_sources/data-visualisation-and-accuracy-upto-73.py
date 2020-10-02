#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import libararies

import re
import pandas as pd # CSV file I/O (pd.read_csv)
from nltk.corpus import stopwords
import numpy as np
import sklearn
import nltk
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score ,confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


news = pd.read_json('../input/News_Category_Dataset.json', lines=True)
#remove_columns_list = ['authors', 'date', 'link', 'short_description', 'headline']
news['information'] = news[['headline', 'short_description']].apply(lambda x: ' '.join(x), axis=1)


# In[ ]:


# Dataset dimension(row, columns)
news.shape


# In[ ]:


# To display entire text
pd.set_option('display.max_colwidth', -1)


# In[ ]:


news.head(1)
#news[['information', 'category']].head(5)


# In[ ]:


#news[news['authors'] == ''].groupby(by='category').size()
#news[(news['authors'] == '') & (news['short_description'] == '' )].index
# Drop those rows which has authors and short_description column as empty.
news.drop(news[(news['authors'] == '') & (news['short_description'] == '' )].index, inplace=True)


# 
# # Some analysis on "category" of news

# In[ ]:


news.groupby(by='category').size()


# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(35,7))
sns.countplot(x = 'category', data = news)


# In[ ]:





# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(15,15))
news['category'].value_counts().plot.pie( autopct = '%1.1f%%')


# ### We can see that almost 38% of the news is of category POLITICS and ENTERTAINMENT. I think we shoud focus more on these 2 category for time being.

# 

# # Some analysis on author to see if "authors" has any relationship with category

# In[ ]:


#count the number of author in the dataset
#news.authors.value_counts()
total_authors = news.authors.nunique()
news_counts = news.shape[0]
print('Total Number of authors : ', total_authors)
print('avg articles written by per author: ' + str(news_counts//total_authors))
print('Total news counts : ' + str(news_counts))


# In[ ]:


authors_news_counts = news.authors.value_counts()
sum_contribution = 0
author_count = 0
for author_contribution in authors_news_counts:
    author_count += 1
    if author_contribution < 80:
        break
    sum_contribution += author_contribution
print('{} of news is contributed by {} authors i.e  {} % of news is contributed by {} % of authors'.
      format(sum_contribution, author_count, format((sum_contribution*100/news_counts), '.2f'), format((author_count*100/total_authors), '.2f')))


# ### We can observe that ~1% of authors are contributing to ~61% of the news. 
# ### If all authors are writing only of few categories of news then we can consider the author feature as well for modeling.

# 

# In[ ]:


news.authors.value_counts()[0:10]


# In[ ]:


author_name = 'Lee Moran'
#author_name = 'Ed Mazza'
particular_author_news = news[news['authors'] == author_name]
df = particular_author_news.groupby(by='category')['information'].count()
df


# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(20,20))
df.plot.pie( autopct = '%1.1f%%')


# ### We can observe that even though authors are  writing for almost all category but majority of their contribution

# 

# In[ ]:





# In[ ]:


# Split the data into train and test.
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(news[['information', 'authors']], news['category'], test_size=0.33)


# In[ ]:


# Convert pandas series into numpy array
X_train = np.array(X_train);
X_test = np.array(X_test);
Y_train = np.array(Y_train);
Y_test = np.array(Y_test);
cleanHeadlines_train = [] #To append processed headlines
cleanHeadlines_test = [] #To append processed headlines
number_reviews_train = len(X_train) #Calculating the number of reviews
number_reviews_test = len(X_test) #Calculating the number of reviews


# In[ ]:


from nltk.stem import PorterStemmer, WordNetLemmatizer
lemmetizer = WordNetLemmatizer()
stemmer = PorterStemmer()
def get_words(headlines_list):
    headlines = headlines_list[0]   
    author_names = [x for x in headlines_list[1].lower().replace('and',',').replace(' ', '').split(',') if x != '']
    headlines_only_letters = re.sub('[^a-zA-Z]', ' ', headlines)
    words = nltk.word_tokenize(headlines_only_letters.lower())
    stops = set(stopwords.words('english'))
    meaningful_words = [lemmetizer.lemmatize(w) for w in words if w not in stops]
    return ' '.join(meaningful_words + author_names)


# In[ ]:


for i in range(0,number_reviews_train):
    cleanHeadline = get_words(X_train[i]) #Processing the data and getting words with no special characters, numbers or html tags
    cleanHeadlines_train.append( cleanHeadline )


# In[ ]:


for i in range(0,number_reviews_test):
    cleanHeadline = get_words(X_test[i]) #Processing the data and getting words with no special characters, numbers or html tags
    cleanHeadlines_test.append( cleanHeadline )


# In[ ]:


vectorize = sklearn.feature_extraction.text.TfidfVectorizer(analyzer = "word", max_features=30000)
tfidwords_train = vectorize.fit_transform(cleanHeadlines_train)
X_train = tfidwords_train.toarray()

tfidwords_test = vectorize.transform(cleanHeadlines_test)
X_test = tfidwords_test.toarray()


# In[ ]:


from sklearn.svm import LinearSVC

model = LinearSVC()
model.fit(X_train,Y_train)
Y_predict = model.predict(X_test)
accuracy = accuracy_score(Y_test,Y_predict)*100
print(format(accuracy, '.2f'))


# In[ ]:


logistic_Regression = LogisticRegression()
logistic_Regression.fit(X_train,Y_train)
Y_predict = logistic_Regression.predict(X_test)
accuracy = accuracy_score(Y_test,Y_predict)*100
print(format(accuracy, '.2f'))


# In[ ]:


from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB(alpha=0.1)
model.fit(X_train,Y_train)
Y_predict = model.predict(X_test)
accuracy = accuracy_score(Y_test,Y_predict)*100
print(format(accuracy, '.2f'))


# In[ ]:


# from sklearn.ensemble import BaggingClassifier
# model = BaggingClassifier(random_state=0, n_estimators=10)
# model.fit(X_train, Y_train)
# prediction = model.predict(X_test)
# print('Accuracy of bagged KNN is :',accuracy_score(prediction, Y_test))


# In[ ]:


# from sklearn.tree import DecisionTreeClassifier

# model = DecisionTreeClassifier()
# model.fit(X_train, Y_train)
# prediction_decision_tree = model.predict(X_test)
# print('The accuracy of Decision Tree is', accuracy_score(prediction_decision_tree, Y_test))


# In[ ]:


# from sklearn.neighbors import KNeighborsClassifier

# model = KNeighborsClassifier()
# model.fit(X_train, Y_train)
# prediction_knn = model.predict(X_test)
# print('The accuracy of the KNN is', metrics.accuracy_score(prediction_knn, Y_test))


# In[ ]:





# In[ ]:


# from sklearn.svm import SVC
# model = SVC(kernel='rbf',C=1,gamma=0.1)
# model.fit(X_train, Y_train)
# predict_rsvm = model.predict(X_test)
# print('Predict accuracy is ',accuracy_score(predict_rsvm,Y_test))


# In[ ]:




