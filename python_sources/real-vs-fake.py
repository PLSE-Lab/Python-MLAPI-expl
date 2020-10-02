#!/usr/bin/env python
# coding: utf-8

# Credits - **https://www.kaggle.com/sreshta140/is-it-authentic-or-not/notebook**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import plotly.offline as pyoff
import plotly.graph_objs as go
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


true = pd.read_csv('../input/fake-and-real-news-dataset/True.csv')
fake = pd.read_csv('../input/fake-and-real-news-dataset/Fake.csv')


# In[ ]:


true.head()


# In[ ]:


fake.head()


# In[ ]:


true.info()


# In[ ]:


fake.info()


# In[ ]:


true['Target'] = 0
fake['Target'] = 1


# In[ ]:


df = pd.concat([true, fake])


# In[ ]:


patternDel = "http"
filter1 = df['date'].str.contains(patternDel)

df = df[~filter1]


# In[ ]:


pattern = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
filter2 = df['date'].str.contains(pattern)

df = df[filter2]


# In[ ]:


df['date'] = pd.to_datetime(df['date'])


# In[ ]:


df.head()


# In[ ]:


plt.figure(figsize=(10,10))
sns.countplot(df['subject'], hue='Target', data=df)


# **Highly imbalanced representation which might create problems later**

# In[ ]:


copy = df.copy()


# In[ ]:


copy = copy.sort_values(by = ['date'])


# In[ ]:


copy = copy.reset_index(drop=True)


# In[ ]:


copy


# In[ ]:


copy1 = copy[copy['Target'] == 1]
copy1 = copy1.groupby(['date'])['Target'].count()


# In[ ]:


copy1 = pd.DataFrame(copy1)
copy1.head()


# In[ ]:


copy0 = copy[copy['Target'] == 0]
copy0 = copy0.groupby(['date'])['Target'].count()


# In[ ]:


copy0 = pd.DataFrame(copy0)
copy0.head()


# In[ ]:


plot_data = [
    go.Scatter(
        x=copy0.index,
        y=copy0['Target'],
        name='True',
        #x_axis="OTI",
        #y_axis="time",
    ),
    go.Scatter(
        x=copy1.index,
        y=copy1['Target'],
        name='Fake'
    )
    
]
plot_layout = go.Layout(
        title='Day-wise',
        yaxis_title='Count',
        xaxis_title='Time',
        plot_bgcolor='rgba(0,0,0,0)'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# We can safely say that data is highly representative of the real scenario. 
# 1. Emergence of fake news before US 2016 Elections
# 2. Huge amount of tweets on 9/Nov 2016 - Trump's Victory
# 3. 7th April 2017 - US missile attack on Syria

# In[ ]:


from wordcloud import WordCloud,STOPWORDS


# In[ ]:


plt.figure(figsize = (20,20))
wc = WordCloud(min_font_size = 3,  max_words = 3000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate((" ".join(df_[df_.target == 1].news)))
plt.imshow(wc,interpolation = 'bilinear')


# In[ ]:


plt.figure(figsize = (20,20))
wc = WordCloud(min_font_size = 3,  max_words = 3000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate((" ".join(df_[df_.target == 0].news)))
plt.imshow(wc,interpolation = 'bilinear')


# In[ ]:


from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from collections import  Counter
plt.style.use('ggplot')
stop=set(stopwords.words('english'))
import re
from nltk.tokenize import word_tokenize
import gensim
import string


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import re,string,unicodedata
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,plot_confusion_matrix
from sklearn.model_selection import train_test_split
from string import punctuation


# In[ ]:


df_['news'] = df['text'] + df['title'] + df['subject']
df_['target'] = df['Target']


# In[ ]:





# In[ ]:



df_.head()


# In[ ]:


def cleaner(phrase):
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can't", 'can not', phrase)
  
  # general
    phrase = re.sub(r"n\'t"," not", phrase)
    phrase = re.sub(r"\'re'"," are", phrase)
    phrase = re.sub(r"\'s"," is", phrase)
    phrase = re.sub(r"\'ll"," will", phrase)
    phrase = re.sub(r"\'d"," would", phrase)
    phrase = re.sub(r"\'t"," not", phrase)
    phrase = re.sub(r"\'ve"," have", phrase)
    phrase = re.sub(r"\'m"," am", phrase)
    
    return phrase


# In[ ]:


from bs4 import BeautifulSoup
from tqdm import tqdm
import re

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords


# In[ ]:


stop = set(stopwords.words('english'))


# In[ ]:


cleaned_text = []

for sentance in tqdm(df_['news'].values):
    sentance = str(sentance)
    sentance = re.sub(r"http\S+", "", sentance)
    sentance = BeautifulSoup(sentance, 'lxml').get_text()
    sentance = cleaner(sentance)
    sentance = re.sub(r'[?|!|\'|"|#|+]', r'', sentance)
    sentance = re.sub("\S*\d\S*", "", sentance).strip()
    sentance = re.sub('[^A-Za-z]+', ' ', sentance)
    sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stop)
    cleaned_text.append(sentance.strip())


# In[ ]:


df_['news'] = cleaned_text


# In[ ]:


df_.head()


# In[ ]:


import gensim
import nltk


# In[ ]:





# In[ ]:


X = []
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
for par in df_['news'].values:
    tmp = []
    sentences = nltk.sent_tokenize(par)
    for sent in sentences:
        sent = sent.lower()
        tokens = tokenizer.tokenize(sent)
        filtered_words = [w.strip() for w in tokens if w not in stop and len(w) > 1]
        tmp.extend(filtered_words)
    X.append(tmp)


# In[ ]:


w2v_model = gensim.models.Word2Vec(sentences=X, size=150, window=5, min_count=2)


# In[ ]:


w2v_model.wv.most_similar(positive = 'trump')


# In[ ]:


w2v_model.wv.most_similar('hillary')


# In[ ]:


w2v_model.wv.similarity('donaldtrump', 'hillaryclinton')


# In[ ]:


w2v_model.wv.similarity('donaldtrump', 'mikepence')


# In[ ]:


w2v_model.wv.doesnt_match(['trump', 'hillary', 'sanders'])


# In[ ]:


w2v_model.wv.doesnt_match(['donaldtrump', 'hillary', 'mikepence'])


# **Perfecto**

# In[ ]:


w2v_model.wv.most_similar(positive = ['woman','trump'], negative=['hillary'], topn=3)


# In[ ]:


w2v_model.wv.most_similar(positive = ['hillary','trump'], negative=['mikepence'], topn=3)


# Ohh My Gawd **'Hillary and her friends at FOX'**

# In[ ]:


w2v_model.wv.most_similar(positive = ['hillary','trump'], negative=['america'], topn=3)


# ***Basket of Deplorables***

# Modelling

# In[ ]:


X = df_['news']
y = df_['target']


# In[ ]:


from sklearn.model_selection import train_test_split
X_Train, X_test, y_Train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# In[ ]:


X_train, X_cross, y_train, y_cross = train_test_split(X_Train, y_Train, test_size=0.1, random_state=42)


# In[ ]:


tf_idf=TfidfVectorizer(min_df=5,use_idf=True,ngram_range=(1,2))
tf_idf.fit(X_train)
Train_TFIDF = tf_idf.transform(X_train)
CrossVal_TFIDF = tf_idf.transform(X_cross)
Test_TFIDF= tf_idf.transform(X_test)


# Multinomial naive Bayes

# In[ ]:


alpha_set=[0.0001,0.001,0.01,0.1,1,10,100,1000]

Train_AUC_TFIDF = []
CrossVal_AUC_TFIDF = []


for i in alpha_set:
    naive_b=MultinomialNB(alpha=i)
    naive_b.fit(Train_TFIDF, y_train)
    Train_y_pred =  naive_b.predict(Train_TFIDF)
    Train_AUC_TFIDF.append(roc_auc_score(y_train,Train_y_pred))
    CrossVal_y_pred =  naive_b.predict(CrossVal_TFIDF)
    CrossVal_AUC_TFIDF.append(roc_auc_score(y_cross,CrossVal_y_pred))


# In[ ]:


Alpha_set=[]
for i in range(len(alpha_set)):
    Alpha_set.append(np.math.log(alpha_set[i]))


# In[ ]:


plt.plot(Alpha_set, Train_AUC_TFIDF, label='Train AUC')
plt.scatter(Alpha_set, Train_AUC_TFIDF)
plt.plot(Alpha_set, CrossVal_AUC_TFIDF, label='CrossVal AUC')
plt.scatter(Alpha_set, CrossVal_AUC_TFIDF)
plt.legend()
plt.xlabel("alpha : hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.show()


# In[ ]:


optimal_alpha=alpha_set[CrossVal_AUC_TFIDF.index(max(CrossVal_AUC_TFIDF))]
print(optimal_alpha)


# In[ ]:


Classifier2 = MultinomialNB(alpha=optimal_alpha)
Classifier2.fit(Train_TFIDF, y_train)


# In[ ]:


print ("Accuracy on Train is: ", accuracy_score(y_train,Classifier2.predict(Train_TFIDF)))

print ("Accuracy on Test is: ", accuracy_score(y_test,Classifier2.predict(Test_TFIDF)))


# In[ ]:


from sklearn import metrics
print(metrics.classification_report(y_test,Classifier2.predict(Test_TFIDF)))


# In[ ]:


plot_confusion_matrix(Classifier2, Test_TFIDF, y_test ,display_labels=['0','1'],cmap="Blues",values_format = '')


# In[ ]:





# **Logistic Regression**

# In[ ]:


c=[0.0001,0.001,0.01,0.1,1,10,100,1000]
Train_AUC_TFIDF = []
CrossVal_AUC_TFIDF = []
for i in c:
    logreg = LogisticRegression(C=i,penalty='l2')
    logreg.fit(Train_TFIDF, y_train)
    Train_y_pred =  logreg.predict(Train_TFIDF)
    Train_AUC_TFIDF.append(roc_auc_score(y_train ,Train_y_pred))
    CrossVal_y_pred =  logreg.predict(CrossVal_TFIDF)
    CrossVal_AUC_TFIDF.append(roc_auc_score(y_cross,CrossVal_y_pred))


# In[ ]:


C=[]
for i in range(len(c)):
    C.append(np.math.log(c[i]))


# In[ ]:


plt.plot(C, Train_AUC_TFIDF, label='Train AUC')
plt.scatter(C, Train_AUC_TFIDF)
plt.plot(C, CrossVal_AUC_TFIDF, label='CrossVal AUC')
plt.scatter(C, CrossVal_AUC_TFIDF)
plt.legend()
plt.xlabel("lambda : hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.show()


# In[ ]:


optimal_inverse_lambda=c[CrossVal_AUC_TFIDF.index(max(CrossVal_AUC_TFIDF))]
print(pow(optimal_inverse_lambda,-1))


# In[ ]:


Classifier=LogisticRegression(C=optimal_inverse_lambda,penalty='l2')
Classifier.fit(Train_TFIDF, y_train)


# In[ ]:


print ("Accuracy is: ", accuracy_score(y_test,Classifier.predict(Test_TFIDF)))


# In[ ]:


from sklearn import metrics
print(metrics.classification_report(y_test,Classifier.predict(Test_TFIDF)))


# In[ ]:


plot_confusion_matrix(Classifier, Test_TFIDF, y_test ,display_labels=['0','1'],cmap="Blues")


# 1. **Logistic Regression using L2 Penalty turns out to be the winner with an accuracy of more than 99.6%**
# 2. **Multinomial Naive Bayes also did a commendable job with the classification**

# **When I first tried, I didn't include the 'subject' column and the model performed similar for MNB model, but the Logistic Regression Model could only achieve an accuracy of 90%. I guess the model suffers from overfitting due to the 'subject' feature**

# We had already seen that the presence of 'Subject' was highly inconsistent, the subjects were very target specific which might the reason for overfitting. 
# **I would prefer the Naive Bayes model without the 'Subject' feature as a perfect model.**

# Please upvote and comment

# In[ ]:




