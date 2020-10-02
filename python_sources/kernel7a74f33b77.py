import numpy as np
import pandas as pd 
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

data_train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
data_test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

# Removing null values column
total = data_train.isnull().sum().sort_values(ascending=False)
percent = (data_train.isnull().sum()/data_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
data_train = data_train.drop(['location','keyword','id'], axis=1)
print("Null Values Removed")

#Corpus - Text Pre-processing
corpus=[]
pstem = PorterStemmer()
for i in range(data_train['text'].shape[0]):
    tweet = re.sub("[^a-zA-Z]",' ',data_train['text'][i])
    tweet = tweet.lower()
    tweet = tweet.split()
    tweet = [pstem.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)
    corpus.append(tweet)
print("Corpus Done !")

uniqueWordFrequents = {}
for tweet in corpus:
    for word in tweet.split():
        if(word in uniqueWordFrequents.keys()):
            uniqueWordFrequents[word] += 1
        else:
            uniqueWordFrequents[word] = 1
            
uniqueWordFrequents = pd.DataFrame.from_dict(uniqueWordFrequents,orient='index',columns=['Word Frequent'])
uniqueWordFrequents.sort_values(by=['Word Frequent'], inplace=True, ascending=False)
uniqueWordFrequents = uniqueWordFrequents[uniqueWordFrequents['Word Frequent'] >= 20]

print("Unique Words Count")

counVec = CountVectorizer(max_features = uniqueWordFrequents.shape[0])
bow = counVec.fit_transform(corpus).toarray()
X = bow
y = data_train['target']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20,random_state = 55,shuffle = True)

print("Vectorization And Splitting Done")

SVClassifier = SVC(kernel= 'linear',
                   degree=3,
                   max_iter=10000,
                   C=2,
                   random_state = 55)
SVClassifier.fit(X_train,y_train)

print("Model Prepared")
print(type(SVClassifier).__name__,' Train Score is   : ' ,SVClassifier.score(X_train, y_train))
print(type(SVClassifier).__name__,' Test Score is    : ' ,SVClassifier.score(X_test, y_test))
y_pred = SVClassifier.predict(X_test)
print(type(SVClassifier).__name__,' F1 Score is      : ' ,f1_score(y_test,y_pred))
print('--------------------------------------------------------------------------')


counVec = CountVectorizer(max_features = uniqueWordFrequents.shape[0])
bow = counVec.fit_transform(data_test['text']).toarray()
testData = bow

results = SVClassifier.predict(testData)
print(type(results))

submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
submission.target = results
submission.to_csv('sample_submission.csv', index = False)