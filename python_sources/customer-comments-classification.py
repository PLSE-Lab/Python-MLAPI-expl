# Kernel written by : Jihed DEROUICHE (Classic ML model exploration)
# Data set: Customer Comments in correlation Clothing E-commerce reviewing 
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import gc
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import train_test_split
# import data
df=pd.read_csv('../input/Womens Clothing E-Commerce Reviews.csv')
#df.isnull().sum()
df=df.fillna({'Title':'none'})
df=df.fillna({'Review Text':''})
df=df.fillna({'Division Name':''})
df=df.fillna({'Department Name':''})
df=df.fillna({'Class Name':''})
#df1=df['Division Name'].value_counts().to_frame()
#df1.head()
# Just for visualization // Establish the most rating topics 
aggregat1=df.groupby(['Division Name'], as_index=False).agg({'Rating': np.average}).sort_values(by='Rating', ascending=False) 
aggregat2=df.groupby(['Department Name'], as_index=False).agg({'Rating': np.average}).sort_values(by='Rating', ascending=False)
aggregat3=df.groupby(['Class Name'], as_index=False).agg({'Rating': np.average}).sort_values(by='Rating', ascending=False)
#Assign True to 4<=Rating<=5 and False to 1<=Rating<=2 // we Admit that 3 is a neutral comment 
df=df[df.Rating!=3] 
df['Sentiment']=df['Rating']>=4  
#df.head()
vector_word=TfidfVectorizer(
    max_features=9000, 
    stop_words='english',
    strip_accents='unicode',
    token_pattern=r'\w{1,}',
    ngram_range=(1,1),
    analyzer='word',lowercase=True,
    sublinear_tf=True)
vector_char=TfidfVectorizer(
    max_features=50000,
    stop_words='english',
    strip_accents='unicode',
    ngram_range=(2,6),
    analyzer='char',
    lowercase=True,
    sublinear_tf=True)  
train_data, test_data=train_test_split(df,test_size=0.1)
train_word_vect=vector_word.fit_transform(train_data['Review Text'])
train_char_vect=vector_char.fit_transform(train_data['Review Text'])
test_word_vect=vector_word.fit_transform(test_data['Review Text'])
test_char_vect=vector_char.fit_transform(test_data['Review Text'])
X_train=hstack([train_word_vect,train_char_vect])
X_test=hstack([test_word_vect,test_char_vect])
Y_train=train_data['Sentiment']
Y_test=test_data['Sentiment']
#Second attempt with Naive Bayes
#classifier=MultinomialNB()
scores=[]
classifier=LogisticRegression(C=0.2, solver='sag')
#classifier=MultinomialNB()
# using 3-fold cross validation and roc_auc scoring
crossv_score=np.mean(cross_val_score(classifier, X_train,Y_train, cv=3, scoring='roc_auc'))
scores.append(crossv_score)
print('CV score for class is {} '.format(crossv_score))
classifier.fit(X_train, Y_train)
print(scores) 
print('the total score of this dataset is {}'.format(np.mean(scores)))
# the mean scores is equivalent to 0.9568 which can be improved by exploring other hyperparameters and ML model
