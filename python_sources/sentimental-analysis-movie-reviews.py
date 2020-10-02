import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer
import re

dataset = pd.read_csv('train.tsv',delimiter='\t')

text = dataset['Phrase']
stop = stopwords.words('english')
stop.remove('not')

data =[]
for i in range(len(dataset)):
    review = text[i]
    #convertig letters into  lowercase 
    review =review.lower()
    # removing special characters and numerics 
    review = re.sub('[^a-zA-Z]',' ',review)
    # converting sentence into words(tokenizing)
    review = review.split()
    #stemming (finding root word)
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in stop]
    
    # join the words
    review = ' '.join(review)
    data.append(review)
data


cv = CountVectorizer()
X= cv.fit_transform(data).toarray()
y =dataset['Sentiment'].values

print(cv.get_feature_names())

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
X_train.shape,X_test.shape,y_train.shape,y_test.shape

model_M = MultinomialNB() 
model_G = GaussianNB()
model_M.fit(X_train,y_train)
model_G.fit(X_train,y_train)
y_pred_M = model_M.predict(X_test)
y_pred_G = model_G.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report
cm_M = confusion_matrix(y_test,y_pred_M)
cm_G = confusion_matrix(y_test,y_pred_G)
cr_M  = classification_report(y_test,y_pred_M)
cr_G = classification_report(y_test,y_pred_G)

import seaborn as sns 
import matplotlib.pyplot as plt 

sns.heatmap(cm_M,cmap='summer')
plt.show()

sns.heatmap(cm_G,cmap='summer')
plt.show()

print(cr_M)
print(cr_G)

reviews = []
dataset2 = pd.read_csv('test.tsv',delimiter='\t')
test =dataset2['Phrase']
for i in range(len(dataset2)):
    review = test[i] 
    review=review.lower()
    review = re.sub('[^a-zA-Z]',' ',review)
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word)  for word in review if not word in set(stop)]
    review = ' '.join(review)
    reviews.append(review)
    
length=len(reviews)

predicted_values=[]
test_arr=[]

for i in range(0,length):
    test_arr.append(cv.transform([reviews[i]]))
    predicted_values.append(model_M.predict(test_arr[i]))
    
for i in range(0,length):
    print(i+1,predicted_values[i])



