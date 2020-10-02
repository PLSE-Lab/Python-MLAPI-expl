import re
import pandas as pd
import numpy as np
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

df=pd.read_csv('../input/spam.csv', encoding='latin-1')
df=df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis='columns')
stemmer=SnowballStemmer('english')
stop=set(stopwords.words('english'))
df['v2']=[re.sub('[^a-zA-Z]', ' ', sms) for sms in df['v2']]
word_list=[sms.split() for sms in df['v2']]
def normalize(words):
    current_words=list()
    for word in words:
        if word.lower() not in stop: 
            updated_word=stemmer.stem(word)
            current_words.append(updated_word.lower())
    return current_words
word_list=[normalize(word) for word in word_list]
df['v2']=[" ".join(word) for word in word_list]
x_train, x_test, y_train, y_test=train_test_split(df['v2'], df['v1'], test_size=0.2, random_state=7)
cv=CountVectorizer()
x_train_df=cv.fit_transform(x_train)
x_test_df=cv.transform(x_test)
clf=MultinomialNB()
clf.fit(x_train_df,y_train)
prediction=clf.predict(x_test_df)
conf_mat=confusion_matrix(y_test, prediction)
print(conf_mat)
print("Accuracy:"+str(accuracy_score(y_test,prediction)))
print(classification_report(y_test, prediction, target_names=["Ham", "Spam"]))