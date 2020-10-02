
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def Lemm(txt):
    lemmatizer = WordNetLemmatizer()
    lem = []
    for word, tag in txt:
        wntag = get_wordnet_pos(tag)
        if wntag is None:# not supply tag in case of None
            lemma = lemmatizer.lemmatize(word) 
        else:
            lemma = lemmatizer.lemmatize(word, pos=wntag)
            
        lem.append(lemma)

    return lem

def ProcessText(sms):
    sms = word_tokenize(sms)
    sms = [w for w in sms if not w in stopwords.words('english')]
    sms = [s for s in sms if s.isalpha()]
    sms = nltk.pos_tag(sms)
    sms = Lemm(sms)

    separator =' '
    sms_str = separator.join(sms)

    return sms_str




df = pd.read_csv('../input/sms-spam-collection-dataset/spam.csv', encoding='latin-1')
df.rename(columns={'v1':'Class','v2':'SMS'},inplace=True)
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
# print(df.head())

df['SMS_Cleaned'] = df['SMS'].apply(lambda x : ProcessText(x))
df['Length'] = df['SMS'].apply(lambda x: len(x))
df.drop(columns=['SMS'],inplace=True)

df['Class'] = df['Class'].map({'ham':0,'spam':1})

X = df['SMS_Cleaned'].values
y = df['Class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train).toarray()
X_test_cv = cv.transform(X_test).toarray()

# tfidf = TfidfVectorizer()
# X_train_tfidf = tfidf.fit_transform(X_train).toarray()
# X_test_tfidf = tfidf.transform(X_test).toarray()

# clf = LogisticRegression(random_state=100)
clf = MultinomialNB()
model = clf.fit(X_train_cv,y_train)

prediction = model.predict(X_test_cv)

print(classification_report(y_test,prediction))
print()
print("Accuracy: {}".format(accuracy_score(y_test,prediction)))
print()
print(confusion_matrix(y_test,prediction))
