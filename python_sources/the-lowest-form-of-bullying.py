from __future__ import division
import sqlite3, time, csv, re, random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix

'''
This script was inspired by Dave Fernig's script "the lowest form of sarcasm." 

The corpus has about 54m posts. I have framed it as a binary classification 
task with uniform class distribution. A logistic regression model scores 
about 72% on unseen data.

'''

#Parameters
srs_lmt = 30100 #serious posts to train on
sar_lmt = 30100 #sarcastic posts to train on
top_k = 30 #features to display
num_ex = 20 #examples displayed per feature
min_ex = 0 #shortest example displayed
max_ex = 120 #longest example displayed
ovr_ex = True #display longer/shorter examples if we run out

print('Querying DB...\n')
sql_conn = sqlite3.connect('../input/database.sqlite')

sarcasmData = sql_conn.execute("SELECT subreddit, body, score FROM May2015\
                                WHERE controversiality == 1\
                                LIMIT " + str(sar_lmt))

seriousData = sql_conn.execute("SELECT subreddit, body, score FROM May2015\
                                WHERE controversiality == 0\
                                LIMIT " + str(srs_lmt))

print('Building Corpora\n')
corpus, raw_corpus, srs_corpus = [], [], []

for sar_post in sarcasmData:
    raw_corpus.append(re.sub('\n', '', sar_post[1]))
    cln_post = re.sub('/s|\n', '', sar_post[1]) #Remove /s and newlines
    corpus.append(re.sub(r'([^\s\w]|_)+', '', cln_post)) #and then non-alpha

for srs_post in seriousData:
    srs_corpus.append(re.sub('\n', '', srs_post[1]))
    cln_post = re.sub('\n', '', srs_post[1]) #Remove newlines
    corpus.append(re.sub(r'([^\s\w]|_)+', '', cln_post)) #and then non-alpha

print('Fitting TF-IDF and Classifier\n')

# from sklearn import svm
# clf = svm.SVC(kernel = 'rbf', C = 10000)
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
# vec = TfidfVectorizer(min_df=5)
vec, clf = TfidfVectorizer(min_df=5, stop_words = 'english'), LogisticRegression()

td_matrix = csr_matrix(vec.fit_transform(corpus).toarray())
labels = [1]*sar_lmt+[-1]*srs_lmt
X_train, X_test, y_train, y_test = train_test_split(td_matrix, labels, 
                                  test_size=0.20, random_state=42)

clf.fit(X_train, y_train)
y_out = clf.predict(X_test)

print("Accuracy on held-out data: ",\
      str(100*accuracy_score(y_out, y_test))[0:5], "%\n")

X_train = y_train = X_test = y_test = y_out = None

print('Folding held-out data back into the training set\n')
clf.fit(td_matrix, labels)

#See what features were informative
feature_weights, feature_names = clf.coef_[0], vec.get_feature_names()
sar_indices = feature_weights.argsort()[-top_k:][::-1]

print("The", top_k, "most informative words for predicting controversiality on reddit:\n")
for k in range(0, top_k):
    
    feature = feature_names[sar_indices[k]]
    all_examples = [post for post in raw_corpus 
                    if ' '+feature+' ' in post]
                    
    srs_examples = [post for post in srs_corpus 
                    if ' '+feature+' ' in post]
    
    print("Feature", str(k+1),':', '"'+feature+'"',\
          "(Appears", len(all_examples), "times controversiality", len(srs_examples), "not controversiality")
    print("Examples:")
    examples = [post for post in all_examples 
                if len(post) <= max_ex and len(post) >= min_ex]
    
    in_range = len(examples)
    extra_examples = [post for post in all_examples if post not in examples]
    random.shuffle(examples)
    extra_examples.sort(key = lambda s: len(s))
    examples += extra_examples

    for i in range(0, min(num_ex, len(examples))):
        if in_range > i or ovr_ex:
            print(str(i+1), ':', examples[i])
    print('')        
