from __future__ import division
import sqlite3, time, csv, re, random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

'''

Pre-fork:

This script was inspired by smerity's script "The Biannual Reddit Sarcasm
Hunt." A natural follow-up question is whether we can detect posts with the
/s flag using a BOW model. 

The corpus has about 54m posts, of which about 30k have the /s flag. It is 
impossible to compete with a majority baseline that strong, so instead I've
framed it as a binary classification task with uniform class distribution.
Realistic, no, but enough to see some pronounced trends in the features. A
logistic regression model scores about 72% on unseen data.

Lots of work has been done on irony detection, here are a couple references:
Bamman, Contextualized Sarcasm Detection on Twitter, ICWSM 2015
Wallace, Humans Require Context to Infer Ironic Intent (so Computers 
Probably do, too), ACL 2014

Post-fork premises:

The uniform class distribution is not the only simplifying assumption in the
pre-forked version. Another is that the items marked "true negatives" may only
be "possible true negatives" -- that is to say: the sarcasm tag may have been
mistakenly ommitted by the author.

We can get a better understanding of this class of "error" with the help of some
follow-on analytics:

   * The confusion matrix shows no assymmetry between "false positives" and
     "false negatives" -- but we know: many have been mis-identified as
     negatives in the gound truth; true positives are 100% correct.
   * Sampling additional SRS examples should show a range of probabilities.
   * Sampling additional SRS examples should show a non-trivial number of
     SAR predictions.

Post-fork observations:

    * When applied to 500K+ unseen "true serious" examples, the two most common
      prediction probability bins are p < 1% and 49% < p < 50% (or 'SERIOUS!!!'
      and 'serious?', respectively)
    * Reviews of both classes of "errors" is likely to reveal mis-classified
      ground truth as a significant source of error -- although this can
      probably only be verified with the help of the original authors

'''

print("Setting query parameters ...")
lmt         = 30100    # sarcastic/serious posts to train on
test_size   = 50000    # serious posts to report prediction probabilities on
srs_size    = lmt + test_size
total_size = 2*lmt + test_size
print("\tSarcasm size: " + str(lmt))
print("\tTest size: " +  str(test_size))
print("\tSerious size: " +  str(srs_size))
print("\tTotal size: " + str(total_size))

print("Connecting to DB ...")
sql_conn = sqlite3.connect('../input/database.sqlite')
all_probabilities = [ ]

for iteration in range(0,10):
    print("Starting iteration " + str(iteration))
    offset = iteration*total_size  # get a fresh set of serious examples
    
    print("Querying DB ...")
    query_stem  = "SELECT body FROM May2015 WHERE body"
    sarcasmData = sql_conn.execute(query_stem + " LIKE '% /s' LIMIT " + str(lmt))
    seriousData = sql_conn.execute(query_stem + " NOT LIKE '%/s%' LIMIT " + str(srs_size) + " OFFSET " + str(offset))
    
    print("Building corpora ...")
    corpus = []
    for sar_post in sarcasmData:
        cln_post = re.sub('/s|\n', '', sar_post[0])          # Remove /s and newlines
        corpus.append(re.sub(r'([^\s\w]|_)+', '', cln_post)) # and then non-alpha
    for srs_post in seriousData:
        cln_post = re.sub('\n', '', srs_post[0])             # Remove newlines
        corpus.append(re.sub(r'([^\s\w]|_)+', '', cln_post)) # and then non-alpha
        
    print("\tCorpus size: " + str(len(corpus)))
    print("Vectorizing features ...")
    vec = TfidfVectorizer(min_df=5)
    td_matrix = csr_matrix(vec.fit_transform(corpus).toarray())  # feature vectors for test and train
    
    labels = [1]*lmt+[0]*srs_size                                # labels for both test and train
    print("Built labels: " + str(len(labels)) + " added ...")
    print("Partitioning vectorized data ...")
    
    ## Make an evenly balanced training corpus (72% accuracy when tested with 1/3 held out):
    X_train = td_matrix[:(2*lmt)]
    y_train = labels[:(2*lmt)]
    
    ## Retrieve the feature vectors for the items to be tested
    X_test  = td_matrix[(2*lmt):]
    
    print("Buidling model ...")    
    clf = LogisticRegression(C=1.25)
    clf.fit(X_train, y_train)
    print("Model classes: ", clf.classes_)  ## 0 and 1, so "sarcastic" is second
    
    print("Checking prediction probabilities ...")
    y_test = [ y[1] for y in clf.predict_proba(X_test) ]
    
    print("Avg Probability: " + str((sum(y_test)/len(y_test))) + " for " + str(len(y_test)) + " examples")
    print("Probability at 72 pctile: ", sorted(y_test)[int(0.72 * len(y_test))])
    all_probabilities.extend(y_test)

print("Probability at 72 pctile (all iterations): ", sorted(y_test)[int(0.72 * len(y_test))])
plt.hist(all_probabilities, 100)
plt.xlabel('Sarcasm Predicition Probability')
plt.ylabel('Probability Likelihood')
plt.show()
