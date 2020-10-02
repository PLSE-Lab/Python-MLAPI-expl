#!/usr/bin/env python
# coding: utf-8

# In other words we try to find questions which were asked before looking for similar questions. A run of the mill classification attempt.
# 
# 1. Import everything we possibly can
# 2. Represent as Vectors
# 3. Classify
# 
# We begin with a showing what the dataset comprises of before we classify.
# 
# ## Exploration

# In[ ]:


import nltk
import scipy
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
get_ipython().run_line_magic('pylab', 'inline')

stop = set(nltk.corpus.stopwords.words('english'))
from tqdm import tqdm
from itertools import chain
from functools import reduce

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.decomposition import PCA

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


# Let's read in the Data. Since it is big enough, we drop the NAs right away
df = pd.read_csv('../input/questions.csv').dropna()
df.info()


# In[ ]:


print(df.is_duplicate.mean(), 'Label skew', df.is_duplicate.sum(), 'ones present')


# In[ ]:


# Let's speed things up a bit. We'll also get rid of the skew in the mean while
# Comment out this block if you want to run the code on the entire dataset.
n_per_label = 50000  # This is ~ 33% of the minorty label

ones = df[df.is_duplicate == 1]
zero = df[df.is_duplicate == 0]
df = pd.concat([ones.sample(n_per_label), zero.sample(n_per_label)])
df = df.sample(df.shape[0]) # Shuffle the data
df.info()


# ## Creating Vectors

# In[ ]:


lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
def clean_sentence(sentence):
    "POS tag the sentence and then lemmatize the words."
    global lemmatizer
    words = nltk.tokenize.word_tokenize(sentence.lower())
    pos_tagged = nltk.pos_tag(words)
    pos_tagged_stems = [(lemmatizer.lemmatize(i[0]), i[1]) for i in pos_tagged]
    return pos_tagged_stems

tqdm.pandas(mininterval=15, ncols=80, desc='Question1')
df['q1_clean'] = df.question1.progress_apply(clean_sentence)
tqdm.pandas(mininterval=15, ncols=80, desc='Question2')
df['q2_clean'] = df.question2.progress_apply(clean_sentence)


# In[ ]:


# A list of all the tags generated for this dataset
tag_set = set()
for tags in chain(df.q1_clean.apply(lambda x: set([i[1] for i in x])),
                  df.q2_clean.apply(lambda x: set([i[1] for i in x]))):
    tag_set = tag_set.union(tags)
tag_set = list(filter(lambda x: x.isalpha(), tag_set))
print(len(tag_set), 'Tags present.', len(tag_set)*3, 'is the approximate number of features expected')
print(tag_set)


# In[ ]:


vectors = []
for q1, q2 in tqdm(zip(df.q1_clean, df.q2_clean),
                   ncols=80,
                   total=df.shape[0],
                   mininterval=20,
                   desc='Generating Vectors'
                  ):
    vec, jaccards = [], []
    for tag in tag_set:
        words1 = set([i[0] for i in q1 if i[1] == tag])
        words2 = set([i[0] for i in q2 if i[1] == tag])
        # Jaccard index
        intersection = len(words1.intersection(words2)) / (len(words1.union(words2)) + 1)
        # Append to this vector
        vec.append(len(words1)); vec.append(len(words2))
        vec.append(intersection); jaccards.append(intersection)
        
    jaccards = np.array(jaccards)
    vec.extend([jaccards.sum(), jaccards.min(), jaccards.max(), jaccards.mean(), jaccards.std()])
    vectors.append(vec)
vectors = np.array(vectors)
print(vectors.shape, 'Samples, Dimension')


# ## Descriptions of our Vectors

# In[ ]:


cols = reduce(lambda x, y: x + y,
             [[i+'-q1', i+'-q2', i+'-jac'] for i in tag_set])
cols += ['.-jSum', '.-jMin', '.-jMax', '.-jMean', '.-jStd']

estimator = RandomForestClassifier(n_jobs=-1, n_estimators=10)
X, Y = vectors, df.is_duplicate
estimator.fit(X, Y)


# In[ ]:


temp = list(zip(estimator.feature_importances_, cols))
temp.sort(reverse=True)
for imp, feature in temp:
    f1, f2 = feature.split('-')
    print('{:5} {:4} {}'.format(round(imp, 3), f1, f2))


# In[ ]:


pca = PCA()
pca.fit(X)
with plt.style.context(('ggplot')):
    plt.plot(np.cumsum(pca.explained_variance_ratio_), '.-')
    plt.xlabel('N- components')
    plt.ylabel('Cumulative variance explained')


# ## Classification

# In[ ]:


estimator = RandomForestClassifier(n_jobs=-1, n_estimators=300)
X, Y = vectors, df.is_duplicate


# In[ ]:


pred = cross_val_predict(estimator, X, Y, cv=10, verbose=10)


# In[ ]:


print(classification_report(Y, pred))
print('Accuracy:', accuracy_score(Y, pred))
print('ROC AUC:', roc_auc_score(Y, pred))
print('Confusion Matrix')


# In[ ]:


cm = confusion_matrix(Y, pred)
cm =cm / cm.sum(axis=1)
cm = pd.DataFrame(cm)
cm.columns=['Not Duplicate', 'Duplicate']
cm.index = ['Not Duplicate', 'Duplicate']
sns.heatmap(cm,
            annot=True,cmap='Reds', linecolor='black', linewidth=1, square=True)
plt.title('Quora Questions Duplication Prediction')


# With a sample size of 100k those are pretty good results I think. I wish there was a benchmark to compare against. 
# 
# Things we've Done
# ------
# We use a Random Forest as the estimator of choice for now as this family has shown itself as a strong contender in many classification tasks. We might look for other classifiers later on.
# 
# - Balance the label imbalance by under sampling the majority.
# - Vector representation of text
#     - Sparse Bag of Words as vectors = **~ 0.61 to ~0.64 f1**
#     - Percentage Common words as vectors = No improvement
#     - Percentage Common stemmed words = No improvement
#     - Similarity measure based on [this paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2007/01/ECIR07-MetzlerDumaisMeek-Final.pdf) = **~0.68 to 0.69 f1**
#     - Jaccard index
#         - POS tagged words for each tag  =**~ 0.71 f1**
#         - +Lemmatized words = **unstable improvement**
#         - +jaccard aggregate metrics for each vector. =  **~0.75 f1**
#         - +PCA **Increased training time**
# 
# To do
# -------
# 
# - I've read about Dense Cohort of Terms. Will attempt to locate the paper and implement.
