#!/usr/bin/env python
# coding: utf-8

# # Text Classification Using Word Embeddings

# ## Introduction
# We are going to use the Donors Choose dataset to classify text, determining if a teacher's proposal was accepted or rejected. Therefore we will be using binary classification techniques. Transforming the essays into numerical data by using pre-trained word embeddings. You need a pre-trained word embedding dataset, I used the Glove.50d dataset. Word embeddings capture the implicit relations between words by determining how often a word appears to other words in the training documents. In this case the embeddings are trained using articles from wikipedia.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.base import TransformerMixin
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import RandomOverSampler
from sklearn.svm import SVC
from sklearn.svm import LinearSVC


# Now some setup, defining the transformer that transforms the essay data into word embeddings. This will be done word by word, and then the word vectors will be averaged to find the document mean vector.

# In[2]:


class MeanEmbeddingTransformer(TransformerMixin):
    
    def __init__(self):
        self._vocab, self._E = self._load_words()
        
    
    def _load_words(self):
        E = {}
        vocab = []

        with open('../input/glove-global-vectors-for-word-representation/glove.6B.50d.txt', 'r', encoding="utf8") as file:
            for i, line in enumerate(file):
                l = line.split(' ')
                if l[0].isalpha():
                    v = [float(i) for i in l[1:]]
                    E[l[0]] = np.array(v)
                    vocab.append(l[0])
        return np.array(vocab), E            

    
    def _get_word(self, v):
        for i, emb in enumerate(self._E):
            if np.array_equal(emb, v):
                return self._vocab[i]
        return None
    
    def _doc_mean(self, doc):
        return np.mean(np.array([self._E[w.lower().strip()] for w in doc if w.lower().strip() in self._E]), axis=0)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.array([self._doc_mean(doc) for doc in X])
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


# In[3]:


def plot_roc(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    
def print_scores(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('F1 score: {:3f}'.format(f1_score(y_test, y_pred)))
    print('AUC score: {:3f}'.format(roc_auc_score(y_test, y_pred)))


# Loading the dataset using pandas

# In[4]:


train_data = pd.read_csv('../input/donorschoose-application-screening/train.csv', sep=',',
                        dtype={'teacher_prefix': 'category', 'school_state': 'category',
                              'project_grade_category': 'category', 'project_subject_categories': 'category',
                              'project_subject_subcategories': 'category', 'teacher_number_of_previously_posted_projects': np.int32,
                              'project_is_approved': np.bool})
train_data.columns


# In[5]:


train_data.head(3)


# We can see how much memory the dataset uses, it looks like it uses a bit, but small enough, so we'll save the transformed data to a file so it can be loaded faster.

# In[ ]:


train_data.memory_usage(deep=True).sum()


# To determine if the dataset is unbalanced we can count the positive and negative cases. The number of positive cases is over five times the number of negative cases, which is very unbalanced, the AUC for the ROC curve or the F1 score would work fine for this dataset, but not regular accuracy.

# In[12]:


train_data['project_is_approved'].value_counts()


# In[13]:


X = train_data[['project_essay_1', 'project_essay_2']].as_matrix()
y = train_data['project_is_approved'].as_matrix()


# The essays must be tokenized first and then they can be transformed.

# In[9]:


def tokenize_and_transform(X, sample_size):
    essays1 = X[:, 0]
    essays2 = X[:, 1]
    tok_es1 = [word_tokenize(doc) for doc in essays1[:sample_size]]
    tok_es2 = [word_tokenize(doc) for doc in essays2[:sample_size]]
    met = MeanEmbeddingTransformer()
    X_transform = np.append(met.fit_transform(tok_es1), met.fit_transform(tok_es2), axis=1)
    return X_transform


# In[15]:


X_transform = tokenize_and_transform(X, 160000)


# Time to save the vector representation of the essays.

# In[16]:


np.savetxt('X_embed.csv', X_transform, delimiter=',')


# We can later on load them using loadtxt.

# In[ ]:


X_transform = np.loadtxt('X_embed.csv', delimiter=',')


# Standardizing the data will center it, making it more useful for some algorithms.

# In[18]:


X_transform = scale(X_transform)


# To prevent the positive class from dominating the negative, undersampling will be used to equalize the two classes.

# In[20]:


rus = RandomUnderSampler(random_state=0)
X_resample, y_resample = rus.fit_sample(X_transform, y[:X_transform.shape[0]])


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X_resample,
                                                    y_resample, stratify=y_resample, random_state=0)


# Let's test the dataset using default parameters and no preprocessing.

# In[22]:


lr = LogisticRegression()
print_scores(lr, X_train, y_train, X_test, y_test)
plot_roc(lr, X_test, y_test)


# In[ ]:


knn = KNeighborsClassifier()
print_scores(knn, X_train, y_train, X_test, y_test)
plot_roc(knn, X_test, y_test)


# In[ ]:


rf = RandomForestClassifier().fit(X_train, y_train)
y_pred = rf.predict(X_test)
print_scores(rf, X_train, y_train, X_test, y_test)
plot_roc(rf, X_test, y_test)


# In[ ]:


svc = SVC().fit(X_train, y_train)
print_scores(svc, X_train, y_train, X_test, y_test)
plot_roc(svc, X_test, y_test)


# In[ ]:


svc = LinearSVC().fit(X_train, y_train)
print_scores(svc, X_train, y_train, X_test, y_test)
plot_roc(svc, X_test, y_test)


# In[ ]:


dtc = DecisionTreeClassifier().fit(X_train, y_train)
y_pred = dtc.predict(X_test)
print_scores(dtc, X_train, y_train, X_test, y_test)
plot_roc(dtc, X_test, y_test)


# In[ ]:


mlp = MLPClassifier().fit(X_train, y_train)
print_scores(mlp, X_train, y_train, X_test, y_test)
plot_roc(mlp, X_test, y_test)


# Let's now use grid search and 4-fold cross validation to find the best parameters on the efficient models.

# In[ ]:


gs = GridSearchCV(LogisticRegression(), 
             param_grid={'C': [0.0001, 0.001, 0.01, 0.1, 1]}, scoring="roc_auc", cv=4)
gs = gs.fit(X_resample, y_resample)
print(gs.best_params_)
print('best score: {:3f}'.format(gs.best_score_))
plot_roc(gs, X_resample, y_resample)


# Comparing the RBF kernel SVC to the linear SVC there's not much of a performance difference.

# In[ ]:


param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
gs = GridSearchCV(LinearSVC(), 
             param_grid=param_grid, scoring="roc_auc", cv=4)
gs = gs.fit(X_resample, y_resample)
print(gs.best_params_)
print('best score: {:3f}'.format(gs.best_score_))
plot_roc(gs, X_resample, y_resample)


# In[ ]:


param_grid = {'activation': ['relu', 'logistic', 'tanh'],
              'alpha': [0.0001, 0.001, 0.01],
              'learning_rate': ['constant', 'invscaling', 'adaptive'], 'tol': [0.01]}
gs = GridSearchCV(MLPClassifier(), 
             param_grid=param_grid, scoring="roc_auc", cv=4)
gs = gs.fit(X_transform, y[:150000])
print(gs.best_params_)
print('best score: {:3f}'.format(gs.best_score_))
plot_roc(gs, X_resample, y_resample)


# First lets try clustering using KMeans, to see if that improves the score.

# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


X_transform_cluster = KMeans(n_clusters=10).fit_transform(X_transform, y[:150000])


# In[ ]:


rus = RandomUnderSampler(random_state=0)
X_resample_cluster, y_resample_cluster = rus.fit_sample(X_transform_cluster, y[:X_transform_cluster.shape[0]])


# In[ ]:


X_train_cluster, X_test_cluster, y_train_cluster, y_test_cluster = train_test_split(X_resample_cluster,
                                                    y_resample_cluster, stratify=y_resample_cluster, random_state=0)


# In[ ]:


lr = LogisticRegression()
print_scores(lr, X_train_cluster, y_train_cluster, X_test_cluster, y_test_cluster)
plot_roc(lr, X_test_cluster, y_test_cluster)


# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


X_transform_pca = PCA().fit_transform(X_transform, y[:150000])


# In[ ]:


rus = RandomUnderSampler(random_state=0)
X_resample_pca, y_resample_pca = rus.fit_sample(X_transform_pca, y[:X_transform_pca.shape[0]])


# In[ ]:


X_train_cluster, X_test_cluster, y_train_cluster, y_test_cluster = train_test_split(X_resample_cluster,
                                                    y_resample_cluster, stratify=y_resample_cluster, random_state=0)


# In[ ]:


lr = LogisticRegression()
print_scores(lr, X_train_cluster, y_train_cluster, X_test_cluster, y_test_cluster)
plot_roc(lr, X_test_cluster, y_test_cluster)


# In[ ]:


test_data = pd.read_csv('../input/donorschoose-application-screening/test.zip')


# In[ ]:


test_data.head(3)


# Now we'll load the test dataset and predict on that.

# In[ ]:


X = test_data[['project_essay_1', 'project_essay_2', 'teacher_number_of_previously_posted_projects']].as_matrix()


# In[ ]:


X_transform = tokenize_and_transform(X, X.shape[0])


# In[ ]:


X_transform = scale(X_transform)


# In[ ]:


X_transform = np.append(X_transform , np.transpose([X[:, 2]]), axis=1)


# Now to predict the probabilities.

# In[ ]:


y_pred = gs.predict_proba(X_transform)[:, 1]


# In[ ]:


out_data = np.append(test_data[['id']].as_matrix(), np.transpose([y_pred]), axis=1)


# In[ ]:


out_data.shape


# In[ ]:


np.savetxt('submission.csv', out_data, fmt='%s, %f', delimiter=',')

