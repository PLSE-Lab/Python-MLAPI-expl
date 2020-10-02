

#EDA
import pandas as pd
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from collections import Counter

# Reading the data
train = pd.read_json('train.json')
test = pd.read_json('test.json')

stemmer = WordNetLemmatizer()

def clean_recipe(recipe):
    # To lowercase
    recipe = [ str.lower(i) for i in recipe ]

    def replacing(i):
        i = i.replace('&', '').replace('(', '').replace(')','')
        i = i.replace('\'', '').replace('\\', '').replace(',','')
        i = i.replace('.', '').replace('%', '').replace('/','')
        i = i.replace('"', '')
        
        return i
    
    # Replacing characters
    recipe = [ replacing(i) for i in recipe ]
    
    # Remove digits
    recipe = [ i for i in recipe if not i.isdigit() ]
    
    # Stem ingredients
    recipe = [ stemmer.lemmatize(i) for i in recipe ]
    
    return recipe

bags_of_words = [ Counter(clean_recipe(recipe)) for recipe in train.ingredients ]
sumbags = sum(bags_of_words, Counter())

plt.style.use(u'ggplot')
fig = pd.DataFrame(sumbags, index=[0])

fig.transpose()[0].sort_values(ascending=False, inplace=False)[:10].plot(kind='barh')
fig = fig.get_figure()
fig.tight_layout()
fig.savefig('top_10_ingredients.jpg')

#naive bayes

from sklearn.naive_bayes import MultinomialNB as nb

train_file = open("train.json")
training_data = json.load(train_file)
train_file.close()
test_file = open("test.json")
test_data = json.load(test_file)
test_file.close()

ingredients = set()
cuisines = set()
recipe_ids_train = list()
recipe_ids_test = list() 

for example in training_data:
    recipe_ids_train.append(example['id'])
    cuisines.add(example['cuisine'])
    for item in example['ingredients']:
        ingredients.add(item)

for example in test_data:
    for item in example['ingredients']:
        ingredients.add(item)

ingredients = list(ingredients)
ingredients = dict([(ingredient, i) for i, ingredient in enumerate(ingredients)])
vec_size = len(ingredients)
cuisines = list(cuisines)
cuisines = dict([(cuisine, i) for i, cuisine in enumerate(cuisines)])
num_to_cuisine = {val: key for key, val in cuisines.items()}

X_list = []
X_test_list = []
y_list = []

for example in training_data:
    ingredients_vector_list = [0 for x in range(vec_size)]
    for ingredient in example['ingredients']:
        dimension = ingredients[ingredient]
        ingredients_vector_list[dimension] = 1.0
    X_list.append(ingredients_vector_list)

    cuisine_int = cuisines[example['cuisine']]
    y_list.append(cuisine_int)

X = np.array(X_list)
y = np.array(y_list)

for example in test_data:
    recipe_ids_test.append(example['id'])
    ingredients_vector_list = [0 for x in range(vec_size)]
    for ingredient in example['ingredients']:
        dimension = ingredients[ingredient]
        ingredients_vector_list[dimension] = 1.0
    X_test_list.append(ingredients_vector_list)

X_test = np.array(X_test_list)

# Train classifier
classifier = nb()
classifier = classifier.fit(X, y)

classifier.score(X, y)

y_pred = classifier.predict(X)

y_pred_test_nb = classifier.predict(X_test)
y_pred_test_nb = [num_to_cuisine[x] for x in y_pred_test_nb]

out_nb = open('naivebayes.csv','w')
out_nb.write('id,cuisine\n')

for x in range(len(recipe_ids_test)):
    out_nb.write(str(recipe_ids_test[x]) + ',' + y_pred_test_nb[x] + '\n')
out_nb.close()

import pickle
pickle.dump(classifier, open("naivebayes.pickle.dat", "wb"))

#linear SVC
import nltk
import re
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import grid_search
from sklearn.linear_model import LogisticRegression
import pickle
stopwords = nltk.corpus.stopwords.words('english')
traindf = pd.read_json("train.json")
traindf['ingredients_clean_string'] = [' , '.join(z).strip() for z in traindf['ingredients']]  
traindf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in traindf['ingredients']]       

testdf = pd.read_json("test.json") 
testdf['ingredients_clean_string'] = [' , '.join(z).strip() for z in testdf['ingredients']]
testdf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in testdf['ingredients']]       



corpustr = traindf['ingredients_string']
vectorizertr = TfidfVectorizer(stop_words=stopwords,
                             ngram_range = ( 1 , 1 ),analyzer="word", 
                             max_df = .75 , binary=False , token_pattern=r'\w+' , sublinear_tf=False)
tfidftr=vectorizertr.fit_transform(corpustr).todense()
corpusts = testdf['ingredients_string']
vectorizerts = TfidfVectorizer(stop_words='english')
tfidfts=vectorizertr.transform(corpusts)

predictors_tr = tfidftr

targets_tr = traindf['cuisine']

predictors_ts = tfidfts

parameters = {'C':[1, 10]}
clf = LinearSVC()

classifier = grid_search.GridSearchCV(clf, parameters)

classifier=classifier.fit(predictors_tr,targets_tr)
classifier.score(predictors_tr, targets_tr)

#ensemble method
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn import grid_search
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np


class EnsembleClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
   
    def __init__(self, clfs, voting='hard', weights=None, verbose=0):

        self.clfs = clfs
        self.named_clfs = {key: value for key, value in _name_estimators(clfs)}
        self.voting = voting
        self.weights = weights
        self.verbose = verbose

    def fit(self, X, y):
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output'
                                      ' classification is not supported.')

        if self.voting not in ('soft', 'hard'):
            raise ValueError("Voting must be 'soft' or 'hard'; got (voting=%r)"
                             % self.voting)

        if self.weights and len(self.weights) != len(self.clfs):
            raise ValueError('Number of classifiers and weights must be equal'
                             '; got %d weights, %d clfs'
                             % (len(self.weights), len(self.clfs)))

        self.le_ = LabelEncoder()
        self.le_.fit(y)
        self.classes_ = self.le_.classes_
        self.clfs_ = [clone(clf) for clf in self.clfs]

        if self.verbose > 0:
            print("Fitting %d classifiers..." % (len(self.clfs)))

        for clf in self.clfs_:

            if self.verbose > 0:
                i = self.clfs_.index(clf) + 1
                print("Fitting clf%d: %s (%d/%d)" %
                      (i, _name_estimators((clf,))[0][0], i, len(self.clfs_)))

            if self.verbose > 2:
                if hasattr(clf, 'verbose'):
                    clf.set_params(verbose=self.verbose - 2)

            if self.verbose > 1:
                print(_name_estimators((clf,))[0][1])

            clf.fit(X, self.le_.transform(y))
        return self

    def predict(self, X):
        if self.voting == 'soft':

            maj = np.argmax(self.predict_proba(X), axis=1)

        else:  # 'hard' voting
            predictions = self._predict(X)

            maj = np.apply_along_axis(
                                      lambda x:
                                      np.argmax(np.bincount(x,
                                                weights=self.weights)),
                                      axis=1,
                                      arr=predictions)

        maj = self.le_.inverse_transform(maj)
        return maj

    def predict_proba(self, X):
        avg = np.average(self._predict_probas(X), axis=0, weights=self.weights)
        return avg

    def transform(self, X):
        if self.voting == 'soft':
            return self._predict_probas(X)
        else:
            return self._predict(X)

    def get_params(self, deep=True):
        """ Return estimator parameter names for GridSearch support"""
        if not deep:
            return super(EnsembleClassifier, self).get_params(deep=False)
        else:
            out = self.named_clfs.copy()
            for name, step in six.iteritems(self.named_clfs):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            return out

    def _predict(self, X):
        """ Collect results from clf.predict calls. """
        return np.asarray([clf.predict(X) for clf in self.clfs_]).T

    def _predict_probas(self, X):
        """ Collect results from clf.predict calls. """
        return np.asarray([clf.predict_proba(X) for clf in self.clfs_])
 
traindf = pd.read_json("train.json")

traindf['ingredients_clean_string'] = [' , '.join(z).strip() for z in traindf['ingredients']]
traindf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in traindf['ingredients']]       

testdf = pd.read_json("test.json") 
testdf['ingredients_clean_string'] = [' , '.join(z).strip() for z in testdf['ingredients']]
testdf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in testdf['ingredients']]       
corpustr = traindf['ingredients_string']
vectorizertr = TfidfVectorizer(stop_words='english', ngram_range = ( 1, 1),analyzer="word", 
                             max_df = .6 , binary=False , token_pattern=r'\w+' , sublinear_tf=False, norm = 'l2')
tfidftr = vectorizertr.fit_transform(corpustr).todense()
corpusts = testdf['ingredients_string']
vectorizerts = TfidfVectorizer(stop_words='english', ngram_range = ( 1, 1),analyzer="word", 
                             max_df = .6 , binary=False , token_pattern=r'\w+' , sublinear_tf=False, norm = 'l2')

tfidfts = vectorizertr.transform(corpusts)

predictors_tr = tfidftr

targets_tr = traindf['cuisine']

predictors_ts = tfidfts

np.random.seed(5)
print("Ensemble: LR - linear SVC")
clf1 = LogisticRegression(random_state=1, C=7)
clf2 = LinearSVC(random_state=1, C=0.4, penalty="l2", dual=False)
nb = BernoulliNB()
rfc = RandomForestClassifier(random_state=1, criterion = 'gini', n_estimators=200)
sgd = SGDClassifier(random_state=1, alpha=0.00001, penalty='l2', max_iter=80)

eclf2 = eclf.fit(predictors_tr,targets_tr)
predictions = eclf2.predict(predictors_ts)
testdf['cuisine'] = predictions
testdf = testdf.sort_values(by='id' , ascending=True)

testdf[['id' , 'cuisine' ]].to_csv("just_python_cooking-vote1.csv", index=False)
pickle.dump(eclf2, open("eclf2.pickle.dat", "wb"))
