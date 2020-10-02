import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV


train = pd.read_csv('../input/Wuzzuf_Job_Posts_Sample.csv')
categories = train.job_category1.unique()
train['job_description'] = train['job_description'].fillna('')
train['text'] = train['job_industry1'] \
                + ' ' + train['career_level'] \
                + ' ' + train['experience_years'] \
                + ' ' + train['job_title'] \
                + ' ' + train['job_description']

y = train['job_category1']
X_train, X_test, y_train, y_test = train_test_split(train.text, y, test_size=0.20)

pipeline = Pipeline([
    ('vect', CountVectorizer(max_df=0.75, max_features=None, ngram_range=(1, 2))),
    ('tfidf', TfidfTransformer(use_idf=True, norm='l2')),
    ('clf', SGDClassifier(loss='hinge', alpha=1e-05, penalty='l2', random_state=None, max_iter=100)),
])
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
print('Accuracy SDG', accuracy_score(y_test, predictions))

# Compare results with simple text based category matching approach
match_count = 0.0
for index, text in X_test.items():
    if text.find(y_test[index]) >= 0:
        match_count += 1
print('Accuracy with simple text based category matching', match_count / len(X_test))

"""
# Can be used to perform a grid search to find the best parameters

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier()),
])

parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2)),
    #'vect__stop_words': ['english'],
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__loss': ['hinge'],
    'clf__alpha': (0.00001, 0.000001, 1e-3),
    'clf__penalty': ('l2', 'elasticnet'),
    'clf__max_iter': (5, 10, 50, 100, 500, 1000),
    'clf__random_state': [None, 42],
}
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print('Best params', grid_search.best_score_)
predictions = grid_search.predict(X_test)
print('Accuracy grid SDG', accuracy_score(y_test, predictions))
"""
