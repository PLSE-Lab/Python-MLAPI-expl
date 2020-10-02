#!/usr/bin/env python
# coding: utf-8

# ## Summary
# We have used different models to predict when a Kickstarter campaign will be successful. For that, we have build pipelines to preprocess the data and get the features we need per each model. We have studied the correlation between models and calculated their performance. At the end, we have created some stacks of these models to try to get a better solution and a voting clasiffier.

# In[ ]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


data = pd.read_csv("../input/ks-projects-201801.csv")


# In[ ]:


data.head()


# In[ ]:


from sklearn.preprocessing import FunctionTransformer, LabelEncoder, LabelBinarizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.metrics import roc_auc_score, precision_score, recall_score, auc
from sklearn.metrics import accuracy_score, roc_curve
from collections import defaultdict


# In[ ]:


class MultiColumnLabelEncoder(TransformerMixin):  
    def __init__(self):
        self.d = defaultdict(LabelEncoder)

    def transform(self, X, **transform_params):

        X = X.fillna('NaN')  
        transformed = X.apply(self._transform_func)
        return transformed

    def fit(self, X, y=None, **fit_params):
        X = X.fillna('NaN')  
        X.apply(self._fit_func)
        return self
    
    
    def _transform_func(self, x):
        return self.d[x.name].transform(x)
    
    def _fit_func(self, x):
        return self.d[x.name].fit(x)


# In[ ]:


def get_categorical_data(x): 
    return x[['category', 'main_category', 'currency', 'country']]

def get_name_lenght_feature(x): 
    return x['name'].str.len().fillna(0).to_frame()
    
def get_duration_feature(x): 
    return (pd.to_datetime(x['deadline']) - pd.to_datetime(x['launched'])).dt.days.to_frame()

def get_deadline_month_feature(x): 
    return pd.to_datetime(x['deadline']).dt.month.to_frame()
    
def get_deadline_weekday_feature(x): 
    return pd.to_datetime(x['deadline']).dt.weekday.to_frame()

def get_launched_month_feature(x): 
    return pd.to_datetime(x['launched']).dt.month.to_frame()

def get_launched_weekday_feature(x): 
    return pd.to_datetime(x['launched']).dt.weekday.to_frame()


preprocess_base_pipeline = FeatureUnion(
         transformer_list = 
         [
            ('name_length', Pipeline([
                ('selector', FunctionTransformer(get_name_lenght_feature, validate=False))
            ])),
            ('duration_feature', Pipeline([
                ('selector', FunctionTransformer(get_duration_feature, validate=False))
            ])),  
            ('deadline_month', Pipeline([
                ('selector', FunctionTransformer(get_deadline_month_feature, validate=False))
            ])),
            ('deadline_weekday', Pipeline([
                ('selector', FunctionTransformer(get_deadline_weekday_feature, validate=False))
            ])),  
            ('launched_month', Pipeline([
                ('selector', FunctionTransformer(get_launched_month_feature, validate=False))
            ])),
            ('launched_weekday', Pipeline([
                ('selector', FunctionTransformer(get_launched_weekday_feature, validate=False))
            ]))                    
        ])


# In[ ]:


preprocess_pipeline = FeatureUnion(
         transformer_list = 
         [
            ('cat_features', Pipeline([
                ('selector', FunctionTransformer(get_categorical_data, validate=False)),
                ('encoder', MultiColumnLabelEncoder())
            ])),

            ('name_length', Pipeline([
                ('preprocess_base_pipeline', preprocess_base_pipeline)
            ])),
                   
        ])


# In[ ]:


def report_results(model, X, y):
    pred_proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)        


    auc = roc_auc_score(y, pred_proba)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred)
    prec = precision_score(y, pred)
    rec = recall_score(y, pred)
    result = {'auc': auc, 'f1': f1, 'acc': acc, 'precision': prec, 'recall': rec}
    return result


# In[ ]:


def get_roc_curve(model, X, y):
    pred_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, pred_proba)

    return fpr, tpr


# In[ ]:


X = preprocess_pipeline.fit_transform(data)
y = (data['state'] == 'successful').astype('int')


# Here is the correlation between features:

# In[ ]:


sns.heatmap(pd.DataFrame(X).corr(), cmap='Blues')
plt.title('Feature Correlations')
plt.show()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, stratify=y, random_state=1)


# ## Models

# ### Logistic Regression

# In[ ]:


model_lr = Pipeline([('preprocess', preprocess_pipeline), 
                     ('estimator', LogisticRegression(solver='liblinear', random_state=0))]) 

model_lr.fit(X_train, y_train)


# In[ ]:


model_lr.score(X_test, y_test)


# In[ ]:


y_pred_lr = model_lr.predict(X_test)
y_pred_proba_lr = model_lr.predict_proba(X_test)[:,1]


# In[ ]:


all_models = {}
all_models['lr'] = {} 
all_models['lr']['model'] = model_lr
all_models['lr']['train_preds'] = model_lr.predict_proba(X_train)[:, 1]
all_models['lr']['result'] = report_results(all_models['lr']['model'], X_test, y_test)
all_models['lr']['roc_curve'] = get_roc_curve(all_models['lr']['model'], X_test, y_test)


# In[ ]:


print(classification_report(y_test, y_pred_lr))


# In[ ]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_lr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
plt.show()


# ### Naive Bayes

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
np.random.seed(1)

model_nb = Pipeline([('preprocess', preprocess_pipeline), 
                     ('estimator', MultinomialNB())]) 


# In[ ]:


model_nb.fit(X_train, y_train)


# In[ ]:


model_nb.score(X_test, y_test)


# In[ ]:


all_models['nb'] = {} 
all_models['nb']['model'] = model_nb
all_models['nb']['train_preds'] = model_nb.predict_proba(X_train)[:, 1]
all_models['nb']['result'] = report_results(all_models['nb']['model'], X_test, y_test)
all_models['nb']['roc_curve'] = get_roc_curve(all_models['nb']['model'], X_test, y_test)


# ### Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

model_rf = Pipeline([('preprocess', preprocess_pipeline), 
                     ('estimator', RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1))])

model_rf.fit(X_train, y_train)


# In[ ]:


model_rf.score(X_test, y_test)


# In[ ]:


all_models['rf'] = {} 
all_models['rf']['model'] = model_rf
all_models['rf']['train_preds'] = model_rf.predict_proba(X_train)[:, 1]
all_models['rf']['result'] = report_results(all_models['rf']['model'], X_test, y_test)
all_models['rf']['roc_curve'] = get_roc_curve(all_models['rf']['model'], X_test, y_test)


# ### k-nearest neighbors

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
np.random.seed(1)

model_knn = Pipeline([('preprocess', preprocess_pipeline), 
                     ('estimator', KNeighborsClassifier(n_neighbors=3, n_jobs=-1))])

model_knn.fit(X_train, y_train)


# In[ ]:


model_knn.score(X_test, y_test)


# In[ ]:


all_models['knn'] = {} 
all_models['knn']['model'] = model_knn
all_models['knn']['train_preds'] = model_knn.predict_proba(X_train)[:, 1]
all_models['knn']['result'] = report_results(all_models['knn']['model'], X_test, y_test)
all_models['knn']['roc_curve'] = get_roc_curve(all_models['knn']['model'], X_test, y_test)


# ### Bag of words
# In this case, we'll obtain features only from the name of the capaign.

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

en_stopwords = set(stopwords.words("english"))  

preprocess_nlp_pipeline = Pipeline(
    [('selector', FunctionTransformer(lambda x: x['name'].fillna(''), validate=False)),
     ('vectorizer', CountVectorizer(stop_words = en_stopwords))
    ]
)

preprocess_full_nlp_pipeline = FeatureUnion(
         transformer_list = [('preprocess_nlp_pipeline', preprocess_nlp_pipeline),
                             ('preprocess_base_pipeline', preprocess_base_pipeline)])


# In[ ]:


model_nlp = Pipeline([('preprocess', preprocess_nlp_pipeline), 
                     ('estimator', LogisticRegression(random_state=0))])


# In[ ]:


model_nlp.fit(X_train, y_train)


# In[ ]:


model_nlp.score(X_test, y_test)


# In[ ]:


all_models['nlp'] = {} 
all_models['nlp']['model'] = model_nlp
all_models['nlp']['train_preds'] = model_nlp.predict_proba(X_train)[:, 1]
all_models['nlp']['result'] = report_results(all_models['nlp']['model'], X_test, y_test)
all_models['nlp']['roc_curve'] = get_roc_curve(all_models['nlp']['model'], X_test, y_test)


# ### Bag of words with additional features
# 

# In[ ]:


model_mix = Pipeline([('preprocess', preprocess_full_nlp_pipeline), 
                     ('estimator', LogisticRegression(random_state=0))])


# In[ ]:


model_mix.fit(X_train, y_train)


# In[ ]:


model_mix.score(X_test, y_test)


# In[ ]:


all_models['mix'] = {} 
all_models['mix']['model'] = model_mix
all_models['mix']['train_preds'] = model_mix.predict_proba(X_train)[:, 1]
all_models['mix']['result'] = report_results(all_models['mix']['model'], X_test, y_test)
all_models['mix']['roc_curve'] = get_roc_curve(all_models['mix']['model'], X_test, y_test)


# Comparasion of all the results

# In[ ]:


all_models_name = all_models.keys()

tmp_list = []
for mo in all_models_name:
    tmp_list.append(all_models[mo]['result'])
models_results = pd.DataFrame(dict(zip(all_models_name, tmp_list))).transpose()


# In[ ]:


models_results = models_results.sort_values(['auc'], ascending=False)
models_results


# In[ ]:


from matplotlib import cm

tmp_models = models_results.index

colors = cm.rainbow(np.linspace(0.0, 1.0, len(tmp_models)))


plt.figure(figsize=(14,8))
lw = 2

for mo, color in zip(tmp_models, colors):
    fpr, tpr = all_models[mo]['roc_curve']
    plt.plot(fpr, tpr, color=color,
         lw=lw, label='{} (auc = {:.4f})'.format(mo, all_models[mo]['result']['auc']))



plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Roc curve')
plt.legend(loc="lower right")

plt.show()


# Let's see the correlation between models:

# In[ ]:


corr_dict = {}
for mo in tmp_models:
    corr_dict[mo] = all_models[mo]['train_preds']


# In[ ]:


kdata_proba = pd.DataFrame(corr_dict) 


# In[ ]:


corr = kdata_proba.corr()
corr


# In[ ]:


plt.figure(figsize=(14,8))
sns.heatmap(corr, cmap="YlGnBu")
plt.show()


# ## Combination of models

# In[ ]:


np.random.seed(1)

from mlxtend.classifier import StackingClassifier

lr_stack = LogisticRegression(random_state=0)
#lr_stack = RandomForestClassifier()

clf_stack = StackingClassifier(
    classifiers = [model_nlp, model_lr],
    use_probas = True, 
    average_probas = False,    
    meta_classifier = lr_stack, verbose=1)


# In[ ]:


clf_stack.fit(X_train, y_train)


# In[ ]:


clf_stack.score(X_test, y_test)


# In[ ]:


report_results(clf_stack, X_test, y_test)


# Let's try to optimize the stacked model:

# In[ ]:


from sklearn.model_selection import GridSearchCV
params = {'meta-logisticregression__C': [0.1, 10.0]}

grid = GridSearchCV(estimator=clf_stack, 
                    param_grid=params, 
                    cv=3,
                    refit=True)
grid.fit(X_train, y_train)


# In[ ]:


print('Best parameters: %s' % grid.best_params_)
print('Accuracy: %f' % grid.best_score_)


# In[ ]:


report_results(grid, X_test, y_test)


# Let's try a votting classifier:

# In[ ]:


from sklearn.ensemble import VotingClassifier


# In[ ]:


eclf1 = VotingClassifier(estimators=[('lr', model_lr), 
                                     ('rf', model_rf), 
                                     ('nb', model_nb),
                                     ('knn', model_knn),
                                     ('nlp', model_nlp)], voting='soft')


# In[ ]:


eclf1.fit(X_train, y_train)


# In[ ]:


report_results(eclf1, X_test, y_test)


# In this case we get a good result, but not better than a simple model like model_mix 
