#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


get_ipython().system('pip install comet_ml')


# In[ ]:


from comet_ml import Experiment


# In[ ]:


comet = Experiment(api_key="KY2FKGd0W5CRZJiGgHVU5nHDu",
                        project_name="general", workspace="jolinda-hub")


# In[ ]:


comet_sec = Experiment(api_key="KY2FKGd0W5CRZJiGgHVU5nHDu",
                        project_name="general", workspace="jolinda-hub")


# In[ ]:


import numpy as np# utilities
import numpy as np
import pandas as pd

# plotting
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()

# nltk
from nltk.tokenize import word_tokenize, TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
import nltk
import re

# sklearn(classifier)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

#Metrics/Evaluation
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE 
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from collections import Counter 
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train=pd.read_csv('../input/climate-change-belief-analysis/train.csv')


# In[ ]:


test=pd.read_csv('../input/climate-change-belief-analysis/test.csv')


# In[ ]:


train


# In[ ]:


all_sents = []
for i, row in train.iterrows():
    for post in row['message'].split('|||'):
        all_sents.append([row['sentiment'], post])
all_sents = pd.DataFrame(all_sents, columns=['sentiment', 'message'])


# In[ ]:


# Remove urls
pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
subs_url = r'url-web'
all_sents['message'] = all_sents['message'].replace(to_replace = pattern_url, value = subs_url, regex = True)


# In[ ]:


# Make lower case
all_sents['message'] = all_sents['message'].str.lower()


# In[ ]:


# Strip out punctuation marks and numerals
import string
def remove_punctuation_numbers(message):
    punc_numbers = string.punctuation + '0123456789'
    return ''.join([l for l in message if l not in punc_numbers])

all_sents['message'] = all_sents['message'].apply(remove_punctuation_numbers)


# In[ ]:


sum_sents = all_sents[['sentiment', 'message']].groupby('sentiment').count()


# In[ ]:


all_sents


# In[ ]:


# Plot messages by sentiment classes
sum_sents.sort_values('message', ascending=False).plot(kind='bar')


# In[ ]:


# Let's use the count vectorizer with its default hyperparameters
vect = CountVectorizer()
X_count = vect.fit_transform(all_sents['message'])


# In[ ]:


vect_new = CountVectorizer(lowercase=True, stop_words='english',max_features=500,analyzer='word', ngram_range=(1, 3))
X_count = vect_new.fit_transform(all_sents['message'])


# In[ ]:


X = X_count.toarray()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# Fit label encoder and return encoded labels
y = le.fit_transform(all_sents['sentiment'])


# In[ ]:


# List of label encoder types to use for lookup 
type_labels = list(le.classes_)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)


# In[ ]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[ ]:


LogisticRegression = LogisticRegression()
##Fitting the model with train dataset
LogisticRegression = LogisticRegression.fit(X_train, y_train)


# In[ ]:


# Getting predicions from the X_test
predict = LogisticRegression.predict(X_test)
#Pritting the classification report
print(metrics.classification_report(y_test,predict))
# Print the overall accuracy
print(metrics.accuracy_score(y_test,predict))


# In[ ]:


f1_logreg = f1_score(y_test, predict, average='macro')
precision_logreg = precision_score(y_test, predict, average='macro')
recall_logreg = recall_score(y_test, predict, average='macro')


# In[ ]:


params2 = {"random_state": 27,
          "model_type": "logreg",
          "scaler": "standard scaler"
          }
metrics2 = {"f1_logreg": f1_logreg,
           "recall_svc": recall_logreg,
           "precision_svc": precision_logreg
           }


# In[ ]:


comet_sec.log_parameters(params2)
comet_sec.log_metrics(metrics2)


# In[ ]:


LinearSVC = LinearSVC()
##Fitting the model with train dataset
LinearSVC = LinearSVC.fit(X_train, y_train)


# In[ ]:


# Getting predicions from the X_test
pred1 = LinearSVC.predict(X_test)
#Printing the classification report
print(metrics.classification_report(y_test, pred1))
# Print the overall accuracy
print(metrics.accuracy_score(y_test,pred1))


# In[ ]:


f1_svc = f1_score(y_test, pred1, average='macro')
precision_svc = precision_score(y_test, pred1, average='macro')
recall_svc = recall_score(y_test, pred1, average='macro')


# In[ ]:


params1 = {"random_state": 27,
          "model_type": "LinearSVC",
          "scaler": "standard scaler"
          }
metrics1 = {"f1_svc": f1_svc,
           "recall_svc": recall_svc,
           "precision_svc": precision_svc
           }


# In[ ]:


comet.log_parameters(params1)
comet.log_metrics(metrics1)


# In[ ]:


BNBmodel = BernoulliNB(alpha = 2)
##Fitting the model with train dataset
BNBmodel.fit(X_train, y_train)


# In[ ]:


# Getting predictions from the X_test
pred4 = BNBmodel.predict(X_test)
#Priting the classification report
print(metrics.classification_report(y_test,pred4))
# Print the overall accuracy
print(metrics.accuracy_score(y_test,pred4))


# In[ ]:


MultinomialNB = MultinomialNB()
##Fitting the model with train dataset
MultinomialNB  = MultinomialNB .fit(X_train, y_train)


# In[ ]:


# Getting predictions from the X_test
pred3 = MultinomialNB.predict(X_test)
#Priting the classification report
print(metrics.classification_report(y_test,pred3))
# Print the overall accuracy
print(metrics.accuracy_score(y_test,pred3))


# In[ ]:


heights = [len(y[y == label]) for label in range(len(type_labels))]
bars = pd.DataFrame(zip(heights,le.transform(type_labels).T, type_labels), columns=['heights','labels','names'])
bars = bars.sort_values(by='heights',ascending=True)

plt.bar(range(len(bars)),bars['heights'],color='grey')
plt.xticks(range(len(bars)),bars['names'])
plt.ylabel("# of observations")


# In[ ]:


# Let's pick a class size of roughly half the size of the largest size
class_size = 30000


# In[ ]:


data = np.concatenate([X, y[:,np.newaxis]], axis=1)


# In[ ]:


bar_label_df = bars.set_index('labels')


# In[ ]:


resampled_classes = []

# For each label
for label in range(len(type_labels)):
    # Get num. of observations from this class
    label_size = bar_label_df.loc[label]['heights']
    
    # If label_size < class size the upsample, else downsample
    if label_size < class_size:
        # Upsample
        label_data = data[data[:,-1] == label]
        label_resampled = resample(label_data,
                                  replace=True, # sample with replacement (we need to duplicate observations)
                                  n_samples=class_size, # number of desired samples
                                  random_state=27) # reproducible results
    else:
        # Downsample
        label_data = data[data[:,-1] == label]
        label_resampled = resample(label_data,
                                  replace=False, # sample without replacement (no need for duplicate observations)
                                  n_samples=class_size, # number of desired samples
                                  random_state=27) # reproducible results
        
    resampled_classes.append(label_resampled)


# In[ ]:


resampled_data = np.concatenate(resampled_classes, axis=0)


# In[ ]:


resampled_data.shape


# In[ ]:


X_resampled = resampled_data[:,:-1]


# In[ ]:


y_resampled = resampled_data[:,-1]


# In[ ]:


heights = [len(y_resampled[y_resampled == label]) for label in range(len(type_labels))]
bars_resampled = pd.DataFrame(zip(heights,le.transform(type_labels).T, type_labels), columns=['heights','labels','names'])
bars_resampled = bars_resampled.sort_values(by='heights',ascending=True)

plt.bar(range(len(bars)),bars['heights'],color='grey')
plt.bar(range(len(bars_resampled)),bars_resampled['heights'],color='orange')
plt.xticks(range(len(bars)),bars['names'])
plt.ylabel("# of observations")
plt.legend(['original','resampled'])
plt.show()


# In[ ]:


# Setting up the train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=27)


# In[ ]:


LogisticRegression = LogisticRegression()
##Fitting the model with train dataset
LogisticRegression = LogisticRegression.fit(X_train, y_train)


# In[ ]:


# Getting predicions from the X_test
predict = LogisticRegression.predict(X_test)
#Pritting the classification report
print(metrics.classification_report(y_test,predict))
# Print the overall accuracy
print(metrics.accuracy_score(y_test,predict))


# In[ ]:


LinearSVC = LinearSVC()
##Fitting the model with train dataset
LinearSVC_up = LinearSVC.fit(X_train, y_train)


# In[ ]:


# Getting predicions from the X_test
predict1 = LinearSVC_up.predict(X_test)
#Printing the classification report
print(metrics.classification_report(y_test, predict1))
# Print the overall accuracy
print(metrics.accuracy_score(y_test,predict1))


# In[ ]:


BNBmodel = BernoulliNB(alpha = 2)
##Fitting the model with train dataset
BNBmodel.fit(X_train, y_train)


# In[ ]:


# Getting predictions from the X_test
predict4 = BNBmodel.predict(X_test)
#Priting the classification report
print(metrics.classification_report(y_test,predict4))
# Print the overall accuracy
print(metrics.accuracy_score(y_test,predict4))


# In[ ]:


MultinomialNB = MultinomialNB()
##Fitting the model with train dataset
MultinomialNB  = MultinomialNB .fit(X_train, y_train)


# In[ ]:


# Getting predictions from the X_test
predict3 = MultinomialNB.predict(X_test)
#Priting the classification report
print(metrics.classification_report(y_test,predict3))
# Print the overall accuracy
print(metrics.accuracy_score(y_test,predict3))


# In[ ]:


comet.end()


# In[ ]:


comet_sec.end()


# In[ ]:


comet_sec.display()

