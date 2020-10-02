# %% [code]
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold

mbti_types = {'I':'Introversion', 'E':'Extroversion', 'N':'Intuition', 
        'S':'Sensing', 'T':'Thinking', 'F': 'Feeling', 
        'J':'Judging', 'P': 'Perceiving'}

# %% [code]
df = pd.read_csv('../input/mbti-type/mbti_1.csv')

# %% [code]
df.head()

# %% [code]
plt.figure(figsize=(40,20))
plt.xticks(fontsize=24, rotation=0)
plt.yticks(fontsize=24, rotation=0)
sns.countplot(data=df, x='type')
plt.ylabel('Number of Occurrences', fontsize=20)
plt.xlabel('Types', fontsize=20)

# %% [code]
df.count()

# %% [code]
def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text) 
    text = re.sub(r'http\S+', r'<URL>', text)
    return text

# %% [code]
df['clean_posts'] = df['posts'].apply(cleanText)

# %% [code]
df.head()

# %% [code]
tfidf2 = CountVectorizer(ngram_range=(1, 1), 
                         stop_words='english',
                         lowercase = True, 
                         max_features = 500)
kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
scoring = {'acc': 'accuracy'}

# %% [code]
model_lr = Pipeline([('tfidf1', tfidf2), ('lr', LogisticRegression(class_weight="balanced", C=0.005))])

results_lr = cross_validate(model_lr, df['clean_posts'], df['type'], cv=kfolds, 
                          scoring=scoring, n_jobs=-1)

# %% [code]
print("Accuracy: {:0.4f} (+/- {:0.4f})".format(np.mean(results_lr['test_acc']),np.std(results_lr['test_acc'])))

# %% [code]
model_lr.fit(df['clean_posts'], df['type'])

# %% [code]
model_lr.predict([''])

# %% [code]
model_lr.predict_proba([''])

# %% [code]
model_lr['lr'].classes_