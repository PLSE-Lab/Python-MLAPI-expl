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


#basic library
import numpy as np
import pandas as pd

#Basic Visualization Libarary
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

#Advanced Visualization Library
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize']=(12,8)
import warnings
warnings.filterwarnings('ignore')
import plotly.offline as py
from plotly.offline import init_notebook_mode,iplot
from plotly import tools
init_notebook_mode(connected=True)
import plotly.figure_factory as ff
import plotly.graph_objs as go


# #### Loading data

# In[ ]:


data=pd.read_csv('/kaggle/input/amazon-alexa-reviews/amazon_alexa.tsv',sep='\t')
data.head()


# In[ ]:


#View data info
data.info()


# In[ ]:


#View data statistics
data.describe()


# In[ ]:


data.groupby('rating').describe()


# In[ ]:


data.groupby('feedback').describe()


# In[ ]:


data['rating'].value_counts().describe()


# ### Data Visualization

# In[ ]:


ratings=data['rating'].value_counts()

label_rating=ratings.index
size_rating=ratings.values

color=['aqua','yellow','green','pink','blue']

rating_piegraph=go.Pie(labels=label_rating,
                      values=size_rating,
                      marker=dict(colors=color),
                      name='Alexa',hole=0.3)
df=[rating_piegraph]

layout=go.Layout(title="Distribution of Alexa Review")

fig=go.Figure(data=df,
             layout=layout)
py.iplot(fig)


# In[ ]:


data['length']=data['verified_reviews'].apply(len)
data.head()


# In[ ]:


color=plt.cm.copper(np.linspace(0,1,15))
data['variation'].value_counts().plot.bar(color=color,figsize=(12,8))
plt.title('Distribution of Alexa review',fontsize=20)
plt.xlabel('variation')
plt.ylabel('count')
plt.show()


# In[ ]:


data['length'].value_counts().plot.hist(color='green',figsize=(12,8),bins=50)
plt.title('Distribution of Alexa Review ',fontsize=20)
plt.xlabel('Length')
plt.ylabel('count')
plt.show()


# In[ ]:


sns.boxenplot(data['variation'],data['rating'],palette='spring')
plt.title('Variation vs Rating')
plt.xticks(rotation=90)
plt.show()


# In[ ]:


sns.swarmplot(data['variation'],data['length'],palette='cool')
plt.title('Variation vs Length')
plt.xticks(rotation=90)
plt.show()


# In[ ]:


sns.violinplot(data['feedback'],data['rating'],palette='deep')
plt.title('Feedback rating')
plt.show()


# In[ ]:


sns.boxplot(data['rating'],data['length'],palette='deep')
plt.title('Rating vs Length')
plt.show()


# ### Convert in Bag Of Words(BOW)

# In[ ]:


import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(STOP_WORDS)
words=cv.fit_transform(data['verified_reviews'])
words_sum=words.sum(axis=0)

words_freq=[(word,words_sum[0,idx]) for word,idx in cv.vocabulary_.items()]
words_freq=sorted(words_freq,key=lambda x:x[1],reverse=True)

frequency=pd.DataFrame(words_freq,columns=['word','freq'])

color=plt.cm.ocean(np.linspace(0,1,20))
frequency.head(20).plot.bar(x='word',y='freq',figsize=(12,8),color=color)
plt.title('Amazaon Alex revew Top-20 words')
plt.show()


# In[ ]:


from wordcloud import WordCloud

wordcloud=WordCloud(background_color='lightcyan',
                   width=2000,
                   height=2000).generate_from_frequencies(dict(words_freq))
plt.style.use('fivethirtyeight')
plt.figure(figsize=(15,12))
plt.title('Vocabulary of word review',fontsize=20)
plt.axis('off')
plt.imshow(wordcloud)
plt.show()


# In[ ]:


import spacy
from spacy import displacy
from spacy.lang.en.stop_words import STOP_WORDS
nlp=spacy.load('en_core_web_sm')

def explain_text_entities(text):
    doc=nlp(text)
    for ent in doc.ents:
        print(f'Entity:{ent},Label:{ent.label_},{spacy.explain.ent.label_}')
for i in range(0,3150):
    one_sentence=data['verified_reviews'][i]
    doc=nlp(one_sentence)
    displacy.render(doc,style='ent')


# In[ ]:


import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import nltk
from nltk.stem.porter import PorterStemmer
import re


# In[ ]:


corpus=[]
for i in range(0,3150):
    review=re.sub('[^a-zA-Z]',' ',data['verified_reviews'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if word not in STOP_WORDS]
    review=' '.join(review)
    corpus.append(review)


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
cv=CountVectorizer(max_features=2500)
X=cv.fit_transform(corpus).toarray()
y=data.iloc[:,4].values


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
mm=MinMaxScaler()
X_train=mm.fit_transform(X_train)
X_test=mm.transform(X_test)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

model=RandomForestClassifier()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

print("Training Accuracy:",model.score(X_train,y_train))
print("Testing Accuracy:",model.score(X_test,y_test))


# In[ ]:


cm=confusion_matrix(y_pred,y_test)


# In[ ]:


print("Confusion Matrix: ",cm)


# ### Applying K fold cross validation

# In[ ]:


from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=model,X=X_train,y=y_train,cv=10)


# In[ ]:


print("Accuracies: ",accuracies.mean())
print("Standard Variance: ",accuracies.std())


# ### Grid Search 

# In[ ]:


params={'bootstrap':['True'],
       'max_depth':[80,100],
       'min_sample_split':[8,12],
       'n_estimator':[100,300]}


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


# In[ ]:


cv_object=StratifiedKFold(n_splits=2)
grid=GridSearchCV(estimator=model,
                 param_grid=params,
                 cv=cv_object,verbose=0,
                 return_train_score=True)


# In[ ]:





# In[ ]:




