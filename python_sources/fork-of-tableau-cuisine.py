#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Loading training And Testing Dataset.....
# The given dataset is in json format

# In[ ]:


x=pd.read_json('../input/train.json')
y=pd.read_json('../input/test.json')


# Loading Various preprocessing element for Natural Language Processing.......

# In[ ]:


from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import string
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
z=x['cuisine']
from sklearn.preprocessing import LabelEncoder as le
from sklearn.preprocessing import OneHotEncoder as ohe


# describe train dataset

# In[ ]:


x.info()


# In[ ]:


x.head()


# describe test dataset

# In[ ]:


y.info()


# In[ ]:


y.head()


# Converting list of ingredients to a sentence where each ingredient is separated by a space

# In[ ]:


x['separated_ing']=x['ingredients'].map(lambda x: ' '.join(x))
y['separated_ing']=y['ingredients'].map(lambda x: ' '.join(x))


# Preprocessing the sentence formed off in the above steps to 
# -coverting all characters of ingredients to smalll letter
# -remove punctuation marks
# -converting multiple spaces to single spaces
# Both Test and Train Data preprocessed
# 

# In[ ]:


import string,re
def purify(f):
    f=f.lower()
    f=re.sub('[%s]' % re.escape(string.punctuation),'',f)
    f=re.sub('\s+',' ',f)
    return f
x['cleared_ing']=x['separated_ing'].map(lambda g :purify(g))
y['cleared_ing']=y['separated_ing'].map(lambda g :purify(g))


# Using a SnowballStemmer with english as parameter representing English language and also WordNetLemmatizer for bringing all ingredients to core form .Like 'works' -->'work'
# This has been used by both training and test dataset

# In[ ]:


sb=SnowballStemmer('english')
def stemmer(f):
    lists=[sb.stem(c) for c in f.split(" ")]
    return lists
l=WordNetLemmatizer()
def lemmar(f):
    lists=[l.lemmatize(g) for g in f.split(" ")]
    return lists
x['separated_ing_stemmed']=[stemmer(l) for l in x['cleared_ing']]
x['separated_ing_stemmed']=x['separated_ing_stemmed'].map(lambda x: ' '.join(x))
x['separated_ing_lemma']=[lemmar(l) for l in x['separated_ing_stemmed']]
x['separated_ing_lemma']=x['separated_ing_lemma'].map(lambda x: ' '.join(x))
y['separated_ing_stemmed']=[stemmer(l) for l in y['cleared_ing']]
y['separated_ing_stemmed']=y['separated_ing_stemmed'].map(lambda x: ' '.join(x))
y['separated_ing_lemma']=[lemmar(l) for l in y['separated_ing_stemmed']]
y['separated_ing_lemma']=y['separated_ing_lemma'].map(lambda x: ' '.join(x))


# Dropping all unnecessary data columns from train and test dataset

# In[ ]:


x=x.drop(['ingredients','separated_ing','cleared_ing','separated_ing_stemmed'],axis=1)
y=y.drop(['ingredients','separated_ing','cleared_ing','separated_ing_stemmed'],axis=1)


# In[ ]:


x.columns


# Getting all the cuisines in a list  allCuisine
# Also we need to separate ingredients for each cuisine

# In[ ]:


allCuisine=[]
for x1 in x['cuisine']:
    if x1 not in allCuisine:
        allCuisine.append(x1)
dic={}
for x1 in allCuisine:
    dic[x1]=[]
def IngredientsForCuisine(x):
    dic[x['cuisine']]=list(dic[x['cuisine']])+[f for f in x['separated_ing_lemma'].split(" ") if f not in dic[x['cuisine']]]
for e,d in x.iterrows():   
    IngredientsForCuisine(d)
    
    


# Detecting correlation between various cuisines

# In[ ]:


xx=pd.DataFrame(index=allCuisine,columns=allCuisine)


# getting all unique ingredients from 'a'

# In[ ]:


for z in allCuisine:
    for z1 in allCuisine:
        xx.loc[z,z1]=len(list(set(dic[z]) & set(dic[z1])))/len(list(set(dic[z]) | set(dic[z1])))
        xx.loc[z1,z]=xx.loc[z,z1]


# In[ ]:


xx.to_csv("cor.csv")


# DATA VISUALIZATIONS

# 1-Correlation between various Cuisine.SIMILAR CUISINES

# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1538489378237' style='position: relative'><noscript><a href='#'><img alt='Sheet 1 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Co&#47;CorForCuisine&#47;Sheet1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='CorForCuisine&#47;Sheet1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Co&#47;CorForCuisine&#47;Sheet1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1538489378237');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1538489639742' style='position: relative'><noscript><a href='#'><img alt='Cuisine and distinct ingredients count ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;NL&#47;NLP_0&#47;Sheet1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='NLP_0&#47;Sheet1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;NL&#47;NLP_0&#47;Sheet1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1538489639742');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1538489685013' style='position: relative'><noscript><a href='#'><img alt='Rare Ingredients ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;NL&#47;NLP_0&#47;Sheet2&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='NLP_0&#47;Sheet2' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;NL&#47;NLP_0&#47;Sheet2&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1538489685013');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1538489718393' style='position: relative'><noscript><a href='#'><img alt='Most Common Ingredients ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;NL&#47;NLP_0&#47;Sheet3&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='NLP_0&#47;Sheet3' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;NL&#47;NLP_0&#47;Sheet3&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1538489718393');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1538556179098' style='position: relative'><noscript><a href='#'><img alt='10 Most Common Ingredient in Cuisines ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;NL&#47;NLP_0&#47;Sheet4&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='NLP_0&#47;Sheet4' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;NL&#47;NLP_0&#47;Sheet4&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1538556179098');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# FINDINGS FROM VISUALIZATION--
# we come to a onclusion that there are ingredients present in more than 1 cuisine
# Maximum distinct ingredients are present in Intalian and Mexican cuisine
# Minimum distinct ingredients are present in Russian and Brazilian
# Certain cuisines are correlated
# salt,onions and olive oils appears to be amongst the most commong element in every ingredient

# Using STOP_WORDS to remove all common ingredients present in the recipes 

# In[ ]:


lists=list(ENGLISH_STOP_WORDS)+stopwords.words()


# Importing TFIDFVectorizer-->Term Frequency*Inverse Document frequency (Count of an element in doc*log(no. of doc/docs in which element present)) and CountVectorizer(count of each element in a row as vector)

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer  as tfidf,CountVectorizer as cv


# Target as series z

# In[ ]:


z=x['cuisine']


# setting parameters for itidf 
# -max_df relates to maximum doc freq to be cosidered
# -stop_words eliminate common words from given text
# -analyzer how the text to be analyzed,word by word or character by character
# fitting tf-idf over train dataset

# In[ ]:


tfidf1=tfidf(max_df=0.9,stop_words=lists,analyzer=u'word')
train=tfidf1.fit_transform(x['separated_ing_lemma'])
test=tfidf1.transform(y['separated_ing_lemma'])


# Importing required ML functions from sklearn

# In[ ]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV as gsc
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier as xgb
from lightgbm import LGBMClassifier as lgb
from sklearn.linear_model import LogisticRegression as lr


# Setting required parameters for svm

# In[ ]:


svm={'C':[6]}


# Fitting  labelencoder over target to convert categorical target to numeric in nature 

# In[ ]:


from sklearn.preprocessing import LabelEncoder as le
p=le().fit(z)


# In[ ]:


z=p.transform(z)


# Performing validation train  split of train dataset in ratio 3:7

# In[ ]:


from sklearn.model_selection import train_test_split as tts
xtrain,xtest,ztrain,ztest=tts(train,z,train_size=0.7)


# Setting  LGBM parameters

# In[ ]:


r1={'n_estimators':[500],'max_depth':[7],'objective':['multiclass'],'metric':['multi_logloss'],'bagging_fraction':[0.6],'feature_fraction':[0.6]}


# In[ ]:


from sklearn.model_selection import GridSearchCV as gsc
a=gsc(lr(),svm)
b=gsc(lgb(),r1)
b.fit(xtrain,ztrain)


# In[ ]:


a.best_params_


# In[ ]:


from sklearn.metrics import accuracy_score as acs
print(acs(ztest,a.predict(xtest)))


# Checking KNearestNeighbors over the data

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier as knn
k={'n_neighbors':[5,7,9]}
k1=gsc(knn(),k)


# Using Voting Classifier for classification purpose with
# -SVM
# -KNN
# -LGBM
# using soft margin for classification

# In[ ]:



from sklearn.ensemble import VotingClassifier as vc


# In[ ]:


v=vc(estimators=[('lr',a),('k1',k1),('lg',b)],voting='soft')


# Fitting train data over votingclassifier

# In[ ]:


v.fit(xtrain,ztrain)


# Checking accuracy_score

# In[ ]:


from sklearn.metrics import accuracy_score 
print(accuracy_score(ztest,v.predict(xtest)))


# Predicting test data target

# In[ ]:


z1=v.predict(test)


# converting  numeric target back to categorical data

# In[ ]:


z=p.inverse_transform(z1)


# Preparing output result file

# In[ ]:


ff=pd.DataFrame(z,index=y['id'],columns=['cuisine'])


# In[ ]:


ff.index.name='id'


# In[ ]:


ff.to_csv('aagya.csv')


# In[ ]:




