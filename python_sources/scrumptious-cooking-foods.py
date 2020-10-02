#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
Using the Reference : 
https://www.kaggle.com/shivamb/what-s-cooking-tf-idf-with-ovr-svm
https://www.kaggle.com/ash316/what-is-the-rock-cooking-ensembling-network
'''
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
from collections import Counter
import matplotlib.pyplot as plt
#import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.pipeline import FeatureUnion

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from sklearn.manifold import TSNE
from keras.optimizers import adam,nadam


# In[ ]:


train = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json')
sub = pd.read_csv('../input/sample_submission.csv')

train['seperated_ingredients'] = train['ingredients'].apply(','.join)
test['seperated_ingredients'] = test['ingredients'].apply(','.join)


# In[ ]:


print('size of train data',train.shape)
print('size of test data',test.shape)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


# checking missing data
total = train.isnull().sum().sort_values(ascending = False)
percent = (train.isnull().sum()/train.isnull().count()*100).sort_values(ascending = False)
missing_train_data  = pd.concat([total, percent], axis=1, keys=['Total missing', 'Percent missing'])
missing_train_data.head(20)


# In[ ]:


# checking missing data
total_test = test.isnull().sum().sort_values(ascending = False)
percent_test = (test.isnull().sum()/test.isnull().count()*100).sort_values(ascending = False)
missing_train_data  = pd.concat([total, percent], axis=1, keys=['Total missing in Test', 'Percent missing in Test'])
missing_train_data.head(20)


# In[ ]:


# train['cuisine'].value_counts()

# data = [go.Bar(y = train['cuisine'].value_counts(), marker = dict(color='rgb(49,130,189)'))]
# layout = go.Layout(
#     xaxis=dict(tickangle=-45),
#     barmode='group',
# )

# fig = go.Figure(data=data, layout=layout)
# iplot(fig, filename='basic-bar')
color_theme = dict(color = ['rgba(221,160,221,1)','rgba(169,169,169,1)','rgba(255,160,122,1)','rgba(176,224,230,1)','rgba(169,169,169,1)','rgba(255,160,122,1)','rgba(176,224,230,1)',
                   'rgba(188,143,143,1)','rgba(221,160,221,1)','rgba(169,169,169,1)','rgba(255,160,122,1)','rgba(176,224,230,1)','rgba(189,183,107,1)','rgba(188,143,143,1)','rgba(221,160,221,1)','rgba(169,169,169,1)','rgba(255,160,122,1)','rgba(176,224,230,1)','rgba(169,169,169,1)','rgba(255,160,122,1)'])
temp = train['cuisine'].value_counts()
trace = go.Bar(y=temp.index[::-1],x=(temp / temp.sum() * 100)[::-1],orientation = 'h',marker=color_theme)
layout = go.Layout(title = "Top cuisine with recipe count (%)",xaxis=dict(title='Recipe count',tickfont=dict(size=14,)),
                   yaxis=dict(title='Cuisine',titlefont=dict(size=16),tickfont=dict(size=14)),margin=dict(l=200,))
data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig,filename='basic-bar')
temp = pd.DataFrame(temp)
temp.head()


# > **Top 5 Famous Food Contain Italian, Mexican, Southern_us, Indian and Chinese**

# In[ ]:


labels = temp.index.tolist()
values = (temp["cuisine"]/temp["cuisine"].sum())*100

trace = go.Pie(labels=labels, values=values,hole = 0.4)

iplot([trace], filename='basic_pie_chart')


# ## All Cuisine by Popularity

# In[ ]:


def cuisine_dish(train, cuisine):
    temp1 = train[train['cuisine'] == cuisine]
    n=6714 # total ingredients in train data
    top = Counter([item for sublist in temp1.ingredients for item in sublist]).most_common(n)
    temp= pd.DataFrame(top)
    temp.columns = ['ingredient','total_count']
    temp = temp.head(20)
    trace0 = go.Pie(labels=temp.ingredient[::-1], values=temp.total_count[::-1],hole = 0.4)
    data = [trace0]
    fig = go.Figure(data = data, layout= dict(title = "Famous Cuisine of '"+cuisine+"'"))
    iplot(fig,filename='basic_pie_chart')


# In[ ]:


cuisine_dish(train,"italian")


# ## Insight from Italian Food
# 
# - **Salt and Olive oil** are **most used ingredients.**
# - **Eggs and Dried oragano** are **least used ingredients**.

# In[ ]:


cuisine_dish(train,"southern_us")


# ## Insight from Southern_us Food
# 
# - **Salt and Butter** are **most used ingredients.**
# - **Garlic Cloves  and Olive Oil** are **least used ingredients**.

# In[ ]:


cuisine_dish(train,"mexican")


# ## Insight from Mexican Food
# 
# - **Salt and Onions** are **most used ingredients.**
# - **Lime  and Vegetable oil** are **least used ingredients**.

# In[ ]:


cuisine_dish(train,"indian")


# ## Insight from Indian Food
# 
# - **Salt , Onions,  Garam Masala, Water** are **most used ingredients.**
# - **Olive Oil  and Fresh Ginger** are **least used ingredients**.

# In[ ]:


cuisine_dish(train,"chinese")


# ## Insight from Chinese Food
# 
# - **Soy sauce,  Seasame Oil, Salt, Corn Starch, Sugar** are **most used ingredients.**
# - **Onions and Eggs** are **least used ingredients**.

# In[ ]:


top_n = Counter([item for sublist in train.ingredients for item in sublist]).most_common(70)
top__n_count = pd.DataFrame(top_n)
top__n_count.columns = ["ingredients","Count"]
labels = top__n_count["ingredients"]
values = (top__n_count["Count"]/top__n_count["Count"].sum())*100

trace = go.Pie(labels=labels, values=values,hole = 0.4)

iplot([trace], filename='basic_pie_chart')


# ## Finding a Similar Dishes

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
tokenizer=lambda x: [i.strip() for i in x.split(',')]
vec = CountVectorizer(tokenizer)
counts = vec.fit_transform(train['seperated_ingredients'])
count=dict(zip(vec.get_feature_names(), counts.sum(axis=0).tolist()[0]))
count=pd.DataFrame(list(count.items()),columns=['Ingredient','Count'])


# In[ ]:


ingreList = []
for index, row in train.iterrows():
    ingre = row['ingredients']
    
    for i in ingre:
        if i not in ingreList:
            ingreList.append(i)

def binary(ingre_list):
    binaryList = []
    
    for item in ingreList:
        if item in ingre_list:
            binaryList.append(1)
        else:
            binaryList.append(0)
    
    return binaryList


# In[ ]:


train['bin ingredients']=train['ingredients'].apply(lambda x: binary(x))


# In[ ]:


from scipy import spatial

def Similarity(Id1, Id2):
    a = train.iloc[Id1]
    b = train.iloc[Id2]
    
    A = a['bin ingredients']
    B = b['bin ingredients']
    distance=spatial.distance.cosine(A,B)
    
    return distance, Id2


# In[ ]:


food=[]
for i in train.index:
    food.append(Similarity(1,i))
common_ingredients=sorted(food,key=lambda x: x[0])[1:10]
indexes=[]
for i in range(len(common_ingredients)):
    indexes.append(common_ingredients[i][1])
train.iloc[indexes]


# In[ ]:


import networkx as nx
import random
from itertools import combinations

G=nx.Graph()

G.clear()

random_pick_from_top_n = random.sample(top_n, 15)

for list_of_nodes in train.ingredients:
    filtered_nodes = set(list_of_nodes).intersection(set([x[0] for x in random_pick_from_top_n]))  
    for node1,node2 in list(combinations(filtered_nodes,2)): 
        G.add_node(node1)
        G.add_node(node2)
        G.add_edge(node1, node2)

plt.figure(figsize=(15, 15))
pos=nx.spring_layout(G, k=0.15)
nx.draw_networkx(G,pos,node_size=2500, node_color = color,
                     node_shape = "h",
                     edgecolor  = "k",
                     linewidths  = 5 ,
                     font_size  = 20 ,
                     alpha=.8)
plt.show()


# In[ ]:


import nltk
from collections import Counter
import plotly.plotly as py
import cufflinks as cf
import seaborn as sns
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
cf.set_config_file(offline=False, world_readable=True, theme='ggplot')


# In[ ]:


train['for ngrams']=train['seperated_ingredients'].str.replace(',',' ')
def ingre_cusine(cuisine):
    frame=train[train['cuisine']==cuisine]
    common=list(nltk.bigrams(nltk.word_tokenize(" ".join(frame['for ngrams']))))
    return pd.DataFrame(Counter(common),index=['count']).T.sort_values('count',ascending=False)[:15]


# In[ ]:


f,ax=plt.subplots(3,2,figsize=(20,20))
ingre_cusine('italian').plot.barh(ax=ax[0,0],width=0.9,color= flatui[0])
ax[0,0].set_title('Italian Cuisine')
ingre_cusine('mexican').plot.barh(ax=ax[0,1],width=0.9,color = flatui[1] )
ax[0,1].set_title('Mexican Cuisine')
ingre_cusine('southern_us').plot.barh(ax=ax[1,0],width=0.9,color=flatui[2])
ax[1,0].set_title('southern_us Cuisine')
ingre_cusine('indian').plot.barh(ax=ax[1,1],width=0.9,color=flatui[3])
ax[1,1].set_title('Indian Cuisine')
ingre_cusine('chinese').plot.barh(ax=ax[2,0],width=0.9,color=flatui[4])
ax[2,0].set_title('Chinese Cuisine')
ingre_cusine('thai').plot.barh(ax=ax[2,1],width=0.9,color=flatui[5])
ax[2,1].set_title('Chinese Cuisine')
plt.subplots_adjust(wspace=0.5)


# In[ ]:


import networkx as nx
def generate_ngrams(text, n):
    words = text.split(' ')
    iterations = len(words) - n + 1
    for i in range(iterations):
        yield words[i:i + n]


def net_diagram(*cuisines):
    ngrams = {}
    for title in train[train.cuisine==cuisines[0]]['for ngrams']:
            for ngram in generate_ngrams(title, 2):
                ngram = ','.join(ngram)
                if ngram in ngrams:
                    ngrams[ngram] += 1
                else:
                    ngrams[ngram] = 1

    ngrams_mws_df = pd.DataFrame.from_dict(ngrams, orient='index')
    ngrams_mws_df.columns = ['count']
    ngrams_mws_df['cusine'] = cuisines[0]
    ngrams_mws_df.reset_index(level=0, inplace=True)

    ngrams = {}
    for title in train[train.cuisine==cuisines[1]]['for ngrams']:
            for ngram in generate_ngrams(title, 2):
                ngram = ','.join(ngram)
                if ngram in ngrams:
                    ngrams[ngram] += 1
                else:
                    ngrams[ngram] = 1

    ngrams_mws_df1 = pd.DataFrame.from_dict(ngrams, orient='index')
    ngrams_mws_df1.columns = ['count']
    ngrams_mws_df1['cusine'] = cuisines[1]
    ngrams_mws_df1.reset_index(level=0, inplace=True)
    cuisine1=ngrams_mws_df.sort_values('count',ascending=False)[:25]
    cuisine2=ngrams_mws_df1.sort_values('count',ascending=False)[:25]
    df_final=pd.concat([cuisine1,cuisine2])
    g = nx.from_pandas_dataframe(df_final,source='cusine',target='index')
    cmap = plt.cm.RdYlGn
    colors = [n for n in range(len(g.nodes()))]
    k = 0.35
    pos=nx.spring_layout(g, k=k)
    nx.draw_networkx(g,pos, node_size=df_final['count'].values*8, cmap = cmap, node_color=colors, edge_color='grey', font_size=20, width=3)
    plt.title("Top 25 Bigrams for %s and %s" %(cuisines[0],cuisines[1]), fontsize=30)
    plt.gcf().set_size_inches(30,30)
    plt.show()
    plt.savefig('network.png')


# In[ ]:


net_diagram('indian','chinese')


# In[ ]:


net_diagram('indian','southern_us')


# In[ ]:


train.head()


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(binary=True).fit(train['seperated_ingredients'].values)

X_train_vectorized = tfidf.transform(train['seperated_ingredients'].values)
X_train_vectorized = X_train_vectorized.astype('float')
Result_transformed = tfidf.transform(test['seperated_ingredients'].values)
Result_transformed = Result_transformed.astype('float')


# In[ ]:


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y_transformed = encoder.fit_transform(train.cuisine)


# In[ ]:


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train_vectorized, y_transformed ,test_size=0.10  ,random_state = 0)


# In[ ]:


from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

classifier = SVC(C=100, # penalty parameter, setting it to a larger value 
                 kernel='rbf', # kernel type, rbf working fine here
                 degree=3, # default value, not tuned yet
                 gamma=1, # kernel coefficient, not tuned yet
                 coef0=1, # change to 1 from default value of 0.0
                 shrinking=True, # using shrinking heuristics
                 tol=0.001, # stopping criterion tolerance 
                 probability=False, # no need to enable probability estimates
                 cache_size=200, # 200 MB cache size
                 class_weight=None, # all classes are treated equally 
                 verbose=True, # print the logs 
                 max_iter=-1, # no limit, let it run
                 decision_function_shape=None, # will use one vs rest explicitly 
                 random_state=None)
model = OneVsRestClassifier(classifier, n_jobs=4)
model.fit(X_train , y_train)
model.score(X_train , y_train)


# In[ ]:


from sklearn.linear_model import LogisticRegression

clf1 = LogisticRegression(C=10,dual=False)
clf1.fit(X_train , y_train)
clf1.score(X_test, y_test)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

# clf = RandomForestClassifier(max_depth=8, random_state=0)
clf = RandomForestClassifier(n_jobs=-1, n_estimators=30,min_samples_leaf=3,max_features=0.7, oob_score=True)

clf.fit(X_train , y_train)
clf1.score(X_test, y_test)


# In[ ]:


from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
vclf=VotingClassifier(estimators=[('clf1',LogisticRegression(C=10,dual=False)),
                                  ('clf2',SVC(C=100,gamma=1,kernel='rbf',probability=True)),
                                  ('clf',RandomForestClassifier(n_jobs=-1, n_estimators=30,min_samples_leaf=3,max_features=0.7, oob_score=True))],voting='soft',weights=[1,2,3])
vclf.fit(X_train , y_train)
vclf.score(X_test, y_test)


# In[ ]:


predict = vclf.predict(Result_transformed)
# pe=np.argmax(predict, axis=1) 
# final=encoder.inverse_transform(pe)
# final.head()
predic


# In[ ]:


final=encoder.inverse_transform(predict)


# In[ ]:


sub["cuisine"] = final
sub.to_csv("Submission.csv", index=False)
sub.head()


# In[ ]:




