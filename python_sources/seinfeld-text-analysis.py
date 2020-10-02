#!/usr/bin/env python
# coding: utf-8

# # Seinfeld Text Analysis

# In this notebook I will analyze the Seinfeld script, for starters i will show how to parse the data into a dataframe, later I will show some data exploration techniques with some visualization with an interactive graph. afterwards i will try to build a weighted network of the relations between some of the characters. and finally will construct a few models that will try to classify if a given text was said by one of the main roles in the TV Show.
# 
# -- this is my first notebook submission, i hope you will like it -- 

# ## The imports:

# In[ ]:


import spacy
import en_core_web_sm
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from bokeh.io import show, output_file
from bokeh.models import Plot, Range1d, MultiLine, Circle, HoverTool, TapTool, BoxSelectTool, BoxZoomTool, ResetTool
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.io import output_file,show,output_notebook,push_notebook

from bokeh.palettes import Spectral4


get_ipython().run_line_magic('matplotlib', 'inline')


# # Parsing

# Parsing the text into a list of all the sentences:

# In[ ]:


path = "../input/corpus.txt"
with open(path) as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content] 
content[:20]


# Now let's clean the data and import it into a datafrmae, I will use SEP in order to seperate between scenes in order to use it for later.

# In[ ]:


speaker_list = []
text_list = []
for c in content:
    temp = c.split(':')
    speaker = temp[0].split()[0]
#     if (speaker[0] == '(') or (speaker[0] == '['):
    if not (speaker[0].isalpha()):
        speaker_list.append("SEP")
        text_list.append("SEP")
        continue
    if 'INT' in speaker:
        speaker_list.append("SEP")
        text_list.append("SEP")
        continue
    if len(temp) < 2:
        try:
            text = temp[0].split(' ', 1)[1]
        except:
            continue
    else:
        text =temp[1]
    speaker_list.append(speaker)
    text_list.append(text)
df = pd.DataFrame(columns=['speaker', 'text'])
df['speaker'] = speaker_list
df['text'] = text_list
df.iloc[:20]


# # Data Exploration and Visualizations

# first let's try to see how much lines did each of the roles had

# In[ ]:


counter = df.copy()
drop_list = counter[counter['speaker'] == 'SEP'].index
counter = counter.drop(drop_list)
counter = counter.groupby(['speaker'], as_index = False).count()
counter = counter.sort_values('text', ascending= False)
counter.columns = ['speaker','lines count']
counter = counter.reset_index(drop = True)
counter.iloc[:20]


# Not suprising that Jerry is leading with almost 15K lines!
# 
# Now to some visualizations

# In[ ]:


plt.figure(figsize = (16, 16), facecolor = None)
sns.set_palette("Paired")
sns.barplot(x="speaker" ,y="lines count", data=counter.iloc[:20])
plt.show()


# Zooming in the top 4 roles:

# In[ ]:


plt.figure(figsize = (8, 8), facecolor = None)
sns.barplot(x="speaker" ,y="lines count", data=counter.iloc[:4])
plt.show()


# # Creating a Weighted Graph

# Let's try to extract the names of persons from each of the lines in order to create the connection in our graph using a spacy entity extraction.
# 
# 

# In[ ]:


nlp = en_core_web_sm.load()
def extract_name(text):
    entities = []
    doc = nlp(text)
    for entity in doc.ents:
        label = entity.label_
        if label != 'PERSON':
            continue
        e = entity.text.upper().strip()
        if len(e) < 3:
            continue
        entities.append(e)
    return entities
df['ppl'] = df['text'].apply(extract_name)
df.iloc[:-15]


# OK, so this didn't go as planned, some misclassification of a person's name such as the name extracted: 'OOOOOO' was extracted by spacy. 
# 
# So, let's try a different approach... I will connect between all the characters that were on the same scene together and spoke in a sequence manner (that's why I left the SEP earlier:

# In[ ]:


df['person2'] = '-'
for index , row in df.iterrows():
    if row['speaker'] == 'SEP':
        continue
    if index+1 == len(df):
        continue
    next_preson = df.iloc[index + 1]['speaker']
    if next_preson == 'SEP':
        continue
    if next_preson == row['speaker']:
        continue
    df.at[index, 'person2'] = next_preson
df.iloc[:10]


# Now let's remove the SEP between scenes

# In[ ]:


drop_list1 = df[df['speaker'] == 'SEP'].index
df2 = df.drop(drop_list1)
drop_list2 = df2[df2['person2'] == '-'].index
df2 = df2.drop(drop_list2)
df2 = df2.reset_index()
df2 = df2[['speaker','person2']]
df2.columns = ['p1', 'p2']
df2.iloc[:10]


# I will use the number of appearances of each edge to determine the weight of each edge in the network

# In[ ]:


df2['count'] = 0
df2 = df2.groupby(['p1', 'p2'], as_index = False).count()
df2 = df2.sort_values('count', ascending = False)
df2.iloc[:10]


# There is some clutter with low frequency connections, I will toss all the connections that appeared less than 3 times.

# In[ ]:


df2 = df2[df2['count'] > 3]
df2.iloc[:10]


# Now let's create the weighted network:

# In[ ]:


G = nx.Graph()

for index , row in df2.iterrows():
    G.add_edge(row['p1'],row['p2'] , weight=row['count'])

print(nx.info(G))


# And now to plot the Graph:

# In[ ]:


plt.figure(figsize = (20, 20), facecolor = None)
nx.draw_kamada_kawai(G, with_labels=True)


# Zooming in on nodes that have more than 15 connections (in addition to limiting up to 300 nodes):

# In[ ]:


df2 = df2[df2['count'] > 15]
G = nx.Graph()

for index , row in df2.iterrows():
    G.add_edge(row['p1'],row['p2'] , weight=row['count'])
    
plt.figure(figsize = (20, 20), facecolor = None)
pos=nx.spring_layout(G)
nx.draw(G, pos, node_size=300, with_labels=True)


# Let's plot a more interactive and modern graph plot.

# # Interactive Plotting

# Each edge is colored as follow:
# 
# JERRY = red
# GEORGE = blue
# KRAMER = black
# ELAINE= green
# Other = orange

# In[ ]:


output_notebook()

color_dict = {"JERRY": "red","GEORGE": "blue","KRAMER": "black","ELAINE": "green"}
edge_attrs = {}

for start_node, end_node, _ in tqdm(G.edges(data=True)):
    if start_node in color_dict:
        edge_color = color_dict[start_node]
    else:
        edge_color = "orange"
    edge_attrs[(start_node, end_node)] = edge_color

nx.set_edge_attributes(G, edge_attrs, "edge_color")

plot = Plot(plot_width=800, plot_height=800,
            x_range=Range1d(-1.1,1.1), y_range=Range1d(-1.1,1.1))

plot.title.text = "Graph Interaction Seinfeld"

node_hover_tool = HoverTool(tooltips=[("Name", "@index")])

plot.add_tools(node_hover_tool, BoxZoomTool(), ResetTool())

graph_renderer = from_networkx(G, nx.spring_layout, scale=1, center=(0, 0))

graph_renderer.node_renderer.glyph = Circle(size=15, fill_color=Spectral4[2])
graph_renderer.edge_renderer.glyph = MultiLine(line_color="edge_color", line_alpha=0.8, line_width=1)
plot.renderers.append(graph_renderer)

output_file("interactive_graphs.html")
show(plot,notebook_handle=True) 


# We can see that Jerry, Elaine, George and Kramer are somewhere in the center nodes and have alot of connections (not surprising...)

# # Line Classifier and Text Analysis

# First I will focus on the lines of the main roles and insert them into a df

# In[ ]:


df3 = df[['speaker', 'text']]
drop_list1 = df3[df3['speaker'] == 'SEP'].index
df3 = df3.drop(drop_list1)
df3 = df3.reset_index(drop = True)
df3 = df3[df3['speaker'].isin({'JERRY' , 'GEORGE', 'KRAMER', 'ELAINE'})]
df3.iloc[:10]


# Now I will try to create some TFIDF and Word count features for the classifier and will try to see which model preduce the best accuracy.
# The classifiers will be Adaboost and Random forest which are quiet popular classifiers.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df3['text'], df3['speaker'], test_size=0.2, random_state=124)
res_list = pd.DataFrame(columns=['Method', 'Model', 'Acc'])


# In[ ]:


def fit_evaluate(x_train, y_train, x_test, y_test, print_report = False):
    rf = RandomForestClassifier(n_estimators = 25)
    ada = AdaBoostClassifier(n_estimators = 25)
    rf.fit(x_train, y_train)
    ada.fit(x_train, y_train)
    pred1 = rf.predict(x_test)
    pred2 = ada.predict(x_test)
    score_rf = accuracy_score(y_test, pred1)
    # auc = roc_auc_score(y_test, pred1)
    print("Random forest \n Accuracy:", score_rf)
    if print_report:
        print(classification_report(y_test, pred1))
    score_ada = accuracy_score(y_test, pred2)
    print("Adaboost \n Accuracy:", score_ada)
    if print_report:
        print(classification_report(y_test, pred2))
    return score_rf, score_ada


# In[ ]:


tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', strip_accents='unicode',analyzer='char', norm='l2')
tfidf = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)
s1 , s2 = fit_evaluate(tfidf, y_train, tfidf_test, y_test, print_report = True)
res_list = res_list.append({'Method':'tfidf - char, ngram 1', 'Model': 'RF', 'Acc': s1}, ignore_index=True)
res_list = res_list.append({'Method':'tfidf - char, ngram 1', 'Model': 'Ada', 'Acc': s2}, ignore_index=True)


# In[ ]:


tfidf_vectorizer = TfidfVectorizer(ngram_range=(2, 3), stop_words='english', strip_accents='unicode',analyzer='char', norm='l2')
tfidf = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)
s1 , s2 = fit_evaluate(tfidf, y_train, tfidf_test, y_test)
res_list = res_list.append({'Method':'tfidf - char, ngram (2,3)', 'Model': 'RF', 'Acc': s1},ignore_index=True)
res_list = res_list.append({'Method':'tfidf - char, ngram (2,3)', 'Model': 'Ada', 'Acc': s2},ignore_index=True)


# In[ ]:


tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', strip_accents='unicode',analyzer='word', norm='l2')
tfidf = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)
s1 , s2 = fit_evaluate(tfidf, y_train, tfidf_test, y_test)
res_list = res_list.append({'Method':'tfidf - word, ngram 1', 'Model': 'RF', 'Acc': s1},ignore_index=True)
res_list = res_list.append({'Method':'tfidf - word, ngram 1', 'Model': 'Ada', 'Acc': s2},ignore_index=True)


# In[ ]:


count_vectorizer = CountVectorizer(ngram_range=(1, 1), stop_words='english', strip_accents='unicode',
                                   analyzer='word')
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)
s1, s2 = fit_evaluate(count_train, y_train, count_test, y_test)
res_list = res_list.append({'Method':'Count - Word, ngram 1', 'Model': 'RF', 'Acc': s1},ignore_index=True)
res_list = res_list.append({'Method':'Count - Word, ngram 1', 'Model': 'Ada', 'Acc': s2},ignore_index=True)


# In[ ]:


count_vectorizer = CountVectorizer(ngram_range=(3, 4), stop_words='english', strip_accents='unicode',
                                   analyzer='char')
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)
s1, s2 = fit_evaluate(count_train, y_train, count_test, y_test)
res_list = res_list.append({'Method':'Count - char, ngram (3,4)', 'Model': 'RF', 'Acc': s1},ignore_index=True)
res_list = res_list.append({'Method':'Count - char, ngram (3,4)', 'Model': 'Ada', 'Acc': s2},ignore_index=True)


# In[ ]:


count_vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words='english', strip_accents='unicode',
                                   analyzer='char')
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)
s1, s2 = fit_evaluate(count_train, y_train, count_test, y_test)
res_list = res_list.append({'Method':'Count - char, ngram 2', 'Model': 'RF', 'Acc': s1},ignore_index=True)
res_list = res_list.append({'Method':'Count - char, ngram 2', 'Model': 'Ada', 'Acc': s2},ignore_index=True)


# Ok the best shot so far was 39% accuracy,,, I think there is still room for improvment
# 
# let's try with Word2Vec:
# 
# *Tip: TQDM is a very nice runtime bar for your code...*

# In[ ]:


nlp = en_core_web_sm.load()
vectors = []
for t in tqdm(df3['text']):
    vectors.append(nlp(t).vector)

df3['vector'] = vectors
vectors = pd.DataFrame(df3.vector.values.tolist(), index= df3.index)
vectors.iloc[:20]


# I noticed that there are a few nans on the data, lets check if I was right...

# In[ ]:


vectors.isnull().sum().sum()


# So there are some Nans apperantly, lets impute it simply by changing the value to 0 (there arent alot of missing data thus probably wont matter that much).

# In[ ]:


vectors = vectors.replace(np.nan , 0)
vectors.isnull().sum().sum()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(vectors, df3['speaker'], test_size=0.2, random_state=124)


# In[ ]:


s1, s2 = fit_evaluate(X_train, y_train, X_test, y_test)
res_list = res_list.append({'Method':'Vector', 'Model': 'RF', 'Acc': s1},ignore_index=True)
res_list = res_list.append({'Method':'Vector', 'Model': 'Ada', 'Acc': s2},ignore_index=True)


# Now let's see what model perfomed the best:

# In[ ]:


res_list.sort_values('Acc', ascending=False)


# We can see that the results are still not that good... 
# let's try to a build model for each main role and see if it improved the accuracy of the model:

# First, one hot encoding of the target value:

# In[ ]:


df3['is_j'] = df3['speaker'].apply(lambda x: int(x == 'JERRY'))
df3['is_e'] = df3['speaker'].apply(lambda x: int(x == 'ELAINE'))
df3['is_k'] = df3['speaker'].apply(lambda x: int(x == 'KRAMER'))
df3['is_g'] = df3['speaker'].apply(lambda x: int(x == 'GEORGE'))
df3.iloc[:10]


# In[ ]:


res_list_2 = pd.DataFrame(columns=['Method', 'Model', 'Acc'])


# Now building a model per role:

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df3['text'], df3['is_g'], test_size=0.2, random_state=124)

tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', strip_accents='unicode',analyzer='word', norm='l2')
tfidf = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

s1 , s2 = fit_evaluate(tfidf, y_train, tfidf_test, y_test)

res_list_2 = res_list_2.append({'Method':'TFIDF - George model', 'Model': 'RF', 'Acc': s1},ignore_index=True)
res_list_2 = res_list_2.append({'Method':'TFIDF - George model', 'Model': 'Ada', 'Acc': s2},ignore_index=True)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df3['text'], df3['is_j'], test_size=0.2, random_state=124)

tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', strip_accents='unicode',analyzer='word', norm='l2')
tfidf = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)
s1 , s2 = fit_evaluate(tfidf, y_train, tfidf_test, y_test)

res_list_2 = res_list_2.append({'Method':'TFIDF - Jerry model', 'Model': 'RF', 'Acc': s1},ignore_index=True)
res_list_2 = res_list_2.append({'Method':'TFIDF - Jerry model', 'Model': 'Ada', 'Acc': s2},ignore_index=True)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df3['text'], df3['is_k'], test_size=0.2, random_state=124)

tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', strip_accents='unicode',analyzer='word', norm='l2')
tfidf = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

s1 , s2 = fit_evaluate(tfidf, y_train, tfidf_test, y_test)

res_list_2 = res_list_2.append({'Method':'TFIDF - Kramer model', 'Model': 'RF', 'Acc': s1},ignore_index=True)
res_list_2 = res_list_2.append({'Method':'TFIDF - Kramer model', 'Model': 'Ada', 'Acc': s2},ignore_index=True)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df3['text'], df3['is_e'], test_size=0.2, random_state=124)

tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', strip_accents='unicode',analyzer='word', norm='l2')
tfidf = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

s1 , s2 = fit_evaluate(tfidf, y_train, tfidf_test, y_test)

res_list_2 = res_list_2.append({'Method':'TFIDF - Elaine model', 'Model': 'RF', 'Acc': s1},ignore_index=True)
res_list_2 = res_list_2.append({'Method':'TFIDF - Elaine model', 'Model': 'Ada', 'Acc': s2},ignore_index=True)


# Let's see how we did...

# In[ ]:


res_list_2.sort_values('Acc', ascending=False)


# ok so the results looks better for each one of the binary classifiers, lets make an esamble of those in order to see if it works better
# 
# I will try 2 versions of the ensamble, one with weights per model (factor by thier performance) and without weights, the ensamble will be decided by the highest predict proba score

# In[ ]:


def ensemble_fit_evaluate(df3, model, weights = False):
    role_list = ['is_j', 'is_g', 'is_e', 'is_k']
    name_dict = {'is_j': 'JERRY', 'is_g': 'GEORGE', 'is_e': 'ELAINE', 'is_k' : 'KRAMER'}
    wieghts_dict = {'is_j': 0.64, 'is_g': 0.74, 'is_e': 0.8, 'is_k' : 0.83}

    X_train, X_test, y_train, y_test = train_test_split(df3['text'], df3['speaker'], test_size=0.2, random_state=124)
    preds = pd.DataFrame(y_test)
    preds = preds.reset_index(drop=True)
    for r in role_list:
        X_train, _, y_train, __ = train_test_split(df3['text'], df3[r], test_size=0.2, random_state=124)
        
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', strip_accents='unicode',analyzer='word', norm='l2')
        tfidf = tfidf_vectorizer.fit_transform(X_train)
        tfidf_test = tfidf_vectorizer.transform(X_test)
        clf = model
        clf.fit(tfidf, y_train)
        if weights:
            pred = clf.predict_proba(tfidf_test) * wieghts_dict[r]
        else:
            pred = clf.predict_proba(tfidf_test)
        preds[name_dict[r]] = pred[:,1]
        
    return preds 


# In[ ]:


model = RandomForestClassifier(n_estimators = 50)
preds = ensemble_fit_evaluate(df3, model = model, weights = False)
preds['pred'] = preds[['JERRY','GEORGE', 'ELAINE','KRAMER']].idxmax(axis=1)
preds.iloc[:10]


# Creating another res list for performance measurement:

# In[ ]:


res_list3 = pd.DataFrame(columns=['Method', 'Model', 'Acc'])


# In[ ]:


score = accuracy_score(preds['speaker'], preds['pred'])
print(classification_report(preds['speaker'], preds['pred']), "\n Acc = ", score)
res_list3 = res_list3.append({'Method':'Ensamble - TFIDF', 'Model': 'RF - no weights', 'Acc': score},ignore_index=True)


# In[ ]:


model = RandomForestClassifier(n_estimators = 50)
preds = ensemble_fit_evaluate(df3, model = model, weights = True)    
preds['pred'] = preds[['JERRY','GEORGE', 'ELAINE','KRAMER']].idxmax(axis=1)

score = accuracy_score(preds['speaker'], preds['pred'])
print(classification_report(preds['speaker'], preds['pred']), "\n Acc = ", score)

res_list3 = res_list3.append({'Method':'Ensamble - TFIDF', 'Model': 'RF - With weights', 'Acc': score},ignore_index=True)


# In[ ]:


model = AdaBoostClassifier(n_estimators = 500)
preds = ensemble_fit_evaluate(df3, model = model, weights = True) 

preds['pred'] = preds[['JERRY','GEORGE', 'ELAINE','KRAMER']].idxmax(axis=1)
score = accuracy_score(preds['speaker'], preds['pred'])

print(classification_report(preds['speaker'], preds['pred']), "\n Acc = ", score)
res_list3 = res_list3.append({'Method':'Ensamble - TFIDF', 'Model': 'Adaboost - With weights', 'Acc': score},ignore_index=True)


# In[ ]:


model = AdaBoostClassifier(n_estimators = 500)
preds = ensemble_fit_evaluate(df3, model = model, weights = False) 

preds['pred'] = preds[['JERRY','GEORGE', 'ELAINE','KRAMER']].idxmax(axis=1)
score = accuracy_score(preds['speaker'], preds['pred'])

print(classification_report(preds['speaker'], preds['pred']), "\n Acc = ", score)

res_list3 = res_list3.append({'Method':'Ensamble - TFIDF', 'Model': 'Adaboost - No weights', 'Acc': score},ignore_index=True)


# This is how we did in the model per role:

# In[ ]:


res_list3.sort_values('Acc', ascending=False)


# And that's the performance of a global classifier:

# In[ ]:


res_list.sort_values('Acc', ascending=False)


# So it looks that using a model per role there was an improvment in accuracy of ~ 1.5% to 40.9% accuracy on the test-set with the Adaboost model.

# ### Some Conclusions:
# The binary classification task had provided better accuracy levels than the merge of the binary classifiers.
# 
# I guess that it might be possible to improve the building of the ensamble and that way to improve the performance but... That's it for now... 

# Hope you enjoyed this notebook and that it will help you in your data endeavours.
# If you liked it or have any questions feel free to leave a comment :)
