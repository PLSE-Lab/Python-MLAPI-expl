#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import seaborn as sns
import spacy
import re
sns.set()
py.init_notebook_mode(connected = True)
nlp = spacy.load('en')


# # Dear Genie, i want to create a new project for kickstarter, can you suggest me what a good project for my future?

# ![alt text](https://vignette.wikia.nocookie.net/2007scape/images/8/8c/Genie.png/revision/latest?cb=20151018052559)
# 
# # As you wish, my Dear. But before that my dear mere mortal human, we need to study the data first.
# 
# I took Runescape Genie picture because I played this game when I was a kid

# # Dear human, I will show you how stack multi-model to give you a prediction.
# 
# # My base models are bayes and SVM. Upper model is Extreme Gradient Boosting

# Dear human, I will create a function to process tags for your data-set.

# In[ ]:


def clearstring(string):
    string = re.sub('[^a-z ]+', '', string)
    string = string.split(' ')
    string = filter(None, string)
    string = ' '.join([y.strip() for y in string])
    tags = ','.join([str(i) for i in nlp(string) if i.pos_ in ['NOUN']])
    return tags


# In[ ]:


# df = pd.read_csv('../input/ks-projects-201801.csv',encoding = "ISO-8859-1", keep_default_na=False)
# df['year'] = pd.DatetimeIndex(df['launched']).year
# tags = []
# for i in range(df.shape[0]):
#     try:
#         tags.append(clearstring(df.iloc[i,1].lower()))
#     except:
#         print(df.iloc[i,1])
# df['tags'] = tags
# df.head()

# Skip this process to reduce time completion. I already uploaded processed CSV in this kernel


# In[ ]:


df = pd.read_csv('../input/dear-genie-kickstarter/dear_jin.csv',encoding = "ISO-8859-1", keep_default_na=False)
df.head()


# In[ ]:


year_unique, year_count = np.unique(df['year'], return_counts = True)
data = [go.Bar(
            x=year_unique[1:],
            y=year_count[1:],
    text=year_count[1:],
    textposition = 'auto',
            marker=dict(
                color='rgb(179, 224, 255)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=0.5),
            ),
            opacity=0.9
)]
layout = go.Layout(
    title = 'Year count',
     margin = dict(
        t = 50
    )
)
fig = go.Figure(data = data, layout = layout)
py.iplot(fig)


# ### See that human, 2015 was the most count you and your human submitted projects in KickStarter, but did you know how many only successful?

# In[ ]:


state = df.state.unique().tolist()
del state[state.index('live')]
data_bar = []
for i in state:
    year_unique, year_count = np.unique(df[df.state==i]['year'], return_counts = True)
    data_bar.append(go.Bar(x=year_unique[1:],y=year_count[1:],name=i))
layout = go.Layout(
    title = 'State per Year count',
     margin = dict(
        t = 50
    )
)
fig = go.Figure(data = data_bar, layout = layout)
py.iplot(fig)


# ### Only 20.971k successful on 2015, the rest you can hover by yourself mere human

# ### Now let we check your human post on what categories

# In[ ]:


main_category_unique, main_category_count = np.unique(df.main_category,return_counts = True)
data = [go.Bar(
            x=main_category_unique,
            y=main_category_count,
    text=main_category_count,
    textposition = 'auto',
            marker=dict(
                color='rgb(179, 204, 255)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=0.5),
            ),
            opacity=0.9
)]
layout = go.Layout(
    title = 'Category count',
     margin = dict(
        t = 50
    )
)
fig = go.Figure(data = data, layout = layout)
py.iplot(fig)


# ### Herm, Film and Video. I spent my entire life watching your human film in my cup, suddenly you found me

# ### But did you know how many it was successful based on category?

# In[ ]:


data_bar = []
for i in state:
    main_category_unique, main_category_count = np.unique(df[df.state==i]['main_category'], return_counts = True)
    data_bar.append(go.Bar(x=main_category_unique,y=main_category_count,name=i))
layout = go.Layout(
    title = 'State per Main Category count',
     margin = dict(
        t = 50
    )
)
fig = go.Figure(data = data_bar, layout = layout)
py.iplot(fig)


# ### Music successful more than failed, good job music, I hope Jazz contributed it

# ### Let we check inside categories of film and video

# In[ ]:


film_category_unique, film_category_count = np.unique(df[df.main_category == 'Film & Video'].category, return_counts=True)
data = [go.Bar(
            x=film_category_unique,
            y=film_category_count,
    text=film_category_count,
    textposition = 'auto',
            marker=dict(
                color='rgb(255, 224, 179)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=0.5),
            ),
            opacity=0.9
)]
layout = go.Layout(
    title = 'Category inside Film & Video',
     margin = dict(
        t = 50
    )
)
fig = go.Figure(data = data, layout = layout)
py.iplot(fig)


# ### Who spent their time watching a documentary? tell me!

# In[ ]:


data_bar = []
for i in state:
    main_category_unique, main_category_count = np.unique(df[(df.state==i) & (df.main_category == 'Film & Video')].category, return_counts = True)
    data_bar.append(go.Bar(x=main_category_unique,y=main_category_count,name=i))
layout = go.Layout(
    title = 'State per Category inside Film & Video count',
     margin = dict(
        t = 50
    )
)
fig = go.Figure(data = data_bar, layout = layout)
py.iplot(fig)


# ### See, documentary tends to failed almost double than successful. Let we check inside Music.

# In[ ]:


film_category_unique, film_category_count = np.unique(df[df.main_category == 'Music'].category, return_counts=True)
data = [go.Bar(
            x=film_category_unique,
            y=film_category_count,
    text=film_category_count,
    textposition = 'auto',
            marker=dict(
                color='rgb(255, 224, 179)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=0.5),
            ),
            opacity=0.9
)]
layout = go.Layout(
    title = 'Category inside Music',
     margin = dict(
        t = 50
    )
)
fig = go.Figure(data = data, layout = layout)
py.iplot(fig)


# ### Nooooo, Jazz is not the main contributor for Music.

# In[ ]:


data_bar = []
for i in state:
    main_category_unique, main_category_count = np.unique(df[(df.state==i) & (df.main_category == 'Music')].category, return_counts = True)
    data_bar.append(go.Bar(x=main_category_unique,y=main_category_count,name=i))
layout = go.Layout(
    title = 'State per Category inside Music count',
     margin = dict(
        t = 50
    )
)
fig = go.Figure(data = data_bar, layout = layout)
py.iplot(fig)


# # I am very glad Jazz successful more than failed

# ### Now we are going to check, which tags most used in Kickstarter

# In[ ]:


tags_succesful=(','.join(df[df.state=='successful'].tags.values.tolist())).split(',')
tags_succesful_unique, tags_succesful_count = np.unique(tags_succesful,return_counts = True)
ids=(-tags_succesful_count).argsort()[:20]


# In[ ]:


data = [go.Bar(
            x=tags_succesful_unique[ids],
            y=tags_succesful_count[ids],
    text=tags_succesful_count[ids],
    textposition = 'auto',
            marker=dict(
                color='rgb(217, 217, 217)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=0.5),
            ),
            opacity=0.9
)]
layout = go.Layout(
    title = 'Top 20 Unigram keywords for successful projects',
     margin = dict(
        t = 50
    )
)
fig = go.Figure(data = data, layout = layout)
py.iplot(fig)


# ### Album, can be related to music. Film, obviously related to film and video.

# ### Now we check, which Trigram always used in Kickstarter

# In[ ]:


trigram = []
for i in range(len(tags_succesful)-3):
    trigram.append(', '.join(tags_succesful[i:i+3]))
trigram_succesful_unique, trigram_succesful_count = np.unique(trigram,return_counts = True)
ids=(-trigram_succesful_count).argsort()[:20]
data = [go.Bar(
            x=trigram_succesful_unique[ids],
            y=trigram_succesful_count[ids],
    text=trigram_succesful_count[ids],
    textposition = 'auto',
            marker=dict(
                color='rgb(217, 217, 217)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=0.5),
            ),
            opacity=0.9
)]
layout = go.Layout(
    title = 'Top 20 Trigram keywords for successful projects',
     margin = dict(
        t = 50
    )
)
fig = go.Figure(data = data, layout = layout)
py.iplot(fig)


# ### Do mere human play a card game while riding a bicycle?

# ### How about 5-gram?

# In[ ]:


gram_5 = []
for i in range(len(tags_succesful)-5):
    gram_5.append(', '.join(tags_succesful[i:i+5]))
gram_5_succesful_unique, gram_5_succesful_count = np.unique(gram_5,return_counts = True)
ids=(-gram_5_succesful_count).argsort()[:20]
data = [go.Bar(
            x=gram_5_succesful_unique[ids],
            y=gram_5_succesful_count[ids],
    text=gram_5_succesful_count[ids],
    textposition = 'auto',
            marker=dict(
                color='rgb(217, 217, 217)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=0.5),
            ),
            opacity=0.9
)]
layout = go.Layout(
    title = 'Top 20 5-Gram keywords for successful projects',
     margin = dict(
        t = 50,
         b= 200
    )
)
fig = go.Figure(data = data, layout = layout)
py.iplot(fig)


# ### Any bowlmaker name david? Now we are going to study the probability of pledged to be successful

# In[ ]:


decision = df.state.values
data_array = df.pledged.values
data_bar = []
for no, k in enumerate(state):
    weights = np.ones_like(data_array[decision == k])/float(len(data_array[decision == k]))
    n, bins, _ = plt.hist(data_array[decision == k], 10,weights=weights)
    loc = np.where(n >= 0.5)[0]
    plt.clf()
    data_bar.append(go.Bar(x=bins[loc],y=n[loc],name=k))
layout = go.Layout(
    title = 'Probability how much pledged to be for states',
     margin = dict(
        t = 50
    )
)
fig = go.Figure(data = data_bar, layout = layout)
py.iplot(fig)


# ### Actually, higher pledged doesnt mean your project will be successful

# In[ ]:


decision = df.state.values
data_array = df.backers.values
data_bar = []
for no, k in enumerate(state):
    weights = np.ones_like(data_array[decision == k])/float(len(data_array[decision == k]))
    n, bins, _ = plt.hist(data_array[decision == k], 10,weights=weights)
    loc = np.where(n >= 0.5)[0]
    plt.clf()
    data_bar.append(go.Bar(x=bins[loc],y=n[loc],name=k))
layout = go.Layout(
    title = 'Probability how much backers to be for states',
     margin = dict(
        t = 50
    )
)
fig = go.Figure(data = data_bar, layout = layout)
py.iplot(fig)


# ### Also same, your got a lot of backers also doesnt confirm your project will be successful

# In[ ]:


sns.pairplot(df[['goal','pledged','state','backers']], hue="state",size=5)
plt.cla()
plt.show()


# ### Now we are going to train classifiers to do our prediction, Genie need this!

# In[ ]:


tags_to_train = df.tags.iloc[np.where((df.tags != '') & (df.state != 'live'))[0]].tolist()
label_to_train = df.state.iloc[np.where((df.tags != '') & (df.state != 'live'))[0]]
# change to binary classification
label_to_train[label_to_train == 'canceled'] = 'failed'
label_to_train[label_to_train == 'undefined'] = 'failed'
label_to_train[label_to_train == 'suspended'] = 'failed'
label_to_train = label_to_train.tolist()
for i in range(len(tags_to_train)):
    tags_to_train[i] = tags_to_train[i].replace(',', ' ')


# Change into bag-of-word vectorization for our tags

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
Y = LabelEncoder().fit_transform(label_to_train)
bow = CountVectorizer().fit(tags_to_train)
X = bow.transform(tags_to_train)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier


# In[ ]:


clf_huber = SGDClassifier(loss = 'modified_huber', 
                                  penalty = 'l2', alpha = 1e-3, 
                                  n_iter = 50).fit(X, Y)


# In[ ]:


clf_bayes = MultinomialNB().fit(X, Y)


# From the output probability of bayes and svm will be stacked column-wise

# In[ ]:


stacked=np.hstack([clf_bayes.predict_proba(X), clf_huber.predict_proba(X)])


# In[ ]:


import xgboost as xgb
params_xgd = {
    'max_depth': 7,
    'objective': 'binary:logistic',
    'learning_rate': 0.05,
    'nthread': -1,
    'silent': False,
    'n_estimators': 100
    }
clf = xgb.XGBClassifier(**params_xgd)
clf.fit(stacked,Y)


# In[ ]:


fig, ax = plt.subplots(figsize=(12,5))
xgb.plot_importance(clf, ax=ax)
plt.show()


# In[ ]:


print(np.mean(clf_bayes.predict(X) == Y))
print(np.mean(clf_huber.predict(X) == Y))
print(np.mean(clf.predict(stacked) == Y))


# ### The stacked model only helped us a little bit, better than none right?

# ### Now I will show to you mere human, i do example to suggest 'bicyle' and a 'shit'

# In[ ]:


from fuzzywuzzy import fuzz
import random
guess = ['bicyle', 'shit']
result_guess = []
tags = df.tags.values.tolist()
for i in guess:
    picked=np.argmax([fuzz.ratio(guess[1], n) for n in tags_succesful_unique])
    results=np.where(np.array([n.find(tags_succesful_unique[picked]) for n in tags]) >=0)[0][:10]
    for k in results:
        count = 0
        while True and count < 10:
            selected = random.choice(tags[k].split(','))
            if selected not in result_guess:
                result_guess.append(selected)
                break
            count+=1


# In[ ]:


result_guess


# ### Now you see, bicycle and a shit give you this nearest tags

# In[ ]:


from sklearn.neighbors import NearestNeighbors
from random import shuffle
def help_me_genie(wish, suggest_count):
    if wish.find('oh genie, suggest me') < 0:
        return "you need to call me by 'oh genie, suggest me'"
    guess = [i.strip() for i in wish[len('oh genie, suggest me '):].split(',')]
    print('your wish is:',guess)
    result_guess = []
    for i in guess:
        picked=np.argmax([fuzz.ratio(guess[1], n) for n in tags_succesful_unique])
        results=np.where(np.array([n.find(tags_succesful_unique[picked]) for n in tags]) >=0)[0][:10]
        for k in results:
            for n in range(20):
                selected = random.choice(tags[k].split(','))
                if selected not in result_guess:
                    result_guess.append(selected)
                    break
                
    print('your result guess:', result_guess)
    jin_guess=np.zeros((df.shape[0], len(result_guess)))
    for i in range(df.shape[0]):
        for k in range(len(result_guess)):
            if tags[i].find(result_guess[k]) >= 0:
                jin_guess[i, k] += 1
    nbrs = NearestNeighbors(n_neighbors=suggest_count, algorithm='auto', metric='sqeuclidean').fit(jin_guess)
    id_entry = np.argmax(np.sum(jin_guess,axis=1)) 
    xtest = jin_guess[id_entry, :].reshape(1, -1)
    distances, indices = nbrs.kneighbors(xtest)
    results = []
    for i in indices[0][:]:
        items = tags[i].split(',') + [random.choice(result_guess)]
        shuffle(items)
        results.append(' '.join(items))
    prob=clf.predict_proba(np.hstack([clf_huber.predict_proba(bow.transform(results)), clf_bayes.predict_proba(bow.transform(results))]))
    for i in range(len(results)):
        print(results[i], ', successful rate:', prob[i,1]*100, '%')


# ### Because you find my cup, you can give me any tags and count to suggest for you.

# ### me: oh okey, I want to study bicycle and invest!

# In[ ]:


help_me_genie('oh genie, suggest me bicycle, invest', 10)


# ### me: how about taylor swift and a broom?

# In[ ]:


help_me_genie('oh genie, suggest me taylor swift, broom', 10)


# ### me: how about a guitar, a shit and a trousers?

# In[ ]:


help_me_genie('oh genie, suggest me guitar, shit, trousers', 30)


# # Thank you genie, now i feel confidence to create a new project related to guitar and a trousers!

# In[ ]:


help_me_genie('guitar, shit, trousers', 20)


# In[ ]:


help_me_genie('oh genie, suggest me guitar, shit, trousers, smart phone, aloe vera', 20)


# In[ ]:




