#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as ply

import missingno
pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)


# In[ ]:


import plotly.express as px


# In[ ]:


data=pd.read_csv("../input/amazonearphonesreviews/AllProductReviews.csv")


# In[ ]:


data1=pd.read_csv("../input/amazonearphonesreviews/ProductInfo.csv")


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


data1.shape


# In[ ]:


data1


# In[ ]:


boat255=data[data.Product=="boAt Rockerz 255"]


# In[ ]:


boat255.head()


# In[ ]:





# In[ ]:


boat255=data[data.Product=="boAt Rockerz 255"]


# In[ ]:


flybotwave=data[data.Product=="Flybot Wave"]


# In[ ]:


flybotboom=data[data.Product=="Flybot Boom"]


# In[ ]:


PTronintunes=data[data.Product=="PTron Intunes"]


# In[ ]:


flybotbeat=data[data.Product=="Flybot Beat"]


# In[ ]:


samsungeo=data[data.Product=="Samsung EO-BG950CBEIN"]


# In[ ]:


jblt2=data[data.Product=="JBL T205BT"]


# In[ ]:


jblt1=data[data.Product=="JBL T110BT"]


# In[ ]:


skullcandy=data[data.Product=="Skullcandy S2PGHW-174"]


# In[ ]:


seinh=data[data.Product=="Sennheiser CX 6.0BT"]


# In[ ]:


dblist=[boat255,flybotwave,flybotboom,flybotbeat,PTronintunes,samsungeo,jblt2,jblt1,skullcandy,seinh]


# In[ ]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()


# In[ ]:


from textblob import TextBlob


# In[ ]:


def scores(x):
    list1=[]
    x.reset_index(inplace=True)
    x.drop("index",axis=1,inplace=True)
    for i in x.ReviewBody:
        list1.append(sia.polarity_scores(i))
    x[["Negative","Neutral","Positive","Compound"]]=pd.DataFrame(list1)


# In[ ]:





# In[ ]:


for j in dblist:
    scores(j)


# # Working on the Boat Rockerz 225

# In[ ]:


boat255.head()


# In[ ]:


px.line(y="Compound",data_frame=boat255,width=20000, height=400)


# In[ ]:


boat255.head()


# In[ ]:


def cls(x):
    list5=[]
    for i in x["Compound"]:
        if i>0:
            list5.append("Positive")
        elif i==0:
            list5.append("Neutral")
        else:
            list5.append("Negative")
    x["Score"]=list5


# In[ ]:


for j in dblist:
    cls(j)


# In[ ]:


boat255.head()


# In[ ]:


boat255["Score"].value_counts()


# In[ ]:


import plotly.graph_objs as go

Mno=boat255[boat255.Score=="Positive"]["Score"].count()
Fno=boat255[boat255.Score=="Negative"]["Score"].count()
Nno=boat255[boat255.Score=="Neutral"]["Score"].count()
labels = ["Positive Comments","Negative Comments","Neutral Comments"]
values = [Mno,Fno,Nno]
fig = go.Figure(data=[go.Pie(labels=labels, values=values,hole=0.5)])
fig.show()


# In[ ]:


def keys(x):
    a = [None] * len(x)
    for i in range(0,(len(x)-1)):
        list3=[]
        blob = TextBlob(x.iloc[i][1])
        for word, tag in blob.tags:
            if (tag=="JJ")| (tag=="VBN")| (tag=="NNS")| (tag=="NN"):
                list3.append(word.lemmatize())
            a[i]=list3
    x["Keywords"]=a    


# In[ ]:


for j in dblist:
    keys(j)


# In[ ]:


boat255.head()


# In[ ]:


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# In[ ]:


stop_words=set(stopwords.words("english"))


# In[ ]:


filtered_sentence=[]


# In[ ]:


def filter(x):
    a = [None] * len(x)
    for i in range(0,(len(x)-1)):
        list3=[]
        blob = word_tokenize(x.iloc[i][1])
        for word in blob:
            if word not in stop_words:
                list3.append(word)
            a[i]=list3
    x["filter"]=a    


# In[ ]:


boat255.head()


# In[ ]:


for j in dblist:
    filter(j)


# In[ ]:


boat255.head()


# In[ ]:


# WordCloud to highlight important keywords used in reviews


# In[ ]:


from wordcloud import WordCloud
import numpy as np
from PIL import Image
list=[]
for i in boat255["filter"] :
    list.append(i)
slist = str(list)

wordcloud = WordCloud(width=1000, height=500).generate(slist)
plt.figure(figsize=(20,20))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()


# In[ ]:


boat255.head() 


# # The Same techniques can be applied for each product reviews

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




