#!/usr/bin/env python
# coding: utf-8

# I created this notebook as a practice session for me to use some cool visualization tools such as matplotlib, seaborn, and bokeh.  At the same time, the data is also robust enough that you can really tap into all fields of data science, image recognition, NLP, etc.

# In[ ]:


import os
import sys
import operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
terrain = sns.color_palette(palette='terrain',n_colors=10)
plasma = sns.color_palette(palette='plasma',n_colors=10)
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import scipy as sp

from bokeh.io import output_notebook
from bokeh.layouts import gridplot,row,column
from bokeh.plotting import figure,show
output_notebook()


# We'll load in the data first and check out its dimensions.  Then we'll look at the first few values to see what columns exist in this dataset.

# In[ ]:


trainDF=pd.read_json('../input/train.json')
testDF=pd.read_json('../input/test.json')
print('Training data dimensions:',trainDF.shape)
print('Testing data dimensions:',testDF.shape)


# In[ ]:


trainDF.head()


# ### Interest level distribution
# Let's take a look at our y-value first to see what the distribution of it looks like

# In[ ]:


int_level = trainDF['interest_level'].value_counts()
plt.figure(figsize=(10,5))
sns.barplot(int_level.index, int_level.values, alpha=0.6, color=terrain[1])
plt.title('distribution of interest level by count')
plt.ylabel('Count')
plt.xlabel('Interest level')
plt.show()


# The number of lows definitely overwhelm the rest of the data points.  We may need to do some correction for this later as we train our models

# ### Interested level by price point
# We know that more people would be interested in lower price points, but we want to look at the difference in price for each interest level.

# In[ ]:


plt.figure(figsize=(10,5))
sns.barplot(trainDF['interest_level'], trainDF['price'], alpha=0.6, color=plasma[6], order=['low','medium','high'])
plt.title('distribution of interest level by price')
plt.ylabel('price')
plt.xlabel('Interest level')
plt.show()


# From this, we can really see the correlation between having a higher price and having less interest which makes sense.  People prefer to keep their money in the long run.  However, we do see that the low price point does give us more fluctation than the rest of the price points.  This is good to keep in mind when reviewing a logistic regression below

# ### Logistic Regression on one-hot encoded price points
# Because we might be using neural networks to do some of these predictions, let's see how a logistic regression would perform for this feature (Since I initially did this on my local PC with jupyter notebook, it didn't have a problem completing but this timed out on Kaggle so we're setting it to no regression fit)

# In[ ]:


s=pd.get_dummies(trainDF['interest_level'])
trainDFhot=pd.concat([trainDF, s],axis=1)
trainDFhot=trainDFhot[trainDFhot['price']<40000]


# ##### Low Interest based on price point
# Looking at low interest first, we can see that low interest really permeates through every level so it really doesn't matter if the price is high, we also have data points where the price is very low and there are no interest.  Seeing our graph from above, we can tell that the datapoints for low interest is significantly higher than the ones with medium or high interest

# In[ ]:


g=sns.lmplot(x='price',y='low',data=trainDFhot,fit_reg=False)


# ##### Medium Interest based on price point
# This graph is really interest, we can see that there's almost a soft cutoff for medium interest for prices around \$10,000 and a hard cutoff at \$15,000.

# In[ ]:


g=sns.lmplot(x='price',y='medium',data=trainDFhot,fit_reg=False)


# ##### High Interest based on price point
# This is where we can almost see with certainty that anything above a \$9000 pricetag would provide no interest to shoppers.  What's also quite funny here is that if you pay the shopper approx ~\$2000, they will have an 80% chance of being highly interested in this place.  Perhaps, that's something to think about doing to increase interest levels :P

# In[ ]:


g=sns.lmplot(x='price',y='high',data=trainDFhot,fit_reg=False)


# ## Interest based on geographical location
# As they always say in real estate, "location, location, location" (Lord Harold Samuel).  We'll know that people will have preferences in terms of where they are looking.  Because of this, we should look at the interest level mapped across different areas.

# In[ ]:


trainDFdist=trainDF[trainDF['latitude']!=0]
trainDFdist=trainDFdist[trainDFdist['latitude']<42]
trainDFdist=trainDFdist[trainDFdist['longitude']>-80]
g=sns.lmplot(x='longitude',y='latitude',data=trainDFdist,hue='interest_level',fit_reg=False)


# Initially attempted to use seaborn to visualize this data.  However, trying to slice through the data to zoom in, I got frustrated and decided to go with Bokeh.

# #### Bokeh to visualize data

# In[ ]:


p = figure(title="interest level based on geography",y_range=(40.65,40.85),x_range=(-74.05,-73.85))
p.xaxis.axis_label = 'latitude'
p.yaxis.axis_label = 'longitude'
lowLat=trainDF['latitude'][trainDF['interest_level']=='low']
lowLong=trainDF['longitude'][trainDF['interest_level']=='low']
medLat=trainDF['latitude'][trainDF['interest_level']=='medium']
medLong=trainDF['longitude'][trainDF['interest_level']=='medium']
highLat=trainDF['latitude'][trainDF['interest_level']=='high']
highLong=trainDF['longitude'][trainDF['interest_level']=='high']
p.circle(lowLong,lowLat,size=3,color=terrain.as_hex()[1],fill_alpha=0.1,line_alpha=0.1,legend='low')
p.circle(medLong,medLat,size=3,color=plasma.as_hex()[9],fill_alpha=0.1,line_alpha=0.1,legend='med')
p.circle(highLong,highLat,size=3,color=plasma.as_hex()[5],fill_alpha=0.1,line_alpha=0.1,legend='high')
show(p, notebook_handle=True)


# If you zoom in on manhattan, you'll quickly realize that people tend to have more interest near East Village, Chelsea, Hell's Kitchen, and Upper West Side.  But the reason why it probably looks brighter in those areas is probably due to the number of requests in those areas.  So I plotted each individual color separately below.  From there, you can really see that even though there's a lot of high interest in the areas I described above, there's still quite a lot of low interests.  We will need to dig deeper into this feature to determine whether it will be useful or not.

# In[ ]:


p1 = figure(width=500, height=500, title=None,y_range=(40.65,40.85),x_range=(-74.05,-73.85))
p1.circle(lowLong,lowLat,size=3,color=terrain.as_hex()[1],fill_alpha=0.1,line_alpha=0.1,legend='low')
p2 = figure(width=500, height=500, title=None,y_range=(40.65,40.85),x_range=(-74.05,-73.85))
p2.circle(medLong,medLat,size=3,color=plasma.as_hex()[9],fill_alpha=0.1,line_alpha=0.1,legend='med')
p3 = figure(width=500, height=500, title=None,y_range=(40.65,40.85),x_range=(-74.05,-73.85))
p3.circle(highLong,highLat,size=3,color=plasma.as_hex()[5],fill_alpha=0.1,line_alpha=0.1,legend='high')
show(column(p1,p2,p3), notebook_handle=True)


# ### KNN on location
# Since I am quite interested in whether longitutde and latitude data will perform well in our prediction task, I've decided to further pursue this option. First I'll need to create a dataframe from just the longitutde and latitude data and our dependent variable, interest level.
# I feel like KNN would perform the best with this kind of data because the assumption is that people interested in apartments in one building may be interested other apartments in that building.

# In[ ]:


X=pd.concat([trainDF['latitude'],trainDF['longitude']],axis=1)
y=trainDF['interest_level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=45)
neigh = KNeighborsClassifier(n_neighbors=9)
neigh.fit(X_train, y_train)


# Since this is a decently balanced dataset, I wanted to check what is the % accuracy in predicting the right interest level.

# In[ ]:


predVal=neigh.predict(X_test)
mat=[predVal,y_test]
df=pd.DataFrame(mat).transpose()
df.columns=('h0','y')
df['diff']=np.where(df.h0==df.y,1,0)
print('% correct =',sum(df['diff'])/len(df['diff'])*100)


# I looked into building out the log loss function to see how much error there is in the predictions

# In[ ]:


PredProb=neigh.predict_proba(X_test)
pred=np.asmatrix(PredProb)
pred.columns=('high','low','medium')
s=np.asmatrix(pd.get_dummies(y_test))
def f(x):
    return sp.log(sp.maximum(sp.minimum(x,1-10**-5),10**-5))
f=np.vectorize(f)
predf=f(pred)
mult=np.multiply(predf,s)
print('log loss =',np.sum(mult)/-len(y_test))


# This log loss is quite high so let's see if we can improve this by increasing our k value. Since, it would be annoying to change the value and run it, I figure it'll be faster to run a for loop through values of k from odd numbers between 3 to 39 (represented by j). I also wanted to have atleast 5 samples in each k to give us a good average (represented by i).

# In[ ]:


accbig=[]
loglossbig=[]

def f(x):
    return sp.log(sp.maximum(sp.minimum(x,1-10**-5),10**-5))
f=np.vectorize(f)

for j in range(3,40,2):
    logloss=[]
    acc=[]
    for i in range(5):
        #split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=i)
        neigh = KNeighborsClassifier(n_neighbors=j)
        #train classifier
        neigh.fit(X_train, y_train)
        
        #find % predicted correctly for this k
        predVal=neigh.predict(X_test)
        mat=[predVal,y_test]
        df=pd.DataFrame(mat).transpose()
        df.columns=('h0','y')
        df['diff']=np.where(df.h0==df.y,1,0)
        acc.append(sum(df['diff'])/len(df['diff']))
        
        #find the logloss for this k
        PredProb=neigh.predict_proba(X_test)
        pred=np.asmatrix(PredProb)
        pred.columns=('high','low','medium')
        s=np.asmatrix(pd.get_dummies(y_test))
        predf=f(pred)
        mult=np.multiply(predf,s)
        logloss.append(np.sum(mult)/-len(y_test))
    loglossbig.append(np.mean(logloss))
    accbig.append(np.mean(acc))
print(accbig)
print(loglossbig)


# In[ ]:


Now let's plot this against every K to see the decrease


# In[ ]:


plt.plot(range(3,40,2),loglossbig)
plt.ylabel('logloss')
plt.xlabel('k value')
plt.title('KNN logloss on longitude and latitude')


# In[ ]:


plt.plot(range(3,40,2),accbig)
plt.ylabel('% predicted correctly')
plt.xlabel('k value')
plt.title('KNN prediction on longitude and latitude')


# Even though the performance for this isn't amazing, we can see that there are some predictive value that we can use from longtitude and latitude data

# ## Apartment Feature Analysis
# Now I want to take a look at the features of the apartment to see which ones would stand out for each interest level.  If I were to go apartment shopping, I would definitely look at what they have first.
# 
# I used [SRK's][1] awesome text parsing code to put each listing's features into a document and vectorized it using CountVectorizer and also TfidfVectorizer.  
# 
# The goals is to find out which features will contribute the most toward an interest level.
#   [1]: https://www.kaggle.com/sudalairajkumar/two-sigma-connect-rental-listing-inquiries/xgb-starter-in-python

# In[ ]:


word=pd.DataFrame(columns=list(['ct','tf']))
trainDF['newfeatures'] = trainDF["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
ctVec = CountVectorizer(stop_words='english', max_features=100)
tr_ctVec = ctVec.fit_transform(trainDF['newfeatures'])
word['ct']=ctVec.get_feature_names()

tfIdfVec=TfidfVectorizer(stop_words='english', max_features=100)
tr_tfidf = tfIdfVec.fit_transform(trainDF['newfeatures'])
word['tf']=tfIdfVec.get_feature_names()

target_num_map = {'high':0, 'medium':1, 'low':2}
train_y = np.array(trainDF['interest_level'].apply(lambda x: target_num_map[x]))


# After generating the vectorized matrix and our labels.  I trained a logistic regression algorithm and pull the weights associated with each class.  Then took the absolute value and sorted them to get the words most related to a certain class.

# In[ ]:


wordDF=pd.DataFrame(tr_tfidf.toarray())
wordDF.columns=tfIdfVec.get_feature_names()
lr=LogisticRegression(solver='lbfgs')
lr.fit(wordDF,train_y)


# In[ ]:


highco=lr.coef_[0,:]
medco=lr.coef_[1,:]
lowco=lr.coef_[2,:]

high=pd.concat([pd.DataFrame(abs(highco)), word['tf']],axis=1)
high.columns=['val','words']
high=high.sort_values('val',ascending=False)

med=pd.concat([pd.DataFrame(abs(medco)), word['tf']],axis=1)
med.columns=['val','words']
med=med.sort_values('val',ascending=False)

low=pd.concat([pd.DataFrame(abs(lowco)), word['tf']],axis=1)
low.columns=['val','words']
low=low.sort_values('val',ascending=False)


# In[ ]:


high.index=range(0,len(high))
med.index=range(0,len(med))
low.index=range(0,len(low))

wordDF=pd.DataFrame(columns=list(['hi-weight','high','med-weight','med','low-weight','low']))
wordDF['hi-weight']=high.val
wordDF['high']=high.words
wordDF['med-weight']=med.val
wordDF['med']=med.words
wordDF['low-weight']=low.val
wordDF['low']=low.words


# I would say having parking is probably one of the most important things to generate high interest.  Having tried to find parking in new york city, I remember it being the worst nightmare for anyone!

# In[ ]:


wordDF


# This is still a work in progress...If you like this, please upvote =)
