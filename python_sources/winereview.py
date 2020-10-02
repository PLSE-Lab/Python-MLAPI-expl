#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud,STOPWORDS
import missingno as msno
import squarify


# In[ ]:


path = '../input/'
wine_150k = pd.read_csv(path + 'winemag-data_first150k.csv',index_col=0)
wine_130k = pd.read_csv(path + 'winemag-data-130k-v2.csv',index_col=0)
print("Count of 150k:",wine_150k.shape)
print("Count of 130k:",wine_130k.shape)

wine = pd.concat([wine_150k,wine_130k],axis=0,sort=False)
print("Number of rows and columns:",wine.shape)
wine.head()


# In[ ]:


#City Distribution

print('Number of country list in data:',wine['country'].nunique())#Count once when repeat
plt.figure(figsize=(14,10))
winecountry = wine['country'].value_counts().to_frame()[0:20]
print(winecountry)

sns.barplot( x = winecountry['country'], y = winecountry.index, data = winecountry, palette = 'ocean',orient = 'h')
plt.xscale('log')
plt.title('Distribution of Wine Reviews by Top 20 Countries');


# In[ ]:


#Price Distribution

plt.figure(figsize=(18,6))
wine['price'].describe()


# In[ ]:


# Price Distribution with Country
avgPrice = wine.groupby(['country',]).mean()['price'].sort_values(ascending = False).to_frame()
 
#Check NaN avgPrice
#fliterTunisia = (wine['country'] == "Tunisia")
#fliterEgypt = (wine['country'] == "Egypt")
#print(wine[fliterTunisia | fliterEgypt]['price'])

plt.figure(figsize=(18,18))
plt.title('Country wise average wine price')
sns.pointplot(x = avgPrice['price'] ,y = avgPrice.index , palette = 'ocean', orient='h', markers='o')

plt.figure(figsize=(18,12))
plt.title('Country wise average wine price')
plt.xlabel('Price')
plt.ylabel('Country');
squarify.plot(avgPrice['price'].fillna(0.1),color = sns.color_palette('rainbow'),label = avgPrice.index)


# In[ ]:


fig,ax = plt.subplots(1,2,figsize=(16,8))
ax1,ax2 = ax.flatten()

cnt = wine.groupby(['country'])['price'].max().sort_values(ascending = False).to_frame()[:15]
sns.barplot(x = cnt['price'], y = cnt.index, palette = 'inferno',ax = ax1)
ax1.set_title('Most expensive wine in country')
ax1.set_ylabel('Country')
ax1.set_xlabel('Price')

cnt = wine.groupby(['country'])['price'].min().sort_values(ascending = True).to_frame()[:15]
sns.barplot(x = cnt['price'], y = cnt.index, palette = 'rainbow_r',ax = ax2)
ax2.set_title('Least price wine by country')
ax2.set_ylabel('Country')
ax2.set_xlabel('Price')

plt.subplots_adjust(wspace = 0.4);


# In[ ]:


plt.figure(figsize=(16,6))
sns.boxplot(x = wine['country'], y = wine['price'])
plt.yscale("log")
plt.title('Country wise boxplot of price (log scale)')
plt.xticks(rotation=90);


# In[ ]:


# Points Distribution with Country
avgPoints = wine.groupby(['country',]).mean()['points'].sort_values(ascending = False).to_frame()

plt.figure(figsize=(18,18))
plt.title('Country wise average wine price')
sns.pointplot(x = avgPoints['points'] ,y = avgPoints.index , palette = 'ocean', orient='h', markers='o')

plt.figure(figsize=(18,12))
plt.title('Country wise average wine price')
plt.xlabel('Points')
plt.ylabel('Country');
squarify.plot(avgPoints['points'].fillna(0.1),color = sns.color_palette('rainbow'),label = avgPoints.index)


# In[ ]:


# Points and Price scatter
plt.figure(figsize=(12,8))
plt.scatter(wine['points'], wine['price'])
plt.xlabel('Points')
plt.ylabel('Price');


# In[ ]:


from __future__ import unicode_literals
from textblob import TextBlob


# In[ ]:


wine = pd.read_csv("../input/winemag-data_first150k.csv",sep=",")


wine = wine.drop('Unnamed: 0',1)
wine = wine.drop_duplicates()

# keeping only varieties with at least 20 reviews
num_reviews = wine.groupby('variety').description.count().to_frame().reset_index()
num_reviews = num_reviews[num_reviews.description > 19]
frequent_varieties = num_reviews.variety.tolist()
wine_f = wine.loc[wine['variety'].isin(frequent_varieties)]

# Randomly choose one of the variety
wine['description'] = wine['description'].str.lower()
porto = wine[wine.variety == "Port"]
#print(porto)


# In[ ]:



def sweetness_score(descriptions):
    sweet_freq = []
    for review in descriptions:
        review = TextBlob(review)
        num_sweet = review.words.count("sweet")
        num_sweetness = review.words.count("sweetness")
        num_sugar = review.words.count("sugar")
        num_sugary = review.words.count("sugary")
        num_caramel = review.words.count("caramel")
        num_caramelized = review.words.count("caramelized")
        total_sweet = num_sweet + num_sweetness + num_sugar + num_sugary + num_caramel + num_caramelized
        sweet_freq.append(total_sweet)
    return float(sum(sweet_freq)/len(sweet_freq))
sweetness_score(porto.description)


# In[ ]:


def bitter_score(descriptions):
    bitter_freq = []
    for review in descriptions:
        review = TextBlob(review)
        num_bitter = review.words.count("bitter")
        num_tannin = review.words.count("tannin")
        num_tobacoo = review.words.count("tobacoo")
        num_dry = review.words.count("dry")
        num_cedar = review.words.count("cedar")
        num_oak = review.words.count("oak")
        num_leather = review.words.count("leather")
        total_bitter = num_bitter+num_tannin+num_tobacoo+num_dry+num_cedar+num_oak+num_leather
        bitter_freq.append(total_bitter)
    return float(sum(bitter_freq)/len(bitter_freq))
bitter_score(porto.description)


# In[ ]:


sweet_list = []
bitter_list = []
for variety in wine_f.variety.unique():
    df_variety = wine_f[wine_f.variety == variety]
    sweet = sweetness_score(df_variety.description)
    sweet_list.append((variety,sweet))
    bitter = bitter_score(df_variety.description)
    bitter_list.append((variety,sweet))

sorted_sweet_list = sorted(sweet_list, key=lambda x: -x[1])
df_sweetness = pd.DataFrame(sorted_sweet_list,columns=["variety","sweetness_score"])
sorted_bitter_list = sorted(bitter_list, key=lambda x: -x[1])
df_bitter = pd.DataFrame(sorted_bitter_list,columns=["variety","bitter_score"])


# In[ ]:


import matplotlib.pyplot as plt
plt.rcdefaults()
fig, ax = plt.subplots()
varieties = tuple(df_sweetness.variety.tolist())[:20]
varieties = [TextBlob(i) for i in varieties]
y_pos = np.arange(len(varieties))
performance = np.array(df_sweetness.sweetness_score)[:20]
error = np.random.rand(len(varieties))

plt.barh(y_pos, performance, align='center', alpha=0.5)
plt.yticks(y_pos, varieties)
plt.xlabel('Sweetness score')
plt.title('Wine-varieties by sweetness')
 
plt.show()
###
varieties = tuple(df_bitter.variety.tolist())[:20]
varieties = [TextBlob(i) for i in varieties]
y_pos = np.arange(len(varieties))
performance = np.array(df_bitter.bitter_score)[:20]
error = np.random.rand(len(varieties))

plt.barh(y_pos, performance, align='center', alpha=0.5)
plt.yticks(y_pos, varieties)
plt.xlabel('Bitter score')
plt.title('Wine-varieties by bitter')
 
plt.show()


# In[ ]:




from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

wine1 = wine.copy()
col = ['province','variety','points']
wine1 = wine[col]
wine1 = wine1.dropna(axis=0)
wine1 = wine1.drop_duplicates(['province','variety'])
wine1 = wine1[wine1['points'] >85]
wine_pivot = wine1.pivot(index= 'variety',columns='province',values='points').fillna(0)
wine_pivot_matrix = csr_matrix(wine_pivot)


knn = NearestNeighbors(n_neighbors=10,algorithm= 'brute', metric= 'cosine')
model_knn = knn.fit(wine_pivot_matrix)


# In[ ]:


query_index = np.random.choice(wine_pivot.shape[0])

distance, indice = model_knn.kneighbors(wine_pivot.iloc[query_index,:].values.reshape(1,-1),n_neighbors=6)
for i in range(0, len(distance.flatten())):
    if  i == 0:
        print('Recmmendation for {0}:\n'.format(wine_pivot.index[query_index]))
    else:
        print('{0}: {1} with distance: {2}'.format(i,wine_pivot.index[indice.flatten()[i]],distance.flatten()[i]))


# In[ ]:




