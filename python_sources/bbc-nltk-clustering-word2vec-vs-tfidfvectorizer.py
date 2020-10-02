#!/usr/bin/env python
# coding: utf-8

# # The BBC News Clustering
# * The dataset has 2225 News and all are labeled.
# * There are 5 different categories for these news.
# * Even though the news are labeled, they will be used to test it after prediction.
# * The all labels will be predicted by unsupervised learning.
# ## Summary
# * The two different libraries are used to predcit news' classes.
# * **1)** Word2Vec 
# * * Firstly, the prediction is done without cleaning the words, just stopwords will be removed. 
# * * Also, the two different stem algorithms will be used to clean words and will be compared all results. 
# * * The purpose of stemming is removing inflection in words. For expample, 'loves' ---> 'love'
# * * The stemming libraries ---> WordNetLemmatizer and PorterStemmer (There are more than these 2)
# * * Lastly, nltk.kmeansclusterer library is used to predict clusters.
# * **2)** TfidfVectorizer
# * * This library more convenient than first one because we didn't use any stemming algorithm because it is not necessary.
# * * Kmeans library is used to predict clusters.

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Importing the libraries
import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
# Stemming Libraries
from nltk.stem import WordNetLemmatizer
lm = WordNetLemmatizer()
from nltk.stem.porter import PorterStemmer
pm = PorterStemmer()
from nltk.probability import FreqDist
from gensim.models import Word2Vec
from nltk.cluster import KMeansClusterer


# # Data 

# In[ ]:


df_bbc = pd.read_csv('../input/bbc-fulltext-and-category/bbc-text.csv')


# In[ ]:


df_bbc.head()


# In[ ]:


category_counts = df_bbc.category.value_counts()
categories = category_counts.index
print(categories)


# In[ ]:



fig = plt.figure(figsize = (12,5))
ax = fig.add_subplot(111)
sns.barplot(x = category_counts.index , y = category_counts)
for a, p in enumerate(ax.patches):
    ax.annotate(f'{categories[a]}\n' + format(p.get_height(), '.0f'), xy = (p.get_x() + p.get_width() / 2.0, p.get_height()), xytext = (0,-25), size = 13, color = 'white' , ha = 'center', va = 'center', textcoords = 'offset points', bbox = dict(boxstyle = 'round', facecolor='none',edgecolor='white', alpha = 0.5) )
plt.xlabel('Categories', size = 15)
plt.ylabel('The Number of News', size= 15)
plt.xticks(size = 12)

plt.title("The number of News by Categories" , size = 18)
plt.show()


# # Text Preprocessing

# In[ ]:


data = np.asarray(df_bbc)
no_cluster = len(categories)

temp_text = []
temp_text_lemma = []
temp_text_stem = []

cleaned_texts = []
cleaned_texts_lemma = []
cleaned_texts_stem = []

for i in range(len(data)):
    temp_text.append([])
    temp_text_lemma.append([])
    temp_text_stem.append([])
    temp_text[i] = re.sub('[^a-zA-Z]', ' ', data[i][1] )                                                                      # Remove all punctuations
    temp_text[i] = temp_text[i].lower()
    temp_text[i] = temp_text[i].split()
    temp_text_lemma[i] = [lm.lemmatize(word) for word in temp_text[i] if not word in set(stopwords.words('english')) ]           # First stemming method 
    temp_text_stem[i] = [pm.stem(word) for word in temp_text[i] if not word in set(stopwords.words('english'))]                  # Second stemming method
    temp_text[i] = [word for word in temp_text[i] if not word in set(stopwords.words('english')) ]                         # we didn't use stemming method, just get rid of stopwords
    cleaned_texts.append(temp_text[i])
    cleaned_texts_lemma.append(temp_text_lemma[i])
    cleaned_texts_stem.append(temp_text_stem[i])


# # Clustering
# # Vectorizing the Words by Word2Vec

# In[ ]:


# Vectorize all words

from gensim.models import Word2Vec
from nltk.cluster import KMeansClusterer

def word_sentinizer(txt, model):
    text_vect = []
    no_words = 0
    for word in txt:
        if no_words ==  0:
            text_vect = model[word]
        else:
            text_vect = np.add(text_vect, model[word])
        no_words += 1
    return np.asarray(text_vect) / no_words

# Vectorizing withot cleaning
X = []
model = Word2Vec(cleaned_texts, min_count = 1)
for text in cleaned_texts:
    X.append(word_sentinizer(text, model))
    
# Vectorizing with WordNetLemmatizer 
X_lemma = []
model_lemma = Word2Vec(cleaned_texts_lemma, min_count = 1)
for text in cleaned_texts_lemma:
    X_lemma.append(word_sentinizer(text, model_lemma))
    

# Vectorizing with PorterStemmer      
X_stem = []
model_stem = Word2Vec(cleaned_texts_stem, min_count = 1)
for text in cleaned_texts_stem:
    X_stem.append(word_sentinizer(text, model_stem))


# In[ ]:


# Clustering vectorized words
kclusterer = KMeansClusterer(no_cluster, distance= nltk.cluster.util.cosine_distance, repeats = 100)

assigned_clusterers = kclusterer.cluster(X, assign_clusters = True)
assigned_clusterers_lemma = kclusterer.cluster(X_lemma, assign_clusters = True)
assigned_clusterers_stem = kclusterer.cluster(X_stem, assign_clusters = True)


# ## Finding Clusters' Topic

# * This functions is created to stack the real news classes and predicted classes.

# In[ ]:


# Stacking output and predicted results
def stack_pred_actual(assigned_clusterers,cleaned_texts,data):
    cluster_results = np.asarray(assigned_clusterers) 
    cluster_results = cluster_results.reshape(len(cluster_results), 1)
    cleaned_texts = np.asarray(cleaned_texts)
    cleaned_texts = cleaned_texts.reshape(len(cleaned_texts), 1)
    results = np.hstack((cleaned_texts,cluster_results, data[:,0].reshape(len(data), 1)))
    return results
results = stack_pred_actual(assigned_clusterers , cleaned_texts , data)
results_lemma = stack_pred_actual(assigned_clusterers_lemma , cleaned_texts_lemma , data)
results_stem = stack_pred_actual(assigned_clusterers_stem , cleaned_texts_stem , data)


# * This function is created to combine the texts based on thier clusters to find clusters' topics. So we will 5 long texts by clusters.
# * We combined the text by clusters because we need to discover which cluster belogns to which topic.

# In[ ]:


def merge_cluster_news(no_cluster , results):  
    text_by_clusters = []
    for i in range(no_cluster):
        text_by_clusters.append([[],[]])

        for k in range(len(results)):
            if results[k,1] == i:
                temp = " ".join(results[k,0])
                text_by_clusters[i][0].append(str(temp))
                text_by_clusters[i][1] = i

        text_by_clusters[i][0] = " ".join(text_by_clusters[i][0])
    return text_by_clusters

text_by_clusters =  merge_cluster_news(no_cluster , results)
text_by_clusters_lemma = merge_cluster_news(no_cluster , results_lemma)
text_by_clusters_stem = merge_cluster_news(no_cluster , results_stem)


# * The word cloud has been created just for the one model that is not cleaned. 

# In[ ]:


# First way to find the clusters' topic. ---> Creating word cloud for each cluster

from wordcloud import WordCloud
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,5, figsize = (25,5))
for i in range(len(text_by_clusters)):
    wordcloud = WordCloud(background_color = 'white',
                              width = 1200,
                              height = 1200).generate(text_by_clusters[i][0]) 
    ax[i].imshow(wordcloud)
    ax[i].grid(False)
    ax[i].axis('off')
    ax[i].title.set_text(str(text_by_clusters[i][1]))


# * The purpose of this function is finding the clusters' topic based of top 20 common words.
# * The key words has been found from these 20 words and it is used for desicion of each classes cluster number and assined.

# In[ ]:


# Second way to find the clusters' topic. ---> Finding most common 20 words and print them based on their cluster
def find_cluster_word2vec(text_by_clusters):
    topic = [[],[],[]]
    business = 0
    sport = 0
    tech = 0
    politics = 0
    entertainment = 0
    array = []
    for i, text in enumerate(text_by_clusters):
        tokenized_words = nltk.tokenize.word_tokenize(text[0])
        word_dist = FreqDist(tokenized_words)
        for word, frequency in word_dist.most_common(20):
            topic[0].append(int(text[1]))
            topic[1].append(word)
            topic[2].append(frequency)

    topic = np.array(topic).T
    topic= pd.DataFrame(topic)
    topic[0] = topic[0].astype(int)

    for i in range(len(np.unique(topic.iloc[:,0]))):
        common_words = topic[topic.iloc[:,0] == i].iloc[:,1]

        print(f'Cluster {i}: most common words are {[common_words.iloc[a] for a in range(len(common_words))]}')
        
        if np.isin(np.array(common_words[:]), ['election','minister','labour']).sum() > 0:
            politics =i
        elif np.isin(np.array(common_words[:]), ['market','growth']).sum() > 0:
            business =i
        elif np.isin(np.array(common_words[:]), ['mobile','technology','technolog','comput']).sum() > 0:
            tech =i
        elif np.isin(np.array(common_words[:]), ['show','club']).sum() > 0:
            entertainment =i
        elif np.isin(np.array(common_words[:]), ['back','england',]).sum() > 0:
            sport =i

    return {'politics' : [politics],'business': [business],'tech': [tech],'entertainment': [entertainment],'sport': [sport]}

print('\n--------Without Cleaning CLuster Predictions--------\n')
predicted_classes = pd.DataFrame(find_cluster_word2vec(text_by_clusters),index = ['cluster_numbers'])
print(predicted_classes)

print('\n--------WordNetLemmatizer CLuster Predictions--------\n')
predicted_classes_lemma = pd.DataFrame(find_cluster_word2vec(text_by_clusters_lemma), index = ['cluster_numbers'])
print(predicted_classes_lemma)

print('\n--------PorterStemmer CLuster Predictions--------\n')
predicted_classes_stem = pd.DataFrame(find_cluster_word2vec(text_by_clusters_stem), index = ['cluster_numbers'])
print(predicted_classes_stem)


# ## Results for Word2Vec

# In[ ]:


def confusion_matrix(results, predicted_classes):
    temp_array = np.zeros((5,5), dtype = int)
    cm = pd.DataFrame(temp_array, index = predicted_classes.keys(), columns = predicted_classes.keys())

    for i in range(len(results)):
        cm.loc[results[i][2], predicted_classes.T[predicted_classes.T == results[i][1]].dropna().index.values[0]] +=1
    return cm


# In[ ]:


cm = confusion_matrix(results, predicted_classes)
cm_lemma = confusion_matrix(results_lemma, predicted_classes_lemma)
cm_stem = confusion_matrix(results_stem, predicted_classes_stem)
import seaborn as sns
axes = []
fig, ax = plt.subplots(1,3, figsize = (15,5))
axes.append(sns.heatmap(cm, annot = True, cmap="YlGnBu",fmt="d", ax = ax[0]))
axes.append(sns.heatmap(cm_lemma, annot = True, cmap="YlGnBu",fmt="d", ax = ax[1]))
axes.append(sns.heatmap(cm_stem, annot = True, cmap="YlGnBu",fmt="d", ax = ax[2]))
for i in range(len(axes)):
    axes[i].set_xlabel('Predicted', fontsize = 16)
    axes[i].set_ylabel('Actual', fontsize = 16)
    axes[i].tick_params('x', rotation = 45)
axes[0].set_title('Without Cleaning', fontsize = 20)
axes[1].set_title('WordNetLemmatizer', fontsize = 20)
axes[2].set_title('PorterStemmer', fontsize = 20)
plt.tight_layout()
plt.show()


# # # Scatter Plot for Predicted Classes
# * TNSE library is used to reduce 100 dimension vector to 2 dimension(coordinats) to demonstrate in scatter plot.

# In[ ]:


from sklearn.manifold import TSNE
import seaborn as sns
def find_coords(X,assigned_clusterers,predicted_classes):
    tsne = TSNE(n_components = 2)                      
    X_tsne = tsne.fit_transform(X)
    df_coords = pd.DataFrame(X_tsne , columns = ['x', 'y'] )
    df_coords['clusters'] = assigned_clusterers
    for i in range(len(df_coords)):
        df_coords.loc[i,'pred_labels'] = predicted_classes.T[predicted_classes.T == df_coords.loc[i, 'clusters']].dropna().index.values[0]
    return df_coords

df_coords = find_coords(X,assigned_clusterers,predicted_classes)
df_coords_lemma =find_coords(X_lemma,assigned_clusterers_lemma,predicted_classes_lemma)
df_coords_stem = find_coords(X_stem,assigned_clusterers_stem,predicted_classes_stem)


# In[ ]:


fig, ax = plt.subplots(1,3, figsize = (21,7))
axes = []
axes.append(sns.scatterplot(x =df_coords.x, y = df_coords.y, hue =df_coords.pred_labels, palette = 'Set2', ax = ax[0]))
axes.append(sns.scatterplot(x =df_coords_lemma.x, y = df_coords_lemma.y, hue =df_coords_lemma.pred_labels, palette = 'Set2', ax = ax[1]))
axes.append(sns.scatterplot(x = df_coords_stem.x, y = df_coords_stem.y, hue = df_coords_stem.pred_labels, palette = 'Set2', ax = ax[2]))
axes[0].set_title('Without Cleaning', fontsize = 18)
axes[1].set_title('WordNetLemmatizer', fontsize = 18)
axes[2].set_title('PorterStemmer', fontsize = 18)
[axes[i].axis(False) for i in range(3)]
plt.show()


# * As you can see, PorterStemmer has perfomed better that fisrt two methods. In PorterStemmer, almost all classes distinguished its own class.
# * On the other hand, in first two methods, entertainment classes is not seperated very well, it is distributed to other classes. 

# In[ ]:


def accuracy(results, business_class =0, sport_class = 0 , entertainment_class =0, tech_class =0, politics_class =0):
    false = 0
    
    evaluating = {'cluster' : [], 'no_record' : [], 'correct_pred' : [], 'wrong_pred' : [] , 'accuracy' : []}
    clusters, counts = np.unique(results[:,2], return_counts = True)
    clusters = np.asarray([clusters, counts]).T
    for i in range(len(clusters)):
        evaluating['cluster'].append(clusters[i,0])
        evaluating['no_record'].append(clusters[i,1])
        evaluating['correct_pred'].append(0)
        evaluating['wrong_pred'].append(0)
        evaluating['accuracy'].append(0)
    evaluating = pd.DataFrame(evaluating)    
    
    for i in range(len(results)):
        if ((results[i][1] == business_class) and (results[i][2]== 'business')):
            evaluating.iloc[0,2] += 1      
        elif ((results[i][1] == sport_class) and (results[i][2]== 'sport')):
            evaluating.iloc[3,2] += 1
        elif ((results[i][1] == entertainment_class) and (results[i][2]== 'entertainment')):
            evaluating.iloc[1,2] += 1
        elif ((results[i][1] == tech_class) and (results[i][2]== 'tech')):
            evaluating.iloc[4,2] += 1
        elif ((results[i][1] == politics_class) and (results[i][2]== 'politics')):
            evaluating.iloc[2,2] += 1
        else:
            false +=1
    evaluating['wrong_pred'] = evaluating['no_record'] - evaluating['correct_pred']
    evaluating['accuracy'] = round(evaluating['correct_pred'] / evaluating['no_record'],2)
    return evaluating


evaluating = accuracy(results, business_class =int(predicted_classes.business), sport_class = int(predicted_classes.sport) , entertainment_class =int(predicted_classes.entertainment), tech_class =int(predicted_classes.tech), politics_class =int(predicted_classes.politics))
evaluating_lemma = accuracy(results_lemma, business_class =int(predicted_classes_lemma.business), sport_class = int(predicted_classes_lemma.sport) , entertainment_class =int(predicted_classes_lemma.entertainment), tech_class =int(predicted_classes_lemma.tech), politics_class =int(predicted_classes_lemma.politics))
evaluating_stem = accuracy(results_stem, business_class =int(predicted_classes_stem.business), sport_class = int(predicted_classes_stem.sport) , entertainment_class =int(predicted_classes_stem.entertainment), tech_class =int(predicted_classes_stem.tech), politics_class =int(predicted_classes_stem.politics))
print('\n             -----Without Cleaning Results-----')
print(evaluating)
print('\n             -----WordNetLemmatizer Results-----')
print(evaluating_lemma)
print('\n             -----PorterStemmer Results-----')
print(evaluating_stem)


# In[ ]:


def print_evaluation(evaluating):
    evaluating.iloc[:,3] = evaluating.iloc[:,1] - evaluating.iloc[:,2]

    print(f'{round((sum(evaluating.iloc[:,2])/sum(evaluating.iloc[:,1]))*100,2)} of --all the news-- are predicted as correctly.')

    for i in range(len(evaluating)):
        print(f'{round((evaluating.iloc[i,2]/evaluating.iloc[i,1])*100,2)} of the --{evaluating.iloc[i,0]}-- news is predicted correctly.')


        
print('\n             -----Without Cleaning Results-----')
print_evaluation(evaluating)
print('\n             -----WordNetLemmatizer Results-----')
print_evaluation(evaluating_lemma)
print('\n             -----PorterStemmer Results-----')
print_evaluation(evaluating_stem)


# # Vectorizing the Word by TfidfVectorizer
# 

# * TfidfVectorizer is another library to vectorize the words.
# * There is a limitation about tokenization but it is performing well and faster.
# * There is no need to clean the data.

# In[ ]:


# sklearn word vectorizing

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

corpus =[]
for i in range(len(data)):
    corpus.append(data[i][1])
vectorizer = TfidfVectorizer(stop_words = "english")
X_tf = vectorizer.fit_transform(corpus)
model_tf = KMeans(n_clusters = no_cluster, init = 'k-means++' , max_iter = 100, n_init = 1)
model_tf.fit(X_tf)

order_centroids = model_tf.cluster_centers_.argsort()[:,::-1]        # order_centroids variable includes all words words' vectors for each cluster and it is ordered
terms = vectorizer.get_feature_names()                               # this variable includes all words


# * In Word2Vec, the vector had 100 dimension but in this method, dimension has been decided based on the number of unique words.

# In[ ]:


print(np.shape(X_lemma))
print(np.shape(X_tf))


# ## Results for TfidfVectorizer 

# * The purpose of this function is finding the clusters' topic based of top 20 common words

# In[ ]:


def find_cluster(no_cluster,terms, order_centroids):     
    business = 0
    sport = 0
    tech = 0
    politics = 0
    entertainment = 0
    for i in range(no_cluster):
        common_words = [terms[word_number] for word_number in order_centroids[i,:20]]

        print(f'Cluster {i}: most common words are {common_words}')
        
        if np.isin(np.array(common_words[:]), ['election','minister','labour']).sum() > 0:
            politics =i
        elif np.isin(np.array(common_words[:]), ['economy','oil']).sum() > 0:
            business =i
        elif np.isin(np.array(common_words[:]), ['software','microsoft','computer','technology']).sum() > 0:
            tech =i
        elif np.isin(np.array(common_words[:]), ['oscar','actress','award']).sum() > 0:
            entertainment =i
        elif np.isin(np.array(common_words[:]), ['match','cup','coach']).sum() > 0:
            sport =i
    return {'politics' : [politics],'business': [business],'tech': [tech],'entertainment': [entertainment],'sport': [sport]}


# In[ ]:


predicted_classes_tf = find_cluster(no_cluster, terms, order_centroids)                    # The clusters' numbers are assigned based on key words in top 20 words.

predicted_classes_tf = pd.DataFrame(predicted_classes_tf, index = ['cluster_numbers'])

assigned_cluster_tf = model_tf.predict(X_tf)                                               # All of the news clusters predictions.                 

results_tf = stack_pred_actual(assigned_cluster_tf,corpus,data)                            # The real and prediction classes are stacked.
print('\n--------TfidfVectorizer CLuster Predictions--------')
print(predicted_classes_tf)


# In[ ]:


cm_tf = confusion_matrix(results_tf, predicted_classes_tf)
sns.heatmap(cm_tf, annot = True, cmap="YlGnBu",fmt="d")
plt.xlabel('Predicted', size = 16)
plt.ylabel("Actual" , size = 16)
plt.title('TfidfVectorizer', size = 20)
plt.show()


# * There are two libraries to reduce numbers of dimension for words vectors which are TSNE and UMAP.
# * Both has been used to demonstrate.

# In[ ]:


import umap
coords_finder = umap.UMAP(metric = 'cosine')
coords = coords_finder.fit_transform(X_tf)
df_coords_tf1 = find_coords(X_tf,assigned_cluster_tf,predicted_classes_tf)
df_coords_tf2 = df_coords_tf1.copy()
df_coords_tf2['x'] = coords[:,0]
df_coords_tf2['y'] = coords[:,1]


# In[ ]:


fig, ax = plt.subplots(1,2,figsize = (20,8))
ax1 = sns.scatterplot(x = df_coords_tf1. x , y = df_coords_tf1.y, hue = df_coords_tf1.pred_labels, palette = "Set2", ax = ax[0])
ax2 = sns.scatterplot(x = df_coords_tf2. x , y = df_coords_tf2.y, hue = df_coords_tf2.pred_labels, palette = "Set2", ax = ax[1])
ax1.axis(False)
ax2.axis(False)
ax1.set_title("TfidfVectorizer Clusters(TSNE)", size = 18)
ax2.set_title("TfidfVectorizer Clusters(UMAP)", size = 18)
plt.show()


# In[ ]:


total_true_pred = 0
for i in range(len(cm_tf)):    
    total_true_pred += cm_tf.iloc[i,i]
print(f'{round((total_true_pred / len(data) ) * 100, 2)}% of --all the news-- is predicted correctly.')
for i in range(len(cm_tf)):
    print(f'{round((cm_tf.iloc[i,i] / cm_tf.iloc[i,:].sum()) * 100, 2)}% of the --{cm_tf.columns[i]}-- news is predicted correctly.')


# # Conclusion
# * In Word2Vec
# * * The accuracy of the model that is not used any stemming is around 75.0% and always the lowest one.
# * * The best clustering has been predicted by PorterStemmer.
# * * Entertainment results are performed worst. The reason is that entertainment contains more general words than others.
# * TfidfVectorizer
# * * Although there is limitation in the method about tokeinizing the words, it performed very well and is definitely better than Word2Vec.
# * * Unlike Word2Vec, in this library, Sport clustering result is great, almost 100% correct prediction.
# * * Most common words are more related to its real clusters, so it is more seperable.
# * Both Vectorizing methods are trained over 10 times, results show that they are not consistent. 
# * They give similar results in every training but sometimes one class can perform worse than previous one especiall in Word2Vec.
# 
