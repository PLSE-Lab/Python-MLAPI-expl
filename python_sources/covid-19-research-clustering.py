#!/usr/bin/env python
# coding: utf-8

# # COVID-19 Literature Clustering
# 
# ### Goal
# Given a large amount of literature and rapidly spreading COVID-19, it is difficult for a scientist to keep up with the research community promptly. Can we cluster similar research articles together to make it easier for health professionals to find relevant research articles? Clustering can be used to create a tool to identify similar articles, given a target article. It can also reduce the number of articles one has to go through as one can focus on a cluster of articles rather than all. 
# 
# **Approach**:
# <ol>
#     <li>Unsupervised Learning task, because we don't have labels for the articles</li>
#     <li>Clustering and Dimensionality Reduction task </li>
#     <li>See how well labels from K-Means classify</li>
#     <li>Use N-Grams with Hash Vectorizer</li>
#     <li>Use plain text with Tfid</li>
#     <li>Use K-Means for clustering</li>
#     <li>Use t-SNE for dimensionality reduction</li>
#     <li>Use PCA for dimensionality reduction</li>
#     <li>There is no continuous flow of data, no need to adjust to changing data, and the data is small enough to fit in memmory: Batch Learning</li>
#     <li>Altough, there is no continuous flow of data, our approach has to be scalable as there will be more literature later</li>
# </ol>
# 
# ### Dataset Description
# 
# >*In response to the COVID-19 pandemic, the White House and a coalition of leading research groups have prepared the COVID-19 Open Research Dataset (CORD-19). CORD-19 is a resource of over 29,000 scholarly articles, including over 13,000 with full text, about COVID-19, SARS-CoV-2, and related coronaviruses. This freely available dataset is provided to the global research community to apply recent advances in natural language processing and other AI techniques to generate new insights in support of the ongoing fight against this infectious disease. There is a growing urgency for these approaches because of the rapid acceleration in new coronavirus literature, making it difficult for the medical research community to keep up.*
# #### Cite: [COVID-19 Open Research Dataset Challenge (CORD-19) | Kaggle](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) <br>
# 
# **Clustering section of the project (cite):** *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, 2nd Edition, by Aurelien Geron (O'Reilly). Copyright 2019 Kiwisoft S.A.S, 978-1-492-03264-9. Machine Learning Practice. Implimenting this section following the Chapter-9 project on O'REILLY's Hands-On Machine Learning. <br>
# 
# ## Load the Data
# Load the data following the notebook by Ivan Ega Pratama, from Kaggle.
# #### Cite: [Dataset Parsing Code | Kaggle, COVID EDA: Initial Exploration Tool](https://www.kaggle.com/ivanegapratama/covid-eda-initial-exploration-tool)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import glob
import json


# ### Loading Metadata

# Let's load the metadata of the dateset. 'title' and 'journal' attributes may be useful later when we cluster the articles to see what kinds of articles cluster together.

# In[ ]:


root_path = '/kaggle/input/CORD-19-research-challenge/2020-03-13'
metadata_path = f'{root_path}/all_sources_metadata_2020-03-13.csv'
meta_df = pd.read_csv(metadata_path, dtype={
    'pubmed_id': str,
    'Microsoft Academic Paper ID': str, 
    'doi': str
})
meta_df.head()


# ### Fetch All of JSON File Path

# Get path to all JSON files:

# In[ ]:


all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)
len(all_json)


# ### Helper: File Reader Class

# In[ ]:


class FileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            self.paper_id = content['paper_id']
            self.abstract = []
            self.body_text = []
            # Abstract
            for entry in content['abstract']:
                self.abstract.append(entry['text'])
            # Body text
            for entry in content['body_text']:
                self.body_text.append(entry['text'])
            self.abstract = '\n'.join(self.abstract)
            self.body_text = '\n'.join(self.body_text)
            # Extend Here
            #
            #
    def __repr__(self):
        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'
first_row = FileReader(all_json[0])
print(first_row)


# ### Load the Data into DataFrame

# Using the helper function, let's read in the articles into a DataFrame that can be used easily:

# In[ ]:


dict_ = {'paper_id': [], 'abstract': [], 'body_text': []}
for idx, entry in enumerate(all_json):
    if idx % (len(all_json) // 10) == 0:
        print(f'Processing index: {idx} of {len(all_json)}')
    content = FileReader(entry)
    dict_['paper_id'].append(content.paper_id)
    dict_['abstract'].append(content.abstract)
    dict_['body_text'].append(content.body_text)
df_covid = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text'])
df_covid.head()


# ### Adding the Word Count Column

# Adding word count columns for both abstract and body_text can be useful parameters later:

# In[ ]:


df_covid['abstract_word_count'] = df_covid['abstract'].apply(lambda x: len(x.strip().split()))
df_covid['body_word_count'] = df_covid['body_text'].apply(lambda x: len(x.strip().split()))
df_covid.head()


# In[ ]:


df_covid.info()


# In[ ]:


df_covid.describe(include='all')


# ### Clean Duplicates

# When we look at the unique values above, we can see that tehre are duplicates. It may have caused because of author submiting the article to multiple journals. Let's remove the duplicats from our dataset:

# In[ ]:


df_covid.drop_duplicates(['abstract'], inplace=True)
df_covid.describe(include='all')


# ## Take a Look at the Data:

# In[ ]:


df_covid.head()


# ## Data Pre-processing

# Now that we have our dataset loaded, we need to clean-up the text to improve any clustering or classification efforts. First, let's remove punctuation from each text:

# In[ ]:


import re

df_covid['body_text'] = df_covid['body_text'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))
df_covid['abstract'] = df_covid['abstract'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))


# Convert each text to lower case:

# In[ ]:


def lower_case(input_str):
    input_str = input_str.lower()
    return input_str

df_covid['body_text'] = df_covid['body_text'].apply(lambda x: lower_case(x))
df_covid['abstract'] = df_covid['abstract'].apply(lambda x: lower_case(x))


# In[ ]:


df_covid.head(4)


# ## Similar Articles
# Now that we have the text cleaned up, we can create our features vector which can be fed into a clustering or dimensionality reduction algorithm. For our first try, we will focus on the text on the body of the articles. Let's grab that:

# In[ ]:


text = df_covid.drop(["paper_id", "abstract", "abstract_word_count", "body_word_count"], axis=1)


# In[ ]:


text.head(5)


# Let's transform 1D DataFrame into 1D list where each index is an article (instance), so that we can work with words from each instance:

# In[ ]:


text_arr = text.stack().tolist()
len(text_arr)


# ### 2-Grams:

# Let's create 2D list, where each row is instance and each column is a word. Meaning, we will separate each instance into words:  

# In[ ]:


words = []
for ii in range(0,len(text)):
    words.append(str(text.iloc[ii]['body_text']).split(" "))


# In[ ]:


print(words[0][0])


# What we want now is n-grams from the words where n=2 (2-gram). We will still have 2D array where each row is an instance; however, each index in that row going to be a 2-gram:

# In[ ]:


n_gram_all = []

for word in words:
    # get n-grams for the instance
    n_gram = []
    for i in range(len(word)-2+1):
        n_gram.append("".join(word[i:i+2]))
    n_gram_all.append(n_gram)


# In[ ]:


n_gram_all[0][0]


# ### Vectorize:

# Now we will use HashVectorizer to create the features vector X. For now, let's limit the feature size to 2**12(4096) to speed up the computation. We might need to increase this later to reduce the collusions and improve the accuracy:

# In[ ]:


from sklearn.feature_extraction.text import HashingVectorizer

# hash vectorizer instance
hvec = HashingVectorizer(lowercase=False, analyzer=lambda l:l, n_features=2**12)

# features matrix X
X = hvec.fit_transform(n_gram_all)


# In[ ]:


X.shape


# #### Separete Training and Test Set

# In[ ]:


from sklearn.model_selection import train_test_split

# test set size of 20% of the data and the random seed 42 <3
X_train, X_test = train_test_split(X.toarray(), test_size=0.2, random_state=42)

print("X_train size:", len(X_train))
print(" X_test size:", len(X_test), "\n")


# ### Dimensionality Reduction:
# #### t-SNE
# Using t-SNE we can reduce our high dimensional features vector into 2 dimensional plane. In the process, t-SNE will keep similar instances together while trying to push different instances far from each other. Resulting 2-D plane can be useful to see which articles cluster near each other:

# In[ ]:


from sklearn.manifold import TSNE

tsne = TSNE(verbose=1)
X_embedded = tsne.fit_transform(X_train)


# Let's plot the result:

# In[ ]:


from matplotlib import pyplot as plt
import seaborn as sns

# sns settings
sns.set(rc={'figure.figsize':(15,15)})

# colors
palette = sns.color_palette("bright", 1)

# plot
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], palette=palette)

plt.title("t-SNE Covid-19 Articles")
plt.show()
#plt.savefig("plots/t-sne_covid19.png")


# We can clearly see few clusters forming. This may be a good sign that we are able to cluster similar articles together using 2-grams and HashVectorizer with 2**10 features. However, without labels it is difficult to see the clusters. For now, it looks like a blob of data... Let's try if we can use K-Means to generate our labels which later we can use to plot this scatterplot later with labels instead to see if we have clusters.

# ### Unsupervised Learning: Clustering

# #### K-Means
# Using K-means we will get the labels we need. For now, we will create 10 clusters. I am choosing this arbitrarily. We can change this later.

# In[ ]:


from sklearn.cluster import KMeans

k = 10
kmeans = KMeans(n_clusters=k, n_jobs=4, verbose=10)
y_pred = kmeans.fit_predict(X_train)


# Labels for the training set:

# In[ ]:


y_train = y_pred


# Labels for the test set:

# In[ ]:


y_test = kmeans.predict(X_test)


# Now that we have the labels, let's plot the t-SNE. scatterplot again and see if we have any obvious clusters:

# In[ ]:


# sns settings
sns.set(rc={'figure.figsize':(15,15)})

# colors
palette = sns.color_palette("bright", len(set(y_pred)))

# plot
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y_pred, legend='full', palette=palette)
plt.title("t-SNE Covid-19 Articles - Clustered")
#plt.savefig("plots/t-sne_covid19_label.png")
plt.show()


# That looks pretty promising. It can be see that articles from the same cluster are near each other forming groups. There are still overlaps. So we will have to see if we can improve this by changing the cluster size, using another clustering algorithm, or different feature size. We can also consider not using 2-grams, or HashVectorizer. We can try 3-grams, 4-grams, or plain text as our instances and vectorize them using either HashVectorizer, TfidVectorizer, or Burrows Wheeler Transform Distance. <br>
# 
# Before we try another method for clustering, I want to see how well it will classify using the labels we just created using K-Means.

# ## Classify

# ### Helper Function:

# In[ ]:


# function to print out classification model report
def classification_report(model_name, test, pred):
    from sklearn.metrics import precision_score, recall_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    
    print(model_name, ":\n")
    print("Accuracy Score: ", '{:,.3f}'.format(float(accuracy_score(test, pred)) * 100), "%")
    print("     Precision: ", '{:,.3f}'.format(float(precision_score(test, pred, average='micro')) * 100), "%")
    print("        Recall: ", '{:,.3f}'.format(float(recall_score(test, pred, average='micro')) * 100), "%")
    print("      F1 score: ", '{:,.3f}'.format(float(f1_score(test, pred, average='micro')) * 100), "%")


# ### Random Forest

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# random forest classifier instance
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=4)

# cross validation on the training set 
forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=3, n_jobs=4)

# print out the mean of the cross validation scores
print("Accuracy: ", '{:,.3f}'.format(float(forest_scores.mean()) * 100), "%")


# In[ ]:


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score

# cross validate predict on the training set
forest_train_pred = cross_val_predict(forest_clf, X_train, y_train, cv=3, n_jobs=4)

# print precision and recall scores
print("Precision: ", '{:,.3f}'.format(float(precision_score(y_train, forest_train_pred, average='macro')) * 100), "%")
print("   Recall: ", '{:,.3f}'.format(float(recall_score(y_train, forest_train_pred, average='macro')) * 100), "%")


# In[ ]:


# first train the model
forest_clf.fit(X_train, y_train)

# make predictions on the test set
forest_pred = forest_clf.predict(X_test)


# In[ ]:


# print out the classification report
classification_report("Random Forest Classifier Report (Test Set)", y_test, forest_pred)


# It looks like it doesn't overfit, which is good news. But results can be better than ~70-80%. So we might want to come back to this later again to see if we can improve it.  

# ## Tfid with Plain Text
# Let's see if we will be able to get better clusters using plain text as instances rather than 2-grams and vectorize it using Tfid.

# ### Vectorize

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=2**12)
X = vectorizer.fit_transform(text_arr)


# In[ ]:


X.shape


# ### Separete Training and Test Set

# In[ ]:


from sklearn.model_selection import train_test_split

# test set size of 20% of the data and the random seed 42 <3
X_train, X_test = train_test_split(X.toarray(), test_size=0.2, random_state=42)

print("X_train size:", len(X_train))
print(" X_test size:", len(X_test), "\n")


# ### K-Means

# Again, let's try to get our labels. We will choose 10 clusters again.

# In[ ]:


k = 10
kmeans = KMeans(n_clusters=k, n_jobs=4, verbose=10)
y_pred = kmeans.fit_predict(X_train)


# Get the training set labels:

# In[ ]:


y_train = y_pred


# Get the test set labels:

# In[ ]:


y_test = kmeans.predict(X_test)


# ### Dimensionality Reduction (t-SNE)

# Let's reduce the dimensionality using t-SNE again:

# In[ ]:


from sklearn.manifold import TSNE

tsne = TSNE(verbose=1)
X_embedded = tsne.fit_transform(X_train)


# ### Plot t-SNE

# In[ ]:


from matplotlib import pyplot as plt
import seaborn as sns

# sns settings
sns.set(rc={'figure.figsize':(15,15)})

# colors
palette = sns.color_palette("bright", len(set(y_pred)))

# plot
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y_pred, legend='full', palette=palette)
plt.title("t-SNE Covid-19 Articles - Clustered(K-Means) - Tfid with Plain Text")
#plt.savefig("plots/t-sne_covid19_label_TFID.png")
plt.show()


# This time we are able to see the clusters more clearly. There are clusters that further apart from each other. I can also start to see that there is possibly more than 10 clusters we need to identify using k-means.

# ### Dimensionality Reduction (PCA)

# In[ ]:


from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca_result = pca.fit_transform(X_train)


# ### Plot PCA

# In[ ]:


# sns settings
sns.set(rc={'figure.figsize':(15,15)})

# colors
palette = sns.color_palette("bright", len(set(y_pred)))

# plot
sns.scatterplot(pca_result[:,0], pca_result[:,1], hue=y_pred, legend='full', palette=palette)
plt.title("PCA Covid-19 Articles - Clustered (K-Means) - Tfid with Plain Text")
#plt.savefig("plots/pca_covid19_label_TFID.png")
plt.show()


# ### More Clusters?
# On our previous plot we could see that there is more clusters than only 10. Let's try to label them:

# In[ ]:


k = 25
kmeans = KMeans(n_clusters=k, n_jobs=4, verbose=10)
y_pred = kmeans.fit_predict(X_train)


# In[ ]:


from matplotlib import pyplot as plt
import seaborn as sns

# sns settings
sns.set(rc={'figure.figsize':(15,15)})

# colors
palette = sns.color_palette("bright", len(set(y_pred)))

# plot
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y_pred, legend='full', palette=palette)
plt.title("t-SNE Covid-19 Articles - Clustered(K-Means 25) - Tfid with Plain Text")
#plt.savefig("plots/t-sne_covid19_20label_TFID.png")
plt.show()

