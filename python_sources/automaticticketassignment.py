#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## Install the necessary Libraries (LANGUAGE DETECTION)
get_ipython().system('pip install fasttext')
get_ipython().system('pip install pycountry')


# In[ ]:


import numpy as np
import pandas as pd
import nltk 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import tqdm as tqdm
import itertools
#from google.colab import drive
import matplotlib.pyplot as plt
import scipy.stats as stats

import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("darkgrid")

import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

import fasttext
from pycountry import languages

from wordcloud import WordCloud, STOPWORDS
import matplotlib.cm as cm

from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# 

# **LOAD THE DATA FROM THE EXCEL**

# drive.mount('/content/drive/', force_remount=True)
# import os
# project_path =  '/content/drive/My Drive/CapstoneProject/'
# os.chdir(project_path)
# os.getcwd()
# ## copy the language model data to a Tmp folder. We will need this later to get
# ## language
# 
# !wget -O /tmp/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

# !ls

# In[ ]:


## Read the data from EXCEL
incidents = pd.read_excel("../input/automatic-ticket-assignment-using-nlp/Input Data Synthetic (created but not used in our project).xlsx")
#model_data = pd.read_excel("modelInput.xlsx")
## Quick View 
incidents.head(3)


# In[ ]:


## basic info
incidents.info()


# In[ ]:


## Shape
incidents.shape


# In[ ]:


incidents.columns


# In[ ]:


incidents["Assignment group"].unique()


# In[ ]:


incidents["Assignment group"].nunique()


# In[ ]:


## find nulls
incidents[incidents.isnull().any(axis=1)]


# In[ ]:


# drop nulls
incidents.dropna(inplace=True)
incidents.shape


# In[ ]:


## Duplicates 
sub_incidents = incidents[['Short description', 'Description', 'Caller','Assignment group']].copy()
duplicateRowsDF = sub_incidents[sub_incidents.duplicated()]
duplicateRowsDF


# In[ ]:


# Remove Duplicates
incidents_upd = incidents.drop_duplicates(['Short description', 'Description', 'Caller', 'Assignment group'])


# In[ ]:


## Group by Categories
df_grp = incidents_upd.groupby(['Assignment group']).size().reset_index(name='counts')
df_grp


# In[ ]:


df_grp.describe()


# **Visualize the Dsitribution of Records across Groups**

# In[ ]:


sns.set_style("darkgrid")

## distibution based on Percentage
df_grp["count_perc"] = round((df_grp["counts"]/incidents.shape[0])*100,2)
df_grp.sort_values(["count_perc"], axis=0, 
                 ascending=False, inplace=True) 


# In[ ]:


## View the Distribution of all Records
plt.subplots(figsize = (20,5))
 
plt.plot(df_grp["Assignment group"], df_grp["count_perc"]) 
plt.xlabel('Assignment Group') 
plt.ylabel('Percentage') 
plt.xticks(rotation=90)
plt.title('Incident Distribution') 
  
 
plt.show() 


# In[ ]:


## View the Distribution of only records that DONOT belong to GRP_0
df_test = df_grp[df_grp["Assignment group"] != 'GRP_0']
plt.subplots(figsize = (20,5))

plt.plot(df_test["Assignment group"], df_test["counts"]) 
plt.xlabel('Assignment Group') 
plt.ylabel('counts') 
plt.xticks(rotation=90)

plt.title('Incident Distribution For non GRP_0') 
  

plt.show() 


# **TEXT Processing - CLEANING and Associated activies**

# In[ ]:


## merging the Short Description and Description Columns
incidents_sum = pd.DataFrame({"Description": incidents_upd["Short description"] + " " + incidents_upd["Description"],
                             "AssignmentGroup": incidents_upd["Assignment group"]}, 
                                                       columns=["Description","AssignmentGroup"])


# Now clean up the Description column to address the following
# * Convert each character in a sentence to lowercase character
# * Remove HTML Tags
# * Remove punctuations
# * Remove stopwords
# * Remove common words like com, hello
# * Remove NUmbers - (r'[0-9] , is used for replace, but there are some numbers which remain in the data.. that needs to be tested further
# * Stemming was causing invalid words, hence used a lemmatizer

# In[ ]:


## NLTK Downloads

nltk.download('stopwords')
stop = set(stopwords.words('english')) 
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


# **Cleaning Data and Lemmatization**

# In[ ]:


import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()

# function to convert nltk tag to wordnet tag
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

def lemmatize_sentence(sentence):
    #tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:        
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)


temp =[]
for sentence in incidents_sum["Description"]:
    sentence = sentence.lower()
    #sentence = sentence.str.replace('\d+', '')
    cleanr = re.compile('<.*?>')
    sentence = re.sub(cleanr, ' ', sentence)        #Removing HTML tags
    sentence = re.sub(r'\S+@\S+', 'EmailId', sentence)
    sentence = re.sub(r'\'', '', sentence, re.I|re.A)
    sentence = re.sub(r'[0-9]', '', sentence, re.I|re.A)
    #print ("Sentence1.5 = ",sentence)
    sentence = re.sub(r'[^a-zA-Z0-9\s]', ' ', sentence)
    #print ("Sentence2 = ",sentence)
    sentence = sentence.lower()
    sentence = re.sub(r'com ', ' ', sentence, re.I|re.A)
    sentence = re.sub(r'hello ', ' ', sentence, re.I|re.A)
    l_sentence = lemmatize_sentence(sentence)

    words = [word for word in l_sentence.split() if word not in stopwords.words('english')]
    temp.append(words)


# In[ ]:


incidents_sum["Lemmatized_clean"] = temp


# **Use Fast Text to detect languages and store it into two additional columns language and accuracy**

# In[ ]:


import fasttext
from pycountry import languages
PRETRAINED_MODEL_PATH = '/tmp/lid.176.bin'
model = fasttext.load_model(PRETRAINED_MODEL_PATH)

temp1 =[]
temp2 = []
for sentence in incidents_sum["Lemmatized_clean"]:
    acc = 0
    try:
      predictions = model.predict(sentence)
      prediction_lang = re.sub('__label__', '',str(predictions[0][0][0]))
      acc = round(predictions[1][0][0],2) * 100
      language_name = languages.get(alpha_2=prediction_lang).name
    except:
      language_name = "NOT DETERMINED"
    temp1.append(language_name)
    temp2.append(acc)


# In[ ]:


incidents_sum["Language"] = temp1
incidents_sum["Accuracy"] = temp2


# In[ ]:


## we load the data into a EXcel to check the output manually (faster)
incidents_sum.to_excel("TempOutput.xlsx")


# In[ ]:


## Additional Text Cleaning
## we noticed that there are several rows that have all junk characters (tried decoding UTF-8 but did not work)
## this indicated that the characters were junk and we planned to remove the 36 rows that had the junk characters
## We initially tried regular expression but not sure why this did not remove all of the junk characters
## We observed that lemmatization of the junk characters returned blank array and hence we use that condition
## to remove the rows with Junk Characters

incidents_sum1 = incidents_sum[incidents_sum['Lemmatized_clean'].map(lambda d: len(d)) > 0]


# **Word Cloud Visualization - We will try to visualize word clouds for GRP_0, Non GRP_0 and some of the Groups apart from GRP_0 that had higher number of Incidents**

# In[ ]:


## GRP_0 Visualization:

## create a column to mark records with GRP_0 and non GRP_0=>GRP_X
incidents_sum1['GRP_MOD'] = incidents_sum1['AssignmentGroup'].apply(lambda x: 'GRP_X' if x != 'GRP_0' else x)


# In[ ]:


stopwords = set(STOPWORDS)
## function to create Word Cloud
def show_wordcloud(data, title):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(15, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()


# In[ ]:


## view word cloud for GRP_0
text_Str = incidents_sum1['Lemmatized_clean'][incidents_sum1['GRP_MOD'].isin(["GRP_0"])].tolist()
show_wordcloud(text_Str, "GRP_0 WORD CLOUD")


# In[ ]:


## GRP_X Visualization:
text_Str = incidents_sum1['Lemmatized_clean'][incidents_sum1['GRP_MOD'].isin(["GRP_X"])].tolist()
show_wordcloud(text_Str, "GRP_X WORD CLOUD")


# In[ ]:


text_Str1 = incidents_sum['Lemmatized_clean'][incidents_sum['AssignmentGroup'].isin(["GRP_1"])].tolist()
show_wordcloud(text_Str1,"GRP_9 WORD CLOUD" )


# In[ ]:


text_Str1 = incidents_sum['Lemmatized_clean'][incidents_sum['AssignmentGroup'].isin(["GRP_39"])].tolist()
show_wordcloud(text_Str1,"GRP_9 WORD CLOUD" )


# In[ ]:


text_Str1 = incidents_sum['Lemmatized_clean'][incidents_sum['AssignmentGroup'].isin(["GRP_1"])].tolist()
show_wordcloud(text_Str1,"GRP_9 WORD CLOUD" )


# **Word Clouds for Groups that have higher incidents other than GRP_0**

# In[ ]:


text_Str1 = incidents_sum['Lemmatized_clean'][incidents_sum['AssignmentGroup'].isin(["GRP_8"])].tolist()
show_wordcloud(text_Str1,"GRP_8 WORD CLOUD" )


# In[ ]:


text_Str1 = incidents_sum['Lemmatized_clean'][incidents_sum['AssignmentGroup'].isin(["GRP_24"])].tolist()
show_wordcloud(text_Str1,"GRP_24 WORD CLOUD" )


# In[ ]:


text_Str1 = incidents_sum['Lemmatized_clean'][incidents_sum['AssignmentGroup'].isin(["GRP_12"])].tolist()
show_wordcloud(text_Str1,"GRP_12 WORD CLOUD" )


# In[ ]:


text_Str1 = incidents_sum['Lemmatized_clean'][incidents_sum['AssignmentGroup'].isin(["GRP_9"])].tolist()
show_wordcloud(text_Str1,"GRP_9 WORD CLOUD" )


# In[ ]:


text_Str1 = incidents_sum['Lemmatized_clean'][incidents_sum['AssignmentGroup'].isin(["GRP_2"])].tolist()
show_wordcloud(text_Str1,"GRP_2 WORD CLOUD" )


# **Word Clouds for some random Groups**

# In[ ]:


text_Str1 = incidents_sum['Lemmatized_clean'][incidents_sum['AssignmentGroup'].isin(["GRP_16"])].tolist()
show_wordcloud(text_Str1,"GRP_16 WORD CLOUD" )


# In[ ]:


text_Str1 = incidents_sum['Lemmatized_clean'][incidents_sum['AssignmentGroup'].isin(["GRP_13"])].tolist()
show_wordcloud(text_Str1,"GRP_13 WORD CLOUD" )


# In[ ]:


text_Str1 = incidents_sum['Lemmatized_clean'][incidents_sum['AssignmentGroup'].isin(["GRP_25"])].tolist()
show_wordcloud(text_Str1,"GRP_25 WORD CLOUD" )


# In[ ]:


text_Str1 = incidents_sum['Lemmatized_clean'][incidents_sum['AssignmentGroup'].isin(["GRP_29"])].tolist()
show_wordcloud(text_Str1,"GRP_29 WORD CLOUD" )


# In[ ]:


text_Str1 = incidents_sum['Lemmatized_clean'][incidents_sum['AssignmentGroup'].isin(["GRP_33"])].tolist()
show_wordcloud(text_Str1,"GRP_33 WORD CLOUD" )


# In[ ]:


text_Str1 = incidents_sum['Lemmatized_clean'][incidents_sum['AssignmentGroup'].isin(["GRP_38"])].tolist()
show_wordcloud(text_Str1,"GRP_38 WORD CLOUD" )


# In[ ]:


text_Str1 = incidents_sum['Lemmatized_clean'][incidents_sum['AssignmentGroup'].isin(["GRP_45"])].tolist()
show_wordcloud(text_Str1,"GRP_45 WORD CLOUD" )


# In[ ]:


text_Str1 = incidents_sum['Lemmatized_clean'][incidents_sum['AssignmentGroup'].isin(["GRP_40"])].tolist()
show_wordcloud(text_Str1,"GRP_40 WORD CLOUD" )


# In[ ]:


text_Str1 = incidents_sum['Lemmatized_clean'][incidents_sum['AssignmentGroup'].isin(["GRP_41"])].tolist()
show_wordcloud(text_Str1,"GRP_41 WORD CLOUD" )


# In[ ]:


text_Str1 = incidents_sum['Lemmatized_clean'][incidents_sum['AssignmentGroup'].isin(["GRP_42"])].tolist()
show_wordcloud(text_Str1,"GRP_40 WORD CLOUD" )


# In[ ]:


text_Str1 = incidents_sum['Lemmatized_clean'][incidents_sum['AssignmentGroup'].isin(["GRP_43"])].tolist()
show_wordcloud(text_Str1,"GRP_40 WORD CLOUD" )


# In[ ]:


text_Str1 = incidents_sum['Lemmatized_clean'][incidents_sum['AssignmentGroup'].isin(["GRP_44"])].tolist()
show_wordcloud(text_Str1,"GRP_40 WORD CLOUD" )


# In[ ]:


text_Str1 = incidents_sum['Lemmatized_clean'][incidents_sum['AssignmentGroup'].isin(["GRP_45"])].tolist()
show_wordcloud(text_Str1,"GRP_40 WORD CLOUD" )


# In[ ]:


text_Str1 = incidents_sum['Lemmatized_clean'][incidents_sum['AssignmentGroup'].isin(["GRP_46"])].tolist()
show_wordcloud(text_Str1,"GRP_40 WORD CLOUD" )


# In[ ]:


text_Str1 = incidents_sum['Lemmatized_clean'][incidents_sum['AssignmentGroup'].isin(["GRP_47"])].tolist()
show_wordcloud(text_Str1,"GRP_47 WORD CLOUD" )


# In[ ]:


text_Str1 = incidents_sum['Lemmatized_clean'][incidents_sum['AssignmentGroup'].isin(["GRP_48"])].tolist()
show_wordcloud(text_Str1,"GRP_40 WORD CLOUD" )


# In[ ]:


text_Str1 = incidents_sum['Lemmatized_clean'][incidents_sum['AssignmentGroup'].isin(["GRP_49"])].tolist()
show_wordcloud(text_Str1,"GRP_49 WORD CLOUD" )


# 

# In[ ]:


text_Str1 = incidents_sum['Lemmatized_clean'][incidents_sum['AssignmentGroup'].isin(["GRP_37"])].tolist()
show_wordcloud(text_Str1,"GRP_30 WORD CLOUD" )


# In[ ]:


text_Str1 = incidents_sum['Lemmatized_clean'][incidents_sum['AssignmentGroup'].isin(["GRP_69","GRP_70","GRP_71","GRP_72","GRP_73"])].tolist()
show_wordcloud(text_Str1,"GRP_8 WORD CLOUD" )


# **Find Optimal Clusters - Visualize Elbow Method**

# In[ ]:


def find_optimal_clusters(data, max_k):
    iters = range(2, max_k+1, 2)
    print ("Iters", iters)
    sse = []
    for k in iters:
        sse.append(MiniBatchKMeans(n_clusters=k, init_size=200, batch_size=300, random_state=20).fit(data).inertia_)
        print('Fit {} clusters'.format(k))
        
    f, ax = plt.subplots(1, 1,figsize=(15,5))
   
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')
    #set_size(50,5)
    plt.show()


# In[ ]:


## Function to PLot the Clusters using Tsne and PCA
def plot_tsne_pca(data, labels, size_d, component_count):
  max_label = max(labels)
  max_items = np.random.choice(range(data.shape[0]), size=size_d, replace=False)
  
  pca = PCA(n_components=component_count).fit_transform(data[max_items,:].todense())
  tsne = TSNE().fit_transform(PCA(n_components=component_count).fit_transform(data[max_items,:].todense()))
  
  
  idx = np.random.choice(range(pca.shape[0]), size=size_d, replace=False)
  label_subset = labels[max_items]
  label_subset = [cm.hsv(i/max_label) for i in label_subset[idx]]
  
  f, ax = plt.subplots(1, 2, figsize=(14, 6))
  
  ax[0].scatter(pca[idx, 0], pca[idx, 1], c=label_subset)
  ax[0].set_title('PCA Cluster Plot')
  
  ax[1].scatter(tsne[idx, 0], tsne[idx, 1], c=label_subset)
  ax[1].set_title('TSNE Cluster Plot')


# **TFIDF Vectorization**
# 
# We now Use TFIDF Vectorization method to convert the Words in the description to number vectors.
# Once done, we will use clustering mechanisms to see if there are clusters that can be visualized. We will use KMeans find out the Optimal number of Clusters and then visualize the same.
# 
# 

# In[ ]:


# create teh vectorizer
tfidf = TfidfVectorizer(min_df=5 ,use_idf=True )

## join the list members of the column
incidents_sum1['Lemmatized_clean_upd']=[" ".join(description) for description in incidents_sum1['Lemmatized_clean'].values]

## From the word clouds it was noticed that the earlier Regular Expression replacement of numbers did not work
incidents_sum1['Lemmatized_clean_upd'] = incidents_sum1['Lemmatized_clean_upd'].str.replace('\d+', '')


## fitting the vectorizer
tfidf.fit(incidents_sum1.Lemmatized_clean_upd)
text = tfidf.transform(incidents_sum1.Lemmatized_clean_upd)

## Get Feature Names and Store the values in a Dataframe
tf_matrix = text.toarray()
vocab = tfidf.get_feature_names()
tf_df = pd.DataFrame(np.round(tf_matrix, 2), columns=vocab)

## View the nunmber of Features
tf_df.columns


# In[ ]:


tf_df.shape


# In[ ]:


find_optimal_clusters(text, 73)


# In[ ]:


clusters = MiniBatchKMeans(n_clusters=34, init_size=100, batch_size=200, random_state=20).fit_predict(text)
clusters.shape


# In[ ]:





# In[ ]:


num_features = 8372
pca_comp_count = 100
plot_tsne_pca(text, clusters,num_features,pca_comp_count )


# In[ ]:


df = incidents_sum1.copy()
df.drop(columns=["Description","Lemmatized_clean","Language","Accuracy"], inplace=True)
df1 = df[df["GRP_MOD"] == "GRP_X"]

tfidf = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
tfidf.fit(df1.Lemmatized_clean_upd)
text1 = tfidf.transform(df1.Lemmatized_clean_upd)

tf_matrix = text1.toarray()
vocab = tfidf.get_feature_names()
tf_df = pd.DataFrame(np.round(tf_matrix, 2), columns=vocab)
find_optimal_clusters(text1, 73)


# In[ ]:


clusters = MiniBatchKMeans(n_clusters=32, init_size=100, batch_size=200, random_state=20).fit_predict(text1)
clusters.shape


# In[ ]:


num_features = 4447
pca_comp_count = 100
plot_tsne_pca(text1, clusters,num_features,pca_comp_count )


# In[ ]:


## copy all the data into a new dataframe. We will use this to select top features 
## for segregrating between Group 0 and Group X. This is because using all the features
## generated by TFIDF will take huge time for any models.

from sklearn.feature_selection import SelectKBest

df = model_data.copy()
#df.drop(columns=["Description","Lemmatized_clean","Language","Accuracy"], inplace=True)


# In[ ]:


df.head()


# In[ ]:


df.head()
get_ipython().system('pwd')

df.to_excel("modelInput.xlsx")


# In[ ]:


from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import  numpy, textblob, string
#from keras.preprocessing import text, sequence
#from keras import layers, models, optimizers

## split the data into Train and Text (0.75)

train_x, test_x, train_y, test_y = model_selection.train_test_split(df['Lemmatized_clean_upd'], df['GRP_MOD'], test_size=0.25, random_state=42)


# In[ ]:


## encode the GRP_X and GRP_0 variable

encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
test_y = encoder.fit_transform(test_y)


# In[ ]:


tfidf_vect = TfidfVectorizer(min_df=5 ,use_idf=True,analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(df['Lemmatized_clean_upd'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(test_x)


# In[ ]:


## common function for Classifiers
def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    #return metrics.accuracy_score(predictions, test_y)
    return predictions


def cal_accuracy(model_name, y_test, y_pred): 

    print ("############  Model Used: ",model_name, " ####################")
    print("Confusion Matrix:\n ", 
        metrics.confusion_matrix(y_test, y_pred)) 
      
    print ("Accuracy : ", 
    metrics.accuracy_score(y_test,y_pred)*100) 
    
    print("Recall: {:.2f}".format(metrics.recall_score(y_test, y_pred)))
    print("Precision: {:.2f}".format(metrics.precision_score(y_test, y_pred)))
      
    print("Report : ", 
    metrics.classification_report(y_test, y_pred))


# In[ ]:


## naive Bayes
# Naive Bayes on Word Level TF IDF Vectors
model_name = "Naive Bayes"
pred_result = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
cal_accuracy (model_name,test_y, pred_result)
#print("NB, WordLevel TF-IDF: ", accuracy)


# In[ ]:


model_name = "Logistic Regression"
pred_result = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
cal_accuracy (model_name,test_y, pred_result)

#accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
#print ("LR, WordLevel TF-IDF: ", accuracy)


# In[ ]:


model_name = "SVM"
pred_result = train_model(svm.SVC(), xtrain_tfidf, train_y, xvalid_tfidf)
svm.SVC()
cal_accuracy (model_name,test_y, pred_result)

#accuracy = train_model(svm.SVC(), xtrain_tfidf, train_y, xvalid_tfidf)
#print ("SVM, N-Gram Vectors: ", accuracy)


# In[ ]:


model_name = "RandomForest"
pred_result = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf, train_y, xvalid_tfidf)
cal_accuracy (model_name,test_y, pred_result)

#accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf, train_y, xvalid_tfidf)
#print ("RF, Count Vectors: ", accuracy)


# In[ ]:


# Grid Search
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import GridSearchCV
# Parameter Grid

param_grid = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                    {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                    {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}
                   ] 
# Make grid search classifier
clf_grid = GridSearchCV(svm.SVC(), param_grid, verbose=1)
 
# Train the classifier
clf_grid.fit(xtrain_tfidf, train_y)
 
# clf = grid.best_estimator_()
print("Best Parameters:\n", clf_grid.best_params_)
print("Best Estimators:\n", clf_grid.best_estimator_)


# In[ ]:


model_name = "SVM"
pred_result = train_model(svm.SVC(C=3, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False), xtrain_tfidf, train_y, xvalid_tfidf)
cal_accuracy (model_name,test_y, pred_result)


# **Hierarchical Cluster for Group X**
# 
# * 

# In[ ]:


df_X = df[df["GRP_MOD"] == "GRP_X"]


# In[ ]:


df_X.drop("Unnamed: 0", axis = 1, inplace=True)


# In[ ]:


df_X


# In[ ]:


tfidf_vect = TfidfVectorizer(min_df=5 ,use_idf=True,analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(df_X['Lemmatized_clean_upd'])
train_x = df_X['Lemmatized_clean_upd']
#test_x = df_X['AssignmentGroup']
xtrain_tfidf =  tfidf_vect.transform(train_x)



## Get Feature Names and Store the values in a Dataframe
tf_matrix = xtrain_tfidf.toarray()
vocab = tfidf_vect.get_feature_names()
tf_df = pd.DataFrame(np.round(tf_matrix, 2), columns=vocab)

## View the nunmber of Features
tf_df.columns
#xvalid_tfidf =  tfidf_vect.transform(test_x)


# In[ ]:


tf_df.head()


# In[ ]:


import scipy.cluster.hierarchy as shc
#plt.figure(figsize=(20, 7))
plt.subplots(figsize = (40,8))  
plt.title("Dendrograms")
plt.xlabel('Clusters')
plt.xticks(rotation=90)
plt.ylabel('counts')

dend = shc.dendrogram(shc.linkage(tf_df, method='ward'))


 
  

plt.show() 


# In[ ]:


from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
cluster.fit_predict(tf_df)


# In[ ]:


plt.figure(figsize=(10, 7))
plt.scatter(tf_df[:,0], c=cluster.labels_, cmap='rainbow')


# **GROUP_X analysis**

# In[ ]:




