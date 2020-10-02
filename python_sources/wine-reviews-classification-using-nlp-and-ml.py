#!/usr/bin/env python
# coding: utf-8

# ## Text Classification using NLP for Various types of Wines -- Part 3

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import nltk
# Importing Natural language Processing toolkit.
from PIL import Image
# from python imaging library
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[ ]:


wines = pd.read_csv("../input/winemag-data-130k-v2.csv",index_col = 0)
wines = wines.dropna(subset = ['points' , 'price'])
df_wines = wines[['country', 'description', 'points', 'price', 'variety']]
df_wines = df_wines.sample(frac = 0.05)
print(df_wines.shape)
df_wines.head()


# <p> For information related to the basics of dataset and the data analysis part of it refer to this link  
#     <a href = "https://www.kaggle.com/bhaargavi/wine-classification-analysis-of-data"> <h3>Link to Wine Classification Analysis of Data </h3> </a>  </p> 
# <p> For information related to the word cloud formations and various basic concepts of NLP  refer to this link  
#    <a href = "https://www.kaggle.com/bhaargavi/wine-review-classification-making-word-clouds "> <h3>Link to Wine Review Classification -- Making Word Clouds</h3> </a> </p>    

# <p> Now we need to divide the points and price of the wines in specific groups so that we can classify the data easily. So here we classify the points in 4 groups -- Average, Good, Great and Perfect and the prices in the six categories 0-10, 10-20, 20-30, 30-50, 50-100, Above 100. </p>

# In[ ]:


df_wines.loc[(df_wines['points'] > 80) & (df_wines['points'] <=85), 'wine_quality'] = 'Average'
df_wines.loc[(df_wines['points'] > 85) & (df_wines['points'] <=90), 'wine_quality'] = 'Good'
df_wines.loc[(df_wines['points'] > 90) & (df_wines['points'] <=95), 'wine_quality'] = 'Great'
df_wines.loc[(df_wines['points'] > 95) & (df_wines['points'] <=100), 'wine_quality'] = 'Perfect'
df_wines = df_wines[df_wines['wine_quality'].apply(lambda x: type(x) == str)]


# In[ ]:


df_wines.loc[(df_wines['price'] > 0) & (df_wines['price'] <=10) , 'price_val'] = '0-10'
df_wines.loc[(df_wines['price'] > 10) & (df_wines['price'] <=20) , 'price_val'] = '10-20'
df_wines.loc[(df_wines['price'] > 20) & (df_wines['price'] <=30) , 'price_val'] = '20-30'
df_wines.loc[(df_wines['price'] > 30) & (df_wines['price'] <=50) , 'price_val'] = '30-50'
df_wines.loc[(df_wines['price'] > 50) & (df_wines['price'] <=100) , 'price_val'] = '50-100'
df_wines.loc[df_wines['price'] > 100  , 'price_val'] = 'Above 100'


# In[ ]:


df_wines = df_wines.drop(columns = ['price', 'points'])
df_wines.head(5)


# ### Removal of less important words for classifcation 
# <p> NOw after the wordclouds we actually need to do some real text analysis and for that the first step will be removal of less important words from the descriptions using function which is as follows </p>

# In[ ]:


sample_data = df_wines
for i in sample_data.description:
    # Importing tokenize library
    from nltk.tokenize import word_tokenize
    # Tokenizing the words not using the treebankTokenizer as it was changing the text and using it with punctuation marks 
    tokens = word_tokenize(i)
    
    # Changing all the letters to lowercase 
    tokens_low = [w.lower() for w in tokens]
    
    
    # Removing all non-alphabetics from the descriptions 
    words = [word for word in tokens_low if word.isalpha()]
    
    # Removing stopwords.
    stopwords = set(STOPWORDS)
    stopwords.update(["drink" , 'now', 'wine' ,'flavour'])
    filter_sen = [w for w in words if not w in stopwords]
    
    #Using stemming for normalisation 
    from nltk.stem.porter import PorterStemmer
    porter  = PorterStemmer()
    stemmed = [porter.stem(word) for word in filter_sen]
    
    sentence = " ".join(w for w in stemmed)
    sample_data = sample_data.replace(i , sentence)


# <p> Now the basic things done while removing these less important words were 
#    <ol>
#     <li> Tokenizing the words </li>
#     <li> Converting all of them to lower cases </li>
#     <li> Removing all stopwords </li>
#     <li> Performing Normalisation only stemming on them. </li>
#     <li>Joining them back to sentences and replacing them with their orignials in the dataframe </li>
#    </ol>
# </p>

# ### Splitting the data into training and test data

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(sample_data, test_size = 0.3)
print("The total training data is {}".format(X_train.shape))
print("The total test data is {}".format(X_test.shape))


# In[ ]:


print("The sample training data has {} tweets".format(X_train.shape))
X_train.head(4)


# ### Checking the number of quality types of wine in training data

# In[ ]:


quality_w = pd.crosstab(index = X_train['wine_quality'], columns = 'count' )
price = pd.crosstab(index = X_train['price_val'], columns = 'count' )
plt.rcParams['figure.figsize'][0] = 14
plt.rcParams['figure.figsize'][1] = 6
fig, axs = plt.subplots(1,2)
quality_w.plot(kind = 'pie', subplots = True, ax=axs[0],autopct='%1.1f%%',shadow=True)
price.plot(kind = 'pie',subplots= True,ax=axs[1],  autopct='%1.1f%%',shadow=True)
plt.suptitle(" A.% Distribution of Qualities of Wines    B.% Distribution of the Price Ranges found",fontsize = 16)
plt.show()


# ### Basic Concepts of NLP 
# <ol>
#     <li> Bag of Words -- It is an approach used in document classification where the (frequency of) occurrence of each word is used as a feature for training a classifier</li>
#  <li> TFIDF Values -- Term frequency and Inverse document frequency helps us to find the importance of each word in the document
#     </li>
# </ol>

# ### Using Bag of Words

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()
training_counts = count.fit_transform(X_train.description)
print("The shape of the data is {}".format(training_counts.shape))
#count.vocabulary_


# ### Using TF-IDF Values

# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(training_counts)
X_train_tfidf.shape


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_model = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
features = tfidf_model.fit_transform(X_train.description)
tfidf_df= pd.DataFrame(features.todense(),columns=tfidf_model.get_feature_names())
tfidf_df.head()


# ### Machine Learning Algorithms 

# ### A. Naive Bayes Algorithm

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
text_clf_multi = Pipeline([('vect' , CountVectorizer()), ('tfidf' , TfidfTransformer()), ('clf', MultinomialNB())])


# In[ ]:


train1 = text_clf_multi.fit(X_train.description, X_train.wine_quality)
predicted1 = train1.predict(X_test.description)
score1_qual = 100 * text_clf_multi.score(X_test['description'], X_test['wine_quality'])
print("The score of Mulitnomial Naive Bayes for the quality of wine is {} %".format(score1_qual))


# In[ ]:


train1 = text_clf_multi.fit(X_train.description, X_train.price_val)
predicted1 = train1.predict(X_test.description)
score1_price = 100 * text_clf_multi.score(X_test['description'], X_test['price_val'])
print("The score of Mulitnomial Naive Bayes for the price value is {} %".format(score1_price))


# ### B. Suppot Vector Machines

# In[ ]:


from sklearn.pipeline import Pipeline 
from sklearn.svm import SVC
text_clf_svm_linear = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),('clf-svm', SVC(kernel = 'linear'))])
text_clf_svm_rbf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),('clf-svm', SVC(kernel = 'rbf'))])


# In[ ]:


text_clf_svm_linear.fit(X_train.description, X_train.wine_quality)
text_clf_svm_rbf.fit(X_train.description, X_train.wine_quality)
predict_svm = text_clf_svm_linear.predict(X_test.description)
predict_svm = text_clf_svm_rbf.predict(X_test.description)
linear_svm_qual = 100 * text_clf_svm_linear.score(X_test['description'] , X_test['wine_quality'])
rbf_svm_qual  = 100  * text_clf_svm_rbf.score(X_test['description'] , X_test['wine_quality'])
print("The score of Support Vector Machine Linear Kernel for the quality of wine is {} %".format(linear_svm_qual))
print("The score of Support Vector Machine RBF Kernel for the quality of wine is {} %".format(rbf_svm_qual))


# In[ ]:


text_clf_svm_linear.fit(X_train.description, X_train.price_val)
text_clf_svm_rbf.fit(X_train.description, X_train.price_val)
predict_svm = text_clf_svm_linear.predict(X_test.description)
predict_svm_price = text_clf_svm_rbf.predict(X_test.description)
linear_svm_price = 100 * text_clf_svm_linear.score(X_test['description'] , X_test['price_val'])
rbf_svm_price  = 100  * text_clf_svm_rbf.score(X_test['description'] , X_test['price_val'])
print("The score of Support Vector Machine Linear Kernel for the price value is {} %".format(linear_svm_price))
print("The score of Support Vector Machine RBF Kernel for the price value is {} %".format(rbf_svm_price))


#  ### C. K Nearest Neighbors 

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
text_clf_knn_3 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),('clf-svm',KNeighborsClassifier(n_neighbors=3))])
text_clf_knn_7 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),('clf-svm',KNeighborsClassifier(n_neighbors=7))])


# In[ ]:


text_clf_knn_3.fit(X_train.description, X_train.wine_quality)
text_clf_knn_7.fit(X_train.description, X_train.wine_quality)
predict_knn = text_clf_knn_3.predict(X_test.description)
predict_knn = text_clf_knn_7.predict(X_test.description)
knn_3_qual = 100 * text_clf_knn_3.score(X_test['description'] , X_test['wine_quality'])
knn_7_qual = 100 * text_clf_knn_7.score(X_test['description'] , X_test['wine_quality'])
print("The score of  3 Neighbor KNN for the quality of wine is {} %".format(knn_3_qual))
print("The score of 7 Neighbor KNN for the quality of wine is {} %".format(knn_7_qual))


# In[ ]:


text_clf_knn_3.fit(X_train.description, X_train.price_val)
text_clf_knn_7.fit(X_train.description, X_train.price_val)
predict_knn = text_clf_knn_3.predict(X_test.description)
predict_knn = text_clf_knn_7.predict(X_test.description)
knn_3_price = 100 * text_clf_knn_3.score(X_test['description'] , X_test['price_val'])
knn_7_price = 100 * text_clf_knn_7.score(X_test['description'] , X_test['price_val'])
print("The score of  3 Neighbor KNN for the price values of wine is {} %".format(knn_3_price))
print("The score of 7 Neighbor KNN for the price values of wine is {} %".format(knn_7_price))


# ## Comparing all these methods and finding the most suitable for both parameters

# In[ ]:


data = [['Naive Bayes Multi', score1_qual] , ['SVM Linear' , linear_svm_qual] , ['SVM RBF', rbf_svm_qual] , 
        ['KNN_3Neighbor' , knn_3_qual] , ['KNN_7Neighbor', knn_7_qual]]
quality_pred = pd.DataFrame(data)
data2 = [['Naive Bayes Multi', score1_price] , ['SVM Linear' , linear_svm_price] , ['SVM RBF', rbf_svm_price] , 
        ['KNN_3Neighbor' , knn_3_price] , ['KNN_7Neighbor', knn_7_price]]
price_pred = pd.DataFrame(data2)


# In[ ]:


plt.rcParams['figure.figsize'][0] = 8
plt.rcParams['figure.figsize'][1] = 6
index = np.arange(len(quality_pred))
width = 0.4
plt.bar(index, quality_pred[1], width = width)
plt.xlabel('Algorithm Used', fontsize=12)
plt.ylabel('Percentage Accuracy', fontsize=12)
plt.xticks(index, quality_pred[0], fontsize=10)
plt.yticks(np.arange(0,100, step = 10))
plt.title('% Accuracy of Various ML Algos in predicting the quality of wine', fontsize = 15)
plt.show()


# In[ ]:


index2 = np.arange(len(price_pred))
width = 0.4
plt.bar(index2, price_pred[1], width = width, color = 'red')
plt.xlabel("Various ML Algos used",fontsize = 12)
plt.ylabel("Percentage Accuracy for Algos", fontsize = 12)
plt.xticks(index2, price_pred[0])
plt.yticks(np.arange(0,100, step = 10))
plt.title('% Accuracy of Various ML Algos in predicting the quality of wine', fontsize = 15)
plt.show()


# >  ## Result Analysis 

# <p> The output of this analysis is that when we are predicting quality the best out of all is SVM Linear Approach with more than 65% accuracy.  KNN with 7 Neighbors is closer to it and can be our second bet. And then we have th 3 Neighbours KNN with accuracy of 62%. But The others including SVM Gaussian and Linear Naive Bayes have lesser accuracy of around 55-58%.
#     So the preferred approach will be SVM Linear Approach</p>
#    
#   
#   <p>The output of the 2nd  analysis is that when we are predicting price the best out of all is Naive Bayes Multinomial with accuracy going to 40%. And then there is SVM Linear Approach around 39% accuracy and third choice among these will be KNN with  7 Neighbors. But others have the accuracy of around30-32%. But the problem here is the accuracy for the best approach is also 40% around which is way too less to be reliable enough and we can't support data on these methods. So there can be various other techniques also that can be applied to it for better results like Neural Networks. and its various Forms  But amongst these for price  prediction our best bet will be Naive Bayes Multinomial Apporach</p>

# In[ ]:




