#!/usr/bin/env python
# coding: utf-8

# # Team NDL: Algorithms and Illnesses
# Hello! We are Team NDL from Penn State's Nittany Data Labs. 
# 
# This kernel was created by Izzidin Oakes, Neil Ashtekar, Will Wright, Suraj Dalsania, and Ming Ju Li.
# 
# Team picture: [imgur link](http://imgur.com/a/Vpg6ZMl)
# 
# Our vision for this project was to first explore our data, and then to dive into sentiment analysis. We wanted to address the following questions:
# * What insights can we gain from exploring and visualizing our data?
# * How does sentiment play into rating and usefulness of reviews?
# * Can we create a way for people to find the best medication for their illness?
# * What machine learning models work best for predicting the sentiment or rating based on review?
# * Is this problem better suited for classification or regression? In other words, should we be trying to sort the reviews into categories based on sentiment or predict the actual rating of the review?
# * What vectorization methods for the reviews are the most efficient and preserve the most data as well as allowing for the most accuracy? 
# * Can we somehow find insight into what features or words are most important for predicting review rating?

# # Drug Ratings Dataset: Preliminary Data Exploration
# 
# <br>
# Our ideas for preliminary exploration:
# - Most common conditions
# - Overall best and worst reviewed drugs
# - The curability of each disease
# - Best drugs for each condition
# - Most useful reviews
# - Usefulness vs review score
# - Bias in reviews
#     - Users tend to review things they really liked or really disliked, fewer reviews in the middle
# 

# In[ ]:


# ALL imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style; style.use('ggplot')
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


# Create dataframes train and test
train = pd.read_csv('../input/drugsComTrain_raw.csv')
test = pd.read_csv('../input/drugsComTest_raw.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


list(train) == list(test)


# Both train and test have the same features. Maybe they are split up to allow us to train/test our models easily.

# In[ ]:


list(train)


# In[ ]:


train.values.shape[0], test.values.shape[0], train.values.shape[0] / test.values.shape[0]


# Yep, the train set is almost exactly 3 times as big as the test set! This is a typical 75:25 train:test split.

# In[ ]:


train.condition.unique().size, test.condition.unique().size


# In[ ]:


train.drugName.unique().size, test.drugName.unique().size


# ## Common Conditions

# In[ ]:


# I previously did this by creating and sorting a dictionary -- here's an easier way with pandas! (Inspiration from Sayan Goswami)
conditions = train.condition.value_counts().sort_values(ascending=False)
conditions[:10]


# In[ ]:


plt.rcParams['figure.figsize'] = [12, 8]


# In[ ]:


conditions[:10].plot(kind='bar')
plt.title('Top 10 Most Common Conditions')
plt.xlabel('Condition')
plt.ylabel('Count');


# We're familiar with these most common conditions. It makes sense that something like birth control would have more reviews than something like anthrax! (Presumably, anthrax antibiotics?)

# ## Rating Distribution

# In[ ]:


# Look at bias in review (also shown on 'Data' page in competition: distribution of ratings)
train.rating.hist(color='skyblue')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.xticks([i for i in range(1, 11)]);


# This distribution illustrates that people generally write reviews for drugs they really like (or those that they really dislike). There are fewer middle ratings as compared to extreme ratings. 

# In[ ]:


rating_avgs = (train['rating'].groupby(train['drugName']).mean())
rating_avgs.hist(color='skyblue')
plt.title('Distribution of average drug ratings')
plt.xlabel('Rating')
plt.ylabel('Count')


# In[ ]:


rating_avgs = (train['rating'].groupby(train['condition']).mean())
rating_avgs.hist(color='skyblue')
plt.title('Averages of medication reviews for each disease')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()


# ## Usefulness vs Rating

# In[ ]:


# Is rating correlated with usefulness of the review?
plt.scatter(train.rating, train.usefulCount, c=train.rating.values, cmap='tab10')
plt.title('Useful Count vs Rating')
plt.xlabel('Rating')
plt.ylabel('Useful Count')
plt.xticks([i for i in range(1, 11)]);


# In[ ]:


# Create a list (cast into an array) containing the average usefulness for given ratings
use_ls = []

for i in range(1, 11):
    use_ls.append([i, np.sum(train[train.rating == i].usefulCount) / np.sum([train.rating == i])])
    
use_arr = np.asarray(use_ls)


# In[ ]:


plt.scatter(use_arr[:, 0], use_arr[:, 1], c=use_arr[:, 0], cmap='tab10', s=200)
plt.title('Average Useful Count vs Rating')
plt.xlabel('Rating')
plt.ylabel('Average Useful Count')
plt.xticks([i for i in range(1, 11)]);


# Looks like people found reviews with higher scores to be more useful! In the sense that reviews with high ratings recieved more 'useful' tags than reviews with low ratings. Interesting...

# We're curious: what makes a review useful? Let's look at some of the most useful reviews:

# In[ ]:


# Sort train dataframe from most to least useful
useful_train = train.sort_values(by='usefulCount', ascending=False)
useful_train.iloc[:10]


# In[ ]:


# Print top 10 most useful reviews
for i in useful_train.review.iloc[:3]:
    print(i, '\n')


# The useful reviews seem positive! Let's see some not-so-useful reviews:

# In[ ]:


# Print 10 of the least useful reviews
for i in useful_train.review.iloc[-3:]:
    print(i, '\n')


# The not-so-useful reviews seem much more negative. The final review listed is barely a review -- just a concerned patient asking questions about the product!
# 
# Our conclusions appear consistent with the above graph -- reviewers find higher ratings/better reviews to be more useful than lower ratings/worse reviews. Does this represent some sort of bias within the useful count?
# 
# We're also interested in quantifying the sentiment of these reviews.

# In[ ]:


sid = SentimentIntensityAnalyzer()


# In[ ]:


# Create list (cast to array) of compound polarity sentiment scores for reviews
sentiments = []

for i in train.review:
    sentiments.append(sid.polarity_scores(i).get('compound'))
    
sentiments = np.asarray(sentiments)


# In[ ]:


sentiments


# In[ ]:


useful_train['sentiment'] = pd.Series(data=sentiments)


# In[ ]:


useful_train = useful_train.reset_index(drop=True)
useful_train.head()


# In[ ]:


useful_train.sentiment.hist(color='skyblue', bins=30)
plt.title('Compound Sentiment Score Distribution')
plt.xlabel('Scores')
plt.ylabel('Count');


# In[ ]:


useful_train.plot(x='sentiment', y='usefulCount', kind='scatter', alpha=0.01)
plt.title('Usefulness vs Sentiment')
plt.ylim(0, 200);


# In[ ]:


temp_ls = []

for i in range(1, 11):
    temp_ls.append(np.sum(useful_train[useful_train.rating == i].sentiment) / np.sum(useful_train.rating == i))


# In[ ]:


plt.scatter(x=range(1, 11), y=temp_ls, c=range(1, 11), cmap='tab10', s=200)
plt.title('Average Sentiment vs Rating')
plt.xlabel('Rating')
plt.ylabel('Average Sentiment')
plt.xticks([i for i in range(1, 11)]);


# Let's see what other meaningful insights we can get from the data! Find the best and worst reviewed drugs overall:

# In[ ]:


# Create a list of all drugs and their average ratings, cast to dataframe
rate_ls = []

for i in train.drugName.unique():
    
    # Only consider drugs that have at least 10 ratings
    if np.sum(train.drugName == i) >= 10:
        rate_ls.append((i, np.sum(train[train.drugName == i].rating) / np.sum(train.drugName == i)))
    
avg_rate = pd.DataFrame(rate_ls)


# In[ ]:


# Sort drugs by their ratings, look at top 10 best and worst rated drugs
avg_rate = avg_rate.sort_values(by=[1], ascending=False).reset_index(drop=True)
avg_rate[:10]


# In[ ]:


avg_rate[-10:]


# A quick Google search seems to confirms our results!

# ## Best and Worst Drugs by Condition
# <br>
# Let's find the highest and lowest rated drugs for each condition! This information will be helpful for a user who is looking for medication for a specific condition.

# In[ ]:


# Make dictionary of conditions, each value will be a dataframe of all of the drugs used to treat the given condition
help_dict = {}

# Iterate over conditions
for i in train.condition.unique():
    
    temp_ls = []
    
    # Iterate over drugs within a given condition
    for j in train[train.condition == i].drugName.unique():
        
        # If there are at least 10 reviews for a drug, save its name and average rating in temporary list
        if np.sum(train.drugName == j) >= 10:
            temp_ls.append((j, np.sum(train[train.drugName == j].rating) / np.sum(train.drugName == j)))
        
    # Save temporary list as a dataframe as a value in help dictionary, sorted best to worst drugs
    help_dict[i] = pd.DataFrame(data=temp_ls, columns=['drug', 'average_rating']).sort_values(by='average_rating', ascending=False).reset_index(drop=True)


# Now we've got a very useful 'help_dict' dictionary! 
# 
# We can simply index the dictionary by a specific condition to see the top rated drugs for that condition. For example, let's look at the top 10 drugs for birth control:

# In[ ]:


help_dict['Birth Control'].iloc[:10]


# Drugs used for birth control are listed, from best to worst average rating. Let's the top 10 best drugs for some other conditions:

# In[ ]:


help_dict['Depression'].iloc[:10]


# In[ ]:


help_dict['Acne'].iloc[:10]


# **This information is really useful! We can easily find the best drugs for any given condition.**

# We can also see the worst rated drugs:

# In[ ]:


help_dict['Acne'].iloc[-10:]


# # Machine Learning Models
# It's now time to begin trying different machine learning models and try treating this first as a classification problem, and then as a regression problem.
# 

# # Classification with sk-learn and Random Forests

# We will apply random forests for the purpose of sentiment analysis of reviews. The obvious starting question for such an approach is how can we convert the raw text of the review into a data representation that can be used by a numerical classifier. To this end, we will use the process of vectorization. By vectorizing the 'review' column, we can allow widely-varying lengths of text to be converted into a numerical format which can be processed by the classifier.
# 
# This is achieved via the TI-IDF method which involves creating tokens (i.e. individual words or groups of words extracted from the text). Once the list of tokens is created, they are assigned an index integer identifier which allows them to be listed. We can then count the number of words in the document and normalize them in such a way that de-emphasizes words that appear frequently (like "a", "the", etc.). This creates what is known as a bag (multi-set) of words. Such a representation associates a real-valued vector to each review representing the importance of the tokens (words) in the review. This represents the entire corpus of reviews as a large matrix where each row of the matrix represents one of the reviews and each column repreents a token occurence.
# 
# Term-Frequency Inverse Document-Frequency (TF-IDF) is a way of handling the excessive noise due to words such as "a", "the", "he", "she", etc. Clearly such common words will appear in many reviews, but do not provide much isight into the sentiment of the text and their high frequency tends to obfuscate words that provide significant insight into sentiment. More details about the method can be found on [wikipedia](http://en.wikipedia.org/wiki/Tf%E2%80%93idf).
# 
# The main limitation of this approach is that it does not take into account the relative position of words within the document. Here we are only using the frequency of occurence. Nevertheless, it works pretty well for this data set.

# In[ ]:


# Creates TF-IDF vectorizer and transforms the corpus
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train.review)

# transforms test reviews to above vectorized format
X_test = vectorizer.transform(test.review)


# The simplest type of model we can attempt to fit on this data is the Naive Bayes classifier. We will first test Naive Bayes on a binarized version of the rating column which attempts to identify which reviews are favorable. We define a favorable review as one which received a rating above 5. Given the sample size involved for our data set, we choose Naive Bayes over other classifiers due to its scalability.

# In[ ]:


# Create a column with binary rating indicating the polarity of a review
train['binary_rating'] = train['rating'] > 5

y_train_rating = train.binary_rating
clf = MultinomialNB().fit(X_train, y_train_rating)

# Evaluates model on test set
test['binary_rating'] = test.rating > 5
y_test_rating = test.binary_rating
pred = clf.predict(X_test)

print("Accuracy: %s" % str(clf.score(X_test, y_test_rating)))
print("Confusion Matrix")
print(confusion_matrix(pred, y_test_rating))


# In[ ]:


# Trains random forest classifier
start = time.time()
rfc_rating = RandomForestClassifier(n_estimators=100, random_state=42, max_depth = 10000, min_samples_split = 0.001)
rfc_rating.fit(X_train, y_train_rating)
end = time.time()
print("Training time: %s" % str(end-start))

# Evaluates model on test set
pred = rfc_rating.predict(X_test)

print("Accuracy: %s" % str(rfc_rating.score(X_test, y_test_rating)))
print("Confusion Matrix")
print(confusion_matrix(pred, y_test_rating))


# We can see that a more complicated classifier gets us a roughly 8% increase in the accuracy of our classifier.

# # Classification with Keras
# * Neural networks usually scale very well with lots of data, and that's exactly what we have here (150,000 training examples, 50,000 test examples). We actually couldn't run all of these through sk-learn as the kernel would just crash after a couple of minutes. However, keras plays with this much data very nicely.
# * While we are sticking to a simple NN in this kernel, other types of NN architecture can deal with natural language processing problems very well (long short term memory models, other types of recurrent networks).
# * Keras gives us lots of freedom to play with hyperparameters and design a network that would be best suited for our data.

# In[ ]:


b = "'@#$%^()&*;!.-"
X_train = np.array(train['review'])
X_test = np.array(test['review'])

def clean(X):
    for index, review in enumerate(X):
        for char in b:
            X[index] = X[index].replace(char, "")
    return(X)

X_train = clean(X_train)
X_test = clean(X_test)
print(X_train[:2])


# Doing some preprocessing by removing all symbols from the reviews.

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from keras.utils import to_categorical
from gensim.models import Word2Vec
from nltk.cluster import KMeansClusterer
import nltk

vectorizer = CountVectorizer(binary=True, stop_words=stopwords.words('english'),lowercase=True, max_features=5000)
#vectorizer = TfidfVectorizer(binary=True, stop_words=stopwords.words('english'), lowercase=True, max_features=5000)
test_train = np.concatenate([X_train, X_test])
print(test_train.shape)
X_onehot = vectorizer.fit_transform(test_train)
stop_words = vectorizer.get_stop_words()
print(type(X_onehot))


# 
# * I use a CountVectorizer to vectorize each of the different reviews into 5000 feature row vectors. This is a very similar approach to vectorization as the TDIF explained above. 
# * This vectorizer is also set to not add in very common words such as "and" and "the" as features. This is specified in stop_words.

# In[ ]:


print(X_onehot.shape)
print(X_onehot.toarray())


# Sanity checking to see if the dimensions make sense, then peeking at what the actual X_onehot matrix looks like.
# We have our trimmed m number of examples, as well as 5000 columns so this looks right!

# In[ ]:


names_list = vectorizer.get_feature_names()
names = [[i] for i in names_list]
names = Word2Vec(names, min_count=1)
print(len(list(names.wv.vocab)))
print(list(names.wv.vocab)[:5])


# Peeking at the first 5 of our vectorized feature words.

# In[ ]:


def score_transform(X):
    y_reshaped = np.reshape(X['rating'].values, (-1, 1))
    for index, val in enumerate(y_reshaped):
        if val >= 8:
            y_reshaped[index] = 1
        elif val >= 5:
            y_reshaped[index] = 2
        else:
            y_reshaped[index] = 0
    y_result = to_categorical(y_reshaped)
    return y_result
    
    print(X_onehot)


# * Creating a helper function to easily create our y labels. We need to transform the review column to a m number of reviews by 3 columns, where index 0 is negative, 1 is positive and 2 is neutral.

# In[ ]:


y_train_test = pd.concat([train, test], ignore_index=True)
y_train = score_transform(y_train_test)
print(y_train)
print(y_train.shape)


# Checking if our helper function works. Seems good.

# In[ ]:


from numpy.random import seed

np.random.seed(1)
model = Sequential()
model.add(Dense(units=256, activation='relu', input_dim=len(vectorizer.get_feature_names())))
model.add(Dense(units=3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary


# I experimented a lot with different kinds of simple NN architecture and this is what I concluded:
# * Softmax was the best activation function for the output layer.
# * Vectorizing with the TFIDF vectorizer took longer to converge but had roughly the same accuracy as when I vectorized with the CountVectorizer. Changing binary and lowercase didn't result in any changes.
# * 256 units was the sweet spot for accuracy without overfitting. 
#     * Any units above would give me less accuracy as the model continued to train as well as overall performing worse, and less units just gave me worse accuracy overall.
#     * I also discovered that adding any additional layers to the model resulted in a loss of accuracy, even when I kept the number of layer units small.
# 
# 

# In[ ]:


history = model.fit(X_onehot[:-53866], y_train[:-53866], epochs=6, batch_size=128, verbose=1, validation_data=(X_onehot[157382:157482], y_train[157382:157482]))


# * I experimented with different numbers of epochs and batch sizes as well, and found roughly 6 epochs and a batch size of 128 to be optimal. 
# * Anything more than 6 epochs resulted in the model beginning to overtrain and lose accuracy.
# * Before running preprocessing on the reviews by eliminating symbols, the validation accuracy stayed around 84%. Now it scores a perfect 100% on the 100 examples.

# In[ ]:


scores = model.evaluate(X_onehot[157482:], y_train[157482:], verbose=1)


# In[ ]:


scores[1]


# In[ ]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# This model predicts the if the rievew is positive, negative or neutral with a ~89.4% accuracy, according to this test set output score.

# # Feature Analysis
# This section was our idea of attempting to derive importance out of our vectorized features by k-clustering by similarity, as keras has no built-in function to check for feature importance.

# In[ ]:


all_names = [i.split() for i in X_train]


# Splitting up our X_train for parsing through.

# In[ ]:


np.random.seed(1)
all_names_rand = [all_names[np.random.randint(low=1, high=150000)] for i in range(5000)]
print(len(all_names_rand))


# Taking a random sample of 5000 reviews for training our kclusterer on.

# In[ ]:


all_names_list = Word2Vec(all_names_rand, min_count=1)
all_names_vec = all_names_list[all_names_list.wv.vocab]
print(all_names[0])


# In[ ]:


kclusterer_all = KMeansClusterer(5, distance=nltk.cluster.util.cosine_distance, repeats=10)
assigned_clusters_all = kclusterer_all.cluster(all_names_vec, assign_clusters=True)
print(len(assigned_clusters_all))


# Running KClustering on every single word from all of the reviews. This takes a while to run usually. I found the algorithm converged around 25 repeats and then stopped clustering further. For this example however, running 25 repeats makes the kernel commit take a very long time, and thus 10 will do the job.
# * Just running kmeans on the features themselves resulted into very inaccurate clustering and things made a lot more sense when I kept the sentences intact, as there was a lot more information retained.
# * I could not find a way to seed this kclusterer to get reproducable results so different cluster numbers are given every time. The overall counts of the clusters are roughly the same though.
# 

# In[ ]:


def generate_df(feature_names): 
    
    #  creates a zipped dictionary with every word from all of the reviews as a key and its assigned cluster as its value
    all_words_dict = dict(zip(all_names_list.wv.vocab, assigned_clusters_all))
    
    #iterates through and deletes any word that isn't a feature.
    for key in list(all_words_dict.keys()):
        if key in list(feature_names):
            pass
        else:
            del all_words_dict[key]
            
    #dictionary is then converted into nested list, with each inner list corresponding to one cluster.
    sorted_names = []
    for cluster in range(5):
        cluster_list = []
        for key, value in all_words_dict.items():
            if value == cluster:
                cluster_list.append(key)
        sorted_names.append(cluster_list)
        
    #inner list word features are sorted alphabetically then converted into a pandas DataFrame.
    for index, entry in enumerate(sorted_names):
        entry.sort()
    
    df_all = pd.DataFrame(sorted_names).T
    print(df_all[:50])
    
    #returns pandas dataframe with each cluster as a column and a list of lists where each list is all of the words assigned to that cluster.
    return df_all, sorted_names
    

df,sorted_names_all = generate_df(names.wv.vocab)


# * This function gives us our list of words per cluster, as well as a pandas DataFrame where each column corresponds to a different cluster.
# * The way how these clusters are split up is interesting. It's cool to see how similar words are clustered together, and words one wouldn't think would be related are grouped together as well.
#     * There is also a big negative side to this: If the features aren't being split up in an easy to understand way, it is much harder to form meaningful conclusions if a certain cluster performs better than another.
#     * In this case, the clusters are not being split up in any way that we can conclude one certain subject is more beneficial for the model than another certain subject.
# 

# In[ ]:


def test_clusters(cluster_list):
    #function iterates through a list of lists. these lists contain names of the feature words we want to test our model with.
    
    #score_list for accuracy of each cluster on model.
    score_list = []
    lens = []
    
    #number of reviews tested on.
    reviewnum = 15000
    
    #random sampling of reviews.
    np.random.seed(3)
    indicies = [np.random.randint(low=1, high=150000) for i in range(reviewnum)]
    X_sample =  test_train[indicies]
    y_sample = y_train[indicies]
    
    #beginning iteration through words per list.
    for cluster in cluster_list:
        
        #appending length for print statement at the end.
        lens.append(len(cluster))

        X_onehot = vectorizer.fit_transform(X_sample)
        
        X_onehot = X_onehot.toarray()

        cluster_indexes = []

        #if the feature name in the feature vocab is found in the cluster, the index is appended to the cluster index list.
        for index, feature_name in enumerate(list(names.wv.vocab)):
            if feature_name in cluster:
                cluster_indexes.append(index)

        #amount of features for the input layer dimension of the neural net
        features = len(cluster_indexes)
        
        #creating specific X_onehot matrix with only columns of corresponding vector columns
        X_onehot = X_onehot[:, cluster_indexes]

        model_cluster = Sequential()
        model_cluster.add(Dense(units=256, activation='relu', input_dim=features))
        model_cluster.add(Dense(units=3, activation='softmax'))

        model_cluster.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_cluster.summary
        
        #indexing so 2000 reviews are saved for testing
        history = model_cluster.fit(X_onehot[:reviewnum-2000], y_sample[:reviewnum-2000], epochs=10, batch_size=128, verbose=2, validation_data=(X_onehot[reviewnum-2000:reviewnum-1900], y_sample[reviewnum-2000:reviewnum-1900]))
        
        y_test = score_transform(test)
        scores = model_cluster.evaluate(X_onehot[reviewnum-2000:], y_sample[reviewnum-2000:], verbose=1)
        
        #score of model is appended to list returned at the end
        score_list.append(scores[1])
    for index, entry in enumerate(score_list):
        print("cluster", index + 1, "accuracy: ", str(entry) + ". number of words for cluster: ", lens[index])


# The idea behind this function was to see how the accuracy between different clusters and features differs.
# 

# In[ ]:


test_clusters(sorted_names_all)


# So.. pretty disappointing results.
# * It seems like the model accuracy just scales linearly with the amount of features it's given.
# * Sometimes I've recieved a higher accuracy with a smaller amount of features with my many times of testing it, but it is rare. Different options of kclustering should have been explored, especially ones that could be seeded.
# * Possibly kclustering was never the right approach to grouping up features in the first place. Maybe trying to manually split them up (Numbers, positive words, negative words, etc) would have been better.
# * Another approach would be seeing if more common words  would be quantified as more important features.
# 
# Regardless, let's see what other insights we can gain.

# In[ ]:


test_clusters([list(names.wv.vocab)[:1000], list(names.wv.vocab)[2000:3000], list(names.wv.vocab)[3000:4000], list(names.wv.vocab)[4000:5000]])


# I was just curious if testing by alphabetical order gave any sort of insight into what words or numbers were contributing the most to model accuracy.

# In[ ]:


vectorizer = CountVectorizer(binary=True, stop_words=stopwords.words('english'),
                             lowercase=True, min_df=3, max_df=0.9, max_features=1000)
X_onehot = vectorizer.fit_transform(test_train)
names_list = vectorizer.get_feature_names()
names = [[i] for i in names_list]
names = Word2Vec(names, min_count=1)

df, sorted_names_all = generate_df(names.wv.vocab)


# Testing if limiting down the max features from 5000 to 1000 changed anything major within our results.

# In[ ]:


test_clusters(sorted_names_all)


# Not really a big difference.
# * Another possible flaw in this approach is that the hyperparameters of the model should've been changed every time to correspond with difference in features. From the training and validation data it didn't really seem like our model was overfitting, but this is something that should have been explored as we could have potentially gotten different results.

# # Simple Regression with sk-learn
# We were curious about framing the problem as a linear regression one, so this was our following exploration of that.

# In[ ]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#Reimporting data due to all of the weird transformations that have been applied to our original X_train as well as different variable names

train = pd.read_csv('../input/drugsComTrain_raw.csv')
test = pd.read_csv('../input/drugsComTest_raw.csv')

# Get review text
reviews = np.vstack((train.review.values.reshape(-1, 1), 
                     test.review.values.reshape(-1, 1)))

# Set up function to re-vectorize reviews. This time binary is set to false, we only have 500 max features and min and max_df arguments have been set.
vectorizer = CountVectorizer(binary=False, stop_words=stopwords.words('english'),
                             lowercase=True, min_df=3, max_df=0.9, max_features=500)

# Vectorize reviews
X = vectorizer.fit_transform(reviews.ravel()).toarray()

# Get ratings
ratings = np.concatenate((train.rating.values, test.rating.values)).reshape(-1, 1)

y = ratings

X_train, X_test = X[:train.values.shape[0], :], X[train.values.shape[0]:, :] 
y_train, y_test = y[:train.values.shape[0]], y[train.values.shape[0]:]


# In[ ]:


X.shape, y.shape, X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[ ]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train[:5000], y_train[:5000])


# In[ ]:


pred = lin_reg.predict(X_train[5000:])


# In[ ]:


np.sum(np.abs(y_train[5000:] - pred[:])) / (161297 - 5000)


# We end predicting the score of the review with only 2.3 error! Considering how little preprocessing, feature engineering, hyperparameter tuning and model experimentation we did, this is pretty impressive.

# # Conclusions
# 
# * It seems like the neural network gives the best overall accuracy with 89.4%.
# * This could've been framed as either a classification or regression problem depending on the approach.
# * In the future, perhaps trying to do some feature exploration in a different way would be helpful in developing insights and meaningful conclusions.
# * Exploring different NN architecture could've been very beneficial, as recurrent nets are known to work very well for NLP problems.
# * Overall, this was a very interesting project and we all learned quite a lot!
# 
# **Thanks for reading through our kernel!**
# 
