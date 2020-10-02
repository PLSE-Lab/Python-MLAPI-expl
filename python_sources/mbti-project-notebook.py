#!/usr/bin/env python
# coding: utf-8

# # EXPLORE Data Science Academy - Team 12 Predict

# ### TEAM_12_EDSA_JHB 
# - Sihle Mtolo
# - Mengezi Dlomo
# - Vuyiso Moyo
# - Wisani Baloyi
# - Sibusiso Ngwenya

# ### Introduction
# 
# We were tasked with determining personality types using the Myers-Briggs personality test. To aid us in doing this we used the MBTI dataset to train and test our model. Some of the major problems faced as the fact that in general it is hard to predict humans. 
# 
# ---
# 
# ### Problem Statement
# 
# Can we classify a person's personality type based on what words they use? In order to answer this we will be using posts collected from a forum as our training data, accompanied by each user's Myers-Briggs Type Indicator(MBTI) personalitiy. In order to process this text we will make use of Natural Language Processing (NLP) techniques. Finally we will build classification models, with MBTI types as our target variable, and then attempt to predict these personality types given unseen data..
# 
# ---

# ### Outline
# 
# 
# 
# In this project we will follow steps we believe are necessary for solving a data science problem. Here is a summary of those steps:
# 
# 1. **Getting Started**
#    - We will begin by importing our modules and data.
#   
# 2. **Exploratory Data Analysis**
#     - Here we will take a deeper look at the data we have imported, using multiple plots.
# 
# 3. **Data Cleaning**
#     - We will use various methods to narrow down the size of our corpus.
# 
# 4. **Feature Extraction**
#     - From our cleaned data we will convert the remaining words into numbers, using vectorisation, in preparion for training our models.
# 
# 5. **Modelling**
#     - With our data in a language our machine can understand, we will build and evaluate our models.
# 
# 6. **Discusion**
#     - We will discuss our results and indicate what we found from the whole process.
#     
# 7. **Conclusion**
#     - This is the end of our notebook where we will discuss our findings and tie everything together before closing this chapter.
#     
# ---
# 
# This notebook took inspiratiion from a number of sources including but not limited to:
#   - Detailed explanations, implementations and examples of accuracy, recall and precision, link can found [here](https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9)
#   - A bunch of Quora posts about the advantage snd disadvantages of pipelining and how to use it.
#   
#   
# All effort has been made to credit the authors and their work and if any errors or ommisions have been made kindly notify us to have this rectified.
# 
# ---
# 
# 

# ## 1. Getting Started 

# What we will do here is to import all the necessary libraries togerther with the train and test data. These will then be used throughout the notebook to make things easier for us.

# ### 1. Importing Libraries

# Below are all the packages we will be using.

# In[ ]:


# General libraries.
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import dill
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# Text mining libraries
import re
import urllib
import nltk
import string
from nltk.tokenize import word_tokenize, TreebankWordTokenizer
from nltk import LancasterStemmer, WordNetLemmatizer
from textblob import TextBlob, Word
from wordcloud import WordCloud, STOPWORDS
from numpy.linalg import svd

# Model building libraries
from imblearn.over_sampling import RandomOverSampler, SMOTE
import sklearn
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, log_loss
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

# Silencing warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
print(os.listdir("../input"))


# Now we have imported all relevant libraries let us now import our train and test data.

# In[ ]:


# Loading the data.
df_train = pd.read_csv("../input/edsambti/train.csv")
df_test = pd.read_csv('../input/edsambti/test.csv')


# We begin tackling our classification problem by importing our datasets and exploring them in order to see what condition they are in. We will also merge the `test` and `train` datasets in order to make it simpler to perform the necessary operations at a later stage.

# It might be a good idea to combine our data inorder to avoid repeating the steps between our datasets. We will add a new column `train` that will help us track whether our data is training data or testing data. It will contain either a 0 or a 1 indicating whether a row is part or training data or testing data. This will make our work easier. 

# In[ ]:


# Defining the 'train' column.
df_train['train'] = 1
df_test['train'] = 0

# Now we will append them
df_all = df_train.append(df_test, sort=False)

# Displaying all dataframes.
display(df_train.head())
display(df_test.head())
display(df_all.head())

# Looking at the shapes of all dataframes.
print(df_train.shape)
print(df_test.shape)
print(df_all.shape)


# Now that we have created a new dataframe let's make sure that all values are present.

# In[ ]:


# Checking for any NANs in the merged dataframe.
print(df_all['type'].isnull().sum(axis=0))
print(df_all['posts'].isnull().sum(axis=0))


# This is not surprising, it makes sense that there are there are 2169 null values in the `type` column as we combined the `train` and `test` dataset for simpler pre-processing, therefore there are no missing values in the `posts` column which require further scrutiny. We may continue.

# A quick note before moving on, we are aware that working with text may require a lot of time to run, so we have created a function which uses the `Dill` library to save the current state of the notebook in order to avoid re-running all cells upon re-opening the kernel at a later stage.

# In[ ]:


def session_file(command, filename='notebook_env.db'):
    """Saves the notebook and provides a checkpoint for 
    when you need to exit the notebook whilst still 
    performing tasks.
    
    Arguments: 
    command - whether to save or reload from the last saved checkpoint. 
    Enter either 'save' or 'load'.
    filename - desired name of the saved file. If user wishes to 
    assign a custom name it should end in the extension '.db'.
    
    Output: a DB file saved to your folder.
    """
    if command == 'save':
        dill.dump_session(filename)
    else:
        # Runs if command is 'load'.
        dill.load_session(filename)


# At this stage we have our data loaded and are ready to gain some meaningful insights. We know that both our datasets have two columns. The test set has an `id` column and `posts` column comprised of posts from users. The training set also has a `posts` column with posts from users however, instead of Ids it has the corresponding MBTI types of each user. 

# # 2 . Explatory Data Analysis

# Now we will proceed to extract some features from our text column to source some information. We need some distinguishing features of each observation, so we will look at word count for each row, along with post length and perform some sentiment analysis to see if it is instructive. Additionally, we will take a look at each category used to summarise personality types.
# 
# Packages used:
# 
#  - Matplotlib.
#  - Seaborn.
#  - Textblob.
#  - Word Cloud.

# Let's take a look at all of the personality types in our dataset.

# In[ ]:


# Viewing the available personality types in our training set.
all_types = df_train['type'].unique().tolist()
all_types


# ### 2.1. Personality Distribution

# Below is a graph displaying the count of the 16 MBTI types. The y-axis is the number of people for each type, and the x-axis is all of the types in the dataset.

# In[ ]:


# Plotting the distribution of personality types after seperating posts from each user.
x = df_train.type.value_counts()
plt.figure(figsize=(10,6))
ax = sns.barplot(x.index, x.values, alpha=0.8)
plt.title('The Distribution of the Different Personality Types')
plt.ylabel('Count')
plt.xlabel('MBTI Types')
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
plt.show();


# It is starkly clear that there are certain types which dominate others in our data set. INTPs are more populated than ESTJs in our dataset.

# ### 2.2. Word Count 

# Let's see the distribution of the average number of words used by each user.

# We will first split the posts by the pipes seperating posts in each observation. Then we will split the text into individual words and sum them. Given that there are 50 posts per user we will divide the word count by 50 in order to get the average number of words. We will then put these values into their own column. 

# In[ ]:


def get_number_of_words(posts):
    # Seperating pipes from posts from each post.  
    parsed_posts = posts.split("|||")
    num_words = sum(len(post.split()) for post in parsed_posts)
    return num_words / 50
df_all['word_count'] = df_all['posts'].apply(get_number_of_words)


# The graph below shows the distribution of post length per user. On the y-axis we have the fraction of all the posts. On the x-axis we have word count. We are able to view average words used against the percentage of posts of that similar length.

# In[ ]:


plt.figure(figsize=(15, 7))
ax = sns.distplot(df_all["word_count"], kde=True)
ax.set_title('Word Distributio')
ax.set_ylabel("Fraction of posts");
print(df_all["word_count"].mean())


# Around 70% of the posts made are around 25 words. 

# ### 2.3. Sentiment Analysis

# We would like to get a feel of the nature of the posts made by users. In order to do this we will use sentiment analysis. Sentiment analysis categorises opinions from text into classifications of positive, neutral or negative. It uses values between 1 and -1. 1 meaning positive, 0 meaning neutral and -1 meaning negative. 

# In[ ]:


#  Creating a column in the training set which has the sentiment score for each user.

df_train['polarity'] = df_train['posts'].map(lambda x:         TextBlob(x).sentiment.polarity)


# In[ ]:


x = round(df_train.groupby('type')['polarity'].mean(), 2)
plt.figure(figsize=(10,6))
ax = sns.barplot(x.index, x.values, alpha=0.8)
plt.title("MBTI_TYPES AVERAGE SEMTIMENT")
plt.ylabel('MEAN POLARITY')
plt.xlabel('MBTI_TYPES')
plt.show();


# We plotted the average sentiments of each personality type. Averall sentiment looks slightly positive and the variations in sentiment are largely similar.

# ### 2.4. Word Usage

# Perhaps some types are more loquacious than others? We tested this hypothesis by viewing the distribution of average words by personality type. The means of each group are relatively close to one another so that is not the case for this data. 

# We will group all personality types in relation to their posts. To generate a set of all available words per type. 

# In[ ]:


def generate_wordcloud(text, title):
    # Create and generate a word cloud image:
    wordcloud = WordCloud(background_color="white").generate(text)

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title, fontsize = 40)
    plt.show()


# In[ ]:


df_by_personality = df_all.groupby("type")['posts'].apply(' '.join).reset_index()
df_by_personality


# Now we iterate over each personality and extract the most used words using "wordcould". 

# In[ ]:


for i, t in enumerate(df_by_personality['type']):
    text = df_by_personality.iloc[i,1]
    generate_wordcloud(text, t)


# # 3 . Data Cleaning
# 

# Next we will go about removing the noise from our text data. By noise we mean elements which will clutter our model, such as words replicated due to the existence of caps on some words, or punctuation which may not add useful information. Furthermore, this function will remove frequent and infrequent words from our text column, among other operations, which will be mentioned further.
# 
# In addition to removing urls we will further clean our data.
# 1. We will make all our posts lower cases in order to avoid making our model see the same word as two different words due to different casing.
# 2. The rows of the 'posts' column are actually made up of multiple posts. These are seperated by three pipes. We will replace the pipes by a space to make our corpus cleaner.
# 3. We will then use regular expressions library to remove punctuation and digits as these are not words and we want to train our model using words.
# 
#  

# Packages used:
# 
#  - TextBlob.
#  - TreebankWordTokenizer.
#  - WordNetLemmatizer.

# ### 3.1 . Handling Links
# 
# We have a few options on how to handle urls and have opted to treat them all as the same word because they are largely unique and may add noise to our dataset. To this end we will replace them with the word 'url-web' to distinguish them from other words.
# 
# 

# In[ ]:


def treat_urls(
    df, 
    text='posts', 
    rename='no',
    delete_url='no'
    ):
    """This function performs specified operations on all the urls.
    
    Arguments:
    df - dataframe which user wants to perform operations on.
    text - column containing the text data.
    rename - renames all urls to 'url-web'.
    delete-urls - removes all the urls from the dataset.
    """
    urls = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
    
    if rename == 'yes':
            
        subs_url = r'url-web'
        df[text] = df[text].replace(to_replace=urls, value=subs_url, regex=True) 
    if delete_url == 'yes':
        
        del_url = ' '
        df[text] = df[text].replace(to_replace=urls, value=del_url, regex=True)
    return df
df_all = treat_urls(df_all, rename='yes')
df_all.head()


# Now we will attempt to correct spelling errors made by users in their posts. This will reduce the number of unique words in our data set since spelling errors will not be classified as new words. The function below helps us achieve this.

# In[ ]:


def correct_spelling(df):
    '''This function corrects spelling errors present in the text.
    and outputs a dataframe with corrected spelling.
    '''
    df['posts'] = df['posts'].apply(lambda x: TextBlob(x).correct()) 
    return df
# Running the function and assigning it to a variable.
# df_all = correct_spelling(df_all)


# ### 3.2 Removing Unnecessary Features

# Now we will remove caps from each post, remove pipes seperating posts from a single user and end off by removing punctuation.

# In[ ]:


def clean_post(post):
    ''' Converts the letters in the posts to lower case 
    and removes punctuation and digits'''

    # Convert all words to lower-case.
    post = post.lower()
    # Removing pipes.
    post = post.replace('|||', ' ')
    # Removing punctuation.
    post = re.sub('[%s]' % re.escape(string.punctuation), '', post)
    post = re.sub('\w*\d\w*', '', post)
    post = re.sub('[''""...@*#*]', '', post)
    return post
# Applying operations to the text column in the dataframe.
df_all['posts'] = df_all['posts'].apply(clean_post)
df_all.head()


# ### 3.3. Tokenizing and Lemmatizing

# The function below follows the theme of pre-processing. It extracts tokens - each element in the text column - which allows us to have all of the words in the dataset as individual strings. The other available operations are Lemmatization and Stemming which attempt to extract root words from each word, removing words which may be redundant. We have opted to use lemmatisation because it outputs meaningful words, in the English sense.

# In[ ]:


def tokenize_lemmatize(df, text='posts'):
    '''Performs tokenisation and lemmatization on 
    the given text column.
    
    Arguments:
    df - dataframe which user wants to perform operations on.
    text - the column which operations should be performed on.
               
    Output:
    returns a dataframe with a two added columns.
    '''
    tokeniser = TreebankWordTokenizer()
    df['tokens'] = df[text].apply(tokeniser.tokenize)
    df['lemm'] = df['tokens'].apply(lambda x: ' '
                                    .join([Word(word).lemmatize()
                                           for word in x]))
    return df
  
# Tokenising and lemmatising the data using our custom function.
df_all = tokenize_lemmatize(df_all)
df_all.head()


# Our data is now clean with much of the noise removed from the posts. We also have 

# # 4. Feature Extraction
# 

# During this step of the process we will be converting our text data into numbers based on their appearance in a document relative the the other documents. This will be handled by Count Vectoriser. Additionally, we will specify our X and y variables leading up to performing a train/test split on the data. Lastly, we will tackle the problem of our imbalanced data.

# ### 4.1. Count Vectorisation

# Our text data has been cleaned, tokenised and words lemmatised. We can now create values using count vectorization.

# In[ ]:


def vectorize_data(df):
    '''Uses the CountVectorizer to converts our data from corpus to
    document term matrix'''
    vect = CountVectorizer(lowercase=True, stop_words='english')
    df_test = (df[df['train'] == 0])
    df_train = (df[df['train'] == 1])
    X_count_train = vect.fit_transform(df_train['posts'])
    X_count_test = vect.transform(df_test['posts'])
    return X_count_train, X_count_test

X, testing_X = vectorize_data(df_all)
y = df_train['type']


# Our dataframe went from consisting of two columns to now having `119227` columns for us to feed into our model.

# ### 4.2. Train Test Split

# Here we are performing a train/test split in order to measure the performance of our models.

# In[ ]:


def data_preprocess(X,y):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,
                                                     random_state=42)
    return (X_train, y_train), (X_test, y_test)


# ### 4.3. Resampling

# We have already seen from our EDA that our data contains imbalanced data. For our predictions to be more accurate, we will need to resample our data and make it balanced. The imbalance learn library is always ready to save the day for us whenever this problem arises. It has classes that are designed to help us with resampling our data. Two of these classes stood our for us. These are the Near-miss class for undersampling and SMOTE for oversampling. Undersampling is a technique where by we reduce the size of the more prevalent class(es) and, consequently, the whole dataset. On the contrary, oversampling increases the size of the less prevalent classes(es) by synthesizing data. In the following steps, we will explore both of these techniques to see which one works best for us

# In[ ]:


def resample_data(X, y):
    '''Resamples the data and returns two tuples for oversampled
    data and undersampled data'''
    sm = SMOTE()
    nm = NearMiss()
    X_sm,y_sm = sm.fit_sample(X, y)
    X_nm,y_nm = nm.fit_sample(X, y)
    return (X_sm, y_sm), (X_nm, y_nm)


# # 6. Modelling

# Now we will begin the process of fitting our models on the data, making observations and prediction. Essentially we are now answering the problem statement we stated at the beginning of our notebook. We have used three models for this process.
# 
# Packages used:
# - LogisticRegression
# - KNeighborsClassifier,
# - Support Vector Machine

# Before we begin we will create a function which can be called when we want to fit our models.

# In[ ]:


def train_model(model, X_train, y_train):
    '''Fits a given model .
    '''
    mod = model()
    return mod.fit(X_train, y_train)


# In[ ]:


def predict(trained_model, X_test):
    '''Takes trained model and
    uses X_test to predict and
    returns the predicted y'''
    
    return trained_model.predict(X_test)


# ## 6.1. Logistic Regression

# Logistic regression is the type of regression that is used when the dependant variable is discrete/categorical in nature. Instead of fitting a straight line it fits an S-curve. The curve's range is [0;1]. It then converts the dependant variable into either 0 or 1 depending on whether it is above or below the threshold

# In[ ]:


logreg_sm_model = train_model(LogisticRegression,
                              resample_data(X_train, y_train)[0][0],
                             resample_data(X_train, y_train)[0][1])

logreg_nm_model = train_model(LogisticRegression,
                              resample_data(X_train, y_train)[1][0],
                             resample_data(X_train, y_train)[1][1])


# ## 6.2. K-Nearest Neighbors

# A K-Nearest-Neighbor is a data classification algorithm that attempts to determine what group a data point is in by looking at the data points around it.
# 
# An algorithm, looking at one point on a grid, trying to determine if a point is in group A or B, looks at the states of the points that are near it. The range is arbitrarily determined, but the point is to take a sample of the data. If the majority of the points are in group A, then it is likely that the data point in question will be A rather than B, and vice versa.
# 
# The k-nearest-neighbor is an example of a "lazy learner" algorithm because it does not generate a model of the data set beforehand. The only calculations it makes are when it is asked to poll the data point's neighbors. This makes k-nn very easy to implement for data mining

# In[ ]:


knn_sm_model = train_model(KNeighborsClassifier,
                              resample_data(X_train, y_train)[0][0],
                             resample_data(X_train, y_train)[0][1])

y_pred_knn_sm_model = predict(knn_sm_model, X_test)

knn_nm_model = train_model(KNeighborsClassifier,
                              resample_data(X_train, y_train)[1][0],
                             resample_data(X_train, y_train)[1][1])

y_pred_knn_nm_model = predict(knn_nm_model, X_test)


# ## 6.3. Support Vector Machines: SVC

# In[ ]:


svc_sm_model = train_model(LogisticRegression,
                              resample_data(X_train, y_train)[0][0],
                             resample_data(X_train, y_train)[0][1])
y_pred_svc_sm_model = predict(svc_sm_model, X_test)


# ## 6.4. Model Evaluation

# In order to make decisions on which model performs the best we will use model evaluation.

# In[ ]:


accuracy_svc_sm_model = metrics.accuracy_score(y_test,y_pred_svc_sm_model)
accuracy_svc_nm_model = metrics.accuracy_score(y_test,y_pred_svc_nm_model)

accuracy_logreg_sm_model = metrics.accuracy_score(y_test,y_pred_logreg_sm_model)
accuracy_logreg_nm_model = metrics.accuracy_score(y_test,y_pred_svc_nm_model)

accuracy_knn_sm_model = metrics.accuracy_score(y_test,y_pred_knn_sm_model)
accuracy_knn_nm_model = metrics.accuracy_score(y_test,y_pred_knn_nm_model)


# In[ ]:


eval_dict = {'SVC SMOTE': accuracy_svc_sm_model,
             'SVC Near_miss': accuracy_svc_nm_model,
             'Logistic SMOTE': accuracy_logreg_sm_model,
             'Logistic Near_miss': accuracy_logreg_nm_model,
             'KNN SMOTE': accuracy_knn_sm_model,
             'KNN Near_miss':accuracy_knn_nm_model}

df_model_eval = pd.DataFrame(eval_dict)
# Let us now view our dataframe to see how each model does in terms accuracy.
df_model_eval


# ## 6.5. Hyper-Parameter Tuning

# Our models use default hyper-parameters; are they the best parameters though? There is only one way to find out - cross validation. This step is essential if we want to ensure the best performance of our model. We want the model with the best predictive ability. We will use grid-search for this step. Grid-search scans the data to configure optimal parameters for a given model. No! It is not a software we will upload our data into. It is a process we need to apply. Sklearn has a Gridsearch class to make this task easier for us.

# In[ ]:


def param_tuning(X_train, y_train):
    '''Applies GridSearchCV to
    find the best parameters for
    Logistic regression and
    returns the fitted 
    GridSearchCV object'''
    
    params = [{'C':np.linspace(0,0.95,10), 'penalty':['l1','l2']}]
    grid_search = GridSearchCV(estimator = LogisticRegression(),
                               param_grid = params,
                               scoring = 'accuracy',
                               cv = 10,
                               n_jobs = -1)
    grid_search = grid_search.fit(X_train, y_train)
    return grid_search

grid_search = param_tuning(X_train, y_train)
print("best params : ", grid_search.best_params_)
print("score : ", grid_search.best_score_)


# In[ ]:


df_test['id'] = [i for i in range(1,len(df_test) + 1)]
df_test.set_index('id', inplace=True)


# Here we will be generating predictions using our fitted model.

# In[ ]:


# generate predictions
prediction = logreg_sm_model.predict(testing_X)
df = pd.DataFrame({"id":df_test.index,"type":prediction})
df.set_index('id', inplace=True)


# Now our dataframe looks good. However for submission, we need to split our data into four binary type columns.

# In[ ]:


def encode_columns(df):
    '''Takes each personality type and split it into its four components
    and encodes them into 0 or 1'''
    df['mind'] = df['type'].str[0]
    df['energy'] = df['type'].str[1]
    df['nature'] = df['type'].str[2]
    df['tactics'] = df['type'].str[3]
    df.replace({"mind" : {"I":0,"E":1},
                 "energy":{"S":0,"N":1},
                 "nature":{"F":0, "T":1},
                 "tactics":{"P":0,"J":1}}, inplace = True)
    return df

df = encode_columns(df).drop('type', axis=1)


# We now convert our dataframe into a submission csv file.

# In[ ]:


df.to_csv("submission.csv")


# # 6. Discussion

# We used various models such as Logistic Regression, Support Vector Machines and K-Nearest Neighbors and we found different results for each them. We tried doing some parameter tuning but the Kaggle kernel was taking too long suggesting that there could have been a problem with our chosen parameters.
# 
# **Above and beyond:**
# we have went as far as finding the sentiment analysis, this is your polarity and subjectivity. We also dealt with class imbalances and tried to fix that.

# # 7. Conclusion

# We have seen that predicting human behaviour from numbers can be hard but the result is somewhat fruiutful. We viewed the data using some EDA and then cleaned it using the neccesary NLP techniques. This allowed us to have the proper data so that we can model, we used 3 models for the modelling part and Logistic Regression perfomed the best based on our Kaggle score of 5.799 which was close to the required base value of 5.
# 
# Some of the key findings include:
#  - The ESXX types is the most rare based on the dataset and from tests online
#  - INXX types seem to have a higher occurence
#  - Oversampling worked better than undersampling.
#  - Logistic Regression was the best model with the default parameters

# In[ ]:




