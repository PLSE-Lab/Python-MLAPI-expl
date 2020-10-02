#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings('ignore')


# # Project Overview
# This project aims to classify questions retrieved from the Stack Overflow database based on the language they are related to. In order to accomplish this, the data will first be retrieved using an SQL query. Following, natural langauge processing(NLP) techniques and feature engineering will be used in conjunction with a deep learning neural network in an attempt to classify the questions. This will be my first kernel here on Kaggle and I'm really excited to share what I've accomplished. 
# 
# **I wished to include more techniques such as k-fold cross validation and use more data, however due to the 13 GB memory limit, I was unable to. However, I just want to thank Kaggle for allowing me to use a GPU compatible with TensorFlow as my own laptop runs on an AMD GPU.*  

# # Retrieving Data from database using SQL

# In[ ]:


import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
stackOverflow = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="stackoverflow")


# In[ ]:


stack_db = BigQueryHelper("bigquery-public-data", "stackoverflow")
stack_db.list_tables()


# In[ ]:


stack_db.head("posts_questions", num_rows=5)


# In[ ]:


stack_db.table_schema("posts_questions")


# For this project, we will be retrieving the title, body, tags, and views from the posts_questions table. Shown below is the query and command used to retrieve this information. The query ensures the retrieved data is tagged as either a python, java, javascript, sql, or R based question. Additionally, the length of the body was limited to only 1000 characters in length, and only 7500 results were retreived. These constraints were placed in order to prevent memory overflow. 

# In[ ]:


query1 = """
         SELECT
             title,
             body as question,
             tags as labels, 
             view_count as views
         FROM
             `bigquery-public-data.stackoverflow.posts_questions`
         WHERE 
             (tags LIKE '%python%' OR
             tags LIKE '%java%' OR 
             tags LIKE '%sql%' OR 
             tags LIKE '%|r|%' OR
             tags LIKE 'r|%') AND
             LENGTH(body) < 1000
         LIMIT
             7500;
         """

questions_df = stackOverflow.query_to_pandas(query1)


# # Data Preprocessing

# In[ ]:


#import necessary packages for analysis 
import nltk
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')


# First, we must split the tags. From the header, we can observe that the tags are seperated by the character '|' and are split accordingly.

# In[ ]:


questions_df['labels'] = questions_df['labels'].str.split('|')


# Next, we find the programming language tags and limit the questions to only those involving a single programming langauge. This is done for simplicity. 
# 

# In[ ]:


def return_tags(labels):
    langauges = [lang for lang in labels if lang in ['python','java','sql','r','javascript']] 
    return langauges


# In[ ]:


questions_df['labels'] = questions_df['labels'].apply(return_tags)


# In[ ]:


# Find rows that contain only a single tag
processed_df = questions_df[(questions_df['labels'].apply(len) > 0) & (questions_df['labels'].apply(len) < 2)]  


# In[ ]:


#verify that only questions with a single language tag are included
processed_df.head()


# In[ ]:


processed_df.info()


# In[ ]:


# In order to properly work with these labels, we must convert them from lists into strings.
def lst_to_str(lst):
    unpacked = ''.join(lst)
    return unpacked

processed_df.loc[:,'labels'] = processed_df.loc[:,'labels'].apply(lst_to_str)

#processed_df.describe()
#processed_df.groupby('labels').describe()


# # Data Visualization
# 
# Now we will begin exploring the data to determine what features to extract and train our neural network on. The first features that come to mind are the number of views each question received as well as the length of each question. For example, perhaps due to the complexity of a language over another, the questions pertaining to that langauge may be longer in length. Additionally, perhaps due to popularity of a certain langauge over another, questions related to one langauge may have more views over another. These features were the first to come to mind due to already being numeric within the dataframe.  

# First, we will explore the distribution of the number of view each question got based on what language they were related to. 

# In[ ]:


grid = sns.FacetGrid(processed_df[processed_df['views'] < 4000], col = 'labels', height = 5, aspect = 0.6)
grid.map(plt.hist, 'views', bins = 75)
axes = grid.axes
axes[0,1].set_xlim([0,4000])

plt.tight_layout()


# It appears that most languages are distrbuted evenly across number of views. There were some outliers amongst the languages however, I removed them for easier visualization. 
# 
# Now we will analyze the length of each question. This will be done by first applying the `len` function to the `question` column and placing the results into a new column labeled `length_of_question`.

# In[ ]:


processed_df.loc[:,'length_of_question'] = processed_df.loc[:,'question'].apply(len)


# In[ ]:


# verify the procedure worked.
processed_df.head(2)


# In[ ]:


# Plot the length of each question based on what language they were written in
grid = sns.FacetGrid(processed_df, hue = 'labels', height = 5, aspect = 2)
grid.map(plt.hist, 'length_of_question', bins = 50, alpha = 0.5)
axes = grid.axes

axes[0,0].set_title('Distribution of question lengths for each programming language')
axes[0,0].set_xlim([0,1000])
axes[0,0].legend()


# Clearly, no obvious trends are visible. The heigth of the bars indicate how many questions had a length within that range, and are affected by how many questions are about each language. From the plot, the questions are fairly similar in length throughout. 
#  
#  ## Extracting Features from Code
# Now that we know that didn't work we have to find different features to train our neural network on. 
# Some background: In typical stack overflow questions, questions are asked by first summarizing the problem, and then including code blocks describing what the user has tried. 
# We know that each language has syntactic difference, for example, in R, values are assigned using a less than sign along with a dash('<-'), whereas in python, values are assigned using the equals sign('='). What if we honed in on these differences? Additional differences that come to mind are the number of brackets used, the number of arithmetic operators, and even punctuation such as periods or slashes. 
# 
# Okay we have a plan of action, how do we actually implement this? Well first we have to parse the question strings for the code blocks. These questions are written in html and as such we can use a html parser to retrieve those code blocks. 

# In[ ]:


# An example of a question posted on stack overflow. Notice the html syntax and code delimiters.
processed_df.head()


# In order to actually implement the html parser onto the question column of our dataframe, we must define a function that retrieves the code. 

# In[ ]:


import lxml.html

def find_code(html_str):
    final_list = []

    dom = lxml.html.fromstring(html_str)
    codes = dom.xpath('//code')

    for code in codes:
        if code.text is None: 
            final_list.append('')
        else:
            final_list.append(code.text)
        
        
    final_list = ' '.join(final_list)
    return final_list 


# In[ ]:


processed_df.loc[:,'code'] = processed_df.loc[:,'question'].apply(find_code)


# Now that we've applied our function and retrieved our code, we need to extract the features of that code. We will now define functions that retrieve these features and apply them to our newly created `code` column. 

# In[ ]:


def count_colons(txt):
    return txt.count(':')

def count_semicolons(txt):
    return txt.count(';')

def count_slashes(txt):
    return txt.count('/')
                                      
def count_cbrackets(txt):
    return txt.count('{') + txt.count('}')

def count_sbrackets(txt):
    return txt.count('[') + txt.count(']')

def count_quotes(txt):
    return txt.count('"') + txt.count("'")

def count_arithmetic(txt):
    return txt.count('<') + txt.count('>') + txt.count('-') + txt.count('+') 

def count_period(txt):
    return txt.count('.')


# In[ ]:


processed_df.loc[:,'colon count']     = processed_df.loc[:,'code'].apply(count_colons)
processed_df.loc[:,'semicolon count'] = processed_df.loc[:,'code'].apply(count_semicolons)
processed_df.loc[:,'slash count']     = processed_df.loc[:,'code'].apply(count_slashes)
processed_df.loc[:,'cbracket count']  = processed_df.loc[:,'code'].apply(count_cbrackets)
processed_df.loc[:,'sbracket count']  = processed_df.loc[:,'code'].apply(count_sbrackets)
processed_df.loc[:,'quote count']     = processed_df.loc[:,'code'].apply(count_quotes)
processed_df.loc[:,'operator count']  = processed_df.loc[:,'code'].apply(count_arithmetic)
processed_df.loc[:,'period count']    = processed_df.loc[:,'code'].apply(count_period)


# In[ ]:


# Verify the functions worked.
processed_df.head()


# Now that we have verified that the functions worked as intended, we will use heatmaps to visualize the results, and potentially determine if there are any noticeable difference in each languages' syntax. 

# In[ ]:


fig, axis = plt.subplots(figsize=(25,30), nrows = 5)

python_features = processed_df[processed_df['labels'] == 'python'].loc[:,'colon count':]
js_features = processed_df[processed_df['labels'] == 'javascript'].loc[:,'colon count':]
java_features = processed_df[processed_df['labels'] == 'java'].loc[:,'colon count':]
sql_features = processed_df[processed_df['labels'] == 'sql'].loc[:,'colon count':]
r_features = processed_df[processed_df['labels'] == 'r'].loc[:,'colon count':]

sql_features.head()

sns.heatmap(python_features, cmap = 'viridis', ax = axis[0])
axis[0].set_title('Python features heatmap')

sns.heatmap(js_features, cmap = 'inferno', ax = axis[1])
axis[1].set_title('Javascript features heatmap')

sns.heatmap(java_features, cmap = 'viridis', ax = axis[2])
axis[2].set_title('Java features heatmap')

sns.heatmap(sql_features, cmap = 'inferno', ax = axis[3])
axis[3].set_title('SQL features heatmap')

sns.heatmap(r_features, cmap = 'viridis', ax = axis[4])
axis[4].set_title('R features heatmap')


# From these heatmaps, a couple of observations can be found:
# * The square bracket count, quote count, operator count and period count occur the most frequently in python.
# * For Javascript, quotes and operators clearly appear the most frequenctly.
# * For Java, the occurrences of each special character is fairly even. 
# * In SQL, these special characters rarely occur aside from arithmetic operatos. This is a largely due to SQL using white space and new lines to denote complete statements. 
# * For R, quote counts, operator counts, and period counts occur most often. 
# 
# Many of these observations make sense, with knowledge of each language's syntactic difference. 
# In order to get a complete overview, a heatmap displaying the total of of each special character grouped by their language is plotted. 

# In[ ]:


total_syntax_features = processed_df.groupby('labels').sum(axis=1).loc[:,'colon count':]

fig, aggregate_axis = plt.subplots(figsize=(15,6))
sns.heatmap(total_syntax_features, cmap = 'plasma', ax = aggregate_axis)
aggregate_axis.set_title('Syntactic Features of Programming Languages')


# It must be noted that this heatmap is skewed based on how many of each type of question was retrieved by the SQL query. Nonetheless, some observations can still be made: 
# * Amongst the special characters in both python and Java, periods occur quite often. This makes sense considering that both languages are object-oriented, where their methods are denonted by periods. 
# * The quote count for javascript is extremely high. This is due to the fact that javascript is used in conjunction with HTML, in which text denoted by quotes occur more frequently. 
# 
# In order to remove the effects of the uneven retrieval of data from the SQL query, we will use sci-kit `preprocessing` library.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()
total_syntax_features_scaled = pd.DataFrame(min_max_scaler.fit_transform(total_syntax_features.T), columns = total_syntax_features.index, index = total_syntax_features.columns)
total_syntax_features_scaled.head(10)

fig, axis_scaled = plt.subplots(figsize=(15,6))
sns.heatmap(total_syntax_features_scaled.T, cmap = 'plasma', ax = axis_scaled)
axis_scaled.set_title('Normalized Code Features')


# Here we are better able to visualize which features are the most prominent amongst their respective languages. 

# ## Extracting Features from Content

# Now that we've extracted features from the `code` column, we can begin processing the actual contents of the questions and their respective titles. In order to do this we will use an html parser to extract the text from those columns, and then use a `CountVectorizer` to convert those words into a sparse matrix to serve as inputs to our neural network. This is also known as converting text into a "bag-of-words" and is a common natural language processing technique. 
# 
# To do this, we will parse the question for any text. For this project, we will only be using english text. We will also exclude numbers and stopwords. 

# In[ ]:


from bs4 import BeautifulSoup

def find_text(html_str):
    full_text = ''

    parsedContent = BeautifulSoup(html_str, 'html.parser')

    text = parsedContent.findAll('p')
    
    for paragraph in text:
        full_text = full_text + paragraph.getText()
        
    return full_text    


# In[ ]:


import re 
import string
from nltk.corpus import stopwords 

stop_words = stopwords.words()
translation_table = dict.fromkeys(map(ord, string.punctuation), None)

def remove_punc_and_stopwords(full_text):
    cleaned_text = full_text.translate(translation_table)
    word_lst = re.findall('[a-zA-Z]+', cleaned_text)
    return " ".join(word_lst)


# In[ ]:


def clean_html_text(text):
    final_text = find_text(text)
    bag_of_words = remove_punc_and_stopwords(final_text)
    return bag_of_words


# In[ ]:


processed_df.head(2)


# In[ ]:


# Remove digits and special characters from title column. Note that it does not require an html parser.
processed_df.loc[:,'title'] = processed_df.loc[:,'title'].apply(remove_punc_and_stopwords)


# In[ ]:


# Extract the text of the question column, and remove digits and special characters.
processed_df.loc[:,'question'] = processed_df.loc[:,'question'].apply(clean_html_text)


# In[ ]:


# Observe that there are no longer any punctuation or digits in the questions.
processed_df.head(10)


# We will now implement the `CountVectorizer` and covert the text into a "bag-of-words". Additionally, we will use tf-idf to scale the weights according to how relevant each word is to the dataset. We will store the results into new dataframes to be concatenated with the rest of our neural network inputs later.
# 
# **Edit: The Multinomial naive-Bayes was experimented with and without using term frequency - inverse document frequency scaling. This process was repeated for the bag of words concatenated with the extracted code features.*

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

cv_title = CountVectorizer().fit(processed_df['title'])
vectorized_title = cv_title.transform(processed_df['title'])
vectorized_title_df = pd.DataFrame(vectorized_title.toarray(), columns = cv_title.get_feature_names())

# implement tfidf
tfidf_title = TfidfTransformer().fit(vectorized_title)
vectorized_tfidf_title = tfidf_title.transform(vectorized_title)
vectorized_tfidf_title_df = pd.DataFrame(vectorized_tfidf_title.toarray(), columns = cv_title.get_feature_names())


# In[ ]:


cv_question = CountVectorizer().fit(processed_df['question'])
vectorized_question = cv_question.transform(processed_df['question'])
vectorized_question_df = pd.DataFrame(vectorized_question.toarray(), columns = cv_question.get_feature_names())

# implement tfidf
tfidf_question = TfidfTransformer().fit(vectorized_question)
vectorized_tfidf_question = tfidf_question.transform(vectorized_question)
vectorized_tfidf_question_df = pd.DataFrame(vectorized_tfidf_question.toarray(), columns = cv_question.get_feature_names())


# In[ ]:


# combine dataframes 
cv_bow = pd.concat([vectorized_title_df, vectorized_question_df], axis = 1)
cv_tfidf_bow = pd.concat([vectorized_tfidf_title_df, vectorized_tfidf_question_df], axis = 1)


# In[ ]:


# import multinomial NB classifier and fit to dataset
from sklearn.naive_bayes import MultinomialNB

nb_cv_words_only = MultinomialNB().fit(cv_bow, processed_df['labels'])
nb_cv_tfidf_words_only = MultinomialNB().fit(cv_tfidf_bow, processed_df['labels'])


# In[ ]:


# make predictions
cv_bow_predictions = nb_cv_words_only.predict(cv_bow)
cv_tfidf_bow_predictions = nb_cv_tfidf_words_only.predict(cv_tfidf_bow)


# In[ ]:


# display classification report + confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
print('Multinomial Naive-Bayes Classification Report: ')
print(classification_report(processed_df['labels'], cv_bow_predictions))
print('\n')
print('Multinomial Naive-Bayes with TFIDF Classification Report: ')
print(classification_report(processed_df['labels'], cv_tfidf_bow_predictions))

fig, axes = plt.subplots(nrows=2,figsize=(10,10))

sns.heatmap(confusion_matrix(processed_df['labels'], cv_bow_predictions), ax = axes[0], annot=True, cmap='magma', fmt = 'g',
            xticklabels=['java', 'javascript','python','r','sql'], yticklabels=['java', 'javascript','python','r','sql'])
axes[0].set_title('Confusion Matrix of Naive-Bayes Classification')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

sns.heatmap(confusion_matrix(processed_df['labels'], cv_tfidf_bow_predictions), ax = axes[1], annot=True, cmap='magma', fmt = 'g',
            xticklabels=['java', 'javascript','python','r','sql'], yticklabels=['java', 'javascript','python','r','sql'])
axes[1].set_title('Confusion Matrix of Naive-Bayes Classification with TFIDF')
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')

plt.tight_layout()


# In[ ]:


# Repeat process for bag of words including code features
code_features = processed_df.loc[:,'colon count':]

cv_words_and_code = pd.concat([cv_bow.reset_index(drop = True), code_features.reset_index(drop = True)], axis = 1)
cv_words_and_code_tfidf = TfidfTransformer().fit_transform(cv_words_and_code)

nb_cv_words_and_code = MultinomialNB().fit(cv_words_and_code, processed_df['labels'])
nb_cv_words_and_code_tfidf = MultinomialNB().fit(cv_words_and_code_tfidf, processed_df['labels'])

nb_complete_predictions = nb_cv_words_and_code.predict(cv_words_and_code)
nb_tfidf_complete_predictions = nb_cv_words_and_code.predict(cv_words_and_code_tfidf)


# In[ ]:


print('Multinomial Naive-Bayes with code features: ')
print(classification_report(processed_df['labels'], nb_complete_predictions))

print('\n')

print('Multinomial Naive-Bayes with TFIDF and code features: ')
print(classification_report(processed_df['labels'], nb_tfidf_complete_predictions))

fig, axes = plt.subplots(nrows=2,figsize=(10,10))
sns.heatmap(confusion_matrix(processed_df['labels'], nb_complete_predictions), ax = axes[0], annot=True, cmap='magma', fmt = 'g',
            xticklabels=['java', 'javascript','python','r','sql'], yticklabels=['java', 'javascript','python','r','sql'])
axes[0].set_title('Confusion Matrix of Naive-Bayes Classification with code features')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')
sns.heatmap(confusion_matrix(processed_df['labels'], nb_tfidf_complete_predictions), ax = axes[1], annot=True, cmap='magma', fmt = 'g',
            xticklabels=['java', 'javascript','python','r','sql'], yticklabels=['java', 'javascript','python','r','sql'])
axes[1].set_title('Confusion Matrix of Naive-Bayes Classification with TFIDF and code features')
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')


# Now to implement the neural network, we need to combine all of the desired features. This includes the bag of words, the code features, the `'length_of_question'` column, and the `'views'` column.
# Clearly, the values of `'length_of_question'`, and `'views'` will far outnumber the ones and zeros of the countvectorized columns, as well as the code syntax totals. This can heavily skew the result of our neural network. In order to avoid this we will use sci-kit learn's StandardScaler to scale the values within an appropriate range. I chose to scale each section seperately within their "category' because I do not know which features have the most impact in determining what language the question pertains to. Perhaps, the language is more heavily dicatacted by the code_features than the content? As such, keep them scaled within their respective categories. 

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


views_and_length = processed_df.loc[:,['views','length_of_question']]
code_features = processed_df.loc[:,'colon count':]

# Scale the views column
ss_view_length = StandardScaler()
scaled_views_and_length = ss_view_length.fit_transform(views_and_length.astype(float)) 
scaled_views_and_length_df = pd.DataFrame(scaled_views_and_length, columns = views_and_length.columns)

# Scale the code features
ss_code = StandardScaler()
scaled_code_features = ss_code.fit_transform(code_features.astype(float))
scaled_code_features_df = pd.DataFrame(scaled_code_features, columns = code_features.columns)

# Concatenate the results into complete preprocessed dataframe.
final_scaled_vectorized_df = pd.concat([vectorized_title_df, vectorized_question_df, scaled_views_and_length_df, scaled_code_features_df], axis = 1)
final_scaled_vectorized_df.head(2)


# In[ ]:


final_scaled_vectorized_df.info()


# # Building the Deep Learning Network
# We will not begin constructing our neural network. We will do this useing Keras, which is a framework for Google's TensorFlow library. Notice that from the `.info()` method above, we observe that our input dataframe has 30,183 entries. This will play a role in determining the number of neurons each hidden layer contains as well as how many hidden layers we have. I chose the number of neurons to gradually decrease until the expected five outputs(one for each classification). Additionally, for the hidden layers the activation function was chosen to be `'relu'` and the output activation function was chosen to be `'softmax'`, which is standard for single label classification problems such as this.   

# **note, I attempted to use k-fold cross validation, however due to memory constraints, I was unable to.*
# 
# **(05/28/19)This kernel was updated to include dropout layers to reduce the possibility of overfitting, however, better results were achieved without the dropout layers.*
# 

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout

classifier = Sequential()
classifier.add(Dense(units = 10000,
                     input_shape = (30040,),
                     kernel_initializer = 'glorot_uniform',
                     activation = 'relu'
                    )
              )  
#classifier.add(Dropout(0.35))
classifier.add(Dense(units = 1150,
                     kernel_initializer = 'glorot_uniform',
                     activation = 'relu'
                    )
              )
#classifier.add(Dropout(0.25))
classifier.add(Dense(units = 130,
                     kernel_initializer = 'glorot_uniform',
                     activation = 'relu'
                    )
              )
#classifier.add(Dropout(0.25))
classifier.add(Dense(units = 50,
                     kernel_initializer = 'glorot_uniform',
                     activation = 'relu'
                    )
              )
classifier.add(Dense(units = 5,
                     kernel_initializer = 'glorot_uniform',
                     activation = 'softmax'
                    )
              )

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In order to properly train our data, we must split our data into a training set and a test set. Again, we use sci-kit learn's `train-test-split` function. We choose `X` to be our preprocessed data and our `y` to be the labels. When setting y, we must remember to dummy encode the labels because the labels are categorical. 

# In[ ]:


from sklearn.model_selection import train_test_split

X = final_scaled_vectorized_df
y = pd.get_dummies(processed_df['labels'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[ ]:


#training the data. A batch size and number of epochs were chosen to remain within the memory constraints.
classifier.fit(X_train, y_train, batch_size = 42, epochs = 75)


# In[ ]:


# store the predictions. 
predictions = classifier.predict(X_test)
predictions


# In[ ]:


# Retrieve labels by taking the max value in each row and convert it to a 1, effectively 'labeling' the resutls.
from sklearn.preprocessing import LabelBinarizer

labels = np.argmax(predictions, axis = 1)
lb = LabelBinarizer()
labeled_predictions = lb.fit_transform(labels)


# # Evaluation
# In order to evaluate the results of the network, we will use the sci-kit learn's `classification_report` and `confusion matrix`. The classification report will give the precision, recall, and f1-score of the results of an neural network, while the confusion matrix will indicate how many rows were properly classified.

# In[ ]:


from sklearn.metrics import classification_report

print(classification_report(y_test.values, labeled_predictions))


# In[ ]:


from sklearn.metrics import confusion_matrix

print('Confusion matrix for Java')
print(confusion_matrix(y_test.loc[:,'sql'], labeled_predictions[:,0]))
print('\n')

print('Confusion matrix for Javascript')
print(confusion_matrix(y_test.loc[:,'javascript'], labeled_predictions[:,1]))
print('\n')

print('Confusion matrix for Python')
print(confusion_matrix(y_test.loc[:,'python'], labeled_predictions[:,2]))
print('\n')

print('Confusion matrix for R')
print(confusion_matrix(y_test.loc[:,'r'], labeled_predictions[:,3]))
print('\n')

print('Confusion matrix for SQL')
print(confusion_matrix(y_test.loc[:,'sql'], labeled_predictions[:,4])) 


# # Conclusion
# Using some feature engineering and natural language processing techniques, I was able to train a neural network to classify questions from Stack Overflow based solely on their titles and content with a precision, recall, and f1-score of approximately 0.93. These results were achieved using the Multinomial Naive Bayes Classifier in conjunction with the extracted features from the code blocks of the text. In the future, I would like to develop an way to gather information from syntactic features that aren't described by characters. These features include spacing and stylistic tendencies users in each language tend towards. Additionally, this project shows how powerful the naive Bayes model is, despite the assumption that features are independent within classes. It is possible this classifier worked so well because the conditional probabilities between words are so small, they are almost negligible. 
