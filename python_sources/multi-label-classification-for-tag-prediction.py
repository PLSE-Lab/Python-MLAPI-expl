#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
import gc
warnings.filterwarnings("ignore")
import pandas as pd
import sqlite3
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
import re
import os
from sqlalchemy import create_engine # database connection
import datetime as dt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from skmultilearn.adapt import mlknn
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import LabelPowerset
from sklearn.naive_bayes import GaussianNB
from datetime import datetime
import pandasql


# # Stack Overflow: Tag Prediction

# <h1>1. Business Problem </h1>

# <h2> 1.1 Description </h2>

# <p style='font-size:18px'><b> Description </b></p>
# <p>
# Stack Overflow is the largest, most trusted online community for developers to learn, share their programming knowledge, and build their careers.<br />
# <br />
# Stack Overflow is something which every programmer use one way or another. Each month, over 50 million developers come to Stack Overflow to learn, share their knowledge, and build their careers. It features questions and answers on a wide range of topics in computer programming. The website serves as a platform for users to ask and answer questions, and, through membership and active participation, to vote questions and answers up or down and edit questions and answers in a fashion similar to a wiki or Digg. As of April 2014 Stack Overflow has over 4,000,000 registered users, and it exceeded 10,000,000 questions in late August 2015. Based on the type of tags assigned to questions, the top eight most discussed topics on the site are: Java, JavaScript, C#, PHP, Android, jQuery, Python and HTML.<br />
# <br />
# </p>

# <p style='font-size:18px'><b> Problem Statemtent </b></p>
# Suggest the tags based on the content that was there in the question posted on Stackoverflow.

# <h2> 1.2 Real World / Business Objectives and Constraints </h2>

# 1. Predict as many tags as possible with high precision and recall.
# 2. Incorrect tags could impact customer experience on StackOverflow.
# 3. No strict latency constraints.

# <h1>2. Machine Learning problem </h1>

# <h2> 2.1 Data </h2>

# <h3> 2.1.1 Data Overview </h3>

# Refer: https://www.kaggle.com/c/facebook-recruiting-iii-keyword-extraction/data
# <br>
# All of the data is in 2 files: Train and Test.<br />
# <pre>
# <b>Train.csv</b> contains 4 columns: Id,Title,Body,Tags.<br />
# <b>Test.csv</b> contains the same columns but without the Tags, which you are to predict.<br />
# <b>Size of Train.csv</b> - 6.75GB<br />
# <b>Size of Test.csv</b> - 2GB<br />
# <b>Number of rows in Train.csv</b> = 6034195<br />
# </pre>
# The questions are randomized and contains a mix of verbose text sites as well as sites related to math and programming. The number of questions from each site may vary, and no filtering has been performed on the questions (such as closed questions).<br />
# <br />
# 

# __Data Field Explaination__
# 
# Dataset contains 6,034,195 rows. The columns in the table are:<br />
# <pre>
# <b>Id</b> - Unique identifier for each question<br />
# <b>Title</b> - The question's title<br />
# <b>Body</b> - The body of the question<br />
# <b>Tags</b> - The tags associated with the question in a space-seperated format (all lowercase, should not contain tabs '\t' or ampersands '&')<br />
# </pre>
# 
# <br />

# <h3>2.1.2 Example Data point </h3>

# <pre>
# <b>Title</b>:  Implementing Boundary Value Analysis of Software Testing in a C++ program?
# <b>Body </b>: <pre><code>
#         #include&lt;
#         iostream&gt;\n
#         #include&lt;
#         stdlib.h&gt;\n\n
#         using namespace std;\n\n
#         int main()\n
#         {\n
#                  int n,a[n],x,c,u[n],m[n],e[n][4];\n         
#                  cout&lt;&lt;"Enter the number of variables";\n         cin&gt;&gt;n;\n\n         
#                  cout&lt;&lt;"Enter the Lower, and Upper Limits of the variables";\n         
#                  for(int y=1; y&lt;n+1; y++)\n         
#                  {\n                 
#                     cin&gt;&gt;m[y];\n                 
#                     cin&gt;&gt;u[y];\n         
#                  }\n         
#                  for(x=1; x&lt;n+1; x++)\n         
#                  {\n                 
#                     a[x] = (m[x] + u[x])/2;\n         
#                  }\n         
#                  c=(n*4)-4;\n         
#                  for(int a1=1; a1&lt;n+1; a1++)\n         
#                  {\n\n             
#                     e[a1][0] = m[a1];\n             
#                     e[a1][1] = m[a1]+1;\n             
#                     e[a1][2] = u[a1]-1;\n             
#                     e[a1][3] = u[a1];\n         
#                  }\n         
#                  for(int i=1; i&lt;n+1; i++)\n         
#                  {\n            
#                     for(int l=1; l&lt;=i; l++)\n            
#                     {\n                 
#                         if(l!=1)\n                 
#                         {\n                    
#                             cout&lt;&lt;a[l]&lt;&lt;"\\t";\n                 
#                         }\n            
#                     }\n            
#                     for(int j=0; j&lt;4; j++)\n            
#                     {\n                
#                         cout&lt;&lt;e[i][j];\n                
#                         for(int k=0; k&lt;n-(i+1); k++)\n                
#                         {\n                    
#                             cout&lt;&lt;a[k]&lt;&lt;"\\t";\n               
#                         }\n                
#                         cout&lt;&lt;"\\n";\n            
#                     }\n        
#                  }    \n\n        
#                  system("PAUSE");\n        
#                  return 0;    \n
#         }\n
#         </code></pre>\n\n
#         <p>The answer should come in the form of a table like</p>\n\n
#         <pre><code>       
#         1            50              50\n       
#         2            50              50\n       
#         99           50              50\n       
#         100          50              50\n       
#         50           1               50\n       
#         50           2               50\n       
#         50           99              50\n       
#         50           100             50\n       
#         50           50              1\n       
#         50           50              2\n       
#         50           50              99\n       
#         50           50              100\n
#         </code></pre>\n\n
#         <p>if the no of inputs is 3 and their ranges are\n
#         1,100\n
#         1,100\n
#         1,100\n
#         (could be varied too)</p>\n\n
#         <p>The output is not coming,can anyone correct the code or tell me what\'s wrong?</p>\n'
# <b>Tags </b>: 'c++ c'
# </pre>

# <h2>2.2 Mapping the real-world problem to a Machine Learning Problem </h2>

# <h3> 2.2.1 Type of Machine Learning Problem </h3>

# <p> It is a multi-label classification problem  <br>
# <b>Multi-label Classification</b>: Multilabel classification assigns to each sample a set of target labels. This can be thought as predicting properties of a data-point that are not mutually exclusive, such as topics that are relevant for a document. A question on Stackoverflow might be about any of C, Pointers, FileIO and/or memory-management at the same time or none of these. <br>
# __Credit__: http://scikit-learn.org/stable/modules/multiclass.html
# </p>

# <h3>2.2.2 Performance metric </h3>

# <b>Micro-Averaged F1-Score (Mean F Score) </b>: 
# The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal. The formula for the F1 score is:
# 
# <i>F1 = 2 * (precision * recall) / (precision + recall)</i><br>
# 
# In the multi-class and multi-label case, this is the weighted average of the F1 score of each class. <br>
# 
# <b>'Micro f1 score': </b><br>
# Calculate metrics globally by counting the total true positives, false negatives and false positives. This is a better metric when we have class imbalance.
# <br>
# 
# <b>'Macro f1 score': </b><br>
# Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
# <br>
# 
# https://www.kaggle.com/wiki/MeanFScore <br>
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html <br>
# <br>
# <b> Hamming loss </b>: The Hamming loss is the fraction of labels that are incorrectly predicted. <br>
# https://www.kaggle.com/wiki/HammingLoss <br>

# <h1> 3. Exploratory Data Analysis </h1>

# <h2> 3.1 Data Loading and Cleaning </h2>

# <h3>3.1.1 Using Pandas with SQLite to Load the data</h3>

# In[ ]:


df = pd.read_csv("/kaggle/input/facebook-recruiting-iii-keyword-extraction/Train.zip",nrows=10000)
df.head()


# <h3> 3.1.2 Counting the number of rows </h3>

# In[ ]:


# df = df_main[:50000]
df.info()
df[df.duplicated()] # printing duplicated rows


# In[ ]:



#delete when no longer needed
# del df_main
#collect residual garbage
# gc.collect()


# <h3>3.1.3 Checking for duplicates </h3>

# In[ ]:



df_no_dup = df.drop_duplicates()


# In[ ]:


df_no_dup.head()
# we can observe that there are duplicates


# In[ ]:


# number of times each question appeared in our database
# df_no_dup.cnt_dup.value_counts()


# In[ ]:


# deleting the rows which have empty tags for further processing, there were only 6 such rows
df_no_dup = df_no_dup.dropna()
df_no_dup.isnull().sum()


# In[ ]:


start = datetime.now()
df_no_dup["tag_count"] = df_no_dup["Tags"].apply(lambda text: len(text.split()))
# adding a new feature number of tags per question
print("Time taken to run this cell :", datetime.now() - start)
df_no_dup.head()


# In[ ]:


# distribution of number of tags per question
df_no_dup.tag_count.value_counts()
# df.info()
# df_no_dup.info()


# In[ ]:


#Creating a new database with no duplicates
# os.remove('train.db')
# if not os.path.isfile('train_no_dup.db'):
#     disk_dup = create_engine("sqlite:///train_no_dup.db")
#     no_dup = pd.DataFrame(df_no_dup, columns=['Title', 'Body', 'Tags'])
#     no_dup.to_sql('no_dup_train',disk_dup)


# <h2> 3.2 Analysis of Tags </h2>

# <h3> 3.2.1 Total number of unique tags </h3>

# In[ ]:


# Importing & Initializing the "CountVectorizer" object, which 
#is scikit-learn's bag of words tool.

#by default 'split()' will tokenize each tag using space.
vectorizer = CountVectorizer(tokenizer = lambda x: x.split())
# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of strings.
tag_dtm = vectorizer.fit_transform(df_no_dup['Tags'])


# In[ ]:


print("Number of data points :", tag_dtm.shape[0])
print("Number of unique tags :", tag_dtm.shape[1])


# In[ ]:


#'get_feature_name()' gives us the vocabulary.
tags = vectorizer.get_feature_names()
#Lets look at the tags we have.
print("Some of the tags we have :", tags[:10])


# <h3> 3.2.3 Number of times a tag appeared </h3>

# In[ ]:


# https://stackoverflow.com/questions/15115765/how-to-access-sparse-matrix-elements
#Lets now store the document term matrix in a dictionary.
freqs = tag_dtm.sum(axis=0).A1
result = dict(zip(tags, freqs))


# In[ ]:


#Saving this dictionary to csv files.
if not os.path.isfile('tag_counts_dict_dtm.csv'):
    with open('tag_counts_dict_dtm.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in result.items():
            writer.writerow([key, value])
tag_df = pd.read_csv("tag_counts_dict_dtm.csv", names=['Tags', 'Counts'])
tag_df.head()


# In[ ]:


tag_df_sorted = tag_df.sort_values(['Counts'], ascending=False)
tag_counts = tag_df_sorted['Counts'].values


# In[ ]:


plt.plot(tag_counts)
plt.title("Distribution of number of times tag appeared questions")
plt.grid()
plt.xlabel("Tag number")
plt.ylabel("Number of times tag appeared")
plt.show()


# In[ ]:


plt.plot(tag_counts[0:10000])
plt.title('first 10k tags: Distribution of number of times tag appeared questions')
plt.grid()
plt.xlabel("Tag number")
plt.ylabel("Number of times tag appeared")
plt.show()
print(len(tag_counts[0:10000:25]), tag_counts[0:10000:25])


# In[ ]:


plt.plot(tag_counts[0:1000])
plt.title('first 1k tags: Distribution of number of times tag appeared questions')
plt.grid()
plt.xlabel("Tag number")
plt.ylabel("Number of times tag appeared")
plt.show()
print(len(tag_counts[0:1000:5]), tag_counts[0:1000:5])


# In[ ]:


plt.plot(tag_counts[0:500])
plt.title('first 500 tags: Distribution of number of times tag appeared questions')
plt.grid()
plt.xlabel("Tag number")
plt.ylabel("Number of times tag appeared")
plt.show()
print(len(tag_counts[0:500:5]), tag_counts[0:500:5])


# In[ ]:


plt.plot(tag_counts[0:100], c='b')
plt.scatter(x=list(range(0,100,5)), y=tag_counts[0:100:5], c='orange', label="quantiles with 0.05 intervals")
# quantiles with 0.25 difference
plt.scatter(x=list(range(0,100,25)), y=tag_counts[0:100:25], c='m', label = "quantiles with 0.25 intervals")

for x,y in zip(list(range(0,100,25)), tag_counts[0:100:25]):
    plt.annotate(s="({} , {})".format(x,y), xy=(x,y), xytext=(x-0.05, y+500))

plt.title('first 100 tags: Distribution of number of times tag appeared questions')
plt.grid()
plt.xlabel("Tag number")
plt.ylabel("Number of times tag appeared")
plt.legend()
plt.show()
print(len(tag_counts[0:100:5]), tag_counts[0:100:5])


# In[ ]:


# Store tags greater than 10K in one list
lst_tags_gt_10k = tag_df[tag_df.Counts>10000].Tags
#Print the length of the list
print ('{} Tags are used more than 10000 times'.format(len(lst_tags_gt_10k)))
# Store tags greater than 100K in one list
lst_tags_gt_100k = tag_df[tag_df.Counts>100000].Tags
#Print the length of the list.
print ('{} Tags are used more than 100000 times'.format(len(lst_tags_gt_100k)))


# <b>Observations:</b><br />
# 1. There are total 153 tags which are used more than 10000 times.
# 2. 14 tags are used more than 100000 times.
# 3. Most frequent tag (i.e. c#) is used 331505 times.
# 4. Since some tags occur much more frequenctly than others, Micro-averaged F1-score is the appropriate metric for this probelm.

# <h3> 3.2.4 Tags Per Question </h3>

# In[ ]:


#Storing the count of tag in each question in list 'tag_count'
tag_quest_count = tag_dtm.sum(axis=1).tolist()
#Converting list of lists into single list, we will get [[3], [4], [2], [2], [3]] and we are converting this to [3, 4, 2, 2, 3]
tag_quest_count=[int(j) for i in tag_quest_count for j in i]
print ('We have total {} datapoints.'.format(len(tag_quest_count)))

print(tag_quest_count[:5])


# In[ ]:


print( "Maximum number of tags per question: %d"%max(tag_quest_count))
print( "Minimum number of tags per question: %d"%min(tag_quest_count))
print( "Avg. number of tags per question: %f"% ((sum(tag_quest_count)*1.0)/len(tag_quest_count)))


# In[ ]:


sns.countplot(tag_quest_count, palette='gist_rainbow')
plt.title("Number of tags in the questions ")
plt.xlabel("Number of Tags")
plt.ylabel("Number of questions")
plt.show()


# <b>Observations:</b><br />
# 1. Maximum number of tags per question: 5
# 2. Minimum number of tags per question: 1
# 3. Avg. number of tags per question: 2.899
# 4. Most of the questions are having 2 or 3 tags

# <h3>3.2.5 Most Frequent Tags </h3>

# In[ ]:


# Ploting word cloud
start = datetime.now()

# Lets first convert the 'result' dictionary to 'list of tuples'
tup = dict(result.items())
#Initializing WordCloud using frequencies of tags.
wordcloud = WordCloud(    background_color='black',
                          width=1600,
                          height=800,
                    ).generate_from_frequencies(tup)

fig = plt.figure(figsize=(30,20))
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
fig.savefig("tag.png")
plt.show()
print("Time taken to run this cell :", datetime.now() - start)


# <b>Observations:</b><br />
# A look at the word cloud shows that "c#", "java", "php", "asp.net", "javascript", "c++" are some of the most frequent tags.

# <h3> 3.2.6 The top 20 tags </h3>

# In[ ]:


i=np.arange(30)
tag_df_sorted.head(30).plot(kind='bar')
plt.title('Frequency of top 20 tags')
plt.xticks(i, tag_df_sorted['Tags'])
plt.xlabel('Tags')
plt.ylabel('Counts')
plt.show()


# <b>Observations:</b><br />
# 1. Majority of the most frequent tags are programming language.
# 2. C# is the top most frequent programming language.
# 3. Android, IOS, Linux and windows are among the top most frequent operating systems.

# <h3> 3.3 Cleaning and preprocessing of Questions </h3>

# <h3> 3.3.1 Preprocessing </h3>

# <ol> 
#     <li> Sample 1M data points </li>
#     <li> Separate out code-snippets from Body </li>
#     <li> Remove Spcial characters from Question title and description (not in code)</li>
#     <li> Remove stop words (Except 'C') </li>
#     <li> Remove HTML Tags </li>
#     <li> Convert all the characters into small letters </li>
#     <li> Use SnowballStemmer to stem the words </li>
# </ol>

# In[ ]:


def striphtml(data):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(data))
    return cleantext
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")


# __ we create a new data base to store the sampled and preprocessed questions __

# qus_list=[]
# qus_with_code = 0
# len_before_preprocessing = 0 
# len_after_preprocessing = 0 
# start = datetime.now()
# for index,row in df_no_dup.iterrows():
#     title, body, tags = row["Title"], row["Body"], row["Tags"]
#     if '<code>' in body:
#         qus_with_code+=1
#     len_before_preprocessing+=len(title) + len(body)
#     body=re.sub('<code>(.*?)</code>', '', body, flags=re.MULTILINE|re.DOTALL)
#     body = re.sub('<.*?>', ' ', str(body.encode('utf-8')))
#     title=title.encode('utf-8')
#     question=str(title)+" "+str(body)
#     question=re.sub(r'[^A-Za-z]+',' ',question)
#     words=word_tokenize(str(question.lower()))
#     question=' '.join(str(stemmer.stem(j)) for j in words if j not in stop_words and (len(j)!=1 or j=='c'))
#     qus_list.append(question)
#     len_after_preprocessing += len(question)
# #     print(index)
# print("Time taken to run this cell :", datetime.now() - start)

# In[ ]:


#http://www.bernzilla.com/2008/05/13/selecting-a-random-row-from-an-sqlite-table/
# with more weights to title
start = datetime.now()
qus_list=[]
qus_with_code = 0
len_before_preprocessing = 0 
len_after_preprocessing = 0 
for index,row in df.iterrows():
    title, body, tags = row["Title"], row["Body"], row["Tags"]
    if '<code>' in body:
        qus_with_code+=1
    len_before_preprocessing+=len(title) + len(body)
    body=re.sub('<code>(.*?)</code>', '', body, flags=re.MULTILINE|re.DOTALL)
    body = re.sub('<.*?>', ' ', str(body.encode('utf-8')))
    title=title.encode('utf-8')
    question=str(title)+" "+str(title)+" "+str(title)+" "+ body
    question=re.sub(r'[^A-Za-z]+',' ',question)
    words=word_tokenize(str(question.lower()))
    question=' '.join(str(stemmer.stem(j)) for j in words if j not in stop_words and (len(j)!=1 or j=='c'))
    qus_list.append(question)
    len_after_preprocessing += len(question)
df["question_with_more_wt_title"] = qus_list
avg_len_before_preprocessing=(len_before_preprocessing*1.0)/df.shape[0]
avg_len_after_preprocessing=(len_after_preprocessing*1.0)/df.shape[0]
print( "Avg. length of questions(Title+Body) before preprocessing: ", avg_len_before_preprocessing)
print( "Avg. length of questions(Title+Body) after preprocessing: ", avg_len_after_preprocessing)
print ("% of questions containing code: ", (qus_with_code*100.0)/df.shape[0])
print("Time taken to run this cell :", datetime.now() - start)


# In[ ]:



df_no_dup["question"] = qus_list
avg_len_before_preprocessing=(len_before_preprocessing*1.0)/df_no_dup.shape[0]
avg_len_after_preprocessing=(len_after_preprocessing*1.0)/df_no_dup.shape[0]
print( "Avg. length of questions(Title+Body) before preprocessing: ", avg_len_before_preprocessing)
print( "Avg. length of questions(Title+Body) after preprocessing: ", avg_len_after_preprocessing)
print ("% of questions containing code: ", (qus_with_code*100.0)/df_no_dup.shape[0])


# In[ ]:


preprocessed_data = df_no_dup[['question','Tags']]
preprocessed_data.head()


# In[ ]:



# import pickle
# with open('preprocessed.pkl', 'wb') as f:
#     pickle.dump(preprocessed_data, f)
# with open('preprocessed.pkl', 'rb') as f:
#     preprocessed_data = pickle.load(f)


# In[ ]:


print("number of data points in sample :", preprocessed_data.shape[0])
print("number of dimensions :", preprocessed_data.shape[1])


# <h1>4. Machine Learning Models </h1>

# <h2> 4.1 Converting tags for multilabel problems </h2>

# <table>
# <tr>
# <th>X</th><th>y1</th><th>y2</th><th>y3</th><th>y4</th>
# </tr>
# <tr>
# <td>x1</td><td>0</td><td>1</td><td>1</td><td>0</td>
# </tr>
# <tr>
# <td>x1</td><td>1</td><td>0</td><td>0</td><td>0</td>
# </tr>
# <tr>
# <td>x1</td><td>0</td><td>1</td><td>0</td><td>0</td>
# </tr>
# </table>

# In[ ]:


# binary='true' will give a binary vectorizer
vectorizer = CountVectorizer(tokenizer = lambda x: x.split(), binary='true')
multilabel_y = vectorizer.fit_transform(preprocessed_data['Tags'])


# __ We will sample the number of tags instead considering all of them (due to limitation of computing power) __

# In[ ]:


def tags_to_choose(n):
    t = multilabel_y.sum(axis=0).tolist()[0]
    sorted_tags_i = sorted(range(len(t)), key=lambda i: t[i], reverse=True)
    multilabel_yn=multilabel_y[:,sorted_tags_i[:n]]
    return multilabel_yn

def questions_explained_fn(n):
    multilabel_yn = tags_to_choose(n)
    x= multilabel_yn.sum(axis=1)
    return (np.count_nonzero(x==0))


# In[ ]:


questions_explained = []
total_tags=multilabel_y.shape[1]
total_qs=preprocessed_data.shape[0]
for i in range(500, total_tags, 100):
    questions_explained.append(np.round(((total_qs-questions_explained_fn(i))/total_qs)*100,3))


# In[ ]:


fig, ax = plt.subplots()
ax.plot(questions_explained)
xlabel = list(500+np.array(range(-50,450,50))*50)
ax.set_xticklabels(xlabel)
plt.xlabel("Number of tags")
plt.ylabel("Number Questions coverd partially")
plt.grid()
plt.show()
# you can choose any number of tags based on your computing power, minimun is 50(it covers 90% of the tags)
print("with ",5500,"tags we are covering ",questions_explained[50],"% of questions")


# In[ ]:


multilabel_yx = tags_to_choose(5500)
print("number of questions that are not covered :", questions_explained_fn(5500),"out of ", total_qs)


# In[ ]:


print("Number of tags in sample :", multilabel_y.shape[1])
print("number of tags taken :", multilabel_yx.shape[1],"(",(multilabel_yx.shape[1]/multilabel_y.shape[1])*100,"%)")


# __ We consider top 15% tags which covers  99% of the questions __

# <h2>4.2 Split the data into test and train (80:20) </h2>

# In[ ]:


total_size=preprocessed_data.shape[0]
train_size=int(0.80*total_size)

x_train=preprocessed_data.head(train_size)
x_test=preprocessed_data.tail(total_size - train_size)

y_train = multilabel_yx[0:train_size,:]
y_test = multilabel_yx[train_size:total_size,:]


# In[ ]:


print("Number of data points in train data :", y_train.shape)
print("Number of data points in test data :", y_test.shape)


# <h2>4.3 Featurizing data </h2>

# In[ ]:


start = datetime.now()
vectorizer = TfidfVectorizer(min_df=0.00009, max_features=200000, smooth_idf=True, norm="l2",                              tokenizer = lambda x: x.split(), sublinear_tf=False, ngram_range=(1,4))
x_train_multilabel = vectorizer.fit_transform(x_train['question'])
x_test_multilabel = vectorizer.transform(x_test['question'])
print("Time taken to run this cell :", datetime.now() - start)


# In[ ]:


print("Dimensions of train data X:",x_train_multilabel.shape, "Y :",y_train.shape)
print("Dimensions of test data X:",x_test_multilabel.shape,"Y:",y_test.shape)


# In[ ]:


# https://www.analyticsvidhya.com/blog/2017/08/introduction-to-multi-label-classification/
#https://stats.stackexchange.com/questions/117796/scikit-multi-label-classification
# classifier = LabelPowerset(GaussianNB())
"""
from skmultilearn.adapt import MLkNN
classifier = MLkNN(k=21)

# train
classifier.fit(x_train_multilabel, y_train)

# predict
predictions = classifier.predict(x_test_multilabel)
print(accuracy_score(y_test,predictions))
print(metrics.f1_score(y_test, predictions, average = 'macro'))
print(metrics.f1_score(y_test, predictions, average = 'micro'))
print(metrics.hamming_loss(y_test,predictions))

"""
# we are getting memory error because the multilearn package 
# is trying to convert the data into dense matrix
# ---------------------------------------------------------------------------
#MemoryError                               Traceback (most recent call last)
#<ipython-input-170-f0e7c7f3e0be> in <module>()
#----> classifier.fit(x_train_multilabel, y_train)


# <h2> 4.4 Applying Logistic Regression with OneVsRest Classifier </h2>

# In[ ]:


# this will be taking so much time try not to run it, download the lr_with_equal_weight.pkl file and use to predict
# This takes about 6-7 hours to run.
classifier = OneVsRestClassifier(SGDClassifier(loss='hinge', alpha=0.00001, penalty='l1'), n_jobs=-1)
classifier.fit(x_train_multilabel, y_train)
predictions = classifier.predict(x_test_multilabel)

print("accuracy :",metrics.accuracy_score(y_test,predictions))
print("macro f1 score :",metrics.f1_score(y_test, predictions, average = 'macro'))
print("micro f1 scoore :",metrics.f1_score(y_test, predictions, average = 'micro'))
print("hamming loss :",metrics.hamming_loss(y_test,predictions))
print("Precision recall report :\n",metrics.classification_report(y_test, predictions))


# In[ ]:


df_test = pd.read_csv('/kaggle/input/facebook-recruiting-iii-keyword-extraction/Test.zip', nrows=10000)
df_test.head()
# x_test_multilabel
# predictions


# In[ ]:


#http://www.bernzilla.com/2008/05/13/selecting-a-random-row-from-an-sqlite-table/
# with more weights to title
start = datetime.now()
qus_list=[]
qus_with_code = 0
len_before_preprocessing = 0 
len_after_preprocessing = 0 
for index,row in df_test.iterrows():
    title, body = row["Title"], row["Body"]#, row["Tags"]
    if '<code>' in body:
        qus_with_code+=1
    len_before_preprocessing+=len(title) + len(body)
    body=re.sub('<code>(.*?)</code>', '', body, flags=re.MULTILINE|re.DOTALL)
    body = re.sub('<.*?>', ' ', str(body.encode('utf-8')))
    title=title.encode('utf-8')
    question=str(title)+" "+str(title)+" "+str(title)+" "+ body
    question=re.sub(r'[^A-Za-z]+',' ',question)
    words=word_tokenize(str(question.lower()))
    question=' '.join(str(stemmer.stem(j)) for j in words if j not in stop_words and (len(j)!=1 or j=='c'))
    qus_list.append(question)
    len_after_preprocessing += len(question)
df_test["question_with_more_wt_title"] = qus_list
avg_len_before_preprocessing=(len_before_preprocessing*1.0)/df_test.shape[0]
avg_len_after_preprocessing=(len_after_preprocessing*1.0)/df_test.shape[0]
print( "Avg. length of questions(Title+Body) before preprocessing: ", avg_len_before_preprocessing)
print( "Avg. length of questions(Title+Body) after preprocessing: ", avg_len_after_preprocessing)
print ("% of questions containing code: ", (qus_with_code*100.0)/df_test.shape[0])
print("Time taken to run this cell :", datetime.now() - start)


# In[ ]:


# df_test[['question_with_more_wt_title','Id']]
df_test.head()


# In[ ]:


preprocessed_data = df_test[["question_with_more_wt_title",'Id']]
print("Shape of preprocessed data :", preprocessed_data.shape)


# In[ ]:


preprocessed_data.head()


# In[ ]:


print("number of data points in sample :", preprocessed_data.shape[0])
print("number of dimensions :", preprocessed_data.shape[1])


# In[ ]:


# vectorizer = TfidfVectorizer(min_df=0.00009, max_features=200000, smooth_idf=True, norm="l2", \
#                              tokenizer = lambda x: x.split(), sublinear_tf=False, ngram_range=(1,4))
# X_train_multilabel = vectorizer.fit_transform(X_train['question_with_more_wt_title'])
X_test_multilabel = vectorizer.transform(preprocessed_data['question_with_more_wt_title'])


# In[ ]:


y_pred = classifier.predict(X_test_multilabel)


# In[ ]:


Y_pred = classifier.predict(x_test_multilabel)
Y_pred


# In[ ]:




