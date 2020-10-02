#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Dependencies and Setup
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np

import requests
import time
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import collections
import timeit
import re
from datetime import  datetime
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df_answers = pd.read_csv('../input/answers.csv')
df_questions = pd.read_csv('../input/questions.csv')


# **Merge Questions with Answers**

# In[ ]:


df_answeredQuestions = pd.merge(df_questions,df_answers, left_on = 'questions_id', right_on = 'answers_question_id', how='outer')
columns=['questions_title','questions_body','answers_body','questions_id','questions_author_id','answers_id','answers_author_id']


# **Questions with No Answer**

# In[ ]:


df_not_answeredQuestions=df_answeredQuestions[df_answeredQuestions['answers_id'].isnull()][columns]
df_not_answeredQuestions.head()


# **Questions with Answer**

# In[ ]:


df_answeredQuestions=df_answeredQuestions[df_answeredQuestions['answers_id'].isnull()==False][columns]
df_answeredQuestions.head()


# **Top Questions with the most Answers**

# In[ ]:


df_MostAnswerdQuestions=pd.DataFrame(df_answeredQuestions.groupby(['questions_id'],as_index=False)                      .count())[['questions_id','answers_id']]                      .rename(columns = {'answers_id':'answer count'})

print(df_MostAnswerdQuestions.info())
print()
df_MostAnswerdQuestions=df_MostAnswerdQuestions[df_MostAnswerdQuestions['answer count']>=5].sort_values('answer count',ascending=False).reset_index(drop=True)
# Questions with atleast 5 answers
df_MostAnswerdQuestions.head()


# In[ ]:


sns.countplot(x='answer count',data=df_MostAnswerdQuestions)


# **Create and generate a word cloud image based on title and body of questions.**
# 
# **and display most Common Words used in Title/body of Questions**

# In[ ]:


# Create stopword list:
stopWord=['really','this','dont','just','was','but','about','being','know','when','does','take','there','most','when','would','best','become','some','want','and','that','should','have','can','get','the','you','your','which','will','what','with','did','into','not','any','more','much','all','has','had','for','how','are']
stopwords = set(STOPWORDS)
stopwords.update(stopWord)

questionTitle = " ".join(review for review in df_questions['questions_title'])
# Create and generate a word cloud image:
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(questionTitle)
# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

wordcount = {} 

def generate_word(text):
    '''fast, easy, and clean way to iterate by saving memory space''' 
    for word in text.lower().split():
        yield word  

def generate_word_count(text):
    for word in generate_word(text):
        word = word.replace(".","")
        word = word.replace("'","")
        word = word.replace(",","")
        word = word.replace(":","")
        word = word.replace("\"","")
        word = word.replace("!","")
        word = word.replace("“","")
        word = word.replace("‘","")
        word = word.replace("*","")
        word = word.replace("?","")
        word = word.replace("#","")
        if len(word)>=3 and word not in stopWord: 
            if word not in wordcount:
                wordcount[word] = 1
            else:
                wordcount[word] += 1
    return wordcount


word_counter = collections.Counter(generate_word_count(questionTitle))
print('Most Common Words used in Title of Questions')
print()
for word, count in word_counter.most_common(20):
    print(word, ": ", count)
   


questionBody = " ".join(review for review in df_questions['questions_body'])
# Create and generate a word cloud image:
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(questionBody)
# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


word_counter = collections.Counter(generate_word_count(questionBody))
print('Most Common Words used in Body of Questions')
print()
for word, count in word_counter.most_common(20):
    print(word, ": ", count)    
    


# In[ ]:


df_professionals=pd.read_csv('../input/professionals.csv')
df_professionals=df_professionals[(df_professionals['professionals_headline'].notnull()) & 
                                  (df_professionals['professionals_industry'].notnull())] \
                    [['professionals_id','professionals_industry','professionals_headline']].reset_index(drop=True)

df_professionals.head()


# In[ ]:


df_tag_questions=pd.read_csv("../input/tag_questions.csv")
df_tag_questions.head()


# In[ ]:


df_tags=pd.read_csv("../input/tags.csv")
df_tags.head()


# In[ ]:


# Dataframe question/tag
df_tag_questions_detail = pd.merge(df_tag_questions,df_tags, left_on='tag_questions_tag_id' ,right_on= 'tags_tag_id', how='left')                                 [['tag_questions_question_id','tags_tag_name']].rename(columns = {'tags_tag_name':'question_tag'})
df_tag_questions_detail.head()


# **Answers can only be posted by users who are registered as Professionals. **
# 
# **Dataframe Answers by professionals**

# In[ ]:


columns=['answers_id','answers_question_id','professionals_id','professionals_industry','professionals_headline']
df_answers_professionals = pd.merge(df_answers,df_professionals, left_on='answers_author_id' ,right_on= 'professionals_id', how='left')[columns]
df_answers_professionals=df_answers_professionals.dropna(subset=['professionals_id']).reset_index(drop=True)
print(df_answers_professionals.info())
df_answers_professionals.head()


# ** Correlation between professionals professionals_industry/professionals_headline to question tag**

# In[ ]:


columns=['answers_id','answers_question_id','professionals_id','professionals_industry','professionals_headline','question_tag']
df_q_p_t = pd.merge(df_answers_professionals,df_tag_questions_detail, left_on='answers_question_id',
                  right_on= 'tag_questions_question_id', how='left')[columns]
df_q_p_t=df_q_p_t.dropna(subset=['question_tag']).reset_index(drop=True)
df_q_p_t.head()


# **Dataframe Professionals Headline by Question Tag**
# 
# **These tags have been answered by these professionals**

# In[ ]:


df_professionalsHeadline_by_questionTag=df_q_p_t[['professionals_headline','question_tag']]
df_professionalsHeadline_by_questionTag.head()


# **Dataframe Professionals Industry by Question Tag**
# 
# **These tags have been answered by these professionals**

# In[ ]:


df_professionalsIndustry_by_questionTag=df_q_p_t[['professionals_industry','question_tag']]
df_professionalsIndustry_by_questionTag.head()


# 
# # Solution number 1 example
# 
# # Analysing historical data shows the correlation between question tag and professionals_headline/professionals_industry
# 
# # To make it simple, these tags have been answered by these professionals.

# In[ ]:


df_correlation_q_p=pd.concat([df_professionalsHeadline_by_questionTag.set_index('question_tag'),df_professionalsIndustry_by_questionTag.set_index('question_tag')],
             axis=1, join='inner').reset_index()

df_correlation_q_p.head()


# 
# # Solution number 1 example
# 
# # You can send questuion tags to dataframe to find out professionals_headline/professionals_industry who answerd the questions before.
# # Based on professionals_headline or professionals_industry you can get list of professionals and email them the question.
# 

# In[ ]:


def searchTag(questionTags):
    '''This function searchs historical question hashtags and returns 
        list of related professional ids
    '''
    df_result_1=(df_correlation_q_p[df_correlation_q_p['question_tag'].isin(questionTags)] 
    .sort_values(by=['question_tag']).reset_index(drop=True))
    if not df_result_1.empty:
        #Join with df_prof based on (professionals_headline/professionals_industry)
        #to get the list of professionals
        columns=['professionals_id','professionals_industry','professionals_headline','question_tag']
        df_result = pd.merge(df_result_1,df_professionals, 
                             on=['professionals_headline','professionals_industry'], 
                             how='left')[columns]
        return df_result
    else:
        return 'No result found.'
   
#questionTags=input("Please enter a comma-separated hashtags included in the question:")
questionTags="police,law"
searchTag(questionTags.split(','))


# 
# **Machine Learning**
# 
# **Implementing a simple Logistic Regression to predict the probability of question ends with our favorite  comment (Our favorite comments tend to have "Thank you" in them).**

# In[ ]:


df_qa = pd.merge(df_questions,df_answers, left_on = 'questions_id', 
                 right_on = 'answers_question_id', how='right')


# **Calculate the period of each question has been answered (in days) **
# 
# **0 day means the question has been answered on the same day**

# In[ ]:


df_qa['answeredInDays']= (pd.to_datetime(df_qa['answers_date_added'])
                    -pd.to_datetime(df_qa['questions_date_added'])).dt.days

df_qa.head()


# **[hasComment]**
# 
# **We create a new column **
# 
# **[hasComment]=1 if the answer has atleast one comment**
# 
# **[hasComment]=0 if the answer has no comment**
# 

# In[ ]:


comments=Path("../input/comments.csv")
df_comments=pd.read_csv(comments)


# In[ ]:


df_qa_comment = pd.merge(df_qa,df_comments, left_on = 'answers_id', 
                 right_on = 'comments_parent_content_id', how='left')


df_qa_comment['hasComment']=df_qa_comment['comments_parent_content_id'].apply(lambda x: 1 if pd.notnull(x) else 0)
df_qa_comment=df_qa_comment[['questions_id','answers_id','comments_id','comments_parent_content_id','hasComment','answeredInDays']]

df_qa_comment.head()


# **Get the last comment of an answer**

# In[ ]:


df_last_comment=df_comments[df_comments['comments_date_added']
            .isin(df_comments.groupby('comments_parent_content_id',as_index=False)
                  .max()['comments_date_added'].values)].reset_index()

df_last_comment=df_last_comment.drop(columns=['index'])
df_last_comment.head()


# **Number of Comments**

# In[ ]:


df_comment_count=pd.DataFrame(
    df_comments.groupby('comments_parent_content_id',as_index=False).count())[['comments_parent_content_id','comments_id']]
df_comment_count.rename(columns = {'comments_id':'number_of_comments'},inplace=True)
df_comment_count.head()


# **Our favorite comments tend to have "Thank you" in them :)**
#     
# **We create a new column [hasFavComment]**
# 
# **[hasFavComment]=1 if comment has ended with "Thank you or Thanks"**
# 
# **[hasFavComment]=0 if the comment has not ended with "Thank you or Thanks"**
# 

# In[ ]:


favComment = ['thank','pleasure']
df_last_comment['hasFavComment']=df_last_comment['comments_body'].apply(lambda x: 
                                                            1 if any(re.findall(r"(?=("+'|'.join(favComment)+r"))",str(x).lower())) else 0)
df_last_comment=df_last_comment[['comments_id','comments_parent_content_id','hasFavComment']]
df_last_comment.head()


# **Merge all the datasets to get one single dataset with the following columns**

# In[ ]:


columns=['questions_id','answers_id','hasComment','number_of_comments','answeredInDays','hasFavComment']
df_qa_comment_summary = pd.merge(df_qa_comment,df_last_comment, on = 'comments_parent_content_id', how='left')

df_qa_comment_summary = pd.merge(df_qa_comment_summary,df_comment_count, left_on = 'answers_id', 
                 right_on = 'comments_parent_content_id', how='left')

df_qa_comment_summary=df_qa_comment_summary[columns]

df_qa_comment_summary.loc[df_qa_comment_summary['hasFavComment'].isnull(),'hasFavComment'] = 0
df_qa_comment_summary.loc[df_qa_comment_summary['number_of_comments'].isnull(),'number_of_comments'] = 0
df_qa_comment_summary.loc[df_qa_comment_summary['answeredInDays'].isnull(),'answeredInDays'] = -1
df_qa_comment_summary.drop_duplicates(inplace=True)

print(df_qa_comment_summary.info())
print()
df_qa_comment_summary.head()


# **Data exploration**

# In[ ]:


sns.countplot(x='hasFavComment',data=df_qa_comment_summary)


# **Implementing Logistic Regression that is used to predict the probability of a categorical dependent variable (hasFavComment). 1 (yes) or 0 (no).**

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

X = df_qa_comment_summary[['hasComment', 'number_of_comments','answeredInDays']]
y = df_qa_comment_summary['hasFavComment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[ ]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[ ]:


predictions = logmodel.predict(X_test)


# **Now lets evaluate the performance of a classification model**

# In[ ]:


cnf_matrix = metrics.confusion_matrix(y_test, predictions)
cnf_matrix


# **The result is telling us that we have 12897+1674 correct predictions 
# **
# 
# **and 732+35 incorrect predictions.
# **
# 
# **Here, you will visualize the confusion matrix using Heatmap.**

# In[ ]:


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# **Confusion Matrix Evaluation Metrics Let's evaluate the model using model evaluation metrics such as accuracy, precision, and recall.**

# In[ ]:


print("Accuracy:",metrics.accuracy_score(y_test, predictions))
print("Precision:",metrics.precision_score(y_test, predictions))
print("Recall:",metrics.recall_score(y_test, predictions))


# **Well, we got a classification rate of 94%, considered as good accuracy.
# **
# 
# **Precision: Precision is about being precise, In our prediction case, the Precision rate is about 69% when question ends with our favourite comment.**
