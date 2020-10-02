#!/usr/bin/env python
# coding: utf-8

# # **Tweets Sentiment Analysis - EDA, Probability determination**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # **Importing Necesseties**

# In[ ]:


import re 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from plotly import graph_objs as go
import plotly.figure_factory as ff
import string
from collections import Counter
from wordcloud import WordCloud,STOPWORDS
import plotly.express as px
import nltk
from nltk.corpus import stopwords


# #  Read train and test data

# In[ ]:


train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
submission = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")


# In[ ]:


train.head()


# Lets understand the dataset, Textid is alpha numereic number generated for each row, Text is raw data and selected text is subset of the text data and sentiment is about postive, negative or neutral.

# Identify the shape and strucure of the train and test dataset 

# In[ ]:


print(train.shape)
print(test.shape)


# we have 27481 obseravation and 4 varaiables in the Train dataset and 3534 observations and 3 variables in Test data, This is because one variable sentiment which is need to be predicted is missing in the test data
# 
# Lets find the structure of the data 

# In[ ]:


train.info()


# We have one null Value in the train , as the test field for value is NAN we will just remove it

# In[ ]:


test.info()


# There is no null value in the test data 

# Lets remove the null observation from train set

# In[ ]:


test.head()


# In[ ]:


train.dropna(inplace=True)


# Exploratary Data Analysis(EDA)

# In[ ]:


train.describe()


# To find the distribution of sentiment from train data 

# In[ ]:


sentiment = train.groupby('sentiment').count()['text'].reset_index().sort_values(by='text',ascending= False)
sentiment.style.background_gradient(cmap='viridis')


# Lets see the disrtubution in the graphical view

# In[ ]:


plt.figure(figsize=(12,6))
sns.countplot(x='sentiment',data=train)


# Funnel chart

# In[ ]:


fig = go.Figure(go.Funnelarea(
    text =sentiment.sentiment,
    values = sentiment.text,
    title = {"position": "top center", "text": "Funnel-Chart of Sentiment Distribution"}
    ))
fig.show()


# let us find the frequency of words in selected text and text 

# In[ ]:


# we are going to consider only text column and selected text column 
def gen_freq(text):
    #Will store the list of words
    word_list = []

    #Loop over all the tweets and extract words into word_list
    for tw_words in text.split():
        word_list.extend(tw_words)

    #Create word frequencies using word_list
    word_freq = pd.Series(word_list).value_counts()

    #Print top 20 words
    word_freq[:20]
    
    return word_freq

gen_freq(train.text.str)


# In[ ]:


gen_freq(train.selected_text.str)


# Lets check the word cloud for the frequency of the words 

# In[ ]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS

#Generate word frequencies
word_freq = gen_freq(train.text.str)

#Generate word cloud
wc = WordCloud(width=400, height=330, max_words=100, background_color='white').generate_from_frequencies(word_freq)

plt.figure(figsize=(12, 8))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()


# To, the, and, a are stop words which have more frequency which is not meaningful, unable to identify the content of the text, let us also look for selected text 

# In[ ]:


#Generate word frequencies
word_freq = gen_freq(train.selected_text.str)

#Generate word cloud
wc = WordCloud(width=400, height=330, max_words=100, background_color='white').generate_from_frequencies(word_freq)

plt.figure(figsize=(12, 8))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()


# Here also content is missing stop word are more frequent. before cleaning the stop words lets generate the meta feature
# 
# * Difference In Number Of words of Selected_text and Text
# * Jaccard Similarity Scores between text and Selected_text

# In[ ]:


def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


# In[ ]:


results_jaccard=[]

for ind,row in train.iterrows():
    sentence1 = row.text
    sentence2 = row.selected_text

    jaccard_score = jaccard(sentence1,sentence2)
    results_jaccard.append([sentence1,sentence2,jaccard_score])


# In[ ]:


jaccard = pd.DataFrame(results_jaccard,columns=["text","selected_text","jaccard_score"])
train = train.merge(jaccard,how='outer')


# In[ ]:


train['Num_words_ST'] = train['selected_text'].apply(lambda x:len(str(x).split())) #Number Of words in Selected Text
train['Num_word_text'] = train['text'].apply(lambda x:len(str(x).split())) #Number Of words in main text
train['difference_in_words'] = train['Num_word_text'] - train['Num_words_ST'] #Difference in Number of words text and Selected Text


# In[ ]:


train.head()


# Lets look the distribuion of meta - features

# In[ ]:


hist_data = [train['Num_words_ST'],train['Num_word_text']]

group_labels = ['Selected_Text','Text']

# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels,show_curve=False)
fig.update_layout(title_text='Distribution of Number Of words')
fig.update_layout(
    autosize=False,
    width=900,
    height=700,
    paper_bgcolor="LightSteelBlue",
)
fig.show()


# The number of words plot is really interesting ,the tweets having number of words greater than 25 are very less and thus the number of words distribution plot is right skewed

# In[ ]:


#Kernal Distribution
plt.figure(figsize=(12,6))
p1=sns.kdeplot(train['Num_words_ST'], shade=True, color="g").set_title('Kernel Distribution of Number Of words')
p1=sns.kdeplot(train['Num_word_text'], shade=True, color="r")


# Now It will be more interesting to see the differnce in number of words and jaccard_scores across different Sentiments

# In[ ]:


plt.figure(figsize=(12,6))
p1=sns.kdeplot(train[train['sentiment']=='positive']['difference_in_words'], shade=True, color="b").set_title('Kernel Distribution of Difference in Number Of words')
p2=sns.kdeplot(train[train['sentiment']=='negative']['difference_in_words'], shade=True, color="r")


# In[ ]:


plt.figure(figsize=(12,6))
sns.distplot(train[train['sentiment']=='neutral']['difference_in_words'],kde=False)


# Unable to plot kde plot for neutral sentiment because most of the values for difference in number of words were zero. We can see it clearly now ,if we had used the feature in the starting we would have known that text and selected text are mostly the same for neutral sentiment,thus its always important to keep the end goal in mind while performing EDA

# In[ ]:


plt.figure(figsize=(12,6))
p1=sns.kdeplot(train[train['sentiment']=='positive']['jaccard_score'], shade=True, color="b").set_title('KDE of Jaccard Scores across different Sentiments')
p2=sns.kdeplot(train[train['sentiment']=='negative']['jaccard_score'], shade=True, color="r")
plt.legend(labels=['positive','negative'])


# In[ ]:


plt.figure(figsize=(12,6))
sns.distplot(train[train['sentiment']=='neutral']['jaccard_score'],kde=False)


# **Conclusion Of EDA**
# 
# We can see from the jaccard score plot that there is peak for negative and positive plot around score of 1 .That means there is a cluster of tweets where there is a high similarity between text and selected texts ,if we can find those clusters then we can predict text for selected texts for those tweets irrespective of segment
# 
# 
# Let's see if we can find those clusters,one interesting idea would be to check tweets which have number of words lesss than 3 in text, because there the text might be completely used as text

# In[ ]:


k = train[train['Num_word_text']<=2]


# In[ ]:


k.groupby('sentiment').mean()['jaccard_score']


# In[ ]:


k[k['sentiment']=='positive']


# In[ ]:


# Text cleaning 
import re 

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


# In[ ]:


train['text'] = train['text'].apply(lambda x:clean_text(x))
train['selected_text'] = train['selected_text'].apply(lambda x:clean_text(x))


# In[ ]:


train.head()


# In[ ]:


# Remove Stop Words 

print(STOPWORDS)


# Lets Remove the stop words 

# In[ ]:


text = train.text.apply(lambda x: clean_text(x))
word_freq = gen_freq(text.str)*100
word_freq_txt = word_freq.drop(labels=STOPWORDS,errors = 'ignore')

#generate word cloud 
wc = WordCloud(width=450, height=330, max_words=200, background_color='white').generate_from_frequencies(word_freq_txt)

plt.figure(figsize=(12, 14))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[ ]:


ST = train.selected_text.apply(lambda x: clean_text(x))
word_freq = gen_freq(ST.str)*100
word_freq_st = word_freq.drop(labels=STOPWORDS,errors = 'ignore')

#generate word cloud 
wc = WordCloud(width=450, height=330, max_words=200, background_color='white').generate_from_frequencies(word_freq_st)

plt.figure(figsize=(12, 14))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()


# Most Common Words in our Target-Selected text

# In[ ]:


train['temp_list'] = train['selected_text'].apply(lambda x:str(x).split())
top = Counter([item for sublist in train['temp_list'] for item in sublist])
temp = pd.DataFrame(top.most_common(20))
temp.columns = ['Common_words','count']
temp.style.background_gradient(cmap='Reds')


# Lets view in Visulaization Form

# In[ ]:


fig = px.bar(temp, x="count", y="Common_words", title='Commmon Words in Selected Text', orientation='h', 
             width=700, height=700,color='Common_words')
fig.show()


# In[ ]:


def remove_stopword(x):
    return [y for y in x if y not in stopwords.words('english')]
train['temp_list'] = train['temp_list'].apply(lambda x:remove_stopword(x))


# In[ ]:


top = Counter([item for sublist in train['temp_list'] for item in sublist])
temp = pd.DataFrame(top.most_common(20))
temp = temp.iloc[1:,:]
temp.columns = ['Common_words','count']
temp.style.background_gradient(cmap='Purples')


# Tree for most common selected text

# In[ ]:


fig = px.treemap(temp, path=['Common_words'], values='count',title='Tree of Most Common Words')
fig.show()


# In[ ]:


train['temp_list1'] = train['text'].apply(lambda x:str(x).split()) #List of words in every row for text
train['temp_list1'] = train['temp_list1'].apply(lambda x:remove_stopword(x)) #Removing Stopwords


# In[ ]:


top = Counter([item for sublist in train['temp_list1'] for item in sublist])
temp = pd.DataFrame(top.most_common(25))
temp = temp.iloc[1:,:]
temp.columns = ['Common_words','count']
temp.style.background_gradient(cmap='Blues')


# In[ ]:


fig = px.bar(temp, x="count", y="Common_words", title='Commmon Words in Text', orientation='h', 
             width=700, height=700,color='Common_words')
fig.show()


# In[ ]:


fig = px.treemap(temp, path=['Common_words'], values='count',title='Tree of Most Common Words')
fig.show()


# Most common words Sentiments Wise

# In[ ]:


Positive_sent = train[train['sentiment']=='positive']
Negative_sent = train[train['sentiment']=='negative']
Neutral_sent = train[train['sentiment']=='neutral']


# In[ ]:


#MosT common positive words
top = Counter([item for sublist in Positive_sent['temp_list'] for item in sublist])
temp_positive = pd.DataFrame(top.most_common(20))
temp_positive.columns = ['Common_words','count']
temp_positive.style.background_gradient(cmap='Greens')


# In[ ]:


fig = px.bar(temp_positive, x="count", y="Common_words", title='Most Commmon Positive Words', orientation='h', 
             width=700, height=700,color='Common_words')
fig.show()


# In[ ]:


#MosT common negative words
top = Counter([item for sublist in Negative_sent['temp_list'] for item in sublist])
temp_negative = pd.DataFrame(top.most_common(20))
temp_negative = temp_negative.iloc[1:,:]
temp_negative.columns = ['Common_words','count']
temp_negative.style.background_gradient(cmap='Reds')


# In[ ]:


fig = px.bar(temp_negative, x="count", y="Common_words", title='Most Commmon Negative Words', orientation='h', 
             width=700, height=700,color='Common_words')
fig.show()


# In[ ]:


fig = px.treemap(temp_negative, path=['Common_words'], values='count',title='Tree Of Most Common Negative Words')
fig.show()


# In[ ]:


#MosT common Neutral words
top = Counter([item for sublist in Neutral_sent['temp_list'] for item in sublist])
temp_neutral = pd.DataFrame(top.most_common(20))
temp_neutral = temp_neutral.loc[1:,:]
temp_neutral.columns = ['Common_words','count']
temp_neutral.style.background_gradient(cmap='Reds')


# In[ ]:


fig = px.bar(temp_neutral, x="count", y="Common_words", title='Most Commmon Neutral Words', orientation='h', 
             width=700, height=700,color='Common_words')
fig.show()


# In[ ]:


fig = px.treemap(temp_neutral, path=['Common_words'], values='count',title='Tree Of Most Common Neutral Words')
fig.show()


# # Let's Look at Unique Words in each Segment
# # We will look at unique words in each segment in the Following Order:

# In[ ]:


raw_text = [word for word_list in train['temp_list1'] for word in word_list]


# In[ ]:


def words_unique(sentiment,numwords,raw_words):
    '''
    Input:
        segment - Segment category (ex. 'Neutral');
        numwords - how many specific words do you want to see in the final result; 
        raw_words - list  for item in train_data[train_data.segments == segments]['temp_list1']:
    Output: 
        dataframe giving information about the name of the specific ingredient and how many times it occurs in the chosen cuisine (in descending order based on their counts)..

    '''
    allother = []
    for item in train[train.sentiment != sentiment]['temp_list1']:
        for word in item:
            allother .append(word)
    allother  = list(set(allother ))
    
    specificnonly = [x for x in raw_text if x not in allother]
    
    mycounter = Counter()
    
    for item in train[train.sentiment == sentiment]['temp_list1']:
        for word in item:
            mycounter[word] += 1
    keep = list(specificnonly)
    
    for word in list(mycounter):
        if word not in keep:
            del mycounter[word]
    
    Unique_words = pd.DataFrame(mycounter.most_common(numwords), columns = ['words','count'])
    
    return Unique_words


# In[ ]:


Unique_Positive= words_unique('positive', 10, raw_text)
print("The top 10 unique words in Positive Tweets are:")
Unique_Positive.style.background_gradient(cmap='Greens')


# In[ ]:


fig = px.treemap(Unique_Positive, path=['words'], values='count',title='Tree Of Unique Positive Words')
fig.show()


# In[ ]:


from palettable.colorbrewer.qualitative import Pastel1_7
plt.figure(figsize=(16,10))
my_circle=plt.Circle((0,0), 0.7, color='white')
plt.pie(Unique_Positive['count'], labels=Unique_Positive.words, colors=Pastel1_7.hex_colors)
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.title('DoNut Plot Of Unique Positive Words')
plt.show()


# In[ ]:


Unique_Negative= words_unique('negative', 10, raw_text)
print("The top 10 unique words in Negative Tweets are:")
Unique_Negative.style.background_gradient(cmap='Blues')


# In[ ]:


fig = px.treemap(Unique_Negative, path=['words'], values='count',title='Tree Of Unique Negative Words')
fig.show()


# In[ ]:


from palettable.colorbrewer.qualitative import Pastel1_7
plt.figure(figsize=(16,10))
my_circle=plt.Circle((0,0), 0.7, color='white')
plt.pie(Unique_Negative['count'], labels=Unique_Negative.words, colors=Pastel1_7.hex_colors)
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.title('DoNut Plot Of Unique Negative Words')
plt.show()


# In[ ]:


Unique_Neutral= words_unique('neutral', 10, raw_text)
print("The top 10 unique words in Neutral Tweets are:")
Unique_Neutral.style.background_gradient(cmap='Reds')


# In[ ]:


fig = px.treemap(Unique_Neutral, path=['words'], values='count',title='Tree Of Unique Neutral Words')
fig.show()


# In[ ]:


from palettable.colorbrewer.qualitative import Pastel1_7
plt.figure(figsize=(16,10))
my_circle=plt.Circle((0,0), 0.7, color='white')
plt.pie(Unique_Neutral['count'], labels=Unique_Neutral.words, colors=Pastel1_7.hex_colors)
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.title('DoNut Plot Of Unique Neutral Words')
plt.show()


# Model Building 
# 
# Lets consider only Selected Text and Sentiment for Model Building, Later we consider text and Sentiment for Model Building and Seleced Text and Sentiment for Model Buiding 

# In[ ]:


import re
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

all_tokens = []

for idx, row in train.iterrows():
    for word in word_tokenize(row.selected_text):
        all_tokens.append(word)
    
print(len(all_tokens), all_tokens)


# In[ ]:


all_tokens_unique = set(all_tokens)
print(len(all_tokens_unique), all_tokens_unique)


# In[ ]:


number_of_class_labels=len(train['sentiment'].unique())
number_of_class_labels


# In[ ]:


class_prob_df = pd.DataFrame(columns=['sentiment', 'probability'], index=range(number_of_class_labels))
class_prob_df


# In[ ]:


Count_Row=train.shape[0] 
Count_Col=train.shape[1] 
print(Count_Col)
print(Count_Row)
print(train.shape)


# In[ ]:


i=0
for val, cnt in train['sentiment'].value_counts().iteritems():
    print ('value', val, 'was found', cnt, 'times')
    class_prob_df.loc[i].sentiment = val
    class_prob_df.loc[i].probability = cnt/Count_Row
    i = i +1
    
class_prob_df


# In[ ]:


stop_words = set(stopwords.words('english'))

tokens = [w for w in all_tokens_unique if not w in stop_words]
print(len(tokens), tokens)

tokens1=[]
tokens = [word for word in tokens if word.isalpha()]
print(len(tokens), tokens)


# In[ ]:


word = ['@', 'rr', '!', '$', '@', 'jfjf', '&','(', ')', ',']
for word in word:
    if word.isalpha():
        print("yes it is alpha: ", word)


# In[ ]:


train.values


# In[ ]:


group_train = train.groupby('sentiment')['selected_text'].apply(' '.join).reset_index()

group_train


# In[ ]:


for idx, row in group_train.iterrows():
    
    temp1_tokens = []
    for word in word_tokenize(row.selected_text):
        temp1_tokens.append(word)
    
    temp1_tokens = set(temp1_tokens)
         
    temp2_tokens = []
    for word in temp1_tokens:
        if not word in stop_words:
            temp2_tokens.append(word)           
    
    temp3_tokens = []
    for word in temp2_tokens:
        if word.isalpha():
            temp3_tokens.append(word)
            
    print(temp3_tokens)
    temp4_tokens = " ".join(temp3_tokens)
    print(temp4_tokens)
    
    group_train.at[idx, 'selected_text'] = temp4_tokens
    group_train.at[idx, 'no_of_words_in_category'] = len(temp3_tokens)


# In[ ]:


group_train.head()


# In[ ]:


group_train = pd.merge(group_train, class_prob_df[['sentiment', 'probability']], on='sentiment')
group_train


# In[ ]:


final_df = pd.DataFrame()

row_counter = 0

for idx, row in group_train.iterrows():
    for token in tokens:
        # find the number of occurances of the token in the current category of documents
        no_of_occurances = row.selected_text.count(token)
        no_of_words_in_category = row.no_of_words_in_category
        no_unique_words_all = len(tokens)
        
        prob_of_token = (no_of_occurances+ 1)/ (no_of_words_in_category+ no_unique_words_all)
        #print(row.class_label, token, no_of_occurances, prob_of_token)
        final_df.at[row_counter, 'Result'] = row.sentiment
        final_df.at[row_counter, 'token'] = token
        final_df.at[row_counter, 'no_of_occurances'] = no_of_occurances
        final_df.at[row_counter, 'no_of_words_in_category'] = no_of_words_in_category
        final_df.at[row_counter, 'no_unique_words_all'] = no_unique_words_all
        final_df.at[row_counter, 'prob_of_token_category'] = prob_of_token
        
        row_counter = row_counter + 1


# In[ ]:


final_df


# In[ ]:


for idx, row in test.iterrows():
    
    # tokenize & unique words
    temp1_tokens = []
    for word in word_tokenize(row.sentiment):
        temp1_tokens.append(word)
        #temp1_tokens = set(temp1_tokens)
        
    # remove stop words
    temp2_tokens = []
    for word in temp1_tokens:
        if not word in stop_words:
            temp2_tokens.append(word)
          
    # remove punctuations
    temp3_tokens = []
    for word in temp2_tokens:
        if word.isalpha():
            temp3_tokens.append(word)
            
    #temp4_tokens = " ".join(temp3_tokens)
    #print(temp4_tokens)
    
    prob = 1 
    
    # process for each class_label
    for idx1, row1 in group_train.iterrows():
        print("class: "+ row1.sentiment)
        for token in temp3_tokens:
            # find the token in final_df for the given category, get the probability
            # row1.class_label & token
        
            print("      : "+ token)  
        
            temp_df = final_df[(final_df['Result'] == row1.sentiment) & (final_df['token'] == token)]

            # process for exception
            if (temp_df.shape[0] == 0):
                token_prob = 1/(row1.no_of_words_in_category+ no_unique_words_all)
                print("       no token found prob :", token_prob)
                prob = prob * token_prob
            else:
                token_prob = temp_df.get_value(temp_df.index[0],'prob_of_token_category')
                print("       token prob          :", token_prob)
                prob = prob * token_prob

            prob = prob * row1.probability

        col_at = 'prob_'+row1.sentiment

        test.at[idx, col_at] = prob


test


# # Prepare Submission File

# In[ ]:


test.to_csv('submission.csv', index=False)


# Based on the probability we can determine the sentiment.
# 
# Please upvote if you find usefull. 
