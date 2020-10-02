#!/usr/bin/env python
# coding: utf-8

# ### This notebook is to do a data processing and consolidation of all the requirement text file and corresponding estimate . I have tried to do basic EDA on the texts and the corresponding estimate to find out if there is any relationship between the requirement texts and estimate .

# In[ ]:


# Importing necessary libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
color = sns.color_palette()
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
import plotly.tools as tls
import warnings
warnings.filterwarnings('ignore')
import os
import glob
import re
import string


# In[ ]:


os.listdir(r'../input/nlp-estimate/nlp estimate/NLP Estimate/')


# #### Loop through the filenames and the load the text content into dataframe . Concatenate all the dataframe at the end to get a single dataframe with all the requirement texts

# In[ ]:


filenames =glob.glob("../input/nlp-estimate/nlp estimate/NLP Estimate/arq*.txt")

data = []
for filename in filenames:
    df = pd.read_csv(filename,delimiter='\n',names=['Text'])
    df['Filename'] = os.path.basename(filename)
    data.append(df)

df_Req = pd.concat(data, ignore_index=True)


# #### The file names are  a string , therefore the sorting order gets changed and arq1 , arq10 , arq100 etc comes before arq2 . This will become a problem while matching the requirmenent file with the estimate . Thefore I have extracted the file id from filename using regex and lamda function. Then added the id as a column in dataframe

# In[ ]:


df_Req['File_ID']=df_Req['Filename'].map(lambda x : re.findall('\d+',x))
df_Req['File_ID']=df_Req['File_ID'].map(lambda x :x[0])
df_Req['File_ID']=pd.to_numeric(df_Req['File_ID'])


# In[ ]:


df_Req.info() # Lets check information about our current dataframe 


# #### Here we are sorting the values based on the derived file ids , to match with the estimation text . 

# In[ ]:


df_Req.sort_values(by='File_ID',inplace=True)
df_Req.set_index('File_ID')


# In[ ]:


df_Req.shape


# #### We are now going to load the estimate data and match with Requirement data to combine in a single dataframe . 

# In[ ]:


df_Est=pd.read_csv("../input/nlp-estimate/nlp estimate/NLP Estimate/estimate.csv",names=['Estimate']) 


# In[ ]:


df_Est.insert(0, 'Est_ID', range(0, len(df_Est)))


# In[ ]:


df_Est.head(10)#Lets verify the data


# #### Based on File_ID and Est_ID both dataframes were joined to have a combined dataframe for further analysis .

# In[ ]:


df_Merge=pd.merge(df_Req, df_Est, how='inner', on=None, left_on='File_ID', right_on='Est_ID',
         left_index=False, right_index=False, sort=False,
         suffixes=('_x', '_y'), copy=True, indicator=False,
         validate=None)


# #### Dropping the columns which does not give any insight towards NLP or estimation

# In[ ]:


df_Merge.drop(columns =['Filename','File_ID','Est_ID'],inplace=True)


# #### Lets have a look at our final data before text analysis

# In[ ]:


df_Merge.head(38)


# ## Data Pre-Processing

# In[ ]:


df=df_Merge


# In[ ]:


df.shape


# In[ ]:


df.dtypes


# In[ ]:


df.isnull().sum()


# #### There are no null values (There were not supposed to be as the data processing was to take the content from each file , and there were no 0 byte file ) so no null handling required 

# #### After analyzing the data we can see that there are special tag as  < div> ,< p> , < code>, < pre> etc present in the requirement text .We dont want those tags to appear in list of words and throw model off track 

# In[ ]:


def CleanText(Text):
    Text = re.sub(r'html',' ',Text)
    Text = re.sub(r'<div>',' ',Text)
    Text = re.sub(r'<p>',' ',Text)
    Text = re.sub(r'<pre>',' ',Text)
    Text = re.sub(r'<code>',' ',Text)
    Text = re.sub(r'html',' ',Text)
    Text = re.sub(r'< div>',' ',Text)
    Text = re.sub(r'< p>',' ',Text)
    Text = re.sub(r'< pre>',' ',Text)
    Text = re.sub(r'< code>',' ',Text)
    Text = re.sub(r'< code>',' ',Text)
    ## Use string method to do further cleanup from punctuation and digits which will may not give any additional insight
    trans_punct = str.maketrans('', '', string.punctuation)
    trans_digit = str.maketrans('', '', string.digits)
    Text = Text.translate(trans_punct)
    Text = Text.translate(trans_digit)
    Text = Text.lower()
    return Text
    


# #### Applying this function to all rows in dataframe and storing the cleaned data in another column

# In[ ]:


df['Cleaned']= df['Text'].apply(CleanText)


# In[ ]:


df.head(10) ### Verifiying the data 


# ### Lets now see what kind of words appear most in writing requirements for software application development 

# In[ ]:


#!pip install wordcloud (This is not required if wordcloud was installed)


# In[ ]:


from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=100,
        max_font_size=50, 
        scale=3,
        random_state=1 
).generate(str(data))

    fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()

show_wordcloud(df['Cleaned'])


# #### Lets see how the estimates are distributed in the dataset 

# In[ ]:


df['Estimate'].value_counts()


# #### We can see 82% of total records fall within estimate 1 to 8. Therefore we are filtering out other classes to avoid class imbalance problem 
# 
# #### "The class imbalance problem typically occurs when, in a classification problem, there are many more instances of some classes than others. In such cases, standard classifiers tend to be overwhelmed by the large classes and ignore the small ones." 
# 

# In[ ]:


labellist=[]
cnt=df['Estimate'].value_counts().head(10)
labellist=cnt.index[::-1]


# In[ ]:


labellist


# #### Let us see the distribution of requirements  for each type of estimates 

# In[ ]:


cnt = df['Estimate'].value_counts().head(5)
print(cnt)

trace = go.Bar(
    y=cnt.index[::-1],
    x=cnt.values[::-1],
    orientation = 'h',
    marker=dict(
        color=cnt.values[::-1],
        colorscale = 'Blues',
        reversescale = False
    ),
)

layout = dict(
    title='Estimate Distribution',
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Estimates")


# #### Lets see if there is a relationship with length of requirement text and the estimate ? (Does longer requirement text means more elaboration and lesser story point ? )

# In[ ]:


df = df.loc[df['Estimate'].isin([1,2,3,5,8])]


# In[ ]:


df['Req_Length']=df['Text'].apply(len)


# In[ ]:


sns.set(font_scale=2.0)

g = sns.FacetGrid(df,col='Estimate',size=5)
g.map(plt.hist,'Req_Length')


# #### **By seeing the plots , we can conclude that there is no significant relationship between requirement text length and the corresponding estimate**

# #### Lets now analyze and see if there is any difference in commonly used words for different estimations. Slicing dataframes by estimations and checking the worldcloud

# In[ ]:


df1 = df.loc[df['Estimate'].isin([1])]
df8 = df.loc[df['Estimate'].isin([8])]
df2 = df.loc[df['Estimate'].isin([2])]
df5 = df.loc[df['Estimate'].isin([5])]
df3 = df.loc[df['Estimate'].isin([3])]


# In[ ]:


show_wordcloud(df1['Cleaned'])


# In[ ]:


show_wordcloud(df2['Cleaned'])


# In[ ]:


show_wordcloud(df8['Cleaned'])


# In[ ]:


show_wordcloud(df5['Cleaned'])


# ### By reviewing the worldcloud we can see mainly 
# ##### Estimate 1 - Talks about desktop , android, allow , titanium , new , mobile , sample , test, confirm
# ##### Estimate 2 - Talks about files ,  Installer ,need , test , editor ,create, enable , studio and titanium 
# ##### Estimate 5- Talks about  Usergrid fix , tool , deploy, project , shard , test , android ,run , create
# ##### Estimate 8- Talks about  Window , Project , Studio and Titanium 
# 
# ## Need to investigate what is Titanium ? Probably an application which occurs in the text for most classes ? Do we need this ? 

# ## We have seen that there is no significant relation between requirement length and the estimate , lets confirm this assumption using correlation plot

# In[ ]:


sns.set(font_scale=1.2)
plt.figure(figsize = (8,4))
sns.heatmap(df.corr(),cmap='coolwarm',annot=True,linewidths=.5)


# ## Check for more insights 

# ## Try deep learning to see whether we can predict a large story from a small one . Here I have considered 1 to 5 story points are small story and 8 as large story . 

# In[ ]:


df['StoryType'] = df['Estimate']<6


# In[ ]:


from sklearn.model_selection import train_test_split
train_text, test_text, train_y, test_y = train_test_split(df['Cleaned'],df['StoryType'],test_size = 0.2,shuffle =True)


# In[ ]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM,Bidirectional,Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


# #### Here we are using  Keras tokenizer to tokenize requirement texts and create word embeddings . 

# In[ ]:


MAX_NB_WORDS = 20000

# get the raw text data
texts_train = train_text.astype(str)
texts_test = test_text.astype(str)

#  vectorize the text samples 
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS, char_level=False)
tokenizer.fit_on_texts(texts_train)
sequences = tokenizer.texts_to_sequences(texts_train)
sequences_test = tokenizer.texts_to_sequences(texts_test)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


# In[ ]:


MAX_SEQUENCE_LENGTH = 200
#pad sequences are used to bring all sentences to same size.
# pad sequences with 0s
x_train = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
x_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', x_train.shape)
print('Shape of data test tensor:', x_test.shape)


# ### Create sequential model . Here I have used Bidirectional LSTM 
# #### Dropout layer as 20% neuron to be dropped 
# #### sigmoid activation function at the last dense layer 
# #### Since there are two output class (large and small) , used the binary crossentropy as the loss function

# In[ ]:


model = Sequential()
model.add(Embedding(MAX_NB_WORDS, 128))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2,input_shape=(1,))))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))


# In[ ]:


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[ ]:


model.fit(x_train, train_y,
          batch_size=128,
          epochs=3,
          validation_data=(x_test, test_y))


# ### We can see that validation accuracy is decreasing layer by layer . Need to check if other architecture and data preprocessing can help in increasing the test accuracy . 
# 
# ### I have already tried bagging and boosting method and the accuracy was only 35/36 % . Will update  in next version. 
