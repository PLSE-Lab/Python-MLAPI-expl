#!/usr/bin/env python
# coding: utf-8

# # Introduction
# **The given kernel explains the basic hands on with nlp. Those who are starting with nlp must first try  to pour in their knowledge in Sentiment Analysis. Those who are  familiar with sentiment analysis, you are good to try your learning on this dataset and for those who are naive,this article will give you a clear bird's eye view 
# ([https://towardsdatascience.com/sentiment-analysis-concept-analysis-and-applications-6c94d6f58c17](http://towardsdatascience.com/sentiment-analysis-concept-analysis-and-applications-6c94d6f58c17)).**
# 
# **For the given dataset (will explain later what's it is about) spaCy's training model with ner(entity recognizer) is used. Further other nlp models such as BERT as a text classification can also be used. You can find the explanation of the above model in the given link ([https://spacy.io/usage/training](http://spacy.io/usage/training)) and for better understanding about entity recognizer ([https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da](http://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da)).**
# 

# # Articles worth to read before starting
# **Kaggle is a great platform for ML and data science enthusiasts that not only provide to try out your skills in different feature competitions but also a great community that lead you to the solution to your problem and enhance more of your skills.**
# **Such great articles are:**
# * **For understanding the EDA of the dataset:[https://www.kaggle.com/tanulsingh077/twitter-sentiment-extaction-analysis-eda-and-model](http://www.kaggle.com/tanulsingh077/twitter-sentiment-extaction-analysis-eda-and-model) **
# * **WORDSCLOUD FUNCTION -[https://www.kaggle.com/aashita/word-clouds-of-various-shapes](http://https://www.kaggle.com/aashita/word-clouds-of-various-shapes)**
# * **Training the model from spaCy NER on inputs [https://www.kaggle.com/rohitsingh9990/ner-training-using-spacy-0-628-lb](http://www.kaggle.com/rohitsingh9990/ner-training-using-spacy-0-628-lb)**
# 

# # About the Dataset 
# **The dataset contains the training set , test set and sample set(on which submission should be made). For training set four columns are given**
# 1. **text id**
# 2. **text**
# 3. **selected_text**
# 4. **sentiment**
# 
# **For a given text a selected text is given and the sentiment it describes. So training of data should be done in such a way that we model should return the selected text for a given sentiment and a text.**
# **The simple and naive way to approach this is to find the similarities between the text and selected text and learn the model using this and then test on the given test set.**
# 
# **[https://medium.com/@adriensieg/text-similarities-da019229c894](http://medium.com/@adriensieg/text-similarities-da019229c894)**
# 
# **The above mentioned link provide the basic understanding for such scenarios.**

# **Helper functions**

# In[ ]:


import numpy as np 
import pandas as pd 
import string

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import nltk
from nltk.corpus import stopwords
import re
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import random
import spacy
from spacy.util import compounding,minibatch
from tqdm import trange

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename)) 


# **Loading the data**

# In[ ]:


# Data loaded to the kernel
train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
ss = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')


# **Reading the data**

# In[ ]:


#Training data
train.tail()


# In[ ]:


train.info()


# **Dropping one null value in the training data**

# In[ ]:


#Removing the null data
train.dropna(inplace=True)


# In[ ]:


#Test data
test.head()


# In[ ]:


test.info()


# **Number of tweets for each sentiment**

# In[ ]:


# distribution of tweets in the training set
temp=train.groupby('sentiment').count()['text'].reset_index().sort_values(by='text',ascending=False)
temp.style.background_gradient(cmap='Greens')


# **Generating the features**
# * The Jaccard index of the text and the selected text (it will depict the similarity between the text and the selected text). Follow up this discussion to know more.[https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/138520](http://www.kaggle.com/c/tweet-sentiment-extraction/discussion/138520)
# * Difference in number of words selected text and text

# In[ ]:


#Defining Jaccard Index
def jaccard(str1,str2):
    a=set(str1.lower().split())
    b=set(str2.lower().split())
    c=a.intersection(b)
    return float(len(c)) / (len(a)+len(b)-len(c))


# In[ ]:


#Applying the Jaccard index function 
result=[]

for ind,row in train.iterrows():
    sent1=row.text
    sent2=row.selected_text
    
    jaccard_score=jaccard(sent1,sent2)
    result.append([sent1,sent2,jaccard_score])


# In[ ]:


jaccard=pd.DataFrame(result,columns=["text","selected_text","jaccard_score"])
train=train.merge(jaccard,how='outer')


# In[ ]:


# Difference between the lengths of text and selected_text
train['Num_of_words_T']=train['text'].apply(lambda x :len(str(x).split()))
train['Num_of_words_ST']=train['selected_text'].apply(lambda x:len(str(x).split()))
train['Diff_Num_of_words']=train['Num_of_words_T']-train['Num_of_words_ST']


# In[ ]:


train.head()


# **Note : By calculatig the jaccard index we infer that for neutral sentiment the text and selecte text are the same.**

# **Cleaning of data**

# In[ ]:


def clean_data(text):
    text=str(text).lower()
    text=re.sub('\[.*?\]','',text)
    text=re.sub(r'\d+','',text)
    text=re.sub('https?://\S+|www\.\S+','',text)
    text=re.sub('<.*?>+','',text)
    text=re.sub('\n','',text)
    text=re.sub('\w*\d\w*','',text)
    text=re.sub('[%s]'% re.escape(string.punctuation),'',text)
    return text
    


# In[ ]:


train['text']=train['text'].apply(lambda x:clean_data(x))
train['selected_text']=train['selected_text'].apply(lambda x:clean_data(x))


# # Data Visualization(Implementing the word clouds)
# 

# In[ ]:


Positive_sent = train[train['sentiment']=='positive']
Negative_sent = train[train['sentiment']=='negative']
Neutral_sent =  train[train['sentiment']=='neutral']


# **Wordcloud function (masked on a twitter logo image)**

# In[ ]:


def plot_wordcloud(text,mask=None,max_words=400,max_font_size=100,figure_size=(24.0,16.0),title=None,title_size=40,image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords={'u',"im"}
    stopwords=stopwords.union(more_stopwords)
    
    wordcloud = WordCloud(background_color='white',
                         stopwords = stopwords,max_words=max_words,
                         max_font_size=max_font_size,random_state=42,mask=mask)
    
    wordcloud.generate(str(text))
    
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors),interpolation="bilinear");
        plt.title(title,fontdict={'size':title_size,
                                  'verticalalignment':'bottom'})
    else:
            plt.imshow(wordcloud);
            plt.title(title,fontdict={'size':title_size,'color':'red',
                                     'verticalalignment':'bottom'})
            plt.axis('off');
    plt.tight_layout()  
    
d = '../input/imagetc/'


# **Wordclouds for positive,neutral,negative tweets**

# In[ ]:


twitter_mask=np.array(Image.open(d+'twitter.png'))
plot_wordcloud(Neutral_sent.text,mask=twitter_mask,max_font_size=80,title_size=30,title="WordCloud for Neutral tweets")


# In[ ]:


plot_wordcloud(Positive_sent.text,mask=twitter_mask,max_font_size=80,title_size=30,title="WordCloud for Positive tweets")


# In[ ]:


plot_wordcloud(Negative_sent.text,mask=twitter_mask,max_font_size=80,title_size=30,title="WordCloud for Negative tweets")


# # Modelling of the data

# In[ ]:


data_train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
data_test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
submission = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')


# In[ ]:


data_train['Num_words_text'] = data_train['text'].apply(lambda x: len(str(x).split()))
data_train  = data_train[data_train['Num_words_text']>=3]
                                              


# * **Saving the model to output directory**

# In[ ]:


def save_model(output_dir,nlp,new_model_name):
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        nlp.meta["name"] = new_model_name
        nlp.to_disk(output_dir)
        print("Saved model to",output_dir)


# * **Training the model**

# In[ ]:


def training(train_data, output_dir, n_iter=20, model=None):
    """Load the model,set up the pipeline and train the entity recognizer"""
    if model is not None:
        nlp=sapcy.load(model) #load existing spaCy model
        print("Loaded model '%s'" %model)
    else:
        nlp = spacy.blank("en") #create blank Language class
        print("Created blank 'en' model ")
        
        # The pipeline execution
        # Create the built-in pipeline components and them to the pipeline
        # nlp.create_pipe works for built-ins that are registered in the spacy
        
        if "ner" not in nlp.pipe_names:
            ner = nlp.create_pipe("ner")
            nlp.add_pipe(ner,last=True)
            
        # otherwise, get it so we can add labels
        
        else:
            ner = nlp.get_pipe("ner")
            
        # add labels 
        for _, annotations in train_data:
                for ent in annotations.get("entities"):
                    ner.add_label(ent[2])
        
        # get names of other pipes to disable them during training
        
        pipe_exceptions = ["ner","trf_wordpiecer","trf_tok2vec"]
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
        
        with nlp.disable_pipes(*other_pipes): # training of only NER
            
            # reset and intialize the weights randoml - but only if we're
            # training a model
            
            if model is None:
                nlp.begin_training()
            else:
                nlp.resume_training()
            
            for itn in trange(n_iter):
                random.shuffle(train_data)
                losses={}
                
                # batch up the example using spaCy's mnibatch
                batches = minibatch(train_data,size=compounding(4.0,1000.0,1.001))
                
                for batch in batches:
                    texts , annotations = zip(*batch)
                    nlp.update(
                        texts, #batch of texts
                        annotations, # batch of annotations
                        drop = 0.5,  # dropout - make it harder to memorise data
                        losses = losses,
                )
            print("Losses", losses)
        save_model(output_dir, nlp, 'st_ner')
                    


# * **Return the model path**

# In[ ]:


def get_model_path(sentiment):
    model_out_path = None 
    if sentiment == 'positive':
        model_out_path = 'models/model_pos'
    elif sentiment == 'negative':
        model_out_path = 'models/model_neg'
    return model_out_path


# In[ ]:


def get_training_data(sentiment):
    train_data=[]
    
    '''
    Returns Training data in the format needed to train spacy NER
    '''
    for index,row in data_train.iterrows():
        if row.sentiment == sentiment:
            selected_text = row.selected_text
            text = row.text
            start = text.find(selected_text)
            end = start + len(selected_text)
            train_data.append((text, {"entities": [[start,end,'selected_text']]}))
    return train_data


# * **Training for Positive and Negative tweets**

# In[ ]:


sentiment ='positive'
train_data = get_training_data(sentiment)
model_path = get_model_path(sentiment)
training(train_data,model_path,n_iter=3,model=None)


# In[ ]:


sentiment ='negative'
train_data = get_training_data(sentiment)
model_path = get_model_path(sentiment)
training(train_data,model_path,n_iter=3,model=None)


# * **Predicting from the trained models**

# In[ ]:


def predict(text,model):
    doc=model(text)
    ent_array=[]
    for ent in doc.ents:
        start = text.find(ent.text)
        end = start + len(ent.text)
        new_int = [start,end,ent.label_]
        if new_int not in ent_array:
            ent_array.append([start,end,ent.label_])
    selected_text = text[ent_array[0][0]:ent_array[0][1]] if len(ent_array)>0 else text
    return selected_text


# In[ ]:


selected_texts = []
MODELS_BASE_PATH = '../input/tse-spacy-model/models/'
if MODELS_BASE_PATH is not None:
    print("Model is loading from",MODELS_BASE_PATH)
    models_pos = spacy.load(MODELS_BASE_PATH + 'model_pos')
    models_neg = spacy.load(MODELS_BASE_PATH + 'model_neg')
    for index,row in data_test.iterrows():
        text = row.text
        output_str = ""
        if row.sentiment == 'neutral' or len(text.split())<=2:
            selected_texts.append(text)
        elif row.sentiment == 'positive':
            selected_texts.append(predict(text,models_pos))
        else:
             selected_texts.append(predict(text,models_neg))
    
data_test['selected_text']=selected_texts        


# **Final result on sample data**

# In[ ]:


submission['selected_text'] = data_test['selected_text']
submission.to_csv("submission.csv",index=False)
print("Hola Done")
display(submission.head())

