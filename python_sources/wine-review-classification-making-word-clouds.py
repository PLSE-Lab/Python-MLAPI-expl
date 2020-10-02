#!/usr/bin/env python
# coding: utf-8

# ## Text Classification using NLP for Various types of Wines -- Part2

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


# <p> This is wine reviews classification. To see the data analysis of the dataset and some basic information related to the dataset the link is further given below.  <a href = 'https://www.kaggle.com/bhaargavi/wine-classification-analysis-of-data'><h3>Link To Text Classification using NLP for Various types of Wines -- Part 1</h3></a> </p> 

# In[ ]:


wines =  pd.read_csv("../input/wine130k/winemag-data-130k-v2.csv",index_col = 0)
df_wines = wines[['country', 'description', 'points', 'price', 'variety']]
df_wines = df_wines.sample(frac = 0.05)
print(df_wines.shape)
df_wines.head()


# ## Understanding the NLP and Making the Word Clouds

# <p> So now our next task is to do some text analysis and NLTK on this and perform some analysis so that we can classify the wines on the basis of number of points i.e. determine its quality and also determine the price values for these wines. So the steps we are going to follow is by removing less important words by using various techniques then we will use the bag of words, tfidf values, then Machine learning classification techniques to predict the values.  </p> 

# ### So, firstly let us get some basic concepts 
# <ol>  
#  <li>Tokenization -- Which involves breaking the sentences into smaller parts called tokens which are generally words.</li>
#  <li> Normalization -- which involves getting to the root of the word. Like talked becomes talk and talks also becomes talk. So this will eliminate a lot of same words. It can be of 2 forms 1. Stemming and 2. Lemmatization. </li> 
#  <li> Word Cloud --  An image composed of words used in a particular text or subject, in which the size of each word indicates its frequency or importance </li>
# </ol> 
# So, these are some basic Concepts of NLP Now lets get started and implement them one by one 
#   

# ### Using Tokenization and Normalization

# In[ ]:


# Firstly let us count the number of words in all in the description texts available to us in our sample data
text = " ".join(review for review in df_wines.description)
print ("There are {} words in the combination of all review.".format(len(text)))


# In[ ]:


# N0w tokenizing the text using TreebankWordTokenizer
tokenizer1 = nltk.tokenize.TreebankWordTokenizer()
tokens = tokenizer1.tokenize(text)
tokens[0:10]


# In[ ]:


# NOw normalization of the text using lemmatization
stemmer = nltk.stem.WordNetLemmatizer()
words_all = " ".join(stemmer.lemmatize(token) for token in tokens)
words_all[0:100]                     


# ### Now removing the stopwords 
# <p> The stopwords are the words that are very commonly repeated like is, are,the, not, I, am etc and they don't need to be there and they don't tell anything about the sentence. Also they are present in the documents in high frequencies so it is very important to remove them otherwise while making the wordclouds or by classification they can affect the predictions in adverse way. Also the list of stopwords is already predefined for certain languages like english and others and it can also be updates based on the type of data we have.</p>

# In[ ]:


stopwords = set(STOPWORDS)
stopwords.update(["drink" , 'now', 'wine' ,'flavour'])
wordcloud = WordCloud(stopwords = stopwords, background_color = "white").generate(words_all)
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.show()


# ### Some other techniques to build word Clouds
# <p> Word Clouds need not be in a rectangular box, it can be inside any picture also if we want and it can also have various gradients also giving it a look of a particular object or picture depending on the context without acually using a picture. So now let us try that out. </p>

# ### A. Using the picture of a wine and a glass and then using it as a background for wordcloud

# In[ ]:


wine_mask = np.array(Image.open('../input/images/wine.jpg'))
def transform_mask(val):
    # For black color inside the wine bottle and glass making it white
    if val == 0:
        return 255
    else:
        return val
transformed_wine_mask = np.ndarray((wine_mask.shape[0], wine_mask.shape[1]))
for i in range(len(wine_mask)):
    transformed_wine_mask[i] = list(map(transform_mask, wine_mask[i]))


# In[ ]:


wc = WordCloud(background_color="white", max_words=1000, mask=transformed_wine_mask,
               stopwords=stopwords, contour_width=3, contour_color='firebrick', max_font_size = 34).generate(words_all)

# show
plt.figure(figsize=[20,10])
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()


# ### B. Now let us do the same thing with color gradients. 
#  #### So let us do this with the flag of india and USA and getting their wine descriptions inside them 

# In[ ]:


def create_color_comb(country, img):
    country = " ".join(review for review in  wines[wines['country'] == country].description)
    mask = np.array(Image.open("../input/images/"+img))
    wordcloud_country = WordCloud(stopwords = STOPWORDS, background_color = 'white',contour_color='firebrick',
                                  max_words = 1000, mask = mask).generate(country)
    image_colors = ImageColorGenerator(mask)
    plt.figure(figsize=[5,8])
    plt.imshow(wordcloud_country.recolor(color_func=image_colors), interpolation="bilinear")
    plt.axis("off")


# In[ ]:


create_color_comb("India", "india.png")
create_color_comb("US", "usa.jpg")


# . <p> Now for the further information on NLP and Various ML algos and their accuracies for the prediction of data . Follow this link <a href = ' https://www.kaggle.com/bhaargavi/wine-reviews-classification-using-nlp-and-ml'><h3> Link to Wine Reviews Classification Using NLP and ML</h3></a> </p>

# In[ ]:




