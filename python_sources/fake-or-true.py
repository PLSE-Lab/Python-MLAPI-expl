#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[ ]:


true = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')
fake = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')


# In[ ]:


from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image 
text = str
text = " ".join(review for review in true.text)
mask = np.array(Image.open("/kaggle/input/image-file-for-fake-news/news.png"))
wc = WordCloud(background_color="white", max_words=1000, mask=mask,max_font_size=200,contour_color='black')
wc.generate(text)
plt.figure(figsize=[20,10])
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:


from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image 
text = str
text = " ".join(review for review in fake.text)
mask = np.array(Image.open("/kaggle/input/image-file-for-fake-news/news.png"))
wc = WordCloud(background_color="white", max_words=1000, mask=mask,max_font_size=200,contour_color='black')
wc.generate(text)
plt.figure(figsize=[20,10])
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:


true.head()


# In[ ]:


true['result'] = [1 for i in range(0,len(true))]


# In[ ]:


true.shape


# In[ ]:


fake['result'] = [0 for i in range(0,len(fake))]


# In[ ]:


fake.head()


# In[ ]:


fake.shape


# In[ ]:


# club both the files
data = true.append(fake,ignore_index = True,sort=False)


# In[ ]:


true.shape


# In[ ]:


data.head()


# In[ ]:


data.tail()


# In[ ]:


data.info()


# info() shows that there is no null object. so this cuts down one of the steps of data preprocessing.
# since columns are object type(strings) feature scaling is also obviously not required.

# lets have a look at our data subject wise

# In[ ]:


data['subject'].nunique()
# this shows we have 8 subjects


# In[ ]:


data['subject'].unique()


# **lets find out that which subject has maximum  true news**
# this can not be done for fake news as subject for all news is 'news' there

# In[ ]:


true_politicnews=0
world_n=0
govt_n =0
us_n =0
middle_n=0
for i,j in zip(data['subject'],data['result']):
    if i==('politicsNews' or 'politics') and j==1:
        true_politicnews = true_politicnews +1
    if i==('worldnews') and j==1:
        world_n+=1
    if i==('Government News') and j==1:
        govt_n+=1
    if i==('US_News') and j==1:
        us_n+=1
    if i==('Middle-east') and j==1:
        middle_n+=1
print(true_politicnews)
print(world_n)
print(us_n)
print(govt_n)
print(middle_n)


# wordcloud for political_news(true)

# In[ ]:


str =" "
for i,j in zip(data['subject'],data['text']):
    if i=='politicsNews':
        str += j


# In[ ]:


from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
# list_comp = [review if data[data['subject']=="politicsnews"] else ' ' for review in data['text']]
politics = str
wordcloud = WordCloud( background_color="white", max_words=1000).generate(politics)

plt.figure(figsize=[7,7])
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[ ]:


# we can have a look at the wordcloud for each of these subjects
# so lets group the data acc to subjects and then make wordclods for fake and true news.


# **LETS WORK ON TEXT NOW**

# In[ ]:


# CLEANING THE TEXT COLUMN


# i will concat titles and text columns and that column would be pre processed

# In[ ]:


# X = pd.concat(data['title'],data['text'])
data['combines'] = data['title']+" "+data['text']


# In[ ]:


df = data[['combines','result']]


# # LOWERCASE

# In[ ]:


data['combines'] = data['combines'].apply(lambda word:word.lower())


# # PUNCTUATIONS

# In[ ]:


import string
print(string.punctuation)


# In[ ]:


def punctuation_removal(str1):
    list1 = [x for x in str1 if x not in string.punctuation]
    str2 = ''.join(list1)
    return str2
data['combines'] = data['combines'].apply(lambda word:punctuation_removal(word))


# # STOPWORDS

# In[ ]:


from nltk.corpus import stopwords
stop = stopwords.words('english')
print(stop)


# In[ ]:


df['combines'].apply(lambda x: [word for word in x if word not in stop])


# In[ ]:


# seperate each word by white space
data['combines'] = data['combines'].apply(lambda word:word.split(','))


# In[ ]:


data['combines'].head(2)


# In[ ]:


# from nltk.tokenize import word_tokenize
# def to_words(text):
#     tokens = word_tokenize(text)
#     return tokens
# data['combines'] = data['combines'].apply(lambda s:to_words(s))


# In[ ]:


def convert(lst):       
    return ' , '.join(lst)

data['combines'] = data['combines'].apply(lambda s:convert(s))


# # **STEMMING**

# In[ ]:


# # split into words
# from nltk.tokenize import word_tokenize
# def splitter(text):
#     tokens = word_tokenize(text)
#     return tokens
# data['combines'] = data['combines'].apply(lambda x:splitter(x))


# In[ ]:


# def convert(lst):       
#     return ' , '.join(lst)

# data['combines'] = data['combines'].apply(lambda s:convert(s))


# In[ ]:


# from nltk.stem.porter import PorterStemmer
# porter = PorterStemmer()
# def stem(tokens):
#     stemmed = [porter.stem(word) for word in tokens]
#     print(stemmed[:100])
# data['combines'] = data['combines'].apply(lambda x:stem(x))


# In[ ]:


data['combines'].head()


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

combine = CountVectorizer()
combine = combine.fit(data['combines'])

combined_vector = combine.transform(data['combines'])


# In[ ]:



from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(combined_vector)
news_tfidf = tfidf_transformer.transform(combined_vector)
print(news_tfidf.shape)


# In[ ]:


y = data['result']


# In[ ]:


xtrain,xtest,ytrain,ytest = train_test_split(news_tfidf,y,test_size =0.25,random_state=7)


# # SVM

# In[ ]:


from sklearn.linear_model import SGDClassifier

svm = SGDClassifier().fit(xtrain, ytrain)
predict_svm = svm.predict(xtest)


# In[ ]:


from sklearn.metrics import classification_report
print (classification_report(ytest, predict_svm))


# In[ ]:


from sklearn.metrics import confusion_matrix
final = confusion_matrix(predict_svm, ytest)


# In[ ]:


print(final)


# In[ ]:


from sklearn.metrics import accuracy_score
print('Accuracy Score :',accuracy_score(ytest, predict_svm) )


# In[ ]:




