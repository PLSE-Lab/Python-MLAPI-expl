#!/usr/bin/env python
# coding: utf-8

# # Google Play Store Apps Visualization

# ## Importing libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

import warnings
warnings.filterwarnings('ignore')

from pylab import rcParams

get_ipython().run_line_magic('matplotlib', 'inline')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')
data.head()


# In[ ]:


data.shape


# The Dataset contains 10841 records with 13 fields

# In[ ]:


data.columns


# In[ ]:


#Finding the missing data
sb.heatmap(pd.isnull(data))


# In[ ]:


#Evaluating the missing values
missing_values = data.isnull().sum().sort_values(ascending = False)
missing_values


# In[ ]:


#Dropping the Missing values
data.dropna(how = 'any', inplace = True)
missing_values = data.isnull().sum().sort_values(ascending = False)
missing_values


# In[ ]:


sb.heatmap(pd.isnull(data))


# In[ ]:


data.shape


# After removing the missing values now the dataset contains 9360 records with 13 fields

# ### Rating

# In[ ]:


#Evaluating the data for Rating field
data['Rating'].describe()


# In[ ]:


plt.rcParams['figure.figsize'] = (15, 10)
count_graph = sb.countplot(data['Rating'])
count_graph.set_xticklabels(count_graph.get_xticklabels(), rotation = 90)
count_graph
plt.title('Count of Apps ', size = 20)


# In[ ]:


plt.rcParams['figure.figsize'] = (15, 10)
sb.distplot(data.Rating, color = 'red', hist = False)
plt.title('Rating Distribution', size = 20)


# From the above graph we can state that most of the apps in the Google Playstore are rated in between 4 to 5 stars.

# ### Category

# In[ ]:


#Categorical Evaluation of Apps
print(data['Category'].unique())
print('\n', len(data['Category'].unique()), 'Categories')


# In[ ]:


plt.rcParams['figure.figsize'] = (15, 10)
count_graph = sb.countplot(data['Category'])
count_graph.set_xticklabels(count_graph.get_xticklabels(), rotation = 90)
count_graph
plt.title('Count of Apps in each Category', size = 20)


# Game and Family category has the highest count of apps in the Play store.

# ### Comparision between Rating and Category

# In[ ]:


plt.rcParams['figure.figsize'] = (15, 10)
graph = sb.boxplot(y = data['Rating'], x = data['Category'])
sb.despine(left = True)
graph.set_xticklabels(graph.get_xticklabels(), rotation = 90)
graph
plt.title('Box Plot of Rating VS Category', size = 20)
plt.show()


# From the above comparision graph we can say that Rating for apps in each Category is not much different and each category has apps rated between 4 to 5 stars in store.

# ### Reviews

# In[ ]:


#Evaluating the data for Reviews
data['Reviews'].describe()


# In[ ]:


plt.rcParams['figure.figsize'] = (15, 10)
sb.distplot(data.Reviews, color = 'red', hist = False)
plt.title('Reviews Distribution', size = 20)
plt.show()


# 
# From the above graph we can state that most of the apps have less than 1M reviews and popular apps have more reviews.

# ### Comparision between Ratings and Reviews

# In[ ]:


#Convertings the Reviews object data into int type to plot comparision graph 
data['Reviews'] = data['Reviews'].apply(lambda x: int(x))


# In[ ]:


rcParams['figure.figsize'] = (15, 10)
sb.jointplot(data = data, x = "Reviews", y = "Rating", size = 10)


# In[ ]:


rcParams['figure.figsize'] = (15, 10)
sb.regplot(data = data, x = 'Reviews', y = 'Rating')
plt.title('Reviews VS Rating', size = 20)


# From the graph we can say that most of the apps that have high rating also have good reviews.

# ### Size

# In[ ]:


data['Size'].head()


# In[ ]:


data['Size'].unique()


# The data is still in object and most the app size varies with device, so we will filter those apps.

# In[ ]:


len(data[data.Size == 'Varies with device'])


# There are 1637 apps whose size may vary depending on the device.

# In[ ]:


data['Size'].replace('Varies with device', np.nan , inplace = True)


# In[ ]:


#Converting the object data type into int 
data.Size = (data.Size.replace(r'[kM]+$', '', regex=True).astype(float) *              data.Size.str.extract(r'[\d\.]+([KM]+)', expand=False)
            .fillna(1)
            .replace(['k','M'], [10**3, 10**6]).astype(int))


# In[ ]:


data['Size'].fillna(data.groupby('Category')['Size'].transform('mean'), inplace = True)


# 'Varies with device' is filled by mean of the size for each category.

# In[ ]:


rcParams['figure.figsize'] = (15, 10)
sb.jointplot(x = 'Size', y = 'Rating', data = data, size = 10 )


# ### Installs

# In[ ]:


data['Installs'].head()


# In[ ]:


data['Installs'].unique()


# In[ ]:


plt.rcParams['figure.figsize'] = (15, 10)
count_graph = sb.countplot(data['Installs'])
count_graph.set_xticklabels(count_graph.get_xticklabels(), rotation = 90)
count_graph
plt.title('Count of Apps ', size = 20)


# The data is still in object type.

# In[ ]:


#Converting the Object data into interger data
data.Installs = data.Installs.apply(lambda x: x.replace(',',''))
data.Installs = data.Installs.apply(lambda x: x.replace('+',''))
data.Installs = data.Installs.apply(lambda x: int(x))


# In[ ]:


data['Installs'].unique()


# In[ ]:


#Sorting the values
sorted_value = sorted(list(data['Installs'].unique()))


# In[ ]:


data['Installs'].replace(sorted_value, range(0, len(sorted_value), 1), inplace = True)


# ### Comparision between Ratings and Installs

# In[ ]:


rcParams['figure.figsize'] = (15, 10)
sb.regplot(x = 'Installs', y = 'Rating', data = data)
plt.title("Ratings VS Installs", size = 20)


# 
# From the above graph we say state that the no of installs affects the rating of the apps

# ### Type

# In[ ]:


data['Type'].unique()


# In[ ]:


plt.rcParams['figure.figsize'] = (15, 10)
count_graph = sb.countplot(data['Type'])
count_graph.set_xticklabels(count_graph.get_xticklabels(), rotation = 90)
count_graph
plt.title('Count of Apps ', size = 20)


# In[ ]:


labels = data['Type'].value_counts(sort = True).index
size = data['Type'].value_counts(sort = True)

explode = (0.1, 0)

rcParams['figure.figsize'] = (10, 10)

plt.pie(size, explode = explode, labels = labels, autopct = '%.2f%%', shadow = True)

plt.title("Perceantage of Free Apps in Playstore", size = 20)
plt.show()


# From the above chart we can say that 93.11% of the apps in the Playstore are Free.

# ### Price

# In[ ]:


#For Evaluation of Paid Apps only, I will consider the all the free apps as a single record
data['Free'] = data['Type'].map(lambda s :1  if s =='Free' else 0)
data.drop(['Type'], axis=1, inplace=True)


# In[ ]:


data['Price'].unique()


# In[ ]:


data.Price = data.Price.apply(lambda x: x.replace('$',''))
data['Price'] = data['Price'].apply(lambda x: float(x))


# In[ ]:


data['Price'].describe()


# Average price of paid apps is 0.96 dollars and most expensive app is at price 400 dollars.

# ### Comparison between Rating and Price

# In[ ]:


rcParams['figure.figsize'] = (15, 10)
sb.regplot(x = 'Price', y = 'Rating', data = data)
plt.title(" Price VS Rating", size = 20)


# If the app price is high but does not match expectations of the user then the app may get low rating.

# ### Content Rating

# In[ ]:


data['Content Rating'].unique()


# In[ ]:


plt.rcParams['figure.figsize'] = (15, 10)
count_graph = sb.countplot(data['Content Rating'])
count_graph.set_xticklabels(count_graph.get_xticklabels(), rotation = 90)
count_graph
plt.title('Count of Apps ', size = 20)


# In[ ]:


rcParams['figure.figsize'] = (15, 10)
sb.boxplot(x = 'Content Rating', y = 'Rating', data = data)
plt.title("Content Rating VS Rating", size = 20)


# Content Rating does not affect much the overall rating of the app, but the adult apps seems to have low rating when compared to other apps.

# ### Genres

# In[ ]:


data['Genres'].unique()


# In[ ]:


len(data['Genres'].unique())


# In[ ]:


data.Genres.value_counts()


# In[ ]:


#Grouping to ignore sub-genre
data['Genres'] = data['Genres'].str.split(';').str[0]


# In[ ]:


print(data['Genres'].unique())
print('\n', len(data['Genres'].unique()), 'genres')


# In[ ]:


plt.rcParams['figure.figsize'] = (20, 10)
count_graph = sb.countplot(data['Genres'])
count_graph.set_xticklabels(count_graph.get_xticklabels(), rotation = 90)
count_graph
plt.title('Count of Apps', size = 20)


# In[ ]:


rcParams['figure.figsize'] = (20,10)
graph = sb.boxplot(x = 'Genres', y = 'Rating', data = data)
graph.set_xticklabels(graph.get_xticklabels(), rotation = 90)
graph
plt.title('Rating VS Genres', size = 20)


# # NLP for Apps Reviews

# In[ ]:


rev = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore_user_reviews.csv')
rev.head()


# ### Selecting Reviews and Types of Reviews

# In[ ]:


rev = pd.concat([rev.Translated_Review, rev.Sentiment], axis = 1)
rev.dropna(axis = 0, inplace = True)
rev.head(10)


# In[ ]:


rev.Sentiment.value_counts()


# ### Converting Review Types

# ###### Positive = 0
# ###### Negative = 1
# ###### Neutral = 2

# In[ ]:


rev.Sentiment = [0 if i == 'Positive' else 1 if i == 'Negative' else 2 for i in rev.Sentiment]
rev.head(10)


# ### Simplifying Reviews for Modelling

# In[ ]:


# Removing characters that are not letters & converting them to lower case
import re
first_text = rev.Translated_Review[0]
text = re.sub('[^a-zA-Z]',' ', first_text)
text = text.lower()


# In[ ]:


print(rev.Translated_Review[0])
print(text)


# In[ ]:


#Tokenize to seperate each word
import nltk as nlp
from nltk.corpus import stopwords
text = nlp.word_tokenize(text)
text


# In[ ]:


#Lemmatization to convert words to their root forms
lemma = nlp.WordNetLemmatizer()
text = [lemma.lemmatize(i) for i in text]
text = " ".join(text)
text


# The sentence is in its simplest form, So now we will apply this each review

# In[ ]:


text_list = []
for i in rev.Translated_Review:
    text = re.sub('[^a-zA-Z]',' ', i)
    text = text.lower()
    text = nlp.word_tokenize(text)
    lemma = nlp.WordNetLemmatizer()
    text = [lemma.lemmatize(i) for i in text]
    text = " ".join(text)
    text_list.append(text)
    
text_list[:10]    


# ### Modelling

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
max_features = 200000
cou_vec = CountVectorizer(max_features = max_features, stop_words = 'english')
sparce_matrix = cou_vec.fit_transform(text_list).toarray()
all_words = cou_vec.get_feature_names()
print('Most used words :', all_words[:100])


# ### Classification for Modelling

# In[ ]:


y = rev.iloc[:,1].values
x = sparce_matrix


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
random = RandomForestClassifier(n_estimators = 10)
random.fit(x_train, y_train)


# In[ ]:


print("Accuracy: ",random.score(x_test,y_test))


# In[ ]:


y_pred = random.predict(x_test)
y_true = y_test


# In[ ]:


#Confusion Matrix
from sklearn.metrics import confusion_matrix
names = ['Positive', 'Negative', 'Neutral']
cm = confusion_matrix(y_true, y_pred)
f, ax = plt.subplots(figsize = (5,5))
sb.heatmap(cm, annot = True, fmt = '0.2f')
plt.xlabel('y_pred')
plt.ylabel('y_true')
ax.set_xticklabels(names)
ax.set_yticklabels(names)

