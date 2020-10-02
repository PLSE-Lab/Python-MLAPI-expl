#!/usr/bin/env python
# coding: utf-8

# In[8]:


# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import sqlite3


# In[9]:


# creating sql connection string
con = sqlite3.connect('../input/database.sqlite')


# In[11]:


#Positive Review - Rating above 3
#Negative Review - Rating below 3
#Ignoring Reviews with 3 Rating

filtered_data = pd.read_sql_query('SELECT * from Reviews WHERE Score != 3',con)


# In[12]:


filtered_data.head(5)


# In[13]:


# mapping ratings above 3 as Positive and below 3 as Negative

actual_scores = filtered_data['Score']
positiveNegative = actual_scores.map(lambda x: 'Positive' if x>3 else 'Negative')
filtered_data['Score'] = positiveNegative


# In[14]:


filtered_data.head(5)


# In[15]:


# Sorting values according to ProductID
sorted_values = filtered_data.sort_values('ProductId',kind = 'quicksort')


# In[17]:


final = sorted_values.drop_duplicates(subset= { 'UserId', 'ProfileName', 'Time',  'Text'})


# In[22]:


print('Rows dropped : ',filtered_data.size - final.size)
print('Percentage Data remaining after dropping duplicates :',(((final.size * 1.0)/(filtered_data.size * 1.0) * 100.0)))


# In[23]:


# Dropping rows where HelpfulnessNumerator < HelpfulnessDenominator
final = final[final.HelpfulnessDenominator >= final.HelpfulnessNumerator]


# In[25]:


print('Number of Rows remaining in the Dataset: ',final.size)


# In[26]:


# Checking the number of positive and negative reviews
final['Score'].value_counts()


# In[31]:


# Taking a smaller Dataset
final_sliced = final.iloc[:10,:]
final_sliced.shape


# In[28]:


final_sliced['Score'].value_counts()


# In[34]:


# Function to Remove HTML Tags
import re
def cleanhtml(sentence):
    cleaner = re.compile('<.*?>')
    cleantext = re.sub(cleaner,"",sentence)
    return cleantext


# In[35]:


# Function to clean punctuations and special characters

def cleanpunct(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    return  cleaned


# In[44]:


# Initialize Stop words and PorterStemmer and Lemmetizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

stop = set(stopwords.words('english'))
sno = SnowballStemmer('english')


print(stop)
print('*' * 100)
print(sno.stem('tasty'))


# In[46]:


# Cleaning HTML and non-Alphanumeric characters from the review text
i=0
str1=' '
final_string=[]
all_positive_words=[] # store words from +ve reviews here
all_negative_words=[] # store words from -ve reviews here.
s=''
for sent in final['Text'].values:
    filtered_sentence=[]
    #print(sent);
    sent=cleanhtml(sent) # remove HTMl tags
    for w in sent.split():
        for cleaned_words in cleanpunct(w).split():
            if((cleaned_words.isalpha()) & (len(cleaned_words)>2)):    
                if(cleaned_words.lower() not in stop):
                    s=(sno.stem(cleaned_words.lower())).encode('utf8')
                    filtered_sentence.append(s)
                    if (final['Score'].values)[i] == 'positive': 
                        all_positive_words.append(s) #list of all words used to describe positive reviews
                    if(final['Score'].values)[i] == 'negative':
                        all_negative_words.append(s) #list of all words used to describe negative reviews reviews
                else:
                    continue
            else:
                continue 
    #print(filtered_sentence)
    str1 = b" ".join(filtered_sentence) #final string of cleaned words
    #print("***********************************************************************")
    
    final_string.append(str1)
    i+=1


# In[ ]:


final['CleanedText']=final_string
final.head(5)


# In[47]:


# Running BoW
from sklearn.feature_extraction.text import CountVectorizer
count_vec = CountVectorizer()

final_counts = count_vec.fit_transform(final['Text'].values)
final_counts.get_shape()
final_counts = np.asarray(final_counts)


# In[51]:


# Standardizing Data before implementing TSNE
from sklearn.preprocessing import StandardScaler
standardized_data = StandardScaler(with_mean=False).fit_transform(final_counts)
#standardized_data = standardized_data.toarray()


# In[ ]:


#Implement TSNE
from sklearn.manifold import TSNE
labels = final['Score']

model = TSNE(n_components= 2,random_state= 0)
tsne = model.fit_transform(standardized_data)
tsne_data  = np.vstack(tsne.T,labels).T

tsne_df = pd.DataFrame(data = tsne_data, columns={'Dim1','Dim2','Labels'})

# PLotting the TSNE Results
sns.FacetGrid(tsne_df,hue = 'Labels', hue_order=['Positive','Negative'],size= 6).map(plt.scatter,'Dim1','Dim2').add_legend()
plt.show()



