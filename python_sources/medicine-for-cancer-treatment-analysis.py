#!/usr/bin/env python
# coding: utf-8

# ```Use Case Details:```
# 
# A lot has been said during the past several years about how precision medicine and, more concretely, how genetic testing is going to disrupt the way diseases like cancer are treated.
# 
# But this is only partially happening due to the huge amount of manual work still required. Memorial Sloan Kettering Cancer Center (MSKCC) launched this competition, accepted by the NIPS 2017 Competition Track,  because we need your help to take personalized medicine to its full potential.
# 
# 
# 
# Once sequenced, a cancer tumor can have thousands of genetic mutations. But the challenge is distinguishing the mutations that contribute to tumor growth (drivers) from the neutral mutations (passengers). 
# 
# Currently this interpretation of genetic mutations is being done manually. This is a very time-consuming task where a clinical pathologist has to manually review and classify every single genetic mutation based on evidence from text-based clinical literature.
# 
# For this competition MSKCC is making available an expert-annotated knowledge base where world-class researchers and oncologists have manually annotated thousands of mutations.
# 
# We need your help to develop a Machine Learning algorithm that, using this knowledge base as a baseline, automatically classifies genetic variations.
# 
# 
# Kaggle is excited to partner with research groups to push forward the frontier of machine learning. Research competitions make use of Kaggle's platform and experience, but are largely organized by the research group's data science team. Any questions or concerns regarding the competition data, quality, or topic will be addressed by them.

# ## Importing and Loading Datasets

# In[ ]:


# loading required libraries for cancer treament analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# To ignore warinings
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# let's check in which directory our data is available so that it will be easy to pull from specific source location
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


# loading datasets
train_variants = pd.read_csv('../input/msk-redefining-cancer-treatment/training_variants')
test_variants = pd.read_csv('../input/msk-redefining-cancer-treatment/test_variants')
train_text = pd.read_csv('../input/msk-redefining-cancer-treatment/training_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
test_text = pd.read_csv('../input/msk-redefining-cancer-treatment/test_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])


# ### Let's pull the top 5 rows value from train variants

# In[ ]:


train_variants.head()


# ### Similarlly let's pull top 5 rows value fromtrain text data

# In[ ]:


train_text.head()


# ### Merging Train variants and Train text into one trai dataset

# In[ ]:


train_merge = pd.merge(train_variants,train_text,how='left',on='ID')
# let's pull train merge dataset and do the analysis on this
train_merge.head()


# In[ ]:


# Let's understand the type of values present in each column of our dataframe 'train_merge' dataframe.
train_merge.info()


# ## Now let's draw a histogram/count plot to see how the classes are distributed

# In[ ]:


# Histogram : To check class distribution
plt.figure(figsize=(12,8))
sns.countplot(x='Class',data=train_variants)
plt.ylabel('Frequency-Counts', fontsize=15)
plt.xlabel('Class',fontsize=13)
plt.xticks(rotation='vertical')
plt.title('Class Counts',fontsize=15)
plt.show()


# ### Now Let's explore the text column and see the text distribution
# 

# In[ ]:


train_merge["Text_num_words"] = train_merge["Text"].apply(lambda x: len(str(x).split()) )
train_merge["Text_num_chars"] = train_merge["Text"].apply(lambda x: len(str(x)) )


# **- Let's look at the distribution of number of words in the text column.**

# In[ ]:


plt.figure(figsize=(12, 8))
sns.distplot(train_merge.Text_num_words.values, bins=50, kde=False, color='red')
plt.xlabel('Number of words in text', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title("Frequency of number of words", fontsize=15)
plt.show()


# ``` The peak is around 4000 words.``` 
# 
# - Now let us look at character level.

# In[ ]:


plt.figure(figsize=(12, 8))
sns.distplot(train_merge.Text_num_chars.values, bins=50, kde=False, color='brown')
plt.xlabel('Number of characters in text', fontsize=12)
plt.ylabel('log of Count', fontsize=12)
plt.title("Frequency of Number of characters", fontsize=15)
plt.show()


# - check if we could use the number of words in the text has predictive power.

# In[ ]:


plt.figure(figsize=(12,8))
sns.boxplot(x='Class', y='Text_num_words', data=train_merge)
plt.xlabel('Class', fontsize=12)
plt.ylabel('Text - Number of words', fontsize=12)
plt.show()


# ## Let's check whether the data is balance or not

# In[ ]:


train_merge.describe()


# - Looks like data is pretty balanced since we didn't see any random pick value. We are good to go for further analysis

# In[ ]:


# putting respon variable to y
#y = train_merge['Class'].values
#train_merge = train_merge.drop('Class',axis=1)


# In[ ]:


train_merge.head(3)


# In[ ]:


#y


# ## Now let's merge the test datasets(variants & text) together to one dataset

# In[ ]:


test_merge = pd.merge(test_variants,test_text,how='left',on='ID')
test_merge.head(3)


# In[ ]:


pid = test_merge['ID'].values
pid


# ## Let's have a quick look whether we have balance data or not in our test datasets

# In[ ]:


test_merge.describe()


# - Awesome! Our test dataset is looks fine. 

# ## Missing Value Analysis
# 
# - Check for missing values in both training and testing data columns

# In[ ]:


# check total number of null/missing value present in whole datasets
train_merge.isnull().sum()


#     - Ohh Ok, We have only 5 missing values present in text feature
#     - Let's remove those since it's only 5 missing in number if we see the percentage of missing values it's like only 0.1% . Since it's very less number we can remove those.

# In[ ]:


# find out percentage of "?" value present across the dataset
percent_missing = train_merge.isnull().sum() * 100 / len(train_merge)
percent_missing


# In[ ]:


# droping missing values
train_merge.dropna(inplace=True)

# let's check again whether we have any further missing values
train_merge.isnull().sum()


# - Awesome our data is cleaned! Good to go for model building

# ### Check test data is clean or not

# In[ ]:


test_merge.isnull().sum()


# - Okay, We have only 1 missing value in text data. let's remove it.

# In[ ]:


# dropping missing values
test_merge.dropna(inplace=True)

# check if our data is clean or not
test_merge.isnull().sum()


# - Awesome we are good to model builing. let's do this.

# 
# # Splitting Datasets into Training and Testing sets
# **- Splitting Datasets into Training & Testing sets by using scikit learn library**
# 

# In[ ]:


from sklearn.model_selection import train_test_split
train,test = train_test_split(train_merge,test_size=0.2)
np.random.seed(0)
train


# In[ ]:


X_train = train['Text'].values
X_test = test['Text'].values
y_train = train['Class'].values
y_test = test['Class'].values


# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm


# - Set pipeline to build a complete text processing model with Vectorizer, Transformer and LinearSVC

# In[ ]:


text_classifier = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', svm.LinearSVC())
])
text_classifier = text_classifier.fit(X_train,y_train)


# In[ ]:


y_test_predicted = text_classifier.predict(X_test)
np.mean(y_test_predicted == y_test)


# - Predicting values for test data

# In[ ]:


X_test_final = test_merge['Text'].values
#X_test_final


# In[ ]:


predicted_class = text_classifier.predict(X_test_final)


# - Appended the predicted class values to the testing data

# In[ ]:


test_merge['predicted_class'] = predicted_class


# In[ ]:


test_merge.head(5)


# ## Onehot encoding to get the predicted class values as columns

# In[ ]:


onehot = pd.get_dummies(test_merge['predicted_class'])
test_merge = test_merge.join(onehot)


# In[ ]:


test_merge.head(5)


# ## Preparing submission data

# In[ ]:


submission_df = test_merge[["ID",1,2,3,4,5,6,7,8,9]]
submission_df.columns = ['ID', 'class1','class2','class3','class4','class5','class6','class7','class8','class9']
submission_df.head(5)


# In[ ]:


submission_df.to_csv('submission.csv', index=False)


# ## Work in Progress.....

# **More to come. Stay tuned.!**

# ## **If you found this notebook helpful or you just liked it , some upvotes would be very much appreciated - That's will keep me motivated :)**

# In[ ]:




