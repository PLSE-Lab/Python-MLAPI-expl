#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis and Prediction of Review Ratings on the Yelp Reviews Dataset using various Machine Learning Algorithms
# Dataset Information: <br>
# (1). Dataset:
#     *   Column 1 - Unique Business ID
#     *   Column 2 - Date of Review
#     *   Column 3 - Review ID
#     *   Column 4 - Stars given by the user
#     *   Column 5 - Review given by the user
#     *   Column 6 - Type of text entered - Review
#     *   Column 7 - Unique User ID
#     *   Column 8 - Cool column: The number of cool votes the review received
#     *   Column 9 - Useful column: The number of useful votes the review received
#     *   Column 10 - Funny Column: The number of funny votes the review received <br>
# (2). Number of entries - 10000

# **(1). Importing all the necessary modules:**

# In[7]:


# IMPORTING ALL THE NECESSARY LIBRARIES AND PACKAGES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import string
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.grid_search import GridSearchCV
get_ipython().run_line_magic('matplotlib', 'inline')


# **(2). Loading and seeing the dataset details:**

# In[8]:


# LOADING THE DATASET AND SEEING THE DETAILS
data = pd.read_csv('../input/yelp.csv')
# SHAPE OF THE DATASET
print("Shape of the dataset:")
print(data.shape)
# COLUMN NAMES
print("Column names:")
print(data.columns)
# DATATYPE OF EACH COLUMN
print("Datatype of each column:")
print(data.dtypes)
# SEEING FEW OF THE ENTRIES
print("Few dataset entries:")
print(data.head())
# DATASET SUMMARY
data.describe(include='all')


# **(3). Creating of a new column:**<br>
# The new column will be - "length". This column will hold the data of the word length of the review.

# In[9]:


#CREATING A NEW COLUMN IN THE DATASET FOR THE NUMBER OF WORDS IN THE REVIEW
data['length'] = data['text'].apply(len)
data.head()


# **(4). Visualization:**<br>
# Let us now visualize the if there is any correlation between stars and the length of the review.

# In[10]:


# COMPARING TEXT LENGTH TO STARS
graph = sns.FacetGrid(data=data,col='stars')
graph.map(plt.hist,'length',bins=50,color='blue')


# **(5). Mean Value of the Vote columns**
# There are 3 voting columns for the reviews - funny, cool and useful. Let us now find the mean values with respect to the stars given to the review.

# In[11]:


# GETTING THE MEAN VALUES OF THE VOTE COLUMNS WRT THE STARS ON THE REVIEW
stval = data.groupby('stars').mean()
stval


# **(6). Correlation between the voting columns:** <br>
# Let us now see what the correlation is between the three voting columns.

# In[12]:


# FINDING THE CORRELATION BETWEEN THE VOTE COLUMNS
stval.corr()


# Thus, we can see that there is negative correlation between:
#     * Cool and Useful
#     * Cool and Funny
#     * Cool and Length  <br>
# Thus, we can say that the reviews marked cool tend to be curt, not very useful to others and short.<br>
# Whereas, there is a positive correlation between:
#     * Funny and Useful    
#     * Funny and Length
#     * Useful and Length    
# Thus, we can say that longer reviews tend to be funny and useful.

# **(7). Classifying the dataset and splitting it into the reviews and stars:**

# In[13]:


# CLASSIFICATION
data_classes = data[(data['stars']==1) | (data['stars']==3) | (data['stars']==5)]
data_classes.head()
print(data_classes.shape)

# Seperate the dataset into X and Y for prediction
x = data_classes['text']
y = data_classes['stars']
print(x.head())
print(y.head())


# **(8). Data Cleaning:** <br>
# We will now, define a function which will clean the dataset by removing stopwords and punctuations.

# In[86]:


# CLEANING THE REVIEWS - REMOVAL OF STOPWORDS AND PUNCTUATION
def text_process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# **(9). Vectorization**<br>
# We will now vectorize a single review and see the results:

# In[87]:


# CONVERTING THE WORDS INTO A VECTOR
vocab = CountVectorizer(analyzer=text_process).fit(x)
print(len(vocab.vocabulary_))
r0 = x[0]
print(r0)
vocab0 = vocab.transform([r0])
print(vocab0)
"""
    Now the words in the review number 78 have been converted into a vector.
    The data that we can see is the transformed words.
    If we now get the feature's name - we can get the word back!
"""
print("Getting the words back:")
print(vocab.get_feature_names()[19648])
print(vocab.get_feature_names()[10643])


# **(10). Vectorization of the whole review set and and checking the sparse matrix:**

# In[88]:


x = vocab.transform(x)
#Shape of the matrix:
print("Shape of the sparse matrix: ", x.shape)
#Non-zero occurences:
print("Non-Zero occurences: ",x.nnz)

# DENSITY OF THE MATRIX
density = (x.nnz/(x.shape[0]*x.shape[1]))*100
print("Density of the matrix = ",density)


# **(11). Splitting the dataset X into training and testing set:**

# In[89]:


# SPLITTING THE DATASET INTO TRAINING SET AND TESTING SET
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=101)


# **(12). Modelling:**<br>
# We will now use multiple Machine Algorithms to see which gives the best performance.

# (1). Multinomial Naive Bayes - We are using Multinomial Naive Bayes over Gaussian because with sparse data, Gaussian Naive Bayes assumptions aren't met and a simple gaussian fit over the data will not give us a good fit or prediction!

# In[90]:


# Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(x_train,y_train)
predmnb = mnb.predict(x_test)
print("Confusion Matrix for Multinomial Naive Bayes:")
print(confusion_matrix(y_test,predmnb))
print("Score:",round(accuracy_score(y_test,predmnb)*100,2))
print("Classification Report:",classification_report(y_test,predmnb))


# (2). Random Forest Classifier

# In[91]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier
rmfr = RandomForestClassifier()
rmfr.fit(x_train,y_train)
predrmfr = rmfr.predict(x_test)
print("Confusion Matrix for Random Forest Classifier:")
print(confusion_matrix(y_test,predrmfr))
print("Score:",round(accuracy_score(y_test,predrmfr)*100,2))
print("Classification Report:",classification_report(y_test,predrmfr))


# (3). Decision Tree

# In[92]:


# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
preddt = dt.predict(x_test)
print("Confusion Matrix for Decision Tree:")
print(confusion_matrix(y_test,preddt))
print("Score:",round(accuracy_score(y_test,preddt)*100,2))
print("Classification Report:",classification_report(y_test,preddt))


# (4). Support Vector Machines

# In[93]:


# Support Vector Machine
from sklearn.svm import SVC
svm = SVC(random_state=101)
svm.fit(x_train,y_train)
predsvm = svm.predict(x_test)
print("Confusion Matrix for Support Vector Machines:")
print(confusion_matrix(y_test,predsvm))
print("Score:",round(accuracy_score(y_test,predsvm)*100,2))
print("Classification Report:",classification_report(y_test,predsvm))


# (5). Gradient Boosting Classifier

# In[94]:


# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
"""# parameter evaluation
gbe = GradientBoostingClassifier(random_state=0)
parameters = {
     'learning_rate': [0.05, 0.1, 0.5],
    'max_features': [0.5, 1],
    'max_depth': [3, 4, 5]}
gridsearch=GridSearchCV(gbe,parameters,cv=100,scoring='roc_auc')
gridsearch.fit(x,y)
print(gridsearch.best_params_)
print(gridsearch.best_score_)"""
#Boosting
gbi = GradientBoostingClassifier(learning_rate=0.1,max_depth=5,max_features=0.5,random_state=999999)
gbi.fit(x_train,y_train)
predgbi = gbi.predict(x_test)
print("Confusion Matrix for Gradient Boosting Classifier:")
print(confusion_matrix(y_test,predgbi))
print("Score:",round(accuracy_score(y_test,predgbi)*100,2))
print("Classification Report:",classification_report(y_test,predgbi))


# In the above GBC code, I have commented the parameter evaluation code because it takes a lot of time for execution. In version 9 of this notebook , I ran only the parameter evaluation code, I got the parameters of: <br>
#     * Learning Rate = 0.1
#     * Max Depth = 5
#     * Max Features = 0.5 
# Hence, I used those features directly from Version 10 onwards for faster execution. If you want to see the running, you can either run version 9 or uncomment that part.
#     

# (6). K - Nearest Neighbor Classifier

# In[95]:


# K Nearest Neighbour Algorithm
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train,y_train)
predknn = knn.predict(x_test)
print("Confusion Matrix for K Neighbors Classifier:")
print(confusion_matrix(y_test,predknn))
print("Score: ",round(accuracy_score(y_test,predknn)*100,2))
print("Classification Report:")
print(classification_report(y_test,predknn))


# (7). XGBoost Classifier

# In[96]:


# XGBoost Classifier
import xgboost
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(x_train,y_train)
predxgb = xgb.predict(x_test)
print("Confusion Matrix for XGBoost Classifier:")
print(confusion_matrix(y_test,predxgb))
print("Score: ",round(accuracy_score(y_test,predxgb)*100,2))
print("Classification Report:")
print(classification_report(y_test,predxgb))


# In[97]:


# MULTILAYER PERCEPTRON CLASSIFIER
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier()
mlp.fit(x_train,y_train)
predmlp = mlp.predict(x_test)
print("Confusion Matrix for Multilayer Perceptron Classifier:")
print(confusion_matrix(y_test,predmlp))
print("Score:",round(accuracy_score(y_test,predmlp)*100,2))
print("Classification Report:")
print(classification_report(y_test,predmlp))


# From the above algorithm modelling, we can see that: 
#     *  Multilayer Perceptron = 77.57%
#     * Multinomial Naive Bayes = 76.94%
#     * Gradient Boosting Classifier = 73.87%
#     * XGBoost Classifier = 70.81%
#     * Random Forest Classifier = 67.57%
#     * Decision Tree = 65.5%
#     * K Neighbor Classifier = 61.35%
#     * Support Vector Machine  = 59.1%
# 

# Since multilayer perceptron classifier has the best score, let us use it to predict a random positive review, a random average review and a random negative review!

# In[ ]:


# POSITIVE REVIEW
pr = data['text'][0]
print(pr)
print("Actual Rating: ",data['stars'][0])
pr_t = vocab.transform([pr])
print("Predicted Rating:")
mlp.predict(pr_t)[0]


# In[84]:


# AVERAGE REVIEW
ar = data['text'][16]
print(ar)
print("Actual Rating: ",data['stars'][16])
ar_t = vocab.transform([ar])
print("Predicted Rating:")
mlp.predict(ar_t)[0]


# In[ ]:


# NEGATIVE REVIEW
nr = data['text'][16]
print(nr)
print("Actual Rating: ",data['stars'][23])
nr_t = vocab.transform([nr])
print("Predicted Rating:")
mlp.predict(nr_t)[0]


# In[16]:


count = data['stars'].value_counts()
print(count)


# From the above, we can see that predictions are biased towards positive reviews. We can see that the dataset has more positive reviews as compared to negative reviews. <br>
# I think I can fix it by normalizing the dataset to have equal number of reviews - thereby removing the bias. 
