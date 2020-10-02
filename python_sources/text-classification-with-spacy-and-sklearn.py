#!/usr/bin/env python
# coding: utf-8

# **What is Spacy?. **
# 
# Spacy is an opensource advanced python module for natural language processing, It supports multiple languages( English, German, Spanish, Portuguese, French, Italian, Dutch).
# 
# **Features** : 
# 1. Non-destructive tokenization
# 2. Named entity recognition
# 3. Support for 50+ languages
# 4. Pre-trained statistical models and word vectors
# 5. State-of-the-art speed
# 6. Easy deep learning integration
# 7. Part-of-speech tagging
# 8. Labelled dependency parsing
# 9. Syntax-driven sentence segmentation
# 10. Built in visualizers for syntax and NER
# 11. Convenient string-to-hash mapping
# 12. Export to numpy data arrays
# 13. Efficient binary serialization
# 14. Easy model packaging and deployment
# 15. Robust, rigorously evaluated accuracy
# 
# For more details [Click here.](https://github.com/explosion/spaCy)

# **This kernel has below topics : **
# 1. [Displacy for Text visualization.](#Displacy-for-Text-visualization.)
# 1. [LinearSVC with 40% test data.](#LinearSVC-with-40%-test-data.)
# 1. [LinearSVC with 10% test data.](#LinearSVC-with-10%-test-data.)
# 1. [Model Evaluation](#4)
# 1. [Testing the final model with new instances.](#Testing-the-final-model-with-new-instances.)

# In[ ]:


# import the libraries
import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from spacy import displacy


# In[ ]:


# Dataset
df = pd.read_csv('../input/reviews/Restaurant_Reviews.tsv',sep = '\t')


# In[ ]:


df.head()


# 1 -> Liked 
# 
# 0 -> Not liked

# In[ ]:


df.shape


# In[ ]:


# check for the missing values
df.isna().sum()


# # Displacy for Text visualization.

# In[ ]:


review_ = []

for i in range(0,11):
    text = df["Review"][i]
    review_.append(text)
print(review_)
# check the word dependancy using spacy.displacy()
for data in review_:
    nlp = spacy.load('en_core_web_sm')
    data = nlp(data)
    displacy.render(data,style = 'dep', options = {'font':'Areal','distance':100
                                              ,'color': 'green','bg':'white','compact' : True,}, jupyter =True)
    


# In[ ]:


# remove the empty string from the review column.
empty_loc  = []
for i, Rv,lk in df.itertuples():
    if type(Rv) == str:
        if Rv.isspace() == True:
            empty_loc.append(i)
print(empty_loc)


# In[ ]:


# check the number of positive and negative reviews
df["Liked"].value_counts()


# # LinearSVC with 40% test data.

# In[ ]:


x = df["Review"]
y = df["Liked"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.4)   # 40% of the data is reserved for testing
print(y_test.value_counts())
print(y_train.value_counts())


# In[ ]:


#Linear Classifier
Classifier_svc = Pipeline([('tfIdf',TfidfVectorizer()),('cl',LinearSVC()),])
Classifier_svc.fit(x_train,y_train)
pred = Classifier_svc.predict(x_test)


# In[ ]:


# model evaluation
cm = confusion_matrix(y_test,pred)
print(cm)
print('\n')

print("Accuracy : ", accuracy_score(y_test,pred))
print('\n')
print(classification_report(y_test,pred))

sns.heatmap(cm,annot =True)


# # LinearSVC with 10% test data.
# Lets train with 90% of the available data(i.e test_size = 10%).

# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1)  # 90% of the data used for training
print(y_train.value_counts())
Classifier_svc.fit(x_train,y_train)
pred = Classifier_svc.predict(x_test)


# In[ ]:


# model evaluation -->  Linearsvc with trainset_percentage == 100
print(y_test.value_counts())

print('\n')
print("confusion matrix : ")
cm = confusion_matrix(y_test,pred)
print(cm)
print('\n')

print("Accuracy : ", accuracy_score(y_test,pred))
print('\n')
print(classification_report(y_test,pred))

sns.heatmap(cm,annot =True)


# There are 8 false positives and 9 false negatives.
# Accuracy is good compare to the previous model.

# # Testing the final model with new instances.

# In[ ]:


print(Classifier_svc.predict(["I was ate just on a whim because the parking lot was full. I had the Irish Skillet and it was Delicious. Not bad prices either between my friend and I we only paid just over 20 dollars. Service here is great even on a full day."]))
print(Classifier_svc.predict(["Stopped here for breakfast because this has been a good restaurant for meals at any time of day for many years now. You can just count on a decent meal when you stop here. I like the breakfast skillets."]))


# Model is predicting well for the positive reviews, Let's check the performance with negative reviews.

# In[ ]:


print(Classifier_svc.predict(["We can't get a decent hamburger, they over cook them & they don't know the difference between cornbread or cake. They only have 1 soup worth eating & the waitresses on Saturdays are terrible, they are rude & don't listen while your trying to place your order"]))
print(Classifier_svc.predict(["Stopped in there one evening while traveling through Monticello. Ordered the fish and chips that the menu AND the waitress said was Cod fillets. What was brought to us was overcooked mincemeat fish sticks that had been cooked for a while and just heated in the microwave. Will NOT stop there again."]))


# First review is predicted correctly and there is a wrong prediction for the second review(False positive).
# The above Reviews are from the another restaurant(not the same restaurant data we used for training the model).

# **Work in progress....**
# 
# please upvote if you like my work...
