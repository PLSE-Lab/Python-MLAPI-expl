#!/usr/bin/env python
# coding: utf-8

# I am just a newbie in this field and this is my first kaggle submission.Constructive Feedback will be appreciated.I am a second year Graduation Student
# 

# In[ ]:


import os 
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# First we import all the required libraries

# In[ ]:


data = pd.read_csv("../input/spam.csv",encoding='latin-1')


# Then we import the data 

# In[ ]:


data.head()


# In[ ]:


data['Unnamed: 2'].count()


# In[ ]:


data['Unnamed: 3'].count()


# In[ ]:


data['Unnamed: 4'].count()


# So we see 3 NaN fields which we drop for our first model as the amount of data in them is very less.In required they can be incorporated later for further analysis

# In[ ]:


data.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)


# In[ ]:


data.head()


# Now lets change the column names for better representation.We change v1 to 'Labels' and v2 to 'Message'

# In[ ]:


data.rename(columns={'v1':'Label','v2':'Message'},inplace=True)


# Now lets look at our data again
# 

# In[ ]:


data


# Now lets clean the data and drop the nan values

# In[ ]:


data.dropna(inplace=True)


# In[ ]:


data.shape


# Now for the main thing we will count the total no of words we have in all of the sms combined.This will be the main step in the classifier

# In[ ]:


words=[]

for i in range(len(data['Message'])):
    blob=data['Message'][i]
    words+=blob.split(" ")


# In[ ]:


len(words)


# So we find out the total number of words to be equal to 86961.
# Now we want to remove those words which have special charecters in them for ease of analysis
# 

# In[ ]:


for i in range(len(words)):
    if not words[i].isalpha():
        words[i]=""


# In[ ]:


word_dict=Counter(words)
word_dict
len(word_dict)


# So we have around 8k individual words . We now remove the words which had special charecters in them or were different

# In[ ]:


del word_dict[""]


# Now taking the words which occur very rarely may increase the amount of noise in the data.So we take ony the top 3000

# In[ ]:


word_dict=word_dict.most_common(3000)


# Now lets form the matrix where we have all the individual words as columns and the message index as rows and the values are filled by the frequency of each word corresponding to the row

# In[ ]:


features=[]
labels=[]

for i in range(len(data['Label'])):

    blob=data['Message'][i].split(" ")
    data1=[]
    for j in word_dict:
        data1.append(blob.count(j[0]))
    features.append(data1)
    
    
   
    
    


# We now convert features into array

# In[ ]:


features=np.array(features)


# In[ ]:


features.shape


# Now lets import our output variable

# In[ ]:


labels=data.iloc[:,0]


# As the training models work much better on numeric data we convert labels in numeric data.We change Spam to 1 and Ham to 0.

# In[ ]:


for i in range(len(labels)):
    if labels[i]=='ham':
        labels[i]=0
    else:
        labels[i]=1


# In[ ]:


labels.shape


# In[ ]:


labels=labels.values
labels=labels.astype(int)


# Now we have our required output

# In[ ]:


labels


# Now we perform train test split on our input and output

# In[ ]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(features,labels,test_size=0.2,random_state=9)


# In[ ]:


xtrain.shape


# Now lets check it using naive bayes classifier cause it works best in these situations

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
nbs=MultinomialNB()


# In[ ]:


nbs.fit(xtrain,ytrain)


# In[ ]:


y_pred=nbs.predict(xtest)


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(y_pred,ytest)


# So thats an accuracy of 97% which isnt bad by any means

# Now lets try with logistic regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(xtrain,ytrain)


# In[ ]:


pred = model.predict(xtest)


# In[ ]:


accuracy_score(ytest,pred)


# So the accuracy isnt bad as well

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model1= RandomForestClassifier()
model1.fit(xtrain,ytrain)


# In[ ]:


prediction = model1.predict(xtest)


# In[ ]:


accuracy_score(ytest,prediction)


# Now for model evaluation

# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report


# For our naive bayes model

# In[ ]:


print(classification_report(ytest, y_pred, target_names = ["Ham", "Spam"]))


# For our Logistic Regression Classifier Model

# In[ ]:


print(classification_report(ytest, pred, target_names = ["Ham", "Spam"]))


# In[ ]:


conf_mat = confusion_matrix(ytest, y_pred)
conf_mat_normalized = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]


# In[ ]:


sns.heatmap(conf_mat_normalized)
plt.ylabel('True label')
plt.xlabel('Predicted label')


# In[ ]:


print(conf_mat)


# By seeing the above confusion matrix, it is clear that 19 Ham are mis classified as Spam, and 12 Spam are misclassified as Ham. 

# In[ ]:





# 

# In[ ]:





# In[ ]:




