#!/usr/bin/env python
# coding: utf-8

# # Quickly trying hard voting classifier in this case 

# ## Fire up

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

## Read Data
df=pd.read_csv('../input/prescriber-info.csv')
df.head()


# In[ ]:


print(df.shape)

#here I will add my standard 3-5 things to get info


# **We have a high dimensional data set again. How can we create an effective predictive model quickily on such a complicated data-set?**

# ## Data Proprecessing

# **Even a quick and dirty classification model requires the transformation of non-numeric feature, in order to enable Python to identify the variable.**

# **Thanks for @ Alan (AJ) Pryor, Jr 's feedback! My method was so 'dirty' that I even included misleading variables in my model. Now I will modify my Kernel to a more completed, accurate one.**

# In[ ]:


opioids=pd.read_csv('../input/opioids.csv')
name=opioids['Drug Name']
import re
new_name=name.apply(lambda x:re.sub("\ |-",".",str(x)))
columns=df.columns
Abandoned_variables = set(columns).intersection(set(new_name))
Kept_variable=[]
for each in columns:
    if each in Abandoned_variables:
        pass
    else:
        Kept_variable.append(each)


# In[ ]:


df=df[Kept_variable]
print(df.shape)


# In[ ]:


train,test = train_test_split(df,test_size=0.2,random_state=42)
print(train.shape)
print(test.shape)


# In[ ]:


Categorical_columns=['Gender','State','Credentials','Specialty']
for col in Categorical_columns:
    train[col]=pd.factorize(train[col], sort=True)[0]
    test[col] =pd.factorize(test[col],sort=True)[0]


# In[ ]:


features=train.iloc[:,1:245]
features.head()


# ## Try Different Classifiers

# In[ ]:


import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier


# In[ ]:


features=train.iloc[:,1:244]
target = train['Opioid.Prescriber']
Name=[]
Accuracy=[]
model1=LogisticRegression(random_state=22,C=0.000000001,solver='liblinear',max_iter=200)
model2=GaussianNB()
model3=RandomForestClassifier(n_estimators=200,random_state=22)
model4=GradientBoostingClassifier(n_estimators=200)
model5=KNeighborsClassifier()
model6=DecisionTreeClassifier()
model7=LinearDiscriminantAnalysis()
Ensembled_model=VotingClassifier(estimators=[('lr', model1), ('gn', model2), ('rf', model3),('gb',model4),('kn',model5),('dt',model6),('lda',model7)], voting='hard')
for model, label in zip([model1, model2, model3, model4,model5,model6,model7,Ensembled_model], ['Logistic Regression','Naive Bayes','Random Forest', 'Gradient Boosting','KNN','Decision Tree','LDA', 'Ensemble']):
    scores = cross_val_score(model, features, target, cv=5, scoring='accuracy')
    Accuracy.append(scores.mean())
    Name.append(model.__class__.__name__)
    print("Accuracy: %f of model %s" % (scores.mean(),label))


# **For the model with the accuracy worse than 80%, we drop them and re_ensemble the model.**

# In[ ]:


Name_2=[]
Accuracy_2=[]
Ensembled_model_2=VotingClassifier(estimators=[('rf', model3),('gb',model4)], voting='hard')
for model, label in zip([model3, model4,Ensembled_model_2], ['Random Forest', 'Gradient Boosting','Ensemble']):
    scores = cross_val_score(model, features, target, cv=5, scoring='accuracy')
    Accuracy_2.append(scores.mean())
    Name_2.append(model.__class__.__name__)
    print("Accuracy: %f of model %s" % (scores.mean(),label))


# **We can see that the ensembled model putting two models together performed better than the other three models. I will further use the test set to compare the performances between models.**

# ## Evaluating with the test set

# In[ ]:


from sklearn.metrics import accuracy_score
classifers=[model3,model4,Ensembled_model_2]
out_sample_accuracy=[]
Name_2=[]
for each in classifers:
    fit=each.fit(features,target)
    pred=fit.predict(test.iloc[:,1:244])
    accuracy=accuracy_score(test['Opioid.Prescriber'],pred)
    Name_2.append(each.__class__.__name__)
    out_sample_accuracy.append(accuracy)


# ## In-sample and out-sample evaluation

# In[ ]:


in_sample_accuracy=Accuracy_2


# In[ ]:


Index = [1,2,3]
plt.bar(Index,in_sample_accuracy)
plt.xticks(Index, Name_2,rotation=45)
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.title('In sample accuracy of models')
plt.show()


# In[ ]:


plt.bar(Index,out_sample_accuracy)
plt.xticks(Index, Name_2,rotation=45)
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.title('Out sample accuracies of models')
plt.show()


# **The hard-voting classifer does not provide a better result in terms of out sample error. Therefore, in this case, the voting classifer cannot reach the satisfied result. A soft voting method can be considered as a substitute.**
