#!/usr/bin/env python
# coding: utf-8

# # Project: Predicting the disease
# The Objective of this Project is to take a closer look at the data and to predict the chance of occurrence based on the various features (Risk Factors) responsible for the Disease.

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


import os
os.listdir("../input")


# In[ ]:


df = pd.read_csv("../input/Training.csv")
df.head()


# In[ ]:


df.columns


# In[ ]:


df.info()


# In[ ]:


len(df.columns)


# In[ ]:


len(df)


# In[ ]:


len(df['prognosis'].unique())


# In[ ]:


cols = df.columns
cols = cols[:-1]
cols


# In[ ]:


len(cols)


# In[ ]:


x = df[cols]
y = df['prognosis']


# In[ ]:


df['loss_of_appetite'].value_counts(normalize ='True').plot(kind="bar")


# In[ ]:


sn=sns.FacetGrid(df, hue="prognosis", size=5) 
sn.map(plt.scatter, "itching","redness_of_eyes") 
sn.add_legend()
plt.show()


# ## Applying Machine Learning Algorithms 

# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# In[ ]:


mnb = MultinomialNB()
mnb = mnb.fit(x_train, y_train)


# In[ ]:


mnb.score(x_test, y_test)


# In[ ]:


from sklearn import model_selection
print ("result")
scores = model_selection.cross_val_score(mnb, x_test, y_test, cv=3)
print (scores)
print (scores.mean())


# In[ ]:


test_data = pd.read_csv("../input/Testing.csv")


# In[ ]:


test_data.head()


# In[ ]:


testx = test_data[cols]
testy = test_data['prognosis']


# In[ ]:


mnb.score(testx, testy)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# In[ ]:


print ("DecisionTree")
dt = DecisionTreeClassifier()
clf_dt=dt.fit(x_train,y_train)
print ("Acurracy: ", clf_dt.score(x_test,y_test))


# In[ ]:


from sklearn import model_selection
print ("result")
scores = model_selection.cross_val_score(dt, x_test, y_test, cv=3)
print (scores)
print (scores.mean())


# In[ ]:


print ("Acurracy on the actual test data: ", clf_dt.score(testx,testy))


# In[ ]:


dt.__getstate__()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt

importances = dt.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking",indices)


# In[ ]:


features = cols


# In[ ]:


for f in range(40):
    print("%d. feature %d - %s (%f)" % (f + 1, indices[f], features[indices[f]] ,importances[indices[f]]))


# **Predicting the disease where the only symptom is loss_of_appetite.**

# In[ ]:


feature_dict = {}
for i,f in enumerate(features):
    feature_dict[f] = i


# In[ ]:


feature_dict['loss_of_appetite']


# In[ ]:


sample_x = [i/35 if i ==35 else i*0 for i in range(len(features))]
#This means predicting the disease where the only symptom is redness_of_eyes.


# In[ ]:


len(sample_x)


# In[ ]:


sample_x = np.array(sample_x).reshape(1,len(sample_x))


# In[ ]:


dt.predict(sample_x)


# In[ ]:


dt.predict_proba(sample_x)


# Hence the disease would be Peptic ulcer diseae.
# 
# 

# **Predicting the disease where the only symptom is enlarged_thyroid.**

# In[ ]:


feature_dict['enlarged_thyroid']


# In[ ]:


sample_x = [i/71 if i ==71 else i*0 for i in range(len(features))]


# In[ ]:


sample_x = np.array(sample_x).reshape(1,len(sample_x))


# In[ ]:


dt.predict(sample_x)


# Hence the disease would be Arthritis.
# 
# 

# In[ ]:


def train_model(X_train, y_train, X_test, y_test, classifier, **kwargs):
    model = classifier(**kwargs)
    model.fit(X_train,y_train)
    
    # check accuracy and print out the results
    fit_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    return model


# In[ ]:


plt.figure(figsize=(30,20))
model = train_model(x_train, y_train, x_test, y_test, RandomForestClassifier, random_state=2606)
pd.Series(model.feature_importances_,x.columns).sort_values(ascending=True).plot.barh()


# ***Finding Accuracy***

# In[ ]:


accuracy = []

# list of algorithms names
classifiers = ['Logistic Regression', 'Random Forests', 'Knn (5 Neighbors)']

# list of algorithms with parameters
models = [LogisticRegression(), RandomForestClassifier(n_estimators=1177, random_state=2), KNeighborsClassifier(n_neighbors=5)]

for i in models:
    model = i
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    accuracy.append(score)


# In[ ]:


# create a dataframe from accuracy results
summary = pd.DataFrame({'accuracy':accuracy}, index=classifiers)       
summary


# In[ ]:




