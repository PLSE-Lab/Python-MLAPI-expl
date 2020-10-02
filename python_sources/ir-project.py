#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_json('../input/train.json')
df.head()


# In[ ]:


df.info()


# In[ ]:


import pandas_profiling
pandas_profiling.ProfileReport(df)


# In[ ]:


f, ax = plt.subplots(figsize=(5,6))
sns.countplot(y = 'cuisine', 
                   data = df,
                  order = df.cuisine.value_counts(ascending=False).index)


# # Analysis Of Ingrediants

# In[ ]:


ingredients_individual = Counter([ingredient for ingredient_list in df.ingredients for ingredient in ingredient_list])
ingredients_individual = pd.DataFrame.from_dict(ingredients_individual,orient='index').reset_index()


ingredients_individual = ingredients_individual.rename(columns={'index':'Ingredient', 0:'Count'})

#Most common ingredients
sns.barplot(x = 'Count', 
            y = 'Ingredient',
            data = ingredients_individual.sort_values('Count', ascending=False).head(20))


# In[ ]:


df.ingredients


# In[ ]:


label = df.cuisine

features = df.drop(['cuisine'], axis = 1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(features, label, test_size = 0.20, random_state =0) 


# In[ ]:


train_ingredients_text = X_train.ingredients.apply(lambda s: ' '.join(w.lower() for w in s)).str.replace('[^\w\s]','')
test_ingredients_text = X_test.ingredients.apply(lambda s: ' '.join(w.lower() for w in s)).str.replace('[^\w\s]','')  


# # Term Frequency, Inverse Document Frequency

# In[ ]:


tfidf = TfidfVectorizer(
    min_df = 3,
    max_df = 0.95,
    stop_words = 'english'
)

tfidf.fit(train_ingredients_text)
text = tfidf.transform(train_ingredients_text)
text


# In[ ]:


traintext = tfidf.transform(test_ingredients_text)


# # Random Forest Classifier (Ensemble Learning)

# In[ ]:


clf = RandomForestClassifier(n_estimators=100, max_depth=16,random_state=0)
clf.fit(text, y_train)


# ### Random Forest Train Data Accuracy

# In[ ]:


y_pred= clf.predict(traintext)


# In[ ]:


accuracy_score(y_test,y_pred)*100 


# In[ ]:


y_pred=clf.predict(text)
accuracy_score(y_train,y_pred)*100 


# In[ ]:


clf.score(text,y_train)


# ### Random Forest Test Data Accuracy

# In[ ]:


clf.score(traintext,y_test)


# # Decision Tree Classifier

# In[ ]:


clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=16, min_samples_leaf=5) 
clf_gini.fit(text, y_train)


# ### Decision Tree Training Data Score

# In[ ]:


clf_gini.score(text,y_train)


# ### Decision Tree Test Data Score

# In[ ]:


clf_gini.score(traintext,y_test)


# ## Cross Validation for Decision Tree

# In[ ]:


crossvalidation = df.ingredients.apply(lambda s: ' '.join(w.lower() for w in s)).str.replace('[^\w\s]','')
text1 = tfidf.transform(crossvalidation)
scores = cross_val_score(clf_gini, text1, label, cv=5)
scores


# ## Cross Validation for Random Forest

# In[ ]:


scores = cross_val_score(clf, text1, label, cv=5)
scores


# # For Testing on the test data provided by kaggle

# In[ ]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
# parameters = {'C': np.arange(1, 100, 5)}
model = LinearSVC()
# model = LogisticRegression(multi_class='multinomial')
# model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
# model = SVC()

model = OneVsRestClassifier(model)
# model = BaggingRegressor(model, n_estimators=100)
# model = GridSearchCV(model, parameters, n_jobs=-1, verbose=2, cv=3)

print(cross_val_score(model, text, y_train, cv=3)) 


# In[ ]:


model.fit(text, y_train)
model.score(traintext, y_test)


# In[ ]:


df1=pd.read_json("../input/test.json")
df1.head()


# In[ ]:


predicting = df1.ingredients.apply(lambda s: ' '.join(w.lower() for w in s)).str.replace('[^\w\s]','')
textpre = tfidf.transform(predicting)
predicted= model.predict(textpre)


# In[ ]:


print(predicted)


# In[ ]:


sub=pd.read_csv("../input/sample_submission.csv")
sub.head()
del sub['cuisine']
sub.head()


# In[ ]:


sub['cuisine']=predicted
sub.head()


# In[ ]:


sub.to_csv("Submission.csv",index=False)


# In[ ]:




