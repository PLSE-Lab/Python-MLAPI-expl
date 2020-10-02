#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings

#Setting the properties to personal preference
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
plt.rcParams["figure.figsize"] = (8,4)
warnings.filterwarnings("ignore")    # No major warnings came out on first run. So, I am ignoring the "deprecation" warnings instead of showing them the first time to keep the code clean

print(os.listdir("../input"))
df = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')


# In[ ]:


df.columns


# In[ ]:


df.drop('Serial No.', inplace=True, axis=1)
df.rename({'Chance of Admit ': 'Chance of Admit', 'LOR ':'LOR'}, axis=1, inplace=True)


# ## Exploratory Data Analysis (EDA)

# In[ ]:


df.tail()


# In[ ]:


df.info()


# >Data looks pretty clean and in the right data type format.

# In[ ]:


df.describe()


# >Lot of "cool" info from the .describe(). Let's dive deeper and see this info for each "University Rating"

# ## Stats for each University Ranking (1-5)

# In[ ]:


for rating in sorted(df['University Rating'].unique()):
    print("For University Rating: ", rating, "\n")
    print(df[df['University Rating']==rating].describe(), 2*"\n")


# >Surprising to see that even at the highest rated universities the students with GRE score as low as 303 have 61% chance of admitting in.
# Let's do some more EDA to dissect the data.

# In[ ]:


for rating in sorted(df['University Rating'].unique()):
    sns.jointplot(data=df[df['University Rating']==rating], x = 'GRE Score', y = 'Chance of Admit')
    print("Jointplot for the University Rating: ", rating)
    plt.show()


# In[ ]:


for rating in sorted(df['University Rating'].unique()):
    sns.distplot(df[df['University Rating']==rating]['GRE Score'], hist=False)
plt.show()


# >'GRE Score' "seems" to be a pretty good indicator for Chances of getting in.
# Let's create a heatmap to confirm our hypothesis.

# In[ ]:


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True)


# > Top 3 factors that influence the Chances of Admit are **'CGPA', 'GRE Score', 'TOEFL Score'** in that order.

# In[ ]:


sns.pairplot(df)


# ## Time to implement some Machine Learning to predict 'Chance of Admit'. 

# Looking at the heatmap (above), it seems like there are multiple factors affecting the 'Chance of Admit'. So, let's start with Logistic Regression and compare the accuracy to that of SVM's

# In[ ]:


features = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']
y = df['Chance of Admit']
X = df[features]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


y_train_label = [1 if each > 0.8 else 0 for each in y_train]
y_test_label  = [1 if each > 0.8 else 0 for each in y_test]


# > I am using 80% 'Chance of Admit' as my cut off point to change Chance of Admit from float to binary labels.

# # GridSearch for the Best Model

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


# Create a pipeline
pipe = Pipeline([('classifier', LogisticRegression())])

# Create space of candidate learning algorithms and their hyperparameters
search_space = [{'classifier': [LogisticRegression()]},
        {'classifier': [SVC()]},
        {'classifier': [KNeighborsClassifier(n_neighbors=5)]}]

# Create grid search 
clf = GridSearchCV(pipe, search_space)

# Fit grid search
best_model = clf.fit(X_train, y_train_label)
# View best model
print(best_model.best_estimator_.get_params()['classifier'], "\n")
print("Accuracy of our best model is", clf.score(X_test, y_test_label)*100, "%", "\n")
print("Classification Report:", "\n", classification_report(y_test_label, best_model.predict(X_test)))


# > Looks like SVC gave the best accuracy (94.55 %) out of the three models.

# # Individual Model Performance
# ## 1. Logistic Regression

# In[ ]:


lg = LogisticRegression()

lg.fit(X_train, y_train_label)
predictions = lg.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test_label, predictions))


# > 92% overall accuracy seems fine. Let's try SVM now.

# ## 2. Support Vector Machine (SVM)

# In[ ]:


svmmodel = SVC()
svmmodel.fit(X_train,y_train_label)
y_pred_svm = svmmodel.predict(X_test)

print(classification_report(y_test_label, y_pred_svm))


# > 95% accuracy!!! Better than our Logistic Regression model. 
# LR is more sensitive to outliers than SVM because the cost function of LR diverges faster than those of SVM. That might be the reason why SVM is giving a better accuracy than Logistic Regression. 

# ## 3. KNeighborsClassifier

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train,y_train_label)
y_pred_knn = knn.predict(X_test)

print(classification_report(y_test_label, y_pred_knn))


# > 93% accuracy gives it the second place behind SVM

# ## Continously improving the notebook. Revisit later for more...

# In[ ]:




