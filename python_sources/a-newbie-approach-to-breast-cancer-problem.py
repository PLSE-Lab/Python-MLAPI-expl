#!/usr/bin/env python
# coding: utf-8
This is my first attempt in machine learning and data science fields. I tried to apply what i learned in books and courses. So if there's an anomaly or somethink i was wrong on , please let me now. So let's jump to our problem.  


# In[ ]:


import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import seaborn as sns 
import warnings

warnings.filterwarnings("ignore")


# # Load data and take a quick look

# In[ ]:


df = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
print(df.head)


# In[ ]:


print(df.info()) 


# The last column is totaly empty so let's drop it.
# 

# In[ ]:


copied_data = df.dropna(axis=1)


# Now we have 31 feature to analyse .I don't think that it's wise to analyse them one by one. So my approach is to define the most correlated features with our target(diagnosis) and then visualise them. 

# # One hot encode the diagnosis 
# 
# I will encode the diagnosis feature into a binary one : 1 for malignant 0 for benign in order to determine the most correlated features. 

# In[ ]:


from sklearn.preprocessing import LabelBinarizer 


encoder = LabelBinarizer()
feature = copied_data['diagnosis']
encoded_feature = encoder.fit_transform(feature)
copied_data['diagnosis'] = encoded_feature
most_correlated = copied_data.corr().abs()['diagnosis'].sort_values(ascending=False)

#We will chose top 10 most correlated features
most_correlated = most_correlated[:10]
training_set = copied_data.loc[:, most_correlated.index]
print(most_correlated)


# # Visualizing 

# In[ ]:


sns.catplot(x="diagnosis", y="radius_mean", data=training_set)
plt.show()


# Okay so it's obvious that the larger the radius is the higher the possibility of the tumor to be malignant
# Don't forget malignant is encoded to 1 !

# In[ ]:


for feature in training_set.columns.values:
    sns.catplot(x='diagnosis', y=feature, data=training_set)
    


# In[ ]:


plt.figure(figsize=(10, 10))
sns.heatmap(training_set.corr(), annot=True, fmt='.0%')


# As you can see all the selected features are positively correlated to our target. Now we will check negatively correlated features.

# In[ ]:


corr_matrix = copied_data.corr()
print(corr_matrix['diagnosis'])


# Hummmm. As you can see there isn't interesting correlations here. Let's move to the next step. 

# # Preprocessing the training set
# 
# We will apply feature scalling and split our data

# In[ ]:


from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split

labels = training_set['diagnosis']
new_training_set = training_set.drop('diagnosis', axis=1)

X_train, X_test, y_train, y_test = train_test_split(new_training_set, labels, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# # Training the model

# In[ ]:


from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import cross_val_score

models = {'Logistic':LogisticRegression(), 'forest':RandomForestClassifier(n_estimators=10, criterion='gini', random_state=0),
         'tree':DecisionTreeClassifier(criterion='gini', random_state=0)}

trained_models = list()

for value in models.values():
    model = value
    model.fit(X_train, y_train)
    acc = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=10)
    print(acc.mean())
    trained_models.append(model)


# Okay so based on the results, i will choose the logistic regression model.

# # Accuracy on test set
# 
# Will choose the confusion matrics to calculate the accuracy on test data.

# In[ ]:


from sklearn.metrics import confusion_matrix 

predictions = trained_models[0].predict(X_test)
conf_matrix = confusion_matrix(y_test, predictions)

dataframe = pd.DataFrame(conf_matrix)
sns.heatmap(dataframe, annot=True, cbar=None, cmap="Reds")
plt.title("Confusion Matrix"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()


# Hum.. Nice! Our classifier is doing pretty well on data it has never seen! 
# Let's calculate the accuracy using the formula:
# accuracy = (True Positive + True Negative) / (True Positive + True Negative + False Positive + False Negative)
# 

# In[ ]:


accuracy = (dataframe[0][0] + dataframe[1][1]) / (dataframe[0][1] + dataframe[1][0]+ dataframe[0][0] + dataframe[1][1]) 
print('Test accuracy:', accuracy)


# 
