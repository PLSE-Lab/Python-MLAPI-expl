#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')


# In[ ]:


data = pd.read_csv('../input/sms-spam-collection-dataset/spam.csv', encoding='latin-1')


# In[ ]:


data.head()


# Removing unnamed columns and renaming the columns v1 and v2

# In[ ]:


data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace=True)
data.rename(columns={'v1':'Label', 'v2':'Message',}, inplace=True)


# In[ ]:


data.head()


# In[ ]:


sns.countplot('Label', data = data)


# In[ ]:


data.groupby('Label').describe()


# There are more 'ham' or 'not-spam' messages than 'spam' messages.
# 
# 

# Selecting the features and labels.

# In[ ]:


X = data['Message']
y = data['Label']


# Splitting the data into training and testing sets before extracting features from the test data and building machine learning models.

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)


# Extracting features from the text data using tfidfvectorizer from sklearn.

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


tfidf_vect = TfidfVectorizer()


# The vectorizer is fit on the training data. Later the training and testing data is transformed using the vectorizer.

# In[ ]:


X_train = tfidf_vect.fit_transform(X_train)
X_test = tfidf_vect.transform(X_test)


# We can also see the features/tokens identified by the vectorizer by accessing the get_feature_names method.

# In[ ]:


len(tfidf_vect.get_feature_names())


# In[ ]:


# tfidf_vect.get_feature_names()


# As we can see most of them are numbers, abbreviations(short cuts). Also notice that we are not restricting the stopwords here as they might play an important role in classifying a message to be 'spam' or 'ham'.  
# 
# We will also see the wether removing stopwords might help with the classification task in the later step.

# Let us import different models and evaluation metrics. 

# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier



lr = LogisticRegression()
nb = MultinomialNB()
knc = KNeighborsClassifier()
svc = SVC(gamma = 'auto')
dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier(n_estimators=100)
gbc = GradientBoostingClassifier()
abc = AdaBoostClassifier()



models = {'Logistic Regression':lr, 'Naive Bayes classifier':nb, 'k-nearest neighbors':knc, 
          'Support Vector Machine':svc, 'Decision Tree Classifier':dtc, 
          'Random Forest Classifier':rfc, 'Gradient Boosting Classifier':gbc, 'AdaBoost Classifier':abc}


# Writing a function to fit the model on training data and make predictions on test data.

# In[ ]:


def eval_model(model):
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    test_accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred), columns=['ham', 'spam'], index=['ham','spam'])
    
    return test_accuracy, conf_matrix


# In[ ]:


test_accuracies = []
confusion_matrices = []
for name, model in models.items():
    test_acc, conf_matrix = eval_model(model) 
    test_accuracies.append(test_acc)
    confusion_matrices.append(conf_matrix)
    print(f'{name} ---> Test accuracy - {test_acc*100:.2f}%')


# In[ ]:


results = pd.DataFrame(test_accuracies, index=list(models.keys()), columns=['test_acc'])
results


# In[ ]:


plt.figure(figsize=(10, 6))
sns.barplot(x ='test_acc', y=results.index, data=results)
plt.xlim(0.85, 1.0)
plt.title('Performance comparision')
plt.show()


# Looks like SVM and K-nearest neighbors are not a good choice for this task. The performance of all the other models is almost similar in this case.

# Now let us include the effect of stopwords by passing the 'english' keyword to the stopwords argument to the tfidf vectorizer.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

tfidf_vect = TfidfVectorizer(stop_words='english')

X_train = tfidf_vect.fit_transform(X_train)
X_test = tfidf_vect.transform(X_test)


# In[ ]:


len(tfidf_vect.get_feature_names())


# Notice a decrease in number of features of the vectorizer after removing the stopwords from 7206 to 6946.
# 
# 
# Lets initialize the same models again to fit them on the features without stopwords.

# In[ ]:


lr = LogisticRegression()
nb = MultinomialNB()
knc = KNeighborsClassifier()
svc = SVC(gamma = 'auto')
dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier(n_estimators=100)
gbc = GradientBoostingClassifier()
abc = AdaBoostClassifier()



models = {'Logistic Regression':lr, 'Naive Bayes classifier':nb, 'k-nearest neighbors':knc, 
          'Support Vector Machine':svc, 'Decision Tree Classifier':dtc, 
          'Random Forest Classifier':rfc, 'Gradient Boosting Classifier':gbc, 'AdaBoost Classifier':abc}


# In[ ]:


test_accuracies_no_stopwords = []
confusion_matrices_no_stopwords = []
for name, model in models.items():
    test_acc, conf_matrix = eval_model(model) 
    test_accuracies_no_stopwords.append(test_acc)
    confusion_matrices_no_stopwords.append(conf_matrix)
    print(f'{name} ---> Test accuracy - {test_acc*100:.2f}%')


# In[ ]:


results['test_acc_without_stopwords'] = pd.Series(test_accuracies_no_stopwords, index=list(models.keys()))
results


# As we can see from the results there is not much of an improvment after removing stopwords.

# In[ ]:


def plot_confusion_matrices(models, confusion_matrices):
    fig, axs = plt.subplots(2,4, figsize=(10,5)) 

    m = 0
    for i, ax_r in enumerate(axs):
        for j, ax in enumerate(ax_r):
            sns.heatmap(confusion_matrices[m], annot=True, cbar=False, cmap='Blues', fmt='g', ax = ax)
            ax.set_xlabel('Predicted label')
            ax.set_ylabel('True label')
            ax.set_title(f'{list(models.keys())[m]}', fontsize=12, fontweight='bold')
            m += 1

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                        wspace=0.35)
    plt.tight_layout()
    plt.show()
    


# In[ ]:


plot_confusion_matrices(models, confusion_matrices)


# Finally let us build a pipeline to fit and predict on raw text data.

# In[ ]:


from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('tfidf_vect', TfidfVectorizer()),
    ('classifier', RandomForestClassifier(n_estimators=100))
])


# Splitting the data oncemore, to make sure the 'eval_model' function uses the right data for fitting and evaluating the model.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# We can use the same function to fit and predict with the pipeline

# In[ ]:


test_acc, conf_matrix = eval_model(pipeline) 

print('Test accuracy - ',test_acc)
print('Confusion matrix - \n', conf_matrix)


# As seen from the confusion matrix, 39 'spam' messages are being predicted as 'ham' messages

# In[ ]:


print('Classification Report \n', classification_report(y_test, pipeline.predict(X_test)))


# Another good thing about pipeline is, we can directly make predictions on new raw text messages.
# 
# Lets create some text messages and predict if they are 'spam' or 'ham'.

# In[ ]:


messages = ['Thank you for subscribing! You will be notified when you win your 1 Million Dollar prize money! Please call our customer service representative on 0800012345 for further details ',
          'Hi, hope you are doing well. Please call me as soon as possible!']


# In[ ]:


pipeline.predict(messages)


# In[ ]:




