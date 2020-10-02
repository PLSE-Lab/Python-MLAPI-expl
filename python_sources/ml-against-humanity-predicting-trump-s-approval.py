#!/usr/bin/env python
# coding: utf-8

# # SEPTEMBER DATASET

# In[1]:




def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


# In[2]:


import pandas as pd
import numpy as np
import altair as alt
import seaborn as sns
from sklearn.metrics import roc_auc_score
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import plotly.plotly as py
df1 = pd.read_csv('../input/201709-CAH_PulseOfTheNation.csv')
df1hot = pd.read_csv('../input/201709-CAH_PulseOfTheNation.csv')
df2 = pd.read_csv('../input/201710-CAH_PulseOfTheNation.csv')
df2hot = pd.read_csv('../input/201710-CAH_PulseOfTheNation.csv')
df3 = pd.read_csv('../input/201711-CAH_PulseOfTheNation.csv')
df3hot = pd.read_csv('../input/201711-CAH_PulseOfTheNation.csv')
df1.head()


# In[3]:


df1['Do you approve or disapprove of how Donald Trump is handling his job as president?'].unique()


# # Visualization

# In[4]:


x = df1['Do you approve or disapprove of how Donald Trump is handling his job as president?'].unique()
y = [len(df1[df1['Do you approve or disapprove of how Donald Trump is handling his job as president?'] == 'Strongly disapprove']), len(df1[df1['Do you approve or disapprove of how Donald Trump is handling his job as president?'] == 'Somewhat Approve']), len(df1[df1['Do you approve or disapprove of how Donald Trump is handling his job as president?'] == 'Strongly Approve']), len(df1[df1['Do you approve or disapprove of how Donald Trump is handling his job as president?'] == 'Somewhat disapprove']), len(df1[df1['Do you approve or disapprove of how Donald Trump is handling his job as president?'] == 'Neither approve nor disapprove']), len(df1[df1['Do you approve or disapprove of how Donald Trump is handling his job as president?'] == 'DK/REF'])]

plt.xticks(rotation='vertical')
plt.bar(x,y)


# In[5]:


x = df1['Political Affiliation'].unique()
y = [len(df1[df1['Political Affiliation'] == 'Democrat']), len(df1[df1['Political Affiliation'] == 'Independent']), len(df1[df1['Political Affiliation'] == 'Republican']), len(df1[df1['Political Affiliation'] == 'DK/REF'])]
plt.xticks(rotation='vertical')
plt.bar(x,y)


# In[6]:


import altair
altair.Chart(df1hot, max_rows=10000).mark_bar().encode(x='Age Range',color='Age Range',column='Do you approve or disapprove of how Donald Trump is handling his job as president?',y='count(*)')
#altair.Chart(df1hot, max_rows=10000).mark_bar().encode(x='Age Range',color='Age Range',y='Do you approve or disapprove of how Donald Trump is handling his job as president?')


# In[7]:


import altair
altair.Chart(df1hot, max_rows=10000).mark_bar().encode(x='Political Affiliation',color='Political Affiliation',column='Do you approve or disapprove of how Donald Trump is handling his job as president?',y='count(*)')


# # Some Transformation

# In[8]:


df1 = df1.drop('Q5OTH1', axis=1)
df1 = df1.drop('Q6OTH1', axis=1)
df1 = df1.drop('Q7OTH1', axis=1)
df1 = df1.drop('q8x', axis=1)
df1 = df1.drop('q11x', axis=1)
df1 = df1.drop('q14x', axis=1)
df1 = df1.drop('q16x', axis=1)
df1.head()
df1hot = df1hot.drop('Q5OTH1', axis=1)
df1hot = df1hot.drop('Q6OTH1', axis=1)
df1hot = df1hot.drop('Q7OTH1', axis=1)
df1hot = df1hot.drop('q8x', axis=1)
df1hot = df1hot.drop('q11x', axis=1)
df1hot = df1hot.drop('q14x', axis=1)
df1hot = df1hot.drop('q16x', axis=1)


# ## Fill in numeric columns with median value

# In[9]:


df1['What percentage of the federal budget would you estimate is spent on scientific research?'] = df1['What percentage of the federal budget would you estimate is spent on scientific research?'].fillna(df1['What percentage of the federal budget would you estimate is spent on scientific research?'].median())

df1['Income'] = df1['Income'].fillna(df1['Income'].median())


df1hot['What percentage of the federal budget would you estimate is spent on scientific research?'] = df1hot['What percentage of the federal budget would you estimate is spent on scientific research?'].fillna(df1hot['What percentage of the federal budget would you estimate is spent on scientific research?'].median())

df1hot['Income'] = df1hot['Income'].fillna(df1hot['Income'].median())


# In[ ]:





# In[10]:


def get_y_list(vals):
    res = []
    for i in vals:
        val = i.lower()
        if 'neither' in val or 'dk' in val:
            res.append(2)
        elif 'disapprove' in val:
            res.append(0)
        else:
            res.append(1)
    return res


# ### Try label encoding

# In[11]:


#cols_to_label_encode = ['Gender', 'Do you approve or disapprove of how Donald Trump is handling his job as president?', 'True or false: the earth is always farther away from the sun in the winter than in the summer.','Do you think it is acceptable or unacceptable to urinate in the shower?','If you had to choose: would you rather be smart and sad, or dumb and happy?','Do you believe in ghosts?']
labelencoder = LabelEncoder()
for col in df1.columns:
    if col == 'Age' or col == 'Age Range' or col == 'Income':
        continue
    df1[col] = labelencoder.fit_transform(df1[col])
df1.head()
df1 = df1.drop('Age Range',axis=1)


# # One Hot - and remove rows that didn't answer the question we're trying to classify
# * This turns the problem from multi-class into a binary classification problem

# In[12]:


df1hot['Do you approve or disapprove of how Donald Trump is handling his job as president?'] = get_y_list(df1hot['Do you approve or disapprove of how Donald Trump is handling his job as president?'])

cols_to_transform = list(df1hot.columns)
cols_remove = ['Age', 'Age Range', 'Income', 'Do you approve or disapprove of how Donald Trump is handling his job as president?']
cols = [col for col in cols_to_transform if col not in cols_remove]

df1hot = pd.get_dummies( columns = cols, data=df1hot )
df1hot = df1hot.drop('Age Range', axis=1)


# In[13]:


df1hot.head()


# In[14]:


accs = []
aucs = []
cms = []


# # Train/Test split

# In[15]:


y = df1['Do you approve or disapprove of how Donald Trump is handling his job as president?'].values
X = df1.drop('Do you approve or disapprove of how Donald Trump is handling his job as president?',axis=1) 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.35,random_state=4) 


# # Decision Tree

# ## Try on label encoded data first (This doesn't go too well)

# In[16]:


model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)
ypred = model.predict(X_test)
accuracy_score(y_test, ypred)


# In[17]:


sns.heatmap(df1.corr())


# ## Use one hot encoded data from now on

# In[18]:


df1hot = df1hot.drop(df1hot[df1hot['Do you approve or disapprove of how Donald Trump is handling his job as president?'] == 2].index)
y = df1hot['Do you approve or disapprove of how Donald Trump is handling his job as president?'].values
X = df1hot.drop('Do you approve or disapprove of how Donald Trump is handling his job as president?',axis=1) 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=5) 
model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)
ypred = model.predict(X_test)
accs.append(accuracy_score(y_test, ypred))


# In[19]:


cms.append(confusion_matrix(y_test, ypred))


# In[20]:


aucs.append(roc_auc_score(y_test,ypred))


# # Random Forest

# In[21]:


model = RandomForestClassifier(max_depth=9)
model.fit(X_train, y_train)
ypred = model.predict(X_test)
accs.append(accuracy_score(y_test, ypred))


# In[22]:


cms.append(confusion_matrix(y_test, ypred))
aucs.append(roc_auc_score(y_test,ypred))


# # Naive Bayes

# In[23]:


model = GaussianNB()
model.fit(X_train, y_train)
ypred = model.predict(X_test)
accs.append(accuracy_score(y_test, ypred))


# In[24]:


cms.append(confusion_matrix(y_test, ypred))
aucs.append(roc_auc_score(y_test,ypred))


# # Support Vector Machine

# In[25]:


model = SVC()
model.fit(X_train, y_train)
ypred = model.predict(X_test)
accs.append(accuracy_score(y_test, ypred))
cms.append(confusion_matrix(y_test, ypred))
aucs.append(roc_auc_score(y_test,ypred))


# # Comparisons for October dataset

# In[26]:


labels = ['Decision Tree', 'Random Forest', 'Naive Bayes', 'Support Vector Machine']
plt.xticks(rotation='vertical')
plt.xlabel('Algorithms')
plt.ylabel('Accuracy Score')
plt.title('Accuracy By Algorithm')
plt.bar(labels,accs)


# In[27]:


plt.xticks(rotation='vertical')
plt.xlabel('Algorithms')
plt.ylabel('AUC Score')
plt.title('AUC Score By Algorithm')
plt.bar(labels,aucs)


# In[28]:


plot_confusion_matrix(cm = cms[0],normalize=True, target_names = ['approve', 'dissaprove'],title = "Confusion Matrix Decision Tree")


# In[29]:


plot_confusion_matrix(cm = cms[1],normalize=True, target_names = ['approve', 'dissaprove'],title = "Confusion Matrix Random Forest")


# In[30]:


plot_confusion_matrix(cm = cms[2],normalize=True, target_names = ['approve', 'dissaprove'],title = "Confusion Matrix Naive Bayes")


# In[31]:


plot_confusion_matrix(cm = cms[3],normalize=True, target_names = ['approve', 'dissaprove'],title = "Confusion Matrix SVM")


# # OCTOBER DATASET

# In[32]:


df2.head()


# ## Data Cleanup Again - this time for October survey

# In[33]:


df2 = df2.drop('Q5OTH1', axis=1)
df2 = df2.drop('Q6OTH1', axis=1)
df2 = df2.drop('q8x', axis=1)
df2 = df2.drop('q10x', axis=1)
df2 = df2.drop('Who would you prefer as president of the United States, Darth Vader or Donald Trump?', axis=1)
df2.head()


# In[34]:


df2['If you had to guess, what percentage of Republicans would say that they mostly agree with the beliefs of White Nationalists?'] = df2['If you had to guess, what percentage of Republicans would say that they mostly agree with the beliefs of White Nationalists?'].fillna(df2['If you had to guess, what percentage of Republicans would say that they mostly agree with the beliefs of White Nationalists?'].median())

df2['Income'] = df2['Income'].fillna(df2['Income'].median())

df2['If you had to guess, what percentage of Republicans would say yes to that question?'] = df2['If you had to guess, what percentage of Republicans would say yes to that question?'].fillna(df2['If you had to guess, what percentage of Republicans would say yes to that question?'].median())

df2['If you had to guess, what percentage of Democrats would say yes to that question?'] = df2['If you had to guess, what percentage of Democrats would say yes to that question?'].fillna(df2['If you had to guess, what percentage of Democrats would say yes to that question?'].median())


# In[ ]:





# # One Hot

# In[35]:


df2['Do you approve or disapprove of how Donald Trump is handling his job as president?'] = get_y_list(df2['Do you approve or disapprove of how Donald Trump is handling his job as president?'])
cols_to_transform = list(df2.columns)
cols_remove = ['Age', 'Age Range', 'Income', 'Do you approve or disapprove of how Donald Trump is handling his job as president?']
cols = [col for col in cols_to_transform if col not in cols_remove]

df2 = pd.get_dummies( columns = cols, data=df2 )
df2 = df2.drop('Age Range', axis=1)


# In[36]:


df2.head()


# ## Only 121 respondents didn't answer the approval question. So let's turn this from a multi-classification problem to a binary-classification problem by dropping all rows with No response - we're not trying to predict who didn't answer the question

# In[37]:


print(len(df2[df2['Do you approve or disapprove of how Donald Trump is handling his job as president?'] == 0]))
print(len(df2[df2['Do you approve or disapprove of how Donald Trump is handling his job as president?'] == 1]))
print(len(df2[df2['Do you approve or disapprove of how Donald Trump is handling his job as president?'] == 2]))


# In[38]:


df2 = df2.drop(df2[df2['Do you approve or disapprove of how Donald Trump is handling his job as president?'] == 2].index)


# In[39]:


df2['Do you approve or disapprove of how Donald Trump is handling his job as president?'].unique()


# # Train/Test split

# In[40]:


y = df2['Do you approve or disapprove of how Donald Trump is handling his job as president?'].values
X = df2.drop('Do you approve or disapprove of how Donald Trump is handling his job as president?',axis=1) 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.35,random_state=4) 


# In[41]:


accs = []
aucs = []
cms = []


# # Decision Tree

# 

# In[42]:


model = DecisionTreeClassifier(max_depth=8)
model.fit(X_train, y_train)
ypred = model.predict(X_test)
ac = accuracy_score(y_test, ypred)
accs.append(ac)
ac


# In[43]:


auc = roc_auc_score(np.array(y_test), np.array(ypred))
aucs.append(auc)
auc


# 

# In[44]:


cm = confusion_matrix(y_test, ypred)
cms.append(cm)
cm


# In[45]:


decision_tree_metrics = [ac,auc]


# # Random Forest

# In[46]:


model = RandomForestClassifier(max_depth=9)
model.fit(X_train, y_train)
ypred = model.predict(X_test)
ac = accuracy_score(y_test, ypred)
accs.append(ac)
ac


# In[47]:


cm = confusion_matrix(y_test, ypred)
cms.append(cm)
cm


# In[48]:


auc = roc_auc_score(y_test, ypred)
aucs.append(auc)
auc


# In[49]:


rf_metrics = [ac,auc]
rf_model = model


# # Naive Bayes

# In[50]:


model = GaussianNB()
model.fit(X_train, y_train)
ypred = model.predict(X_test)
ac = accuracy_score(y_test, ypred)
accs.append(ac)
ac


# In[51]:


cm = confusion_matrix(y_test, ypred)
cms.append(cm)
cm


# In[52]:


auc = roc_auc_score(np.array(y_test), np.array(ypred))
aucs.append(auc)
auc


# In[53]:


nb_metrics = [ac,auc]


# # Support Vector Machine

# In[54]:


model = SVC()
model.fit(X_train, y_train)
ypred = model.predict(X_test)
ac = accuracy_score(y_test, ypred)
accs.append(ac)
ac


# In[55]:


cm = confusion_matrix(y_test, ypred)
cms.append(cm)
cm


# In[56]:


auc = roc_auc_score(np.array(y_test), np.array(ypred))
aucs.append(auc)
auc


# In[57]:


sv_metrics = [ac,auc]


# # Comparisons for September dataset

# In[ ]:





# In[58]:


labels = ['Decision Tree', 'Random Forest', 'Naive Bayes', 'Support Vector Machine']
plt.xticks(rotation='vertical')
plt.xlabel('Algorithms')
plt.ylabel('Accuracy Score')
plt.title('Accuracy By Algorithm')
plt.bar(labels,accs)


# In[59]:


plt.xticks(rotation='vertical')
plt.xlabel('Algorithms')
plt.ylabel('AUC Score')
plt.title('AUC Score By Algorithm')
plt.bar(labels,aucs)


# In[ ]:





# In[60]:


plot_confusion_matrix(cm = cms[0],normalize=True, target_names = ['approve', 'dissaprove'],title = "Confusion Matrix Decision Tree")


# In[61]:


plot_confusion_matrix(cm = cms[1],normalize=True, target_names = ['approve', 'dissaprove'],title = "Confusion Matrix Random Forest")


# In[62]:


plot_confusion_matrix(cm = cms[2],normalize=True, target_names = ['approve', 'dissaprove'],title = "Confusion Matrix Naive Bayes")


# In[63]:


plot_confusion_matrix(cm = cms[3],normalize=True, target_names = ['approve', 'dissaprove'],title = "Confusion Matrix SVM")


# In[ ]:





# In[ ]:




