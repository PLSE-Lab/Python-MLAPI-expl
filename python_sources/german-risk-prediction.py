#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# <h1>Introduction:</h1>
# <p>The original dataset contains 1000 entries with 20 categorial/symbolic attributes prepared by Prof. Hofmann. In this dataset, each entry represents a person who takes a credit by a bank. Each person is classified as good or bad credit risks according to the set of attributes.

# <h1>Content</h1>
# It is almost impossible to understand the original dataset due to its complicated system of categories and symbols. Thus, I wrote a small Python script to convert it into a readable CSV file. Several columns are simply ignored, because in my opinion either they are not important or their descriptions are obscure. The selected attributes are:
# <div>
# <p>Age (numeric)
# <p>Sex (text: male, female)
# <p>Job (numeric: 0 - unskilled and non-resident, 1 - unskilled and resident, 2 - skilled, 3 - highly skilled)
# <p>Housing (text: own, rent, or free)
# <p>Saving accounts (text - little, moderate, quite rich, rich)
# <p>Checking account (numeric, in DM - Deutsch Mark)
# <p>Credit amount (numeric, in DM)
# <p>Duration (numeric, in month)
# <p>Purpose(text: car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others
# <p>Risk (Value target - Good or Bad Risk)

# <h1>load Modules</h1>

# In[ ]:


#Load the librarys
import pandas as pd #To work with dataset
import numpy as np #Math library
import seaborn as sns #Graph library that use matplot in background
import matplotlib.pyplot as plt #to plot some parameters in seaborn

#Importing the data
df = pd.read_csv("../input/german_credit_data.csv",index_col=0)


# In[ ]:



# Other Libraries
#from imblearn.datasets import fetch_datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections


# In[ ]:



df.head()


# In[ ]:


df.describe()


# In[ ]:


df.describe(include=["O"])


# In[ ]:


df.info()


# In[ ]:


df.shape


# In[ ]:


df.columns.values


# In[ ]:


print("Total number of case: {}".format(df.shape[0]))
print("Number of high Risk: {}".format(df[df.Risk == "bad"].shape[0]))
print("Number of low Risk: {}".format(df[df.Risk == "good"].shape[0]))


# In[ ]:


sns.countplot('Risk', data=df)
plt.title('Class Distributions \n ( No Risk ||  Risk)', fontsize=18)


# In[ ]:


# it's a library that we work with plotly
import plotly.offline as py 
py.init_notebook_mode(connected=True) # this code, allow us to work with offline plotly version
import plotly.graph_objs as go # it's like "plt" of matplot
import plotly.tools as tls # It's useful to we get some tools of plotly
import warnings # This library will be used to ignore some warnings
from collections import Counter # To do counter of some features


trace0 = go.Bar(
            x = df[df["Risk"]== 'good']["Risk"].value_counts().index.values,
            y = df[df["Risk"]== 'good']["Risk"].value_counts().values,
            name='Good credit'
    )

trace1 = go.Bar(
            x = df[df["Risk"]== 'bad']["Risk"].value_counts().index.values,
            y = df[df["Risk"]== 'bad']["Risk"].value_counts().values,
            name='Bad credit'
    )

data = [trace0, trace1]

layout = go.Layout(
    
)

layout = go.Layout(
    yaxis=dict(
        title='Count'
    ),
    xaxis=dict(
        title='Risk Variable'
    ),
    title='Target variable distribution'
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='grouped-bar')


# In[ ]:


df_good = df.loc[df["Risk"] == 'good']['Age'].values.tolist()
df_bad = df.loc[df["Risk"] == 'bad']['Age'].values.tolist()
df_age = df['Age'].values.tolist()

#First plot
trace0 = go.Histogram(
    x=df_good,
    histnorm='probability',
    name="Good Credit"
)
#Second plot
trace1 = go.Histogram(
    x=df_bad,
    histnorm='probability',
    name="Bad Credit"
)
#Third plot
trace2 = go.Histogram(
    x=df_age,
    histnorm='probability',
    name="Overall Age"
)

#Creating the grid
fig = tls.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],
                          subplot_titles=('Good','Bad', 'General Distribuition'))

#setting the figs
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig.append_trace(trace2, 2, 1)

fig['layout'].update(showlegend=True, title='Age Distribuition', bargap=0.05)
py.iplot(fig, filename='custom-sized-subplot-with-subplot-titles')


# In[ ]:


labels = ['good','bad']
size = df['Risk'].value_counts()
colors = ['lightgreen', 'orange']
explode = [0, 0.1]

plt.rcParams['figure.figsize'] = (7, 7)
plt.pie(size, colors = colors, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')
plt.title('A pie chart Representing the risk')
plt.axis('off')
plt.legend()
plt.show()


# In[ ]:


fig, ax = plt.subplots(2, 1, figsize=(18,10))

amount_val = df['Age'].values
time_val = df['Credit amount'].values

sns.distplot(amount_val, ax=ax[0], color='g')
ax[0].set_title('Distribution of Age', fontsize=14)
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.distplot(time_val, ax=ax[1], color='y')
ax[1].set_title('Distribution of credit amount', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])



plt.show()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

X = df.drop('Risk', axis=1)
y = df['Risk']
sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

for train_index, test_index in sss.split(X, y):
    print("Train:", train_index, "Test:", test_index)
    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]
    
    
    # Turn into an array
original_Xtrain = original_Xtrain.values
original_Xtest = original_Xtest.values
original_ytrain = original_ytrain.values
original_ytest = original_ytest.values


# See if both the train and test label distribution are similarly distributed
train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)
print('-' * 100)

print('Label Distributions: \n')
print(train_counts_label/ len(original_ytrain))
print(test_counts_label/ len(original_ytest))


# In[ ]:


df = df.sample(frac=1)

# amount of fraud classes 492 rows.
bad_df = df.loc[df['Risk'] == "bad"]
good_df = df.loc[df['Risk'] == "good"][:300]

normal_distributed_df = pd.concat([bad_df, good_df])

# Shuffle dataframe rows
new_df = normal_distributed_df.sample(frac=1, random_state=42)

new_df.head()


# In[ ]:


print('Distribution of the Classes in the subsample dataset')
print(new_df['Risk'].value_counts()/len(new_df))



sns.countplot('Risk', data=new_df, palette=colors)
plt.title('Equally Distributed Classes', fontsize=14)
plt.show()


# In[ ]:


df_good = new_df[new_df["Risk"] == 'good']
df_bad = new_df[new_df["Risk"] == 'bad']

fig, ax = plt.subplots(nrows=2, figsize=(12,8))
plt.subplots_adjust(hspace = 0.4, top = 0.8)

g1 = sns.distplot(df_good["Age"], ax=ax[0], 
             color="g")
g1 = sns.distplot(df_bad["Age"], ax=ax[0], 
             color='r')
g1.set_title("Age Distribuition", fontsize=15)
g1.set_xlabel("Age")
g1.set_xlabel("Frequency")

g2 = sns.countplot(x="Age",data=new_df, 
              palette="hls", ax=ax[1], 
              hue = "Risk")
g2.set_title("Age Counting by Risk", fontsize=15)
g2.set_xlabel("Age")
g2.set_xlabel("Count")
plt.show()


# In[ ]:


#Let's look the Credit Amount column
interval = (18, 25, 35, 60, 120)

cats = ['Student', 'Young', 'Adult', 'Senior']
new_df["Age_cat"] = pd.cut(new_df.Age, interval, labels=cats)


df_good = new_df[new_df["Risk"] == 'good']
df_bad = new_df[new_df["Risk"] == 'bad']


# In[ ]:


trace0 = go.Box(
    y=df_good["Credit amount"],
    x=df_good["Age_cat"],
    name='Good credit',
    marker=dict(
        color='#3D9970'
    )
)

trace1 = go.Box(
    y=df_bad['Credit amount'],
    x=df_bad['Age_cat'],
    name='Bad credit',
    marker=dict(
        color='#FF4136'
    )
)
    
data = [trace0, trace1]

layout = go.Layout(
    yaxis=dict(
        title='Credit Amount (US Dollar)',
        zeroline=False
    ),
    xaxis=dict(
        title='Age Categorical'
    ),
    boxmode='group'
)
fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='box-age-cat')


# In[ ]:


#First plot
trace0 = go.Bar(
    x = new_df[new_df["Risk"]== 'good']["Housing"].value_counts().index.values,
    y = new_df[new_df["Risk"]== 'good']["Housing"].value_counts().values,
    name='Good credit'
)

#Second plot
trace1 = go.Bar(
    x = new_df[new_df["Risk"]== 'bad']["Housing"].value_counts().index.values,
    y = new_df[new_df["Risk"]== 'bad']["Housing"].value_counts().values,
    name="Bad Credit"
)

data = [trace0, trace1]

layout = go.Layout(
    title='Housing Distribuition'
)


fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='Housing-Grouped')


# In[ ]:


f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24,24))

# Entire DataFrame
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax1)
ax1.set_title("Imbalanced Correlation Matrix \n (don't use for reference)", fontsize=14)


sub_sample_corr = new_df.corr()
sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax2)
ax2.set_title('SubSample Correlation Matrix \n (use for reference)', fontsize=14)
plt.show()


# In[ ]:


f, axes = plt.subplots(ncols=2, figsize=(20,4))

# Positive correlations (The higher the feature the probability increases that it will be a fraud transaction)
sns.boxplot(x="Risk", y="Duration", data=new_df, palette=colors, ax=axes[0])
axes[0].set_title('Duration vs Risk Positive Correlation')

sns.boxplot(x="Risk", y="Credit amount", data=new_df, palette=colors, ax=axes[1])
axes[1].set_title('credit vs Risk Positive Correlation')

plt.show()


# In[ ]:


from scipy.stats import norm

f, (ax1, ax2) = plt.subplots(1,2, figsize=(20, 6))

Duration_Risk_dist = new_df['Duration'].loc[new_df['Risk'] == "bad"].values
sns.distplot(Duration_Risk_dist,ax=ax1, fit=norm, color='#FB8861')
ax1.set_title('Duration Distribution \n (Risk)', fontsize=14)

credit_risk_dist = new_df['Credit amount'].loc[new_df['Risk'] == "bad"].values
sns.distplot(credit_risk_dist,ax=ax2, fit=norm, color='#56F9BB')
ax2.set_title('credit Distribution \n (Risk)', fontsize=14)

plt.show()


# In[ ]:


# Undersampling before cross validating (prone to overfit)

X = new_df.drop('Risk', axis=1)
X=pd.get_dummies(X)
y = new_df['Risk']


# In[ ]:


# Our data is already scaled we should split our training and test sets
from sklearn.model_selection import train_test_split

# This is explicitly used for undersampling.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:



# Turn the values into an array for feeding the classification algorithms.
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values


# In[ ]:



# Let's implement simple classifiers

classifiers = {
    "LogisiticRegression": LogisticRegression(),
    "KNearest": KNeighborsClassifier(),
    "Support Vector Classifier": SVC(),
    "DecisionTreeClassifier": DecisionTreeClassifier()
}


# In[ ]:


from sklearn.model_selection import cross_val_score


for key, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    training_score = cross_val_score(classifier, X_train, y_train, cv=5)
    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")


# we will improve it further.

# In[ ]:




