#!/usr/bin/env python
# coding: utf-8

# # Predicting Fraudulency based on Job Posts.
# 1. Visualizing **Missing Values**
# 2. **WORDCLOUD** on Job Titles
# 3. **Splitting** Locations into Country, State & City
# 4. Converting salary ranges into **Min & Max**
# 5. **Label Encoding** Categorical Features
# 6. Cleaning Text Features by removing **STOPWORDS** and **Lemmatizing** Words using **NLP**
# 7. **OVERSAMPLING** Target Variable
# 8. Scaling Data using **MINMAXSCALER**
# 9. Plotting **AUC** and **Accuracies** of following Models:
#     * **Logistic Regression**
#     * **Support Vector Classifier**
#     * **MultiLayer Perceptron Classifier**
#     * **KNN Classifier**
#     * **Decision Tree Classifier**
#     * **XGBoost Classifier**
#     * **Random Forest Classifier**

# # IMPORTING LIBRARIES

# In[ ]:


import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random,matplotlib
import missingno as msno
import warnings
warnings.filterwarnings('ignore')
import nltk as nlp
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv('/kaggle/input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv')
df.head()


# # Visualizing Missing Values

# In[ ]:


msno.matrix(df)


# Plotting Barchart for Missing Values in our dataset

# In[ ]:


msno.bar(df)


# # WORDCLOUD

# In[ ]:


text = " ".join(title for title in df.title)
print ("There are {} words in the combination of all available job titles.".format(len(text)))
stopwords=set(STOPWORDS)
wordcloud = WordCloud(background_color="black",max_font_size=100, max_words=10000,width=1600, height=800,stopwords=stopwords,colormap=matplotlib.cm.cool).generate(text)
plt.figure(figsize=(20,10))
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis("off")


# In[ ]:


#Dropping 'job_id' as it is irrelevant to fraudulent
df.drop('job_id', axis=1, inplace=True)
text_features = ['title', 'company_profile', 'description', 'requirements', 'benefits']
complex_features = ['location', 'salary_range']
bin_features = ['telecommuting', 'has_company_logo', 'has_questions']
cat_features = ['department', 'employment_type', 'required_experience', 
                'required_education', 'industry', 'function']


# In[ ]:


df.isnull().sum()


# Filling Null values from Text Features with 'Unspecified'

# In[ ]:


for feature_name in text_features[1:]:
    df[feature_name].fillna('Unspecified', inplace=True)


# Filling Null values from location with 'Unspecified' <br>
# Splitting location into 3 seperate columns: country, state, city<br>
# Dropping location column<br>

# # Splitting Locations

# In[ ]:


location = df['location'].copy()
#splitting location
location_splitted = list(location.str.split(', ').values)
for loc_ind, loc in enumerate(location_splitted):
    if loc is np.nan:
        location_splitted[loc_ind] = ['Unpecified'] * 3
    else:
        for el_ind, el in enumerate(loc):
            if el == '':
                loc[el_ind] = 'Unpecified'
                
location_splitted = list(map(lambda loc: list(loc), location_splitted))
for loc_ind, loc in enumerate(location_splitted):
    if len(loc) > 3:
        location_splitted[loc_ind] = loc[:2] + [', '.join(loc[2:])]
    if len(loc) < 3:
        location_splitted[loc_ind] += ['Unpecified'] * 2
        
data_location = pd.DataFrame(location_splitted, columns=['country', 'state', 'city'])
cat_features += ['country', 'state', 'city']
df= pd.concat([df, data_location], axis=1)
df.drop('location', axis=1, inplace=True)
df.head()


# # Converting Salary Ranges

# Filling Null values from salary_range with '0-0'
# Splitting salary_range into 2 seperate columns: min_salary and max_salary
# Dropping salary_range

# In[ ]:


salary_range = df.salary_range.copy()
salary_range.fillna('0-0', inplace=True)
salary_range_sep = list(salary_range.str.split('-').values)
salary_range_sep[5538] = ['40000', '40000']
error_range_inds = []
for range_ind, s_range in enumerate(salary_range_sep):
    min_value, max_value = s_range
    if not min_value.isdigit() or not max_value.isdigit():
        error_range_inds += [range_ind]
for range_ind in error_range_inds:
    salary_range_sep[range_ind] = ['0', '0']
data_salary_range = pd.DataFrame(np.array(salary_range_sep, dtype='int64'), 
                                 columns=['min_salary', 'max_salary'])

num_features = ['min_salary', 'max_salary']
df = pd.concat([df, data_salary_range], axis=1)
df.drop('salary_range', axis=1, inplace=True)
df.head()


# Filling Null values from remaining features with 'Unspecified'

# In[ ]:


df.fillna('Unspecified', inplace=True)


# All the null values are cleaned.

# In[ ]:


df.info()


# # Cleaning text using NLP

# *clean_text* function to receive column as argument, apply regex functions, tokenizing and lemmatizing and returning the modified dataframe

# In[ ]:


def clean_text(data):
    description_list = []
    for description in data:
        description = re.sub("[^a-zA-Z]"," ",description)
        description = description.lower()
        description = nlp.word_tokenize(description)
        description = [word for word in description if not word in stopwords]
        lemma = nlp.WordNetLemmatizer()
        description = [lemma.lemmatize(word) for word in description ]
        description =" ".join(description)
        description_list.append(description)
    return description_list


# Applying this function on text variables

# In[ ]:


df['description_cleaned']= clean_text(df.description)
df['company_profile_cleaned']=clean_text(df.company_profile)
df['requirements_cleaned']= clean_text(df.requirements)
df['benefits_cleaned']=clean_text(df.benefits)


# Calculating length of each text feature's entries and saving them into feature_length

# In[ ]:


df['title_length']=df['title'].astype(str).str.split(' ').apply(len)
df['company_profile_length']=df['company_profile_cleaned'].astype(str).str.split(' ').apply(len)
df['benefits_length']=df['benefits_cleaned'].astype(str).str.split(' ').apply(len)
df['description_length']=df['description_cleaned'].astype(str).str.split(' ').apply(len)
df['requirements_length']=df['requirements_cleaned'].astype(str).str.split(' ').apply(len)


# # Label Encoding

# In[ ]:


label=LabelEncoder()
df['employment_type']=label.fit_transform(df['employment_type'])
df['required_experience']=label.fit_transform(df['required_experience'])
df['required_education']=label.fit_transform(df['required_education'])
df['industry']=label.fit_transform(df['industry'])
df['function']=label.fit_transform(df['function'])
df['country']=label.fit_transform(df['country'])
df['state']=label.fit_transform(df['state'])
df['city']=label.fit_transform(df['city'])


# Plotting Target Variable

# In[ ]:


plt.figure(figsize=(10, 5))
ax = sns.countplot(df.fraudulent)
plt.title('The distribution of the target feature (fraudulent)')
for p in ax.patches:
    ax.annotate(p.get_height(), (p.get_x()+0.33, p.get_height()))


# # Oversampling Target Variable

# In[ ]:


data_1f = df[df.fraudulent == 1]
original_data = df.copy()
df = pd.concat([df] + [data_1f] * 7, axis=0)


# In[ ]:


plt.figure(figsize=(10, 5))
ax = sns.countplot(df.fraudulent)
plt.title('The distribution of the target feature (fraudulent)')
for p in ax.patches:
    ax.annotate(p.get_height(), (p.get_x()+0.33, p.get_height()))
plt.show()


# Preparing X and y variables by removing unwanted features

# In[ ]:


X=check=df.drop(['title','department','company_profile','description','requirements','benefits','description_cleaned','company_profile_cleaned','requirements_cleaned','benefits_cleaned','fraudulent'],axis=1)
y=df.fraudulent


# In[ ]:


X


# In[ ]:


y


# # SCALING DATA with MINMAXSCALER

# In[ ]:


scaler=MinMaxScaler()
X=scaler.fit_transform(X)


# X

# # MODELING
# **1. LOGISTIC REGRESSION** <br>
# **2. XGBOOST**<br>
# **3. Logistic Regression**<br>
# **4. Support Vector Classifier**<br>
# **5. MultiLayer Perceptron Classifier**<br>
# **6. KNN Classifier**<br>
# **7. Decision Tree Classifier**<br>
# **8. XGBoost Classifier**<br>
# **9. Random Forest Classifier**<br>

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# Creating a function 'roc_plotter' for passing model object, model name and plotting its roc auc curve with accuracy

# In[ ]:


def roc_plotter(model_object,model_name):     
        model_object.fit(X_train, y_train)
        y_pred=model_object.predict(X_test)
        ns_probs = [0 for _ in range(len(y_test))]

        # predict probabilities
        model_probs = model_object.predict_proba(X_test)[:, 1]

        # calculate scores
        ns_auc = roc_auc_score(y_test, ns_probs)
        model_auc = roc_auc_score(y_test, model_probs)
    
        fig = plt.figure(figsize=(12,5))

        # calculate roc curves
        ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
        lr_fpr, lr_tpr, _ = roc_curve(y_test, model_probs)
        # plot the roc curve for the model
        plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        plt.plot(lr_fpr, lr_tpr, marker='.', label=model_name)

        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.legend()
        # show the plot
        score= accuracy_score(y_test, y_pred)
        txt1='ROC AUC = {}'.format(round(model_auc,2))
        txt2='Accuracy = {}%'.format(round(score*100,2))
        
        plt.text(0.3,0.2,model_name,fontsize=25, fontweight='bold',color='red')
        plt.text(0.3,0.1,txt1,bbox={'facecolor': 'orange','pad': 10},fontsize=15)
        plt.text(0.5,0.1,txt2,bbox={'facecolor': 'red', 'pad': 10},fontsize=15,color='white')
        plt.tight_layout()


# # Logistic Regression

# In[ ]:


lrmodel=LogisticRegression()
roc_plotter(lrmodel,'Logistic Regression')


# # Support Vector Classifier

# In[ ]:


svmmodel=SVC(probability=True)
roc_plotter(svmmodel,'Support Vector Classifier')


# # MultiLayer Perceptron Classifier

# In[ ]:


nnmodel=MLPClassifier()
roc_plotter(nnmodel,'MultiLayer Perceptron Classifier')


# # KNN Classifier

# In[ ]:


knnmodel=KNeighborsClassifier()
roc_plotter(knnmodel,'KNN Classifier')


# # Decision Tree Classifier

# In[ ]:


dtmodel=DecisionTreeClassifier()
roc_plotter(dtmodel,'Decision Tree Classifier')


# # XGBoost Classifier

# In[ ]:


xgb = XGBClassifier()
roc_plotter(xgb,'XGBoost Classifier')


# # Random Forest Classifier

# In[ ]:


rfmodel=RandomForestClassifier()
roc_plotter(rfmodel,'Random Forest Classifier')


# In[ ]:


models=['Logistic Regression','Support Vector Classifier','MultiLayer Perceptron Classifier','KNN Classifier','Decision Tree Classifier','XGBoost Classifier','Random Forest Classifier']
accuracies=[80.79,87.76,93.3,96.1,98.43,99.25,99.83]
plt.figure(figsize=(18,10))
plt.scatter(x=models, y=accuracies,s=200)
plt.plot(models,accuracies)
for x,y in zip(models,accuracies):
    label = "{:.2f}%".format(y)
    plt.annotate(label, (x,y), textcoords="offset points", xytext=(0,10), ha='center',fontsize=20)

