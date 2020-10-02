#!/usr/bin/env python
# coding: utf-8

# <h1><center><u>Google Play Store Exploratory Data analysis and Modelling</u></center></h1> 

# <b> Link-: </b> https://www.kaggle.com/lava18/google-play-store-apps/home

# <h2>2.0 Exploratory Data Analysis </h2>

# In[ ]:


#Importing Necessary Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix


# In[ ]:


#Reading data frame
df_app=pd.read_csv('../input/googleplaystore.csv')
df_app.head(5)


# In[ ]:


print("The Google play store dataset contains %d rows and %d columns." %(df_app.shape[0],df_app.shape[1]))


# <h2>2.1 Dropping Missing Values and Duplicates from data </h2>

# In[ ]:


#Dropping Data frame which has NAN values
df_app=df_app.dropna()
print("The Google play store dataset contains %d rows and %d columns after dropping NAN." %(df_app.shape[0],df_app.shape[1]))


# In[ ]:


#Checking if there are any duplicates rows present in dataset that has same App
# False= No duplicate
# True=Duplicate
df_app.duplicated(subset='App').value_counts()


# In[ ]:


#Dropping the duplicates
df_app=df_app.drop_duplicates(subset='App')


# In[ ]:


print("The Google play store dataset contains %d rows and %d columns after dropping NAN and duplicates." %(df_app.shape[0],df_app.shape[1]))


# <h2> 2.2 Data Pre-processing </h2>

# In[ ]:


#Checking the data types of dataset
df_app.dtypes


# Since the columns 'Installs','Price' are string we convert them into integer and float format respectively.

# In[ ]:


#Converting the Installs column into integer
df_app['Installs']=df_app['Installs'].apply(lambda a:a.split('+')[0])   #Removes '+' from Installs
se=df_app['Installs'].apply(lambda a:a.split(','))                      #Removes ',' from Installs 

def add_list(x):
    sum=' '
    for i in range(0,len(x)):
        sum+=x[i]
    return int(sum)  

df_app['Installs']=se.apply(lambda a:add_list(a))                      #Convert str to int values 
df_app.head(5)


# In[ ]:


#Removing Currency symbol from the Price and making it float
def remove_curr(x):
    if x !='0':
        x=x.split('$')[1]
    return float(x)   

df_app['Price']=df_app['Price'].apply(lambda a:remove_curr(a))  #Removes '$' from Price
df_app.head(5)


# <h2> 2.3.0 Plotting </h2>

# In[ ]:


#Checking the number of apps that available based on type: Free v/s Paid
df_app['Type'].value_counts()


# <h3> 2.3.1 Total Number of Apps available and are installed </h3> 

# In[ ]:


#Number of free and paid Apps available
plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
sns.countplot(x='Type',data=df_app)
plt.title("Number of Apps Available: Free v/s Paid")

#Most installed apps based on Category
plt.subplot(1,2,2)
sns.barplot(x='Type',y='Installs',data=df_app,ci=None)
plt.title("Number of Apps installed: Free v/s Paid")
plt.tight_layout()


# <h3> 2.3.2 Number of Apps available based on Catgeory </h3>

# In[ ]:


#Checking the number of Apps available on playstore based on category
plt.figure(figsize=(12,12))
sns.countplot(y='Category',data=df_app)
plt.title("Number of Apps available based on Category")


# <h3> 2.3.3 Number of Apps installed based on Category </h3>

# In[ ]:


#Most installed apps based on Category
plt.figure(figsize=(12,12))
sns.barplot(x='Installs',y='Category',data=df_app,ci=None)
plt.title("Number of Apps installed based on Category")


# <h3> 2.3.4 Number of Apps available based on Content rating </h3>

# In[ ]:


#Apps available based on Content rating
plt.figure(figsize=(10,10))
sns.countplot(x='Content Rating',data=df_app,)
plt.xticks(rotation=45)
plt.title("Number of Apps available based on Content rating")


# <h3> 2.3.5 Number of Apps installed based on Content rating </h3>

# In[ ]:


#Apps installed based on Content rating
plt.figure(figsize=(10,10))
sns.barplot(x='Content Rating',y='Installs',data=df_app,ci=None)
plt.xticks(rotation=45)
plt.title("Number of Apps installed based on Content rating")


# <h3>2.3.6 Android Versions available for most apps </h3>

# In[ ]:


#Android Version of the most available apps
plt.figure(figsize=(15,15))
sns.countplot(y='Android Ver',data=df_app)
plt.title("Android Version's available")


# <h3> 2.3.7 Android Versions of installed Apps </h3>

# In[ ]:


#Android  version of most installed apps
plt.figure(figsize=(15,15))
sns.barplot(x='Installs',y='Android Ver',data=df_app,ci=None)
plt.title("Android Versions of installed Apps")


# <h3> 2.3.8 Ratings of Apps v/s Number of installed </h3>

# In[ ]:


#Ratings of Apps and the number of installed
plt.figure(figsize=(15,15))
sns.barplot(y='Installs',x='Rating',data=df_app,ci=None)
plt.xticks(rotation=45)
plt.title("Number of Apps and ratings ")


# <h3>2.3.9 Top 20 Most downloaded Paid Apps </h3>

# In[ ]:


#Most download  Paid apps
df_type=df_app[df_app['Type']=='Paid']
df_type.sort_values(by='Installs',ascending=False)['App'].head(20)


# <h2> 3.0 Top 3 most common category of Apps that are installed </h2> <br>
#         1. Communication  
#         2. Social  
#         3. Video Players 

# <h3> 3.1 Top 20 apps that are most installed bases on category Communication </h3>

# In[ ]:


#Top 20 apps that are installed most in Category Communication
df_com=df_app[df_app['Category']=='COMMUNICATION']
df_com.sort_values(by='Installs',ascending=False)['App'].head(20)


# <h3>3.2 Top 20 Apps that are most installed based on category Social </h3>

# In[ ]:


#Top 20 apps that are installed most in Category Social
df_soc=df_app[df_app['Category']=='SOCIAL']
df_soc.sort_values(by='Installs',ascending=False)['App'].head(20)


# <h3>3.3 Top 20 Apps that are most installed based on category Video player </h3>

# In[ ]:


#Top 20 apps that are installed most in Category Video Player
df_vp=df_app[df_app['Category']=='VIDEO_PLAYERS']
df_vp.sort_values(by='Installs',ascending=False)['App'].head(20)


# <h2> 4.0 Machine Learning Modelling </h2>

# In[ ]:


#Reading CSV file that contains reviews for Apps
df=pd.read_csv('../input/googleplaystore_user_reviews.csv')
df.head(10)


# In[ ]:


#Size of data frame
df.shape


# In[ ]:


#Class labels available
df['Sentiment'].value_counts()


# <h3> 4.1 Reviews Pre-Processing </h3>

# In[ ]:


#Checking if there are any missing values
sns.heatmap(df.isna())


# In[ ]:


#Dropping the missing values from the data frame
df=df.dropna()
df.shape


# In[ ]:


#Reviews and Labels
reviews=df['Translated_Review']
labels=df['Sentiment']


# In[ ]:


# User-defined function
def cleanhtml(sentence): #function to clean the word of any html-tags
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', sentence)
    return cleantext
def cleanpunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    return cleaned


# In[ ]:


#Stop words and Lemmatizer
Stop=set(stopwords.words('english'))
WrdLem=WordNetLemmatizer()
print(Stop)


# In[ ]:


#Cleaning the reviews(removing html tags,punctuation,Lemmatizations)
Cleaned_sent=[]
for sent in reviews:
    r1=[]
    sent=cleanhtml(sent)
    sent=cleanpunc(sent)
    sent=sent.lower()
    for  word in sent.split():
        if ((word.isalpha()) & (len(word)>2)):
            if word not in Stop:
                w=WrdLem.lemmatize(word)
                r1.append(w)
            else:
                continue
        else:
            continue
    str1 = (" ".join(r1))        
     
    Cleaned_sent.append(str1)

df['Cleaned_text']=Cleaned_sent
df.head(5)    


# In[ ]:


#Defining some user defined function

def plot_cm_rates(y_test, Y_pred):

    #Plotting Confusion matrix
    x=confusion_matrix(y_test,Y_pred)
    cm_df=pd.DataFrame(x,index=['Negative','Neutral','Positive'],columns=['Negative','Neutral','Positive'])

    sns.set(font_scale=1,color_codes=True,palette='deep')
    sns.heatmap(cm_df,annot=True,annot_kws={"size":16},fmt='d',cmap="YlGnBu")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix ")


def plot_miss_error(cv_scores,hyperparam):
    
    # changing to misclassification error
    MSE = [1 - x for x in cv_scores]

    # determining best k
    optimal_k = hyperparam[MSE.index(min(MSE))]
    print('\nThe optimal value of hyper parameter is %f.' % optimal_k)
    
    # plot misclassification error vs K 
    plt.figure(figsize=(8,8))
    plt.plot(hyperparam, MSE)

    for xy in zip(hyperparam, np.round(MSE,3)):
        plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')

    plt.xlabel('Values of Hyperparameter')
    plt.ylabel('Misclassification Error')
    plt.title("Missclassification error v/s Hyperparameter")
    plt.show()
    
    return optimal_k


def train_test_accuracy(Classifier,X_train,y_train,X_test,y_test):
    
    #Train Model Fitting
    Classifier.fit(X_train,y_train)
    pred_train = Classifier.predict(X_train)
    
    #Train Accuracy
    train_acc = accuracy_score(y_train, pred_train, normalize=True) * float(100)
    
    #Test Accuracy
    pred_test = Classifier.predict(X_test)
    test_acc = accuracy_score(y_test, pred_test, normalize=True) * float(100)
    
    #Printing train and test accuracy
    print('\n****Train accuracy = %f%%' % (train_acc))
    print('\n****Test accuracy =  %f%%' % (test_acc))
    
    #plotting Confusion matrix
    plot_cm_rates(y_test,pred_test)


# <h3> 4.3 Splitting the data into train and test split </h3>

# In[ ]:


#Splitting the data into train and test
X_train,X_test,y_train,y_test=train_test_split(df['Cleaned_text'].values,labels,test_size=0.3,random_state=0)


# In[ ]:


#Size of training and test data
print("The number of data points used in  training model is %d "%(X_train.shape[0]))
print("The number of data points used in testing model is %d" %(X_test.shape[0]))


# <h3>4.4.0 Bag of Words (Converting text into vectors) </h3>

# In[ ]:


#Train Vector
bow=CountVectorizer()
X_train_bow=bow.fit_transform(X_train)

#Test vector
X_test_bow=bow.transform(X_test)


# <h3> 4.4.1 Logistic Regression </h3>

# In[ ]:


#Hyper-Parameter 
C=[10**-4,10**-2,10**0,10**2,10**4]


# In[ ]:


#Hyper Parameter tunning
cv_scores=[]
for c in C:
    LR=LogisticRegression(C=c,solver='newton-cg',multi_class='ovr')
    scores=cross_val_score(LR,X_train_bow,y_train,cv=3,scoring='accuracy')
    cv_scores.append(scores.mean())
    


# In[ ]:


#Plotting Misclassification error
optimal=plot_miss_error(cv_scores,C)


# In[ ]:


#Model Fitting based on  optimal value and Plotting Confusion Matrix
classifier1=LogisticRegression(C=optimal,solver='newton-cg',multi_class='ovr')

train_test_accuracy(classifier1,X_train_bow,y_train,X_test_bow,y_test)


# > <h3> 4.4.2 XGB Classifier </h3>

# In[ ]:


#Hyper parameter 
lr=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]


# In[ ]:


#Hyper paramete tunning
cv_scores=[]
for l in lr:
    XGB=XGBClassifier(learning_rate=l)
    scores=cross_val_score(XGB,X_train_bow,y_train,cv=3,scoring='accuracy')
    cv_scores.append(scores.mean())
    


# In[ ]:


#Plotting misclassification error
optimal=plot_miss_error(cv_scores,lr)


# In[ ]:


#Model Fitting based on  optimal value and plotting the confusion matrix
classifier1=XGBClassifier(learning_rate=optimal)
train_test_accuracy(classifier1,X_train_bow,y_train,X_test_bow,y_test)


# <h3> 4.5.0 TF-IDF (Converting text to vectors) </h3>

# In[ ]:


#Train vector
tfidf=TfidfVectorizer()
X_train_tfidf=tfidf.fit_transform(X_train)

#Test Vector
X_test_tfidf=tfidf.transform(X_test)


# <h3> 4.5.1 Logistic Regression </h3>

# In[ ]:


#Hyper Parameter tunning
cv_scores=[]
for c in C:
    LR=LogisticRegression(C=c,multi_class='ovr',solver='newton-cg')
    scores=cross_val_score(LR,X_train_tfidf,y_train,cv=3,scoring='accuracy')
    cv_scores.append(scores.mean())
    


# In[ ]:


#Plotting misclassification error
optimal=plot_miss_error(cv_scores,C)


# In[ ]:


#Model Fitting based on  optimal value and Confusion matrix
classifier1=LogisticRegression(C=optimal,multi_class='ovr',solver='newton-cg')

train_test_accuracy(classifier1,X_train_tfidf,y_train,X_test_tfidf,y_test)


# <h3> 4.5.2 XGB Classifier </h3>

# In[ ]:


#Hyper parameter
cv_scores=[]
for l in lr:
    XGB=XGBClassifier(learning_rate=l)
    scores=cross_val_score(XGB,X_train_tfidf,y_train,cv=3,scoring='accuracy')
    cv_scores.append(scores.mean())
    


# In[ ]:


#Plotting misclassification error
optimal=plot_miss_error(cv_scores,lr)


# In[ ]:


#Model Fitting based on  optimal value and Confusion Matrix
classifier1=XGBClassifier(learning_rate=optimal)

train_test_accuracy(classifier1,X_train_tfidf,y_train,X_test_tfidf,y_test)


# <h2> 5.0 Conclusion: </h2>

# * Exploratory data analyis of the Google play store data was done.
# * Top Apps that are most downloaded is found out.
# * Machine Learning models are applied to the reviews of the App's to predict whether given review is Positive,Neutral or Negative.
# * Also hyper parameter tuning of the model was also done to find the best parameter. The details of it is shown below.

# <h3> With Bag of Words represetation </h3>

# <table style="width:75%">
#   <tr>
#     <th>Model</th>
#     <th>Hyperparameter</th> 
#     <th>Train Accuracy(%)</th> 
#     <th>Test Accuracy(%)</th>
#   </tr>
#   <tr>
#     <td>Logistic Regression</td>
#     <td>C=1</td> 
#     <td>96.77</td> 
#     <td>91.82</td>
#   </tr>
#   <tr>
#     <td>XGB Classifier</td>
#     <td>learning_rate=1</td> 
#     <td>93.11</td>
#     <td>91.09</td>
#   </tr>
# </table>

# <h3> With TF-IDF representation </h3>

# <table style="width:75%">
#   <tr>
#     <th>Model</th>
#     <th>Hyperparameter</th> 
#     <th>Train Accuracy(%)</th> 
#     <th>Test Accuracy(%)</th>
#   </tr>
#   <tr>
#     <td>Logistic Regression</td>
#     <td>C=100</td> 
#     <td>98.74</td> 
#     <td>89.14</td>
#   </tr>
#   <tr>
#     <td>XGB Classifier</td>
#     <td>learning_rate=1</td> 
#     <td>93.88</td>
#     <td>90.79</td>
#   </tr>
# </table>

# In[ ]:




