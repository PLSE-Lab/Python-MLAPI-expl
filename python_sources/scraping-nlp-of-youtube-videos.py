#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv
#from apiclient.discovery import build
#DEVELOPER_KEY = 'API KEY'
#youtube = build('youtube', 'v3', developerKey=DEVELOPER_KEY)


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import re 
import nltk 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer 


# In[ ]:


def clean_data(col):
    try:
        col = re.sub('[^a-zA-Z]', ' ',col) 
        col = col.lower()            
        col = col.split()            
        lemmatizer = WordNetLemmatizer()             
        col = [lemmatizer.lemmatize(word) for word in col if not word in set(stopwords.words('english'))]            
        col = ' '.join(col)
    except Exception as e:
        print('col- ',col)
        print(e)
        col = 'Error'
    return(col)
    
#data['Title'] = data['Title'].apply(clean_data)


# In[ ]:


list_title = []
list_description = []
def get_list(cols):
    list_title.append(cols[0])
    list_description.append(cols[1])


# In[ ]:


#below is the code to scrap youtube videos
'''
#using youtube api to get to the list of videos
from apiclient.discovery import build
from apiclient.errors import HttpError


DEVELOPER_KEY = "API KEY"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"




def youtube_search(q, max_results=50,order="relevance", token=None, location=None, location_radius=None):
    
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
                            developerKey=DEVELOPER_KEY)

    search_response = youtube.search().list(
        q=q,
        type="video",
        pageToken=token,
        order = order,
        part="id,snippet",
        maxResults=max_results,
        location=location,
        locationRadius=location_radius
        ).execute()



    videos = []

    for search_result in search_response.get("items", []):
        
        if search_result["id"]["kind"] == "youtube#video":
                        
            videos.append(search_result)
    
    try:
        #token is used to jump to the next page on youtube to search for more videos
        nexttok = search_response["nextPageToken"]
        return(nexttok, videos)
    except Exception as e:
        nexttok = "last_page"
        return(nexttok, videos)
'''


# In[ ]:


#test = youtube_search("spinners")


# In[ ]:


#print(test[1][2])


# In[ ]:


#token = test[0]
#test_2 = youtube_search("spinners", token=token)


# In[ ]:


#making a csv file and writing a header
'''csv_file = open('YouTube Titles and description using youtube api.csv','w',encoding="utf-8",newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Video Id','Title', 'Description','Category'])'''


# In[ ]:


'''
#the idea was to get 1700 videos but because there is limit to free youtube api queries, I was bound to stop at 200 videos
def get_1700vid(list_of_strings):
    for string in list_of_strings:
        counter=0
        category = string
        token = None
        while((token!="last_page") & (counter<=200)):
            result = youtube_search(string,token=token)
            token = result[0]
            for i in range(0,len(result[1])):
                counter+=1
                title = result[1][i]['snippet']['title']
                vid_id = result[1][i]['id']['videoId']
                videos = youtube.videos().list(id=vid_id, part='snippet').execute()
                for video in videos.get('items', []):
                    description = video['snippet']['description']
                print(vid_id,'-',title,'-',category,'-',counter)
                csv_writer.writerow([vid_id,title, description,category])
    return    
'''


# In[ ]:


#list_of_topics=['travel','science','food','manufacturing','history','music']
#get_1700vid(list_of_topics)


# In[ ]:


#csv_file.close()


# In[ ]:


#this is the csv file which was populated with the scraped youtuble videos
df = pd.read_csv('../input/YouTube Titles and description using youtube api.csv')


# In[ ]:


df.info()


# In[ ]:


df.head()


# In[ ]:


df.isnull().values.any()


# In[ ]:


null_columns=df.columns[df.isnull().any()]
df[null_columns].isnull().sum()


# In[ ]:


#wherever there is null description, I've used the title to replace it. This was the most appropriate way I could think of
df['Description'] = df.apply(lambda row: row['Title'] if pd.isnull(row['Description']) else row['Description'],axis=1)


# In[ ]:


df.isnull().values.any()


# In[ ]:


df['Title'] = df['Title'].apply(clean_data)


# In[ ]:


df.head()


# In[ ]:


df['Description'] = df['Description'].apply(clean_data)


# In[ ]:


df.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[ ]:


le_cat = le.fit_transform(df['Category'])


# In[ ]:


series = pd.Series(le_cat)


# In[ ]:


df = pd.concat([df,series],axis=1)


# In[ ]:


df.head()


# In[ ]:


df.columns=['Video_id','Title','Description','Category','category']


# In[ ]:


df = df.drop('Category',axis=1)


# In[ ]:


df.head()


# In[ ]:


#to get a list of title and description to be used for bag of words later
df[['Title','Description']].apply(get_list,axis=1)


# In[ ]:


len(list_description)


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer   
cv = CountVectorizer(max_features = 1500) 
X = cv.fit_transform(list_description, list_title).toarray() 
y = df['category'].values


# In[ ]:


X.shape


# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit_transform(X)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tfidf_transformer, y, test_size = 0.20, random_state = 0)


# In[ ]:


#Used RFC
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy')
classifier.fit(X_train, y_train)


# In[ ]:


y_pred = classifier.predict(X_test)
classifier.score(X_test, y_test)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[ ]:


#sing SVC
from sklearn.svm import SVC


# In[ ]:


param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}


# In[ ]:


from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,verbose=3)


# In[ ]:


grid.fit(X_train,y_train)


# In[ ]:


grid.best_params_


# In[ ]:


grid.best_estimator_


# In[ ]:


grid_predictions = grid.predict(X_test)


# In[ ]:


print(classification_report(y_test,grid_predictions))


# In[ ]:


#using logistic regression
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()


# In[ ]:


log_model.fit(X_train,y_train)


# In[ ]:


predictions_log = log_model.predict(X_test)


# In[ ]:


print(classification_report(y_test,predictions_log))


# SVM proves to be the most superior of them. 

# In[ ]:


X_train.shape


# In[ ]:


#using neural network
import tensorflow as tf
from tensorflow import keras


# In[ ]:


#building the model
model = keras.Sequential([
    keras.layers.Dense(128,input_shape=(1500,),activation=tf.nn.relu),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(128,activation=tf.nn.relu),
    keras.layers.Dense(6, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


model.fit(X_train,y_train,epochs=10)


# In[ ]:


test_loss, test_acc = model.evaluate(X_test, y_test)

print('Test accuracy:', test_acc)


# In[ ]:


from sklearn.ensemble import BaggingClassifier
from sklearn import tree
model = BaggingClassifier(tree.DecisionTreeClassifier(random_state=1))
model.fit(X_train, y_train)
bagging_pred = model.predict(X_test)


# In[ ]:


print(classification_report(y_test,bagging_pred))


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(random_state=1)
model.fit(X_train, y_train)
model.score(X_test,y_test)


# In[ ]:


boost_pred = model.predict(X_test)


# In[ ]:


print(classification_report(y_test,boost_pred))


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
model= GradientBoostingClassifier(learning_rate=0.01,random_state=1)
model.fit(X_train, y_train)
model.score(X_test,y_test)


# In[ ]:


grad_boost_pred = model.predict(X_test)


# In[ ]:


print(classification_report(y_test,grad_boost_pred))


# Bagging algo proves to be better than Boosting. But overall, SVM dominates all of them.
