#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')


# In[ ]:


train.head()


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


test_passengerid=test.PassengerId


# In[ ]:


df=train.append(test,ignore_index=True,sort=False)


# In[ ]:


df.info()


# **Lets look at null value first!!**

# In[ ]:


df.isnull().sum()


# **Lets start with filling age first.**

# In[ ]:


male_avg_age=df[df.Sex=='male'].Age.mean()


# In[ ]:


female_avg_age=df[df.Sex=='female'].Age.mean()


# In[ ]:


df[df.Sex=='male'].Age.hist(bins=[0,10,20,30,40,50,60,70,80,90,100])
pl.xlabel('Male')
pl.text(male_avg_age, 200,male_avg_age)
pl.show()
df[df.Sex=='female'].Age.hist(bins=[0,10,20,30,40,50,60,70,80,90,100])
pl.text(female_avg_age, 100,female_avg_age)
pl.xlabel('Female')
pl.show()


# **As age has quite a large number of null values and it can seriusly effect the accuracy of model as child was most likely to be survived or escaped first than a grown adult., so instead of replacing age with median or mean we can anlyse other coulmns and get better replacement for null values in age.** 

# In[ ]:


df['Name']


# In[ ]:


df['title']=df.Name.apply(lambda n: n.split(',')[1].split('.')[0].strip())


# In[ ]:


df['title'].unique()


# **Lets divide this titles in some categories.**

# In[ ]:


titles_cat = {
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royal",
    "Don":        "Royal",
    "Sir" :       "Royal",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "the Countess":"Royal",
    "Dona":       "Royal",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royal"
}


# In[ ]:


df.title=df.title.map(titles_cat)
df.title.value_counts()


# **Looking at all the features, most probably 'Sex', 'Pclass' & 'title' can be used to fill null values in Age.**

# In[ ]:


ag_group=df.groupby(['Sex','Pclass','title'])
ag_group.Age.median()


# In[ ]:


df.Age=ag_group.Age.apply(lambda a:a.fillna(a.median()))


# **Lets check Cabin now**

# In[ ]:


df.Cabin.unique()


# In[ ]:


df.Cabin=df.Cabin.fillna('U')


# In[ ]:


df.Embarked.value_counts()


# In[ ]:


df.Embarked=df.Embarked.fillna(df.Embarked.value_counts().index[0])


# In[ ]:


df.Fare=df.Fare.fillna(df.Fare.median())


# In[ ]:


df.isnull().sum()


# **Now we have all the data, lets get the features out ditinctively i.e. Feature Engineering**

# **There is no such importance on number in cabin, but the cabin's first letter depicts in which compartment the cabin is, so lets just get out the first char out of the Cabin.**

# In[ ]:


df.Cabin=df.Cabin.apply(lambda x:x[0])


# In[ ]:


df.head()


# In[ ]:


df.Fare.hist()


# **As the distribution of Fare is quite bad, we can normalize it.**

# In[ ]:


from sklearn import preprocessing


# In[ ]:


mm_scaler=preprocessing.MinMaxScaler()


# In[ ]:


data=df['Fare'].values
scaled_fare=mm_scaler.fit_transform(pd.DataFrame(data))


# In[ ]:


scaled_fare=pd.DataFrame(scaled_fare,columns=['Nm_Fare'])


# In[ ]:


prepared_data=pd.concat([df,scaled_fare],axis=1)


# In[ ]:


prepared_data.head()


# **Lets convert the categorical variables to dummy variable now:**

# In[ ]:


prepared_data.Sex=prepared_data.Sex.map({'male':0,'female':1})


# In[ ]:


pclass_dum=pd.get_dummies(prepared_data.Pclass,prefix='Pclass')
title_dum=pd.get_dummies(prepared_data.title,prefix='title')
cabin_dum=pd.get_dummies(prepared_data.Cabin,prefix='Cabin')
emb_dum=pd.get_dummies(prepared_data.Embarked,prefix='Embarked')


# In[ ]:


prepared_data=pd.concat([prepared_data,pclass_dum,title_dum,cabin_dum,emb_dum],axis=1)


# In[ ]:


prepared_data.drop(['Fare','Pclass','title','Cabin','Embarked','Name','Ticket'],axis=1,inplace=True)
prepared_data.head()


# **Ok now lets devide the data back to train and test. And split the train data into features and label.**

# In[ ]:


train_len=len(train)
train_len


# In[ ]:


train=prepared_data[ :train_len]
test=prepared_data[train_len: ]


# In[ ]:


train.Survived.isnull().any()


# In[ ]:


test.Survived.notnull().any()


# In[ ]:


train.Survived=train.Survived.astype(int)


# In[ ]:


X=train.drop('Survived',axis=1).values
y=train.Survived.values


# In[ ]:


X_test=test.drop('Survived',axis=1).values


# **Now lets see which of the classification model 'Logistic Regression', 'Gradient Boost Classifier' & 'Random Forest Classifier' fits best in this situation. Usually this judgement of model done by the accuracy score between predicted label data and the known test label data. But in this situation we don't have any know test label data, so we just devide the train data in 90-10 train and test & then by checking the score of the known 10% test data we can choose the best classifier algorithm.**

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_train_test, y_train, y_train_test = train_test_split(X,y,test_size=0.1,random_state=0)


# In[ ]:


print(X_train.shape,X_train_test.shape,y_train.shape,y_train_test.shape)


# In[ ]:


from sklearn.metrics import accuracy_score,fbeta_score
from time import time


# **In below train_predict function we training the model given in the parameter. And also passing sample size, that is lenght of the training size data, with which we want to traun the data. And after training we calculating the accuaracy score and f-beta score to compare in future.**

# In[ ]:


def train_predict(model, sample_size, X_train,y_train, X_test, y_test):
    result={}
    start=time()
    model=model.fit(X_train[:sample_size],y_train[:sample_size])
    end=time()
    
    result['train_time']=end-start
    
    start=time()
    pred_test=model.predict(X_test)
    pred_train=model.predict(X_train)
    end=time()
    
    result['pred_time']=end-start
    
    result['acc_train'] = accuracy_score(y_train, pred_train)
    result['acc_test'] = accuracy_score(y_test, pred_test)
    
    result['f_train'] = fbeta_score(y_train, pred_train, beta = 0.5)
    result['f_test'] = fbeta_score(y_test, pred_test, beta = 0.5)
    
    return(result)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


model1=LogisticRegression(random_state=0)
model2=GradientBoostingClassifier(random_state=0)
model3=RandomForestClassifier(random_state=0)


# **Here we are taking 3 different sample size 5%,25% and 100% of training data and iterating through all the 3 models with all 3 sample size and storing them in result variable to plot them later see the differnce in training time, predicting time and accuracy scores.**

# In[ ]:


result={}
sample_5=int(len(y_train)*0.05)
sample_25=int(len(y_train)*0.25)
sample_100=len(y_train)

for model in [model1, model2, model3]:
    model_name=model.__class__.__name__
    result[model_name]={}
    for i,sample in enumerate([sample_5,sample_25,sample_100]):
        result[model_name][i]=train_predict(model,sample,X_train, y_train,X_train_test,y_train_test)


# In[ ]:


print(result)


# In[ ]:


import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
def evaluate(results):
    
    # Create figure
    fig, ax = pl.subplots(2, 3, figsize = (11,7))

    # Constants
    bar_width = 0.3
    colors = ['#A00000','#00A0A0','#00A000']
    
    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):
            for i in np.arange(3):
                
                # Creative plot code
                ax[j//3, j%3].bar(i+k*bar_width, results[learner][i][metric], width = bar_width, color = colors[k])
                ax[j//3, j%3].set_xticks([0.45, 1.45, 2.45])
                ax[j//3, j%3].set_xticklabels(["1%", "10%", "100%"])
                #ax[j//3, j%3].set_xlabel("Training Set Size")
                ax[j//3, j%3].set_xlim((-0.1, 3.0))
    
    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")
    
    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")
    
    
    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))

    # Create patches for the legend
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color = colors[i], label = learner))
    pl.legend(handles = patches, bbox_to_anchor = (-.80, 2.53),                loc = 'upper center', borderaxespad = 0., ncol = 3, fontsize = 'x-large')
    
    # Aesthetics
    pl.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize = 16, y = 1.10)
    pl.tight_layout()
    pl.show()


# In[ ]:


evaluate(result)


# **By looking at the charts for the test data(2nd axis) we can see that Logistic regression has the maximum accuracy score as compared to other two classifiers. So we are choosing Logisic regression model to predict the result on test data.**

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer


# In[ ]:


mymodel= model1


# In[ ]:


dual=[True,False]
max_iter=[100,110,120,130,140]
penalty=['l2']
param_grid=dict(dual=dual,max_iter=max_iter,penalty=penalty)


# **We are setting up the vallues for hyperparameters and then using GridSearch we gonna itterate over all possible combination of those parameters, and then we are gonna choose the model with the best score. As here we setting up quite a few parameters the Gridsearch won't take so much time, but geneally grid search is very time taking process and can be replaced bt baysian technique to get the best fit model.**

# In[ ]:


grid=GridSearchCV(estimator=mymodel,param_grid=param_grid,cv=3,n_jobs=-1)
grid.fit(X,y)


# In[ ]:


best_model=grid.best_estimator_


# In[ ]:


print(best_model)
print(grid.best_score_)


# In[ ]:


predictions=best_model.predict(X_test)


# In[ ]:


load_kaggle=pd.DataFrame({'PassengerId':test_passengerid,'Survived':predictions})

load_kaggle.to_csv('./Titanic_Logistic_Regression.csv',index=False)


# In[ ]:


load_kaggle


# In[ ]:




