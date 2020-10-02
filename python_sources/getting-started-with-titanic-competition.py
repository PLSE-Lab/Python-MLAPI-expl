#!/usr/bin/env python
# coding: utf-8

# So this is my first kernel with the titanic challange, i am a begginner with this, after reading some articles and learning data analysis in cognitive class I think its time to test my skills and try to learn more. 
# 
# Sorry for the bad english 

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import seaborn as sns 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.


# In[2]:


train_df = pd.read_csv('../input/train.csv',index_col='PassengerId')
test_df = pd.read_csv('../input/test.csv',index_col='PassengerId')
test_df['Survived'] = -1 
df = pd.concat([train_df,test_df],axis=0)


# In[3]:


df.head() 


# # Data Basic Structure

# In[5]:


df.info()


# ### Results 
# Here we can see that there are different data types like int,float, and object, also thera are some null values in the columns  Age, embarked and Cabin. 
# We will work with these columns in order to not have null values

# ## Sumary Statistics of the train dataset 

# In[6]:


df.describe(include='all')


# ### Results 
# Here we have some information about the values of each column, also some agregation function. We can see thera some columns that are categorical while others are continuos. We need to know more about the Pclass, SibSp	and Parch , Sex, Embarked and Survived columns. 
# 

# In[7]:


columns = ['Pclass','SibSp','Parch','Sex','Embarked','Survived']
for column in columns:
    print(f'unique elements of {column}: {df[column].unique()}')
    print(f'value count of {column} :\n {df[column].value_counts()}')
    


# ### Result 
# We can see here that the columns listed before are categorical. 
# Thera some interesting results
# *  there are more passenger in the third class than the others. 
# *  many of the passengers does not have spouse or siblings aboard 
# *  Many of the passengers does not have parents or children aboard.
# *  more than the 50% of the passangers are males. 
# *  there are more passengers that have embarked in S 
# *  less than the 50% of the passengers in the train dataset have survived. 
# 

# ## Working with Missing values
# 

# #### Age 

# In[8]:


df[df.Age.isnull()==True].head()


# Here we can do two things: delete the nan rows or replace the age with average age or replace it with age average by pclass. 
# 
# we will replace nan values with average by sex or we can do it by replacing with the averega by Pclass 

# In[9]:


age_median_Pclass = df.groupby('Pclass').Age.transform('median')
df.Age.fillna(age_median_Pclass,inplace=True)


# #### Cabin 

# In[11]:


df[df.Cabin.isnull()==True].head() 


# In[12]:


df.Cabin.unique() 


# we see that each value start wit a prefix D, F, E, etc. so we can create a new colunm called deck to specified the deck of the cabin 

# In[13]:


df.loc[df.Cabin=='T','Cabin']= np.NaN
# extract first character of cabin string to the deck 
def get_deck(cabin):
    return np.where(pd.notnull(cabin),str(cabin)[0].upper(),'Z')
df['Deck'] = df.Cabin.map(lambda x : get_deck(x))


# In[14]:


df.Deck.value_counts()


# #### Embarked

# In[16]:


df[df.Embarked.isnull()==True].head() 


# In[17]:


# we can see the value counts for embarked 

df.Embarked.value_counts() 


# In[18]:


# we can replace nan with S 
df.Embarked.fillna('S',inplace=True)


# ### Fare 

# In[20]:


mean_fare = df.Fare.mean() 
df.Fare.fillna(mean_fare,inplace=True)


# In[21]:


df.info() 


# ## Searching for outliers

# #### Age

# In[22]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
df.Age.plot(kind='hist',bins=5); 


# In[23]:


df.Age.plot(kind='box'); 


# In[24]:


train_df[df.Age==df.Age.max()].head() 


# In[25]:


df[df.Age>df.Age.quantile(0.75)].head() 


# So there some values that are greather that the 75% of the data and some are outliers. 
# Some we can create a new coloumn in which say if the passenger is a child or and adult 

# In[26]:


df['Child_Adult'] = np.where(df['Age']>18,'Adult','Child')


# #### Fare

# In[27]:


df.Fare.plot(kind='hist',bins=5)


# In[28]:


df.Fare.plot(kind='box'); 


# In[29]:


df[df.Fare==df.Fare.max()].head()


# Here we can see that there is also some values that are outliers.
# We will create a new a new column with fare binning, that is we will create this column with fare range: very_low, low, high,very_high

# In[30]:


df['Fare_binning'] = pd.qcut(df.Fare,4,labels=['very_low','low','high','very_high'])
#df['Fare_binning'] = pd.qcut(df.Fare,3,labels=['low','medium','high'])


# In[31]:


# now we see 
df.Fare_binning.value_counts().plot(kind='bar',rot=0); 


# ## Sear ching for a new feature in name
# 

# In[33]:


df.Name.head() 


# In[34]:


df.Name.tail() 


# we see that there are title with each passenger, like Mr, Dona, Master, etc. Lets see if every passenger has a title, in order to create a new feature called title 

# In[36]:


titles = [name.split(',')[1].split('.')[0] for name in df.Name]
print(f"number of titles {len(titles)} ")
names = pd.DataFrame(titles,columns=['Title'])
print("unique values: ",names.Title.unique())


# And we see that every passenger has a title  

# In[37]:


# lets create the new feature 
# but we have to create new values for title that are relate to others
def get_title(name):
    title_group = {
        'mr':'Mr',
        'mrs':'Mrs',
        'miss':'Miss',
        'master':'Master',
        'don':'Sir',
        'rev':'Sir',
        'dr':'Officer',  
        'mme':'Mrs',
        'ms':'Mrs',
        'major':'Officer',
        'lady':'Lady',
        'sir':'Sir',
        'mlle':'Miss',
        'col':'Officer',
        'capt':'Officer',
        'the countess':'Lady',
        'jonkheer':'Sir',
        'dona':'Lady'
    }
    first_name_with_title = name.split(',')[1]
    title = first_name_with_title.split('.')[0]
    title = title.strip().lower()
    return title_group[title]


# In[38]:


df['Title'] = df.Name.map(lambda x : get_title(x))


# ## Searching for Predictors

# ### Pclass

# In[40]:


pd.crosstab(df[df.Survived!=-1].Pclass,df[df.Survived!=-1].Survived)


# In[41]:


pd.crosstab(df[df.Survived!=-1].Pclass,df[df.Survived!=-1].Survived).plot(kind='bar',rot=0);    


# #### Result 
# So the Pclass is a column that we can use to train the model, this is because, if you were from first class you have had more chance to survived, while the second class there will be almost a 50% of survive or die, and if you we from the third class you certainly will die. 

# ### Sex

# In[42]:


pd.crosstab(df[df.Survived!=-1].Sex,df[df.Survived!=-1].Survived).plot(kind='bar');


# #### Result 
# This is another good predictor, because the female has more chances to survive rather than male. 

# ### SibSp

# In[43]:


pd.crosstab(df[df.Survived!=-1].SibSp,df[df.Survived!=-1].Survived).plot(kind='bar'); 


# ### Parch 

# In[44]:


pd.crosstab(df[df.Survived!=-1].Parch,df[df.Survived!=-1].Survived).plot(kind='bar'); 


# With two plots above, we see that people who were alone has more chances to survived, so we can have a new column 

# In[45]:


df['Travel_alone'] = np.where((df.Parch + df.SibSp)==0,1,0)


# In[46]:


pd.crosstab(df[df.Survived!=-1].Travel_alone,df[df.Survived!=-1].Survived).plot(kind='bar',rot=0); 


# Here we can see that the passengers that have traveled alone has less chances of survive

# ### Embarked

# In[47]:


pd.crosstab(df[df.Survived!=-1].Embarked,df[df.Survived!=-1].Survived).plot(kind='bar',rot=0); 


# This is also another column that we can use in the model. 

# ### Child_Adult

# In[48]:


pd.crosstab(df[df.Survived!=-1].Child_Adult,df[df.Survived!=-1].Survived).plot(kind='bar',rot=0); 


# ## Fare

# In[49]:


pd.crosstab(df[df.Survived!=-1].Fare_binning,df[df.Survived!=-1].Survived).plot(kind='bar',rot=0); 


# Fare and child_adult are also some of the columns that we can use. 

# ### Deck 

# In[50]:


pd.crosstab(df[df.Survived!=-1].Deck,df[df.Survived!=-1].Survived).plot(kind='bar',rot=0); 


# In[51]:


has_cabin = np.where(df.Deck[df.Survived!=-1]=='Z',0,1)
pd.crosstab(has_cabin,df[df.Survived!=-1].Survived).plot(kind='bar',rot=0); 


# So we see now that passengers that not have cabin information has more chance to die in the disaster so, create another useful column called has_cabin_information

# In[54]:


df['has_cabin_information'] = has_cabin = np.where(df.Deck=='Z',0,1)


# #### Title
# 

# In[55]:


pd.crosstab(df[df.Survived!=-1].Title,df[df.Survived!=-1].Survived).plot(kind='bar',rot=0); 


# Mr has more probability of die, while Miss and Mrs has more probability of survived, because of that we can use Title has part of the columns to use in the model. 

# ### General Results 
# so far we see that the columns that we will use in the model are: 
# 
# - Sex 
# - Pclass
# - Travel_alone
# - Embarked 
# - Child_Adult
# - Fare_binning 
# - has_cabin_information
# - Title 
# 
# 

# In[56]:


columns = ['Survived','Sex','Pclass','Travel_alone','Embarked','Fare_binning','has_cabin_information','Title']  
proccesed_df = df[columns].copy()    
proccesed_df.head() 


# In[57]:


proccesed_df.shape


# In[58]:


from sklearn.preprocessing import LabelEncoder
columns_to_transform = ['Sex','Embarked','Fare_binning','Title']
for column in columns_to_transform:
    encoder = LabelEncoder()
    proccesed_df[column] = encoder.fit_transform( proccesed_df[column])
proccesed_df.head() 
        
        


# In[59]:


proccesed_df.info() 


# In[60]:


model_train_df = proccesed_df[proccesed_df.Survived!=-1]
model_test_df = proccesed_df[proccesed_df.Survived==-1]
model_train_df.head() 


# In[62]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 

columns = [column for column in model_train_df.columns if column!='Survived']
X = model_train_df[columns].values 
y = model_train_df.Survived.values 
X_train,x_validation,y_train, y_validation = train_test_split(X,y,test_size=0.2,random_state=25)


# ### Logistic Regression

# In[63]:


logistic = LogisticRegression(solver='liblinear',multi_class='ovr')
logistic.fit(X_train,y_train)


# In[64]:


from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

print("logistic validation score ",logistic.score(x_validation,y_validation))
print("logistic train score ",logistic.score(X_train,y_train))
print("logistic accuracy score ",accuracy_score(y_validation,logistic.predict(x_validation)))
print("logistic precision score ",precision_score(y_validation,logistic.predict(x_validation)))
print("logistic recall socre ",recall_score(y_validation,logistic.predict(x_validation)))


# In[65]:


confusion_matrix(y_validation,logistic.predict(x_validation))


# In[66]:


x_test = model_test_df[columns].values 
y_hat = logistic.predict(x_test) 
y_hat[:10]


# In[67]:


from sklearn.neighbors import KNeighborsClassifier


# In[68]:


knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train,y_train)


# In[69]:


knn.score(X_train,y_train)


# In[70]:


knn.score(x_validation,y_validation)


# In[71]:


score_train  = []
score_validation = [] 
for n in range(2,9):
    knn1 = KNeighborsClassifier(n_neighbors=n)
    knn1.fit(X_train,y_train)
    score_train.append(knn1.score(X_train,y_train))
    score_validation.append(knn1.score(x_validation,y_validation))
    
import matplotlib.pyplot as plt 
neighbours = np.array(range(2,9))
plt.plot(neighbours,np.array(score_train),'bo',label='Training score')
plt.plot(neighbours,np.array(score_validation),'b',label='Validation score')
plt.xlabel('neighbours')
plt.ylabel('score')
plt.legend()
plt.show() 


# In[73]:


# the prediction 
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
y_hat_knn = knn.predict(x_test)
y_hat_knn[:10]


# In[74]:


print("knn validation score ",logistic.score(x_validation,y_validation))
print("knn train score ",logistic.score(X_train,y_train))
print("knn accuracy score ",accuracy_score(y_validation,knn.predict(x_validation)))
print("knn precision score ",precision_score(y_validation,knn.predict(x_validation)))
print("knn recall socre ",recall_score(y_validation,knn.predict(x_validation)))


# ### Support Vector Machine

# In[76]:


from sklearn.svm import SVC
model = SVC() 
model.fit(X_train,y_train)
y_hat_SVC =  model.predict(x_test)
print(y_hat_SVC[:10])


# In[79]:


print("SVC validation score ",model.score(x_validation,y_validation))
print("SVC train score ",model.score(X_train,y_train))
print("SVC accuracy score ",accuracy_score(y_validation,model.predict(x_validation)))
print("SVC precision score ",precision_score(y_validation,model.predict(x_validation)))
print("SVC recall socre ",recall_score(y_validation,model.predict(x_validation)))


# ### Neural Network

# In[80]:


from sklearn.neural_network import MLPClassifier
network = MLPClassifier(solver='sgd',learning_rate_init=0.15)
network.fit(X_train,y_train)


# In[85]:


learning_rates = [0.0015,0.015,0.1,0.15,0.20,0.25,0.35,0.45,0.55,0.66,0.75,0.85]
training_score = []
validation_score = []
loss = []
for learnin_rate in learning_rates:
    net = MLPClassifier(solver='sgd',learning_rate_init=learnin_rate)
    net.fit(X_train,y_train)
    training_score.append(net.score(X_train,y_train))
    validation_score.append(net.score(x_validation,y_validation))
    loss.append(net.loss_)


# In[86]:


plt.plot(np.array(learning_rates),np.array(training_score),'b',label='training score')
plt.plot(np.array(learning_rates),np.array(validation_score),'bo',label='validation score')
plt.xlabel('learning rate')
plt.ylabel('score')
plt.title('Neural network with many learning rate values')
plt.legend()
plt.plot() 


# In[87]:


plt.plot(np.array(learning_rates),np.array(loss),'b',label='training loss')
plt.xlabel('learning rate')
plt.ylabel('loss')
plt.title('Neural network with many learning rate values')
plt.legend()
plt.plot() 


# In[88]:


print("network validation score ",network.score(x_validation,y_validation))
print("network train score ",network.score(X_train,y_train))
print("network accuracy score ",accuracy_score(y_validation,network.predict(x_validation)))
print("network precision score ",precision_score(y_validation,network.predict(x_validation)))
print("network recall socre ",recall_score(y_validation,network.predict(x_validation)))


# ## Keras model 

# In[89]:


from keras import models
from keras import layers
keras_model = models.Sequential()
keras_model.add(layers.Dense(64,activation='relu',input_shape=(7,)))
keras_model.add(layers.Dense(32,activation='relu'))
keras_model.add(layers.Dense(16,activation='relu'))
keras_model.add(layers.Dense(1,activation='sigmoid'))
keras_model.compile(optimizer='rmsprop',
             loss='binary_crossentropy',
             metrics=['accuracy'])
history = keras_model.fit(X_train,
                   y_train,
                   epochs=28,
                   batch_size=100,
                   validation_data=[x_validation,y_validation])


# In[90]:



print("keras_model accuracy score ",accuracy_score(y_validation,keras_model.predict_classes(x_validation)))
print("keras_model precision score ",precision_score(y_validation,keras_model.predict_classes(x_validation)))
print("keras_model recall socre ",recall_score(y_validation,network.predict(x_validation)))


# ## Decision Tree

# In[91]:


from sklearn.tree import DecisionTreeClassifier
titanic_tree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
titanic_tree.fit(X_train,y_train)


# In[100]:


score_train  = []
score_validation = [] 
for n in range(2,9):
    tree = DecisionTreeClassifier(criterion="entropy", max_depth = n)
    tree.fit(X_train,y_train)
    score_train.append(tree.score(X_train,y_train))
    score_validation.append(tree.score(x_validation,y_validation))
    
import matplotlib.pyplot as plt 
neighbours = np.array(range(2,9))
plt.plot(neighbours,np.array(score_train),'bo',label='Training score')
plt.plot(neighbours,np.array(score_validation),'b',label='Validation score')
plt.xlabel('max depth')
plt.ylabel('score')
plt.legend()
plt.show() 


# In[93]:


print("tree validation score ",titanic_tree.score(x_validation,y_validation))
print("tree train score ",titanic_tree.score(X_train,y_train))
print("tree accuracy score ",accuracy_score(y_validation,titanic_tree.predict(x_validation)))
print("tree precision score ",precision_score(y_validation,titanic_tree.predict(x_validation)))
print("tree recall socre ",recall_score(y_validation,titanic_tree.predict(x_validation)))


# ## Random Fores
# 

# In[102]:


from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=50, max_depth=4,
                                  random_state=0,criterion='entropy') 

random_forest.fit(X_train,y_train)


# In[103]:


print("random_forest validation score ",random_forest.score(x_validation,y_validation))
print("random_forest train score ",random_forest.score(X_train,y_train))
print("random_forest accuracy score ",accuracy_score(y_validation,random_forest.predict(x_validation)))
print("random_forest precision score ",precision_score(y_validation,random_forest.predict(x_validation)))
print("random_forest recall socre ",recall_score(y_validation,random_forest.predict(x_validation)))


# In[101]:


score_train  = []
score_validation = [] 
for n in range(2,9):
    forest = RandomForestClassifier(n_estimators=50, max_depth=n,
                                  random_state=0) 
    forest.fit(X_train,y_train)
    score_train.append(forest.score(X_train,y_train))
    score_validation.append(forest.score(x_validation,y_validation))
    
import matplotlib.pyplot as plt 
neighbours = np.array(range(2,9))
plt.plot(neighbours,np.array(score_train),'bo',label='Training score')
plt.plot(neighbours,np.array(score_validation),'b',label='Validation score')
plt.xlabel('max depth')
plt.ylabel('score')
plt.legend()
plt.show() 


# In[ ]:




