#!/usr/bin/env python
# coding: utf-8

# # Titanic Survival Prediction 
# ### In this i have covered most of ensemble models. Hope you guys like it. Don't forget to give Upvote :)
# 
# 

# In[1]:


get_ipython().run_cell_magic('time', '', 'import numpy as np \nimport pandas as pd\nimport warnings \nwarnings.filterwarnings("ignore")\nimport os\nprint(os.listdir("../input"))\nimport matplotlib.pyplot as plt\nimport seaborn as sns')


# ## Importing and exploring data

# In[2]:


train=pd.read_csv("../input/train.csv")


# In[3]:


train.head()


# In[4]:


train.describe()


# ## Don't forget to explore object type columns

# In[5]:


train.describe(include=['O'])


# In[6]:


train.hist(figsize=(10,8))


# In[7]:


train.info()


# ## So column Age, Cabin, Embarked have missing values. Let's Start with Age column

# In[8]:


train.Age.hist(bins=20)


# ## Age distribution is positive skewed. Need more information to fill missing data. Let's plot Age with PClass

# In[9]:


sns.boxplot(x='Pclass',y='Age',data=train,hue='Survived')


# In[10]:


print (train.groupby(['Pclass']).get_group(1).Age.mean())
print (train.groupby(['Pclass']).get_group(2).Age.mean())
print (train.groupby(['Pclass']).get_group(3).Age.mean())


# ## We can fill missing age with mean but age varies for each Pclass so filling missing age with mean will not be proper. Lets fill Age according to Pclass 

# In[11]:


train['Age']=train.groupby(['Pclass','Survived'])['Age'].transform(lambda x:x.fillna(x.mean()))


# In[12]:


train.info()


# # Now lets take a look at Fare column. May be it want to something to us

# In[13]:


sns.stripplot(y='Fare',x='Pclass',hue='Survived',data=train)


# ## From above figure we can say that people who paid higher got 1st Pclass and there chanced of survival are better than other Pclass. 
# ## Let's see average Fare based on class

# In[14]:


train.groupby(['Pclass','Survived'])['Fare'].mean()


# In[15]:


plt.hist(train.Fare,bins=30)
plt.xlabel('Fare')
plt.ylabel('count')


# ## Most of the people paid 0-80 Fare. Fare varies based on Pclass and Survival. Survived people paid higher fare than people who died. So we need to utilise fare column. Since Fare as an integer column will not be usefull. Lets make it Categorical 

# In[16]:


train.Fare=np.ceil(train.Fare)
train['fare']=pd.cut(train.Fare,bins=[0,8,13,20,30,50,80,600],labels=['a','b','c','d','e','f','g'],right=False)


# In[17]:


sns.countplot(x='fare',hue='Survived',data=train)


# ## Thats look nice!!! As Fare increases (a to g) chances of survival increases.
# ## Fare really wanted to tell us something :)

# ## Lets see now SibSp and Parch Columns

# In[18]:


sns.countplot(x='SibSp',hue='Survived',data=train)


# In[19]:


sns.countplot(x='Parch',hue='Survived',data=train)


# ## Lets combine both columns. As both column represent members

# In[20]:


train['members']=train['SibSp']+train['Parch']


# In[21]:


sns.countplot(x='members',hue='Survived',data=train)


# In[22]:


train.members.value_counts()


# In[23]:


train[train.members>6].Survived.value_counts()


# ## Members with head count of more than 6 never survived in our train dataset so lets make 6+ members that is 7 and 10 members as 7 members

# In[24]:


train.members.replace({10:7},inplace=True)


# In[25]:


train.head()


# ## Now lets choose our feature attributes. Name is not giving us any proper info so lets drop it. Cabin column have various missing values and filling it may affect our prediction so drop it to. Ticket also not needed so drop it.

# In[26]:


attributes=['Survived','Pclass','Sex','Age','Embarked','fare','members']


# In[27]:


train=train[attributes]


# In[28]:


train.head()


# ## Wait !!! Embarked also have 2 mising values. So lets do filling. But first we need to explore Embarked column

# In[29]:


sns.countplot(x='Embarked',hue='Survived',data=train)


# In[30]:


train[train.Embarked.isnull()]


# ## Two missing values belong to same Pclass and Same Sex with same Fare category ie g. Lets explore further more

# In[31]:


sns.catplot(kind='point',x='Embarked',y='Pclass',hue='Sex',data=train)


# In[32]:


train.groupby(['Pclass','Sex']).get_group((1,'female')).Embarked.value_counts()


# ## So with above exploration we can say that female which belong to Pclass 1 have C Embarked most probably. Lets fill it

# In[33]:


train.Embarked.fillna('C',inplace=True)


# In[34]:


train.info()


# ## No missing value finally. Now lets do type conversion

# In[35]:


def func(x):
    if(x.dtype=='O'):
        x=x.astype('category')
    return(x)


# In[36]:


train=train.apply(func,axis=0)


# In[37]:


train.info()


# In[38]:


train.members=train.members.astype('category')
train.Survived=train.Survived.astype('category')
train.Pclass=train.Pclass.astype('category')
train.Age=train.Age.astype('int64')


# In[39]:


train.info()


# ## Now lets convert categorical values into dummy variable and Scaling 

# In[40]:


df_label=train.Survived
del train['Survived']
df=pd.get_dummies(train)


# In[41]:


from sklearn.preprocessing import StandardScaler


# In[42]:


scaled=StandardScaler().fit_transform(df)
df=pd.DataFrame(scaled,index=df.index,columns=df.columns)


# In[43]:


df=pd.concat([df,df_label],axis=1)


# In[44]:


df.head()


# # Now our data is ready now its time to use it for model building and prediction

# In[45]:


train=df
train.shape


# In[46]:


index=np.random.permutation(891)
train=train.loc[index,:]
train.shape


# In[47]:


train_label=train.Survived
del train['Survived']


# # 1. Linear Classifier

# In[48]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict


# In[49]:


from sklearn.linear_model import SGDClassifier
sgd=SGDClassifier(n_iter=200,penalty='l1',epsilon=1e-20,random_state=8349)
score=cross_val_predict(sgd,train,train_label,cv=10)
confusion_matrix(train_label,score)


# In[50]:


from sklearn.metrics import accuracy_score
acc_lc=accuracy_score(train_label,score)
acc_lc


# # 2. logistic Regression

# In[51]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(random_state=73289471,class_weight='balanced')
score=cross_val_predict(lr,train,train_label,cv=10)
confusion_matrix(train_label,score)


# In[52]:


from sklearn.metrics import accuracy_score
acc_lr=accuracy_score(train_label,score)
acc_lr


# # 3. KNN

# In[53]:


from sklearn.neighbors import KNeighborsClassifier
value=[]
for k in range(1,20):
    knn=KNeighborsClassifier(k,algorithm='brute')
    score=cross_val_predict(knn,train,train_label,cv=10)
    value.append(accuracy_score(train_label,score))


# In[54]:


df=pd.DataFrame(value,index=range(1,20),columns=['accuracy'])


# In[55]:


df.set_index='K value'
df.sort_values(ascending=False,by='accuracy')


# ## So KNN give best result when k=7. lets train with k=7

# In[56]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(7,algorithm='brute')
score=cross_val_predict(knn,train,train_label,cv=10)
acc_knn=accuracy_score(train_label,score)
acc_knn


# # 4. Decision Tree

# In[57]:


from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(random_state=1341)
score=cross_val_predict(dtc,train,train_label,cv=10)
confusion_matrix(train_label,score)


# In[58]:


acc_dtc=accuracy_score(train_label,score)
acc_dtc


# # 5. SVM

# In[59]:


from sklearn.svm import SVC
svm=SVC(kernel='rbf',C=20,gamma=0.05,random_state=2317)
score=cross_val_predict(svm,train,train_label,cv=10)
confusion_matrix(train_label,score)


# In[60]:


acc_svm=accuracy_score(train_label,score)
acc_svm


# # 6.Random Forest

# In[61]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=200,random_state=167123)
score=cross_val_predict(rf,train,train_label,cv=10)
confusion_matrix(train_label,score)


# In[62]:


acc_rf=accuracy_score(train_label,score)
acc_rf


# # 7 Extra Tree

# In[63]:


from sklearn.ensemble import ExtraTreesClassifier
etc=ExtraTreesClassifier(n_estimators=200,random_state=67)
score=cross_val_predict(etc,train,train_label,cv=10)
confusion_matrix(train_label,score)


# In[64]:


acc_etc=accuracy_score(train_label,score)
acc_etc


# # 8. ADA BOOSTING

# In[65]:


from sklearn.ensemble import AdaBoostClassifier
ada=AdaBoostClassifier(dtc,n_estimators=200,
algorithm='SAMME.R',learning_rate=0.01,random_state=13247)
score=cross_val_predict(ada,train,train_label,cv=10)
confusion_matrix(train_label,score)


# In[66]:


acc_ada=accuracy_score(train_label,score)
acc_ada


# # 9. Gradient Boosting

# In[67]:


from sklearn.ensemble import GradientBoostingClassifier
gb=GradientBoostingClassifier(n_estimators=200,learning_rate=0.01,random_state=11233)
score=cross_val_predict(gb,train,train_label,cv=10)
confusion_matrix(train_label,score)


# In[68]:


acc_gb=accuracy_score(train_label,score)
acc_gb


# # 9. Bagging And Pasting

# In[69]:


from sklearn.ensemble import BaggingClassifier
bp=BaggingClassifier(SVC(kernel='rbf',C=20,gamma=0.05,random_state=87),n_estimators=200, bootstrap=False ,
                     n_jobs=-1,random_state=82139 )
score=cross_val_predict(bp,train,train_label,cv=10)
confusion_matrix(train_label,score)


# In[70]:


acc_bp=accuracy_score(train_label,score)
acc_bp


# In[71]:


df=pd.DataFrame([acc_lc*100,acc_lr*100,acc_knn*100,acc_dtc*100,acc_svm*100,acc_rf*100,
             acc_etc,acc_ada*100,acc_gb*100,acc_bp*100],
            index=['Linear Classifier','Logistic','KNN','Decision Tree','SVM','Random Forest',
                  'Extra Trees','ADA boost','Gradient Boost','Bagging and pasting'],columns=['Accuracy'])


# In[72]:


df=df.sort_values(ascending=False,by='Accuracy')


# In[73]:


color=sns.color_palette
sns.barplot(data=df, y=df.index,x='Accuracy')
#plt.xticks(rotation=90)


# # Now Random Search on Best Models

# # Gradient Boost

# In[77]:


from sklearn.model_selection import RandomizedSearchCV
def r_search(classifier,param,data,data_label,fold):
    rs=RandomizedSearchCV(classifier,param_distributions=param,cv=fold,n_jobs=-1)
    rs.fit(data,data_label)
    return(rs.best_params_ , rs.best_score_, rs.best_estimator_)


# In[78]:


param={'max_features':[7,9,13,],'max_depth':[5,7,9,12],'min_samples_split':[25,40,55],
       'min_samples_leaf':[3,5,13,23],'max_leaf_nodes':[3,7,13,19],
      'n_estimators':[100,200,500,1000],'learning_rate':[1,0.1,0.01,0.001]}
best_param , best_score , best_estimator= r_search(GradientBoostingClassifier(random_state=9248309),
                                 param,train,train_label,10)


# In[79]:


print(best_param,'\n' ,best_score)


# In[81]:


gb=best_estimator


# # 10.Voting Classifier

# In[82]:


from sklearn.ensemble import VotingClassifier
vc=VotingClassifier(estimators=[('rf',svm),('gb',gb),
                                ('svm',lr)],voting='hard')
score=cross_val_predict(vc,train,train_label,cv=10)
confusion_matrix(train_label,score)


# In[83]:


acc_vc=accuracy_score(train_label,score)
acc_vc


# # So Finally we got our best algorithm with accuracy of 84.73% Gradient Boosting

# In[84]:


gb.fit(train,train_label)


# # Now we have to import test file and process it before prediction

# In[85]:


test=pd.read_csv('../input/test.csv')


# In[86]:


test.head()


# In[87]:


attributes=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
test=test[attributes]


# In[88]:


test.info()


# In[89]:


test['Age']=test.groupby('Pclass')['Age'].transform(lambda x:x.fillna(x.mean()))


# In[90]:


test.Embarked.fillna('C',inplace=True)


# In[91]:


test.info()


# In[92]:


test['members']=test['SibSp']+test['Parch']
del test['SibSp']
del test['Parch']


# In[93]:


test.members.replace({10:7},inplace=True)


# In[94]:


test.Fare=np.ceil(test.Fare)
test['fare']=pd.cut(test.Fare,bins=[0,8,13,20,30,50,80,600],labels=['a','b','c','d','e','f','g'],right=False)


# In[95]:


test.members=test.members.astype('category')
test.Pclass=test.Pclass.astype('category')
test.Age=test.Age.astype('int64')
test.fare=test.fare.astype('category')
test.Embarked=test.Embarked.astype('category')


# In[96]:


test.info()


# In[97]:


test.fare.value_counts()


# In[98]:


test.fare.fillna('b',inplace=True)
test.info()


# In[99]:


del test['Fare']


# In[100]:


test.head()


# In[101]:


test=pd.get_dummies(test)


# In[102]:


scaled=StandardScaler().fit_transform(test)
test=pd.DataFrame(scaled,index=test.index,columns=test.columns)
test.head()


# # Prediction Time

# In[103]:


test.shape


# In[104]:


prediction=gb.predict(test)


# In[107]:


sample=pd.read_csv('../input/gender_submission.csv')


# In[108]:


sample.head()


# In[109]:


s=pd.DataFrame({'PassengerId':sample.PassengerId,'Survived':prediction})
s.head()


# In[110]:


s.to_csv('submission.csv',index=False)


# ## Thanks and don't forget to upvote it :)
