#!/usr/bin/env python
# coding: utf-8

# In this project we will be doing exploratory data analysis of data obtained from an app.We will also make a recommender system based on which company can try to convert sustomers from free to paid users.In this kernel we will cover topics like Plotting,Data Maniplation,Classification models,K-Fold Cross Validation,Grid Search and Feature Selection.This kernel is a work in process.If you like my work please vote.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **Exploring the data**

# In[ ]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil import parser
import time


# In[ ]:


df=pd.read_csv('../input/appdata10/appdata10.csv')
df.head()


# Dataset has information on the customer when they started using the app and when they enrolled for paid service.0 is Sunday and 6 is Saturday.Screen List is the detail of different screens opened by the customer.

# In[ ]:


df.describe().T


# We can conclude from the describe that there are 50000 users.Mean of day of week is 6 is as expected as days are counted from 0 to 6.Mean age of the users is 32.Number of screens used is around 21.Minigame is used by 10 % of the users.Premium feature is used by 17% of the users.62% of the customers have enrolled into the pay and use.16 % of users have liked a feature.

# **Cleaning the data**

# In[ ]:


df['hour']=df.hour.str.slice(1,3).astype(int)


# In[ ]:


df.head()


# We have replaced the hour column from string to Numerical values

# **Plotting histograms**

# In[ ]:


df1=df.copy().drop(columns=['user','screen_list','enrolled_date','first_open','enrolled'])


# In[ ]:


df1.head()


# In[ ]:


plt.figure(figsize=(20,10))
plt.suptitle('Histogram of Numerical Columns',fontsize=20)
for i in range(1,df1.shape[1]+1):
    plt.subplot(3,3,i)
    f=plt.gca()
    f.set_title(df1.columns.values[i-1])
    vals=np.size(df1.iloc[:,i-1].unique())
    plt.hist(df1.iloc[:i-1],bins=vals)
    


# In[ ]:


df1.shape[1]


# **Correlation Plot**

# In[ ]:


df1.corrwith(df.enrolled).plot.bar(figsize=(20,10),title='Correlation with Response Variable',fontsize=15,rot=45,grid=True)
plt.ioff()


# We can see that the day of the week has very small correlation to enrollement.
# Hour has negative correlation that mean earlier the time of login more the chance of enrollement.
# Age has negtive correlation which means younger people are more likely to enroll for paid usage.
# More number of screens browsed more is change of enrollement.
# People who played minigame on the app have high chance of enrollement.
# People who use premium feature have less chance of enrolling.
# Even Like has a small negative correlation with enrollement.

# **Correlation Matrix**

# In[ ]:


#Set up plot style
sns.set(style='white',font_scale=2)

#Compute Correlation Matrix
corr=df1.corr()

#Generate mask for upper traingle
mask=np.zeros_like(corr,dtype=np.bool)
mask[np.triu_indices_from(mask)]=True

#Set up the matplotlib figure
f,ax=plt.subplots(figsize=(18,15))
f.suptitle("Correlation Matrix",fontsize=40)

#Generate a custom diverging Colormap 
cmap=sns.diverging_palette(220,10,as_cmap=True)

#Draw the heat map with the mask and correct aspect ratio
sns.heatmap(corr,mask=mask,cmap=cmap,vmax=.3,center=0,square=True,linewidths=0.5,cbar_kws={'shrink':.5})
plt.ioff()


# We can see that age and num of screen as negatively correlated as age increase people will browse less.
# Also negative correlation between day of the hour and num os screen means people browse screens more at nigh early morning.
# Mini game and Premium faeature are highly correlated 

# **Feature Engineering-Response **

# In[ ]:


df.dtypes


# **Convert data to datetime object**

# In[ ]:


df.head()


# In[ ]:


df['first_open']=[parser.parse(row_data) for row_data in df['first_open']]


# In[ ]:


df['enrolled_date']=[parser.parse(row_data) if isinstance(row_data,str) else row_data for row_data in df['enrolled_date']]


# There are sum NaN values in the enrolled date column.So we update the code to consider the NaN Values

# In[ ]:


df.info()


# In[ ]:


df['Difference']=(df.enrolled_date-df.first_open).astype('timedelta64[h]')


# In[ ]:


df.head()


# **Finding out the best time for enrollement from open date**

# In[ ]:


plt.hist(df['Difference'].dropna(),color='r')
plt.title('Distribution of Time-Since-Enrolled')
plt.ioff()


# We see that most enrollements happen within 100 hours.

# In[ ]:


plt.hist(df['Difference'].dropna(),color='r',range=[0,100])
plt.title('Distribution of Time-Since-Enrolled')
plt.ioff()


# We can see that most of the enrollements happen within 25 hours.For our analysis we will consider 2 days from opening that will be equaivalent to 48 hours.So we will remove all the people who have not enrolled and have been using the app for more than 48 hours.

# In[ ]:


df.shape


# In[ ]:


df.loc[df.Difference>48,'enrolled']=0


# In[ ]:


df=df.drop(columns=['Difference','enrolled_date','first_open'])


# **Feature enginnering Screens**

# In[ ]:


top_screens=pd.read_csv('../input/appdata10/top_screens.csv')
top_screens.head()


# In[ ]:


top_screens=top_screens.top_screens.values


# In[ ]:


for sc in top_screens:
    df[sc]=df.screen_list.str.contains(sc).astype(int)
    df['screen_list']=df.screen_list.str.replace(sc+",","")


# In[ ]:


df['Other']=df.screen_list.str.count(",")
df=df.drop(columns=['screen_list'])


# In[ ]:


df.head()


# In[ ]:


savings_screens=['Saving1','Saving2','Saving2Amount','Saving4','Saving5','Saving6','Saving7','Saving8','Saving9','Saving10']
df['SavingsCount']=df[savings_screens].sum(axis=1)


# In[ ]:


df=df.drop(columns=savings_screens)


# In[ ]:


cm_screens=['Credit1','Credit2','Credit3','Credit3Container','Credit3Dashboard']


# In[ ]:


df['CMCOunt']=df[cm_screens].sum(axis=1)
df=df.drop(columns=cm_screens)


# In[ ]:


loan_screens=['Loan','Loan2','Loan3','Loan4']


# In[ ]:


df['LoansCount']=df[loan_screens].sum(axis=1)
df=df.drop(columns=loan_screens)


# In[ ]:


df.columns


# In[ ]:


df.to_csv('new_appdata10.csv',index=False)


# **Data Preprocessing**

# In[ ]:


response=df['enrolled']


# In[ ]:


df=df.drop(columns='enrolled')


# **Splitting data into Test Train **

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df,response,test_size=0.2,random_state=0)


# In[ ]:


train_identifier=X_train['user']
X_train=X_train.drop(columns='user')
test_identifier=X_test['user']
X_test=X_test.drop(columns='user')


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train2=pd.DataFrame(sc_X.fit_transform(X_train))
X_test2=pd.DataFrame(sc_X.transform(X_test))
X_train2.columns=X_test.columns.values
X_train2.index=X_train.index.values
X_test2.index=X_test.index.values
X_train=X_train2
X_test=X_test2


# **Model Built**

# In[ ]:


from sklearn.linear_model import LogisticRegression 
classifier=LogisticRegression(random_state=0,penalty='l1')
classifier.fit(X_train,y_train)


# In[ ]:


y_pred=classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,precision_score,recall_score
cm=confusion_matrix(y_test,y_pred)
cm


# In[ ]:


accuracy_score=(y_test,y_pred)


# In[ ]:


precision_score=(y_test,y_pred)


# In[ ]:


recall_score(y_test,y_pred)


# In[ ]:


f1_score(y_test,y_pred)


# In[ ]:


df_cm=pd.DataFrame(cm,index=(0,1),columns=(0,1))
plt.figure(figsize=(10,7))
sns.set(font_scale=1.4)
sns.heatmap(df_cm,annot=True,fmt='g')
#print('Test Data Accuracy:%0.4f' % accuracy_score(y_test,y_pred))
plt.ioff()


# In[ ]:


from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
print('Logistic Accuracy:%0.3f (+/- %0.3f)' % (accuracies.mean(),accuracies.std()*2))


# In[ ]:


final_results=pd.concat([y_test,test_identifier],axis=1).dropna()
final_results['predicted_results']=y_pred
final_results[['user','enrolled','predicted_results']].reset_index(drop=True)


# From this data we can identify the people who have not yet enrolled into the app.Based on this information we can do targetted advertising to get more people to enroll into paid usage

# In[ ]:




