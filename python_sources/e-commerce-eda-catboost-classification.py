#!/usr/bin/env python
# coding: utf-8

# ![](https://i.imgur.com/UHbrtdO.png)

# ## Janata Hack - E Commerce Analytics ML Model
# 
# #### Contents
# 
# - Problem Statement.
# - Import Packages
# - Loading Dataset
# - Data Pre-Processing
# - Basic EDA 
# - Model Building
# - Model Evaluation 
# - Prediction on Test Data
# - Submissions

# ## Problem Statement - 
# 
# Prediction of User Gender,as the E commerce market is growing and it's necessary to tap in the User Data to enhance their customer services, to provide better product predictions which in turn will help the company to grow in terms of customer base and profit. Hence User Details are very much required. Having said that, we will be working on prediction of gender where we are provided with a set of Categorical Columns to work with.If you are looking for how to handle categorical variables this is a good start !
# 
# 
# * In the dataset, Column ProductList Contains products seperated by ( ; ) and which in turn every product has a category, sub category and sub sub category which are seperated by (/) you can refer the notebook to understand clearly.
# 
# 
# 
# - ##### Columns
#    - session_id   :       Object
#    - startTime    :       Object
#    - endTime      :       Object 
#    - ProductList  :       Object
#    - gender       :       Object

# ### Import Packages here - 

# In[ ]:


import pandas as pd
import numpy as np 
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from catboost import Pool, CatBoostClassifier, cv
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Importing Training Data And Test Data 

# In[ ]:


train_df = pd.read_csv('../input/train_8wry4cB.csv')
test_df = pd.read_csv('../input/test_Yix80N0.csv')

target = train_df['gender'].copy() ## Target !
display(train_df.head())
display(test_df.head())
display(target.head())


#Displaying Train Data !
#Displaying Test Data !
#Displaying Target Column !


# ### Data Info

# In[ ]:


train_df.dtypes


# In[ ]:


test_df.dtypes


# In[ ]:


train_df.describe()


# We Have a biased dataset, as you can see in the Gender Column we have 8192 observations only for females. Alongwith, we have 9402 unique products out of which one product has been viewed the most like by 25 users. Also there are users which were viewing same prodcuts at the same time.

# In[ ]:


test_df.describe()


# In this also, you can see we have the same product viewed maximum number of times.

# ### Data Preprocessing

# In[ ]:


X = train_df.copy()
X_test = test_df.copy()


# In[ ]:


#Dropping Gender and Session_id Columns.

X.drop(['session_id', 'gender'], axis = 1, inplace = True)
X_test.drop(['session_id'], axis = 1, inplace = True)
display(X.head())
display(X_test.head())


# In[ ]:


# call this to get group of product codes for each user_id

def product_group(x):
    prod_list = []
    for i in x.split(';'):
        prod_list.append(i.split('/')[-2])
    return prod_list


# In[ ]:


# call this to fetch the most frequent D code used.
def frequency(List): 
    return max(set(List), key = List.count)


# Extracting the D codes. 
def final_items(x):
    prod_code = []
    for i in x:
        prod_code.append(i[:4])
    return frequency(prod_code)


# In[ ]:


def CleanIt(data):

    #PRODUCTS
    data['Category_A'] = data['ProductList'].apply(lambda x : x.split(';')[0].split('/')[0])
    data['Category_B'] = data['ProductList'].apply(lambda x : x.split(';')[0].split('/')[1])
    data['Category_C'] = data['ProductList'].apply(lambda x : x.split(';')[0].split('/')[2])
    
    #Calling Function on ProductList to populate Product table.
    data['Items'] =  data['ProductList'].apply(lambda x: product_group(x))
    data['Product Code'] = data['Items'].apply(lambda x: final_items(x))
    data['Total_Products_Viewed'] = data['ProductList'].apply(lambda x: len(x.split(';'))) 
    
    #TIME
                
    data['startTime'] = pd.to_datetime(data['startTime'], format = "%d/%m/%y %H:%M")
    data['endTime']   = pd.to_datetime(data['endTime'], format = "%d/%m/%y %H:%M")
    data['StartHour'] = data['startTime'].dt.hour
    data['StartMinute'] = data['startTime'].dt.minute
    data['EndHour'] = data['endTime'].dt.hour
    data['EndMinute'] = data['endTime'].dt.minute
    data['Duration_InSeconds']  = (data['endTime']-data['startTime'])/np.timedelta64(1,'s')
    
    
    return data 
    
    


# In[ ]:


trainset = X.copy()
testset = X_test.copy()
trainset = CleanIt(trainset)
testset = CleanIt(testset)
print("----------------------------- Info on Processed Training Data--------------------")
display(trainset.shape)
display(trainset.dtypes)
display(trainset.describe())
display(trainset.info())
display(trainset.head())
print("******************************Info on Processed Testing Data***********************")
display(testset.shape)
display(testset.dtypes)
display(testset.describe())
display(testset.info())
display(testset.head())


# In[ ]:


#Dropping Unecessary Columns now - ProductList and Items.

trainset.drop(['ProductList', 'Items', 'startTime', 'endTime'], axis = 1 , inplace = True)
testset.drop(['ProductList', 'Items', 'startTime','endTime'], axis = 1, inplace = True)
display(trainset.head())
display(testset.head())


# ## *************************************** Basic EDA ****************************************************

# In[ ]:


trainset1 = trainset.copy()
trainset1['gender'] = train_df['gender'].copy()
plt.figure(figsize=(25,15))
sns.countplot('Category_A', data = trainset1, hue = 'gender')
plt.legend(loc = 'center')


# #### As, you can see clearly category 'A00002' is mostly preferred category, among females. Whereas Category 'A00001' is favoured by mostly males

# In[ ]:


plt.figure(figsize=(25,15))
sns.countplot('Category_B', data = trainset1.head(30), hue = 'gender')
plt.legend(loc = 'center')


# #### As, you can see clearly category 'B00001' is mostly preferred category, amongst  females. Also Category 'B00001' is favoured by mostly males, but Category 'B00031' is also preferred by males over female users.

# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot('StartHour', data = trainset1, hue = 'gender')
plt.legend(loc = 'upper left')


# #### Most Viewership by females is in hour 10th of the day, whereas males prefer pretty late in 20th hour.

# In[ ]:


plt.figure(figsize=(20,5))
sns.lineplot(x='StartHour', y = 'Total_Products_Viewed', data = trainset1, hue = 'gender')


# In[ ]:


sns.lmplot(x='StartHour', y = 'Total_Products_Viewed', data = trainset1, hue = 'gender')


# ### Let's Analyse How much reliable customers website have based on time spend amd Viewed Products

# In[ ]:


display(trainset1.Duration_InSeconds.describe())
display(trainset1.Duration_InSeconds.max())
display(trainset1.Duration_InSeconds.min())


# #### For Males.

# In[ ]:


#You will Get a warning message if you try to index this group with Multiple Keys. Avoid it by passing keys into a list.

group_AvgTimeSpend = trainset1.groupby('gender')[['Duration_InSeconds','Total_Products_Viewed']]

#For Males.
group_male = pd.DataFrame(group_AvgTimeSpend.get_group('male'))
display(group_male.sort_values('Total_Products_Viewed', ascending=False))
t1 = group_male.sort_values('Total_Products_Viewed', ascending=False)
plt.figure(figsize=(20,10))
sns.lineplot(x='Duration_InSeconds', y = 'Total_Products_Viewed', sort = False, data = t1)
plt.title('For Males')
display(group_male.sort_values('Duration_InSeconds', ascending = True))
t2 = group_male.sort_values('Duration_InSeconds', ascending = True)
sns.lineplot(x='Duration_InSeconds', y = 'Total_Products_Viewed', sort = False, data = t2)


# ##### For Females : 

# In[ ]:


group_female = pd.DataFrame(group_AvgTimeSpend.get_group('female'))
display(group_female.sort_values('Total_Products_Viewed', ascending=False))
t3 = group_female.sort_values('Total_Products_Viewed', ascending=False)
plt.figure(figsize=(20,10))
sns.lineplot(x='Duration_InSeconds', y = 'Total_Products_Viewed', sort = False, data = t3)
plt.title('For Females')
display(group_female.sort_values('Duration_InSeconds', ascending = True))
t4 = group_female.sort_values('Duration_InSeconds', ascending = True)
sns.lineplot(x='Duration_InSeconds', y = 'Total_Products_Viewed', sort = False, data = t4)


# #### For Both ------------------------------------- Analysing Viewing Behaviour----------------------------------------------------

# In[ ]:


plt.figure(figsize=(10,5))
sns.lineplot(x='Duration_InSeconds', y = 'Total_Products_Viewed', data = trainset1, hue = 'gender')


# #### Basic Analysis.
# 
# Clearly from the above lineplot you can say that the females users are pretty slow in decision making,as compared to males because the product viewing sessions are long for a few number of products for females. Whereas, males have viewed good number of product in limited time. Hence Female Users Spend more time on the Website than males. So company will have more viewership if invested in products frequently bought and viewed by females, provided such products have been put up to the caraousel.

# ## Model Building****
# 
# * We will be using CatBoostClassifier as for training the data and later on we will check how it is performing on the test set. Evaluation and Accuracy will be checked for the model. As, we have Categorical Variables to be dealt with so going with CatBoostClassifier will be good enough, as Catboost handles the Categorical features directly, no explicit OHE or encoding scheme is required.
# 
# 
# - trainset (Training Data) 
# - testset  (Prediction Data)

# In[ ]:


print(trainset.shape, testset.shape)


# In[ ]:


display(trainset.head())
display(trainset.dtypes)
display(testset.head())
display(trainset.dtypes)


# #### As You can see training data and Testing data are in same order, required for CatBoostClassifier to be in  correct order.

# In[ ]:


#### MODEL 

cate_features_index = np.where(trainset.dtypes != float)[0]
X1_train, X1_test, y1_train, y1_test = train_test_split(trainset, target, train_size=0.85,random_state=1200)
from catboost import CatBoostClassifier


cat = CatBoostClassifier(eval_metric='Accuracy',
                         use_best_model=True,random_seed=40,loss_function='MultiClass',
                         learning_rate = 0.674 ,iterations = 700,depth = 4,
                         bagging_temperature=3,one_hot_max_size=2)



#Parameters to be tuned :- 
#1. learning Rate
#2. Training Size(As data is biased)
#3. Iterations
#4. OHE size
#5. depth of tree

cat.fit(X1_train,y1_train ,cat_features=cate_features_index,eval_set=(X1_test,y1_test),use_best_model=True)
print('the test accuracy is :{:.6f}'.format(accuracy_score(y1_test,cat.predict(X1_test))))
predcat = cat.predict(X1_test)
print("----------------------------------------------------------------------------")
print('Training set score: {:.4f}'.format(cat.score(X1_train, y1_train)))
print('Test set score: {:.4f}'.format(cat.score(X1_test, y1_test)))


matrix = confusion_matrix(y1_test, predcat)
print("--------------------------------------------------------------------------------------------")
print('Confusion matrix\n\n', matrix)
print('\nTrue Positives(TP) Females  = ', matrix[0,0])
print('\nTrue Negatives(TN)  Males = ', matrix[1,1])
print('\nFalse Positives(FP) = ', matrix[0,1])
print('\nFalse Negatives(FN) = ', matrix[1,0])


# ##### Also You can Observe before we make final predictions. After evaluation we have found that training set score and test set score is almost equal. Hence We can say there is no Overfitting in the Data.

# ### Predictions And Submissions. !

# In[ ]:


preds = cat.predict(testset)
pred1 = preds.flatten()
predlst = pred1.tolist()
output = pd.DataFrame({'session_id': test_df.session_id,'gender': predlst})
output.to_csv('cleaned.csv', index=False)
sns.countplot(predlst)
pd.Series(predlst).value_counts()


# ### Results And Analysis -------------- 

# * So We Were able to Predict 3742 as females and  758 as males. Which gave us an Model Accuracy of 0.897 ! 
# * During The Hackathon We got a public score of 0.86833 and Private Score as of 0.8733 fairly close. 
# * The Dataset is unfairly unbalanced, Advanced Feature Engineering and Ensembling may help in Boosting the Accuracy.
# * You Guys can try with Ensembling by stacking up and building a baseline Model.
# 

# #### ******************************************************** THANKS ! ***********************************************************************************

# In[ ]:




