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
from scipy import stats
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
# Any results you write to the current directory are saved as output.


# In[ ]:


from kmodes.kmodes import KModes
df_allData=pd.read_csv('../input/BlackFriday.csv')
print(df_allData.sample(n=2))


# In our opinion, we think our model would do cluster on different people, so our key is people.
# 
# But the dataset includes different records of one-single people. So at the beginning we use gruopby to get each people's whole records.
# 
# Then we try to get the mode of each one's "Product_Category_1" to represent the main product category and get the mean of one people's whole "Purchase" as a feature of "average purchase". (We are not sure weather this is a good way buy we have to do this because we cannot keep all the data to train)
# 
# What's more, we change the "Gender" attribute to 0-1 attribute.

# In[ ]:


groupByUserData=df_allData.groupby(['User_ID'])

times=df_allData['User_ID'].value_counts()
times=times.sort_index()

#get the mean
meanData=groupByUserData.mean()

#get the mode
modeData=groupByUserData.agg(lambda x: stats.mode(x)[0][0])

mean_mode_data={'Gender':modeData['Gender'],'Occupation':modeData['Occupation'],'Age':modeData['Age'],'City_Category':modeData['City_Category'],'Marital_Status':modeData['Marital_Status'],'Product_CateGory_1':modeData['Product_Category_1'],'Stay_In_Current_City_Years':modeData['Stay_In_Current_City_Years']}
mean_mode_data=pd.DataFrame(mean_mode_data)
mean_mode_data['times']=times
mean_mode_data['Gender_M']=pd.get_dummies(mean_mode_data['Gender'])['M']
mean_mode_data=mean_mode_data.drop(['Gender'],axis=1)
mean_mode_data['Purchase']=meanData['Purchase']

print (mean_mode_data.sample(2))


# To cluster the categorical attributes, we came up with ideas.
# 
# 1. do one-hot encoding on the discrete attributes
# 
# 1. use k-modes or k-prototype model
# 
# 1. drop those categorical attributes
# 
# 

# In[ ]:


X=pd.DataFrame({'Gender':modeData['Gender'],'Occupation':modeData['Occupation'],'Age':modeData['Age'],'City_Category':modeData['City_Category'],'Marital_Status':modeData['Marital_Status'],'Product_CateGory_1':modeData['Product_Category_1'],"Stay_In_Current_City_Years":modeData["Stay_In_Current_City_Years"]})

one_hot_city=pd.get_dummies(mean_mode_data['City_Category'])
one_hot_age=pd.get_dummies(mean_mode_data['Age'])
one_hot_occupation=pd.get_dummies(mean_mode_data['Occupation'])
one_hot_years=pd.get_dummies(mean_mode_data['Stay_In_Current_City_Years'])
one_hot_product=pd.get_dummies(mean_mode_data['Product_CateGory_1'])
XX=pd.concat([one_hot_age,one_hot_city,one_hot_occupation,one_hot_years,one_hot_product],axis=1)
XX['Gender_M']=mean_mode_data['Gender_M']
XX['Marital_Status']=mean_mode_data['Marital_Status']

print ("categorical data:")
print(X.sample(2))
print("one-hot encoding data:")
print(XX.sample(2))


# At first, I think the key is finding the best clusters number n, but I don't have ideas about how to evalue the output. I just thought maybe different cluster's average price can reflect some difference.
# 
# I try to use jaccard distance and eulicdean distance, but the jaccard distance inside a cluster would alway decrease when n increase.

# In[ ]:


from sklearn.metrics import jaccard_similarity_score
ecArr=[]
jcArr=[]
jcXArr=[]
for i in range(2,10):
    km=KModes(n_clusters=i)
    y=km.fit_predict(X)
    tempArrjc=[]
    tempArrec=[]
    tempArrjcX=[]
    for j in range(i):
        #print(sum(y==j))
        #print(XX[y==j].mode())
        jcscore=[]
        ecscore=[]
        jcXscore=[]
        for k in XX[y==j].T:
            try:
                #jcscore.append(jaccard_similarity_score(XX.loc[k],XX[y==j].mode().T[0]))
                
                ecscore.append(np.linalg.norm(np.array(XX.loc[k])-np.array(XX[y==j].mode().T[0])))
                
                jcXscore.append(jaccard_similarity_score(list(X.loc[k]),list(X[y==j].mode().T[0])))

            except:
                #print(XX.loc[k].T)
                #print(XX[y==j].mode())
                print(k)
                break;
        #print(np.mean(jcscore))
        #tempArrjc.append(np.mean(jcscore))
        #tempArrec.append(np.mean(ecscore))
        tempArrjcX.append(np.mean(jcXscore))

    print("n_cluster =",i,":",np.mean(tempArrjcX))
    #jcArr.append(np.mean(tempArrjc))
    #ecArr.append(np.mean(tempArrec))
    jcXArr.append(np.mean(tempArrjcX))


# Lack of the ways to evaluate better n, I decide to focus on features more rather than n.
# 
# Then I camp up with an idea that I can calculate the eulidean distance between a cluster's mode point and the other point's mode point.
# 
# So I do some tries.

# In[ ]:


XXXX=X.drop(['Marital_Status','Product_CateGory_1','Stay_In_Current_City_Years','Age'],axis=1)
print(XXXX.sample(2))
from sklearn.metrics import jaccard_similarity_score
ecArr=[]
jcArr=[]
jcXArr=[]
for i in range(10,11):
    km=KModes(n_clusters=i)
    y=km.fit_predict(XXXX)
dis_jc=[]
dis_ec=[]
for i in range(10):
    dis_jc.append(jaccard_similarity_score(list(XXXX[y==i].mode().T[0]),list(XXXX[y!=i].mode().T[0])))
    
print("average jc distance in selected features:",np.mean(dis_jc))
    
for i in range(10):
    dis_ec.append(np.linalg.norm((np.array(XX[y==i].mode().T[0])-np.array(XX[y!=i].mode().T[0]))))
    
print("average ec distance in all one-hot features:",np.mean(dis_ec))   


# In[ ]:


purchase_y=pd.DataFrame({"y":y,"Purchase":mean_mode_data["Purchase"]})
plt.scatter(purchase_y['y'],purchase_y['Purchase'])
for i in range(10):
    plt.scatter(i,purchase_y[purchase_y['y']==i].Purchase.mean(),c='r')


# In[ ]:


XXXXX=X.drop(['Stay_In_Current_City_Years'],axis=1)
print(XXXXX.sample(2))
ecArr=[]
jcArr=[]
jcXArr=[]
for i in range(10,11):
    km=KModes(n_clusters=i)
    y=km.fit_predict(XXXXX)

dis_jc=[]
dis_ec=[]
for i in range(10):
    dis_jc.append(jaccard_similarity_score(list(XXXXX[y==i].mode().T[0]),list(XXXXX[y!=i].mode().T[0])))
    
for i in range(10):
    dis_ec.append(np.linalg.norm((np.array(XX[y==i].mode().T[0])-np.array(XX[y!=i].mode().T[0]))))
    
print("average jc distance in selected features:",np.mean(dis_jc))
print("average ec distance in all one-hot features:",np.mean(dis_ec))


# In[ ]:


purchase_y=pd.DataFrame({"y":y,"Purchase":mean_mode_data["Purchase"]})
plt.scatter(purchase_y['y'],purchase_y['Purchase'])
for i in range(10):
    plt.scatter(i,purchase_y[purchase_y['y']==i].Purchase.mean(),c='r')


# ## conclusion
# 
# Thought the process is hard, I think I get something surpringly at the end.
# 
# I paid too much attentation on the cluster number n and the distance between a cluster but I haven't had a great evaluation way.
# 
# But when I campared different features as input to k-modes with the same n, I got some pretty things.
# 
# The change in average euclidean distance showed the feature 'Marital_Status' , 'Product_CateGory_1', 'Age' do influence on the whole cluster performance.
# 
# What't more, the 'Purchase' would really reflect the performance of cluster in some way.
# 
# With this conclusion, it means we can do further work about how different categorial features influence the cluster. The best feature may have the biggest influence on average euclidean distance.
