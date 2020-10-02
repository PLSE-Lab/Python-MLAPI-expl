#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import numpy as np
import math
import re
import os
import csv
import datetime
import matplotlib.pyplot as plt


# In[31]:


from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# 

# # Sort by user and parse into the movie pairwise matrix

# # Calculate Movie Bias and User Bias

# # Join movie_bias and user_bias to df so that the new table containts the following
#     Rating, Date, Movie_Id, Cust_Id, Movie_Bias, User_Bias, ...
#             ... Neighbor_Movie_Rating_1, Neighbor_Movie_Rating_2 (controlled by K)
# - K is the total number of neighbor movies taken into consideration
# - Net_rating can be calculated via total_avg+Movie_bias+User_Bias
# - Or maybe just use "Rating" as Y and make sure the two bias have similar weight

# # Loading training dataset

# In[ ]:


# save df_final to file!!!
#df_final.to_csv('movie_df_final.csv',sep=',',float_format='%.4f')

# read from here

#print(os.listdir("../input/qualifying-dataset"))
df_final = pd.read_csv('../input/training-dataset/df_final_small.csv',sep=',')
df_final['Movie_Id'] = df_final['Movie_Id'].astype(int)
df_final['Cust_Id'] = df_final['Cust_Id'].astype(int)

print(datetime.datetime.now())


# In[ ]:


print(df_final.shape)
df_final.head()


# # Loading probe dataset

# In[ ]:



# read from file
df_probe=pd.read_csv('../input/qualifying-dataset/df_probe.csv',sep=',',index_col=0)
df_probe['Movie_Id'] = df_probe['Movie_Id'].astype(int)
df_probe['Cust_Id'] = df_probe['Cust_Id'].astype(int)
#df_probe.head()


# In[ ]:


print(df_probe.shape)
df_probe.head()


# # define validation set

# In[ ]:


#X_Val = df_probe[['Movie_Bias','User_Bias','nb_rating_1','nb_rating_2','nb_rating_3']].copy() 
X_Val = df_probe[['Movie_Bias','User_Bias','nb_rating_1']].copy() 
# no Y


# # I.b linear regression - use entire training set and make prediction on validation set

# In[ ]:


# use entire dataset, no need to randomize
#X_train = df_final[['Movie_Bias','User_Bias','nb_rat_off_1','nb_rat_off_2','nb_rat_off_3']].copy()
X_train = df_final[['Movie_Bias','User_Bias','nb_rat_off_1']].copy()

Y_train = df_final[['Rating']].copy()


# In[ ]:


# linear regression
reg = LinearRegression()
# fit
reg.fit(X_train, Y_train)
# make prediction
Y_Val=reg.predict(X_Val)


# In[ ]:


# Capping between 1 and 5 improves the MSE
Y_Val_cap=np.clip(Y_Val, 1, 5)
#print(math.sqrt(mean_squared_error(Y_Val_cap, Y_pred_linear_cap)))


# In[ ]:


# print to file
N=int(df_final.iloc[-1]['Movie_Id'])
with open('./output.txt', 'w') as output_file:

    for mov_id in range(1,N+1):

        print('%d:'%mov_id, file=output_file)
        index=df_probe.index[df_probe['Movie_Id']==mov_id]
        #print('%.4f:'%Y_Val[index].tolist(),sep = "\n",  file=output_file)
        print('\n'.join(' '.join(map('{:.4f}'.format,sl)) for sl in Y_Val[index].tolist()), file=output_file)
output_file.close()

