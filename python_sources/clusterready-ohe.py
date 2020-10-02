#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
import sklearn
import re


# In[ ]:


revdata = pd.read_csv("./Cluster-Input.csv") 
#df_2 = df_1_6.iloc[0:2] #use rows 0 through 2
#df_2


# In[ ]:


rawdict = pd.read_csv("../input/english-word-frequency/unigram_freq.csv",usecols =["word"],low_memory = False)

alphadict = rawdict.sort_values('word',ascending = True) #sort alphabetically

finaldict = alphadict.reset_index(drop = True) #reset the index numbering


# In[ ]:


word = finaldict['word'] #rename this column


# In[ ]:


X = [['and']] #The example for OHE to recognize

enc = preprocessing.OneHotEncoder(categories= [word], handle_unknown = 'ignore') #Use this column and ignore any unknown values

enc.fit(X) #do the thing


# In[ ]:


product = []

for line in revdata['comment'].values: #for every line, make it lowercase and take out all punctuation
    
    y = str.lower(line)
    
    REG = re.split('[,-.-" "]',y)
    
    product.append(np.array(REG).reshape(-1,1)) #put it in the previously defined empty array and make it columnar
    
    
print(product) #let's put it somewhere safe


# In[ ]:


#import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning) #suppress those annoying and unimportant warnings
 

P = []
for row in product: #for every row in product, transform each review using the dictionary and append to P
    T = enc.transform(row).toarray() 
    P.append(T)
#print(P)


# In[ ]:


N = []
for row in P: #for every row in P, add each line and make it 1 line for each review
    b = [sum(row[:,i]) for i in range(len(row[0]))]
    N.append(b)
df3 = pd.DataFrame(N)  #Convert N to a pandas dataframe


# In[ ]:


datarev.reset_index(drop=True)

df4 = pd.concat([df3, datarev['above 8?']], axis=1,ignore_index = False)

df4.rename(columns = {'above 8?':'Target'}, inplace = True) #append a new column that has the binary numbers called Target

df4.to_csv("Binrevs.csv")

