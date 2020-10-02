#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import io, bson
import matplotlib.pyplot as plt
from skimage.data import imread   # or, whatever image library you prefer

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# The code below will read the BSON files into a pandas dataframe, and then read the category_id that we are trying to predict into a numpy matrix Y, the images that we are using to predict into a matrix X, and the unique ID's into a matrix X_ids.

# In[ ]:


from scipy.misc import imshow
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


n = 5000
pix_x =180
pix_y =180
rgb = 3
X_ids = np.zeros((n,1)).astype(int)
Y = np.zeros((n,1)).astype(int) #category_id for each row
X_images = np.zeros((n,pix_x,pix_y,rgb)) #m images are 180 by 180 by 3


with open('../input/train.bson', 'rb') as f:
    data = bson.decode_file_iter(f)
    counter = 0
    for c, d in enumerate(data):


        X_ids[counter] = d['_id'] 
        Y[counter] = d['category_id'] 
        for e, pic in enumerate(d['imgs']):
            picture = imread(io.BytesIO(pic['picture']))


        print(counter)
        X_images[counter] = picture #add only the last image 

        counter+=1
        if counter >= n:
            break


# In[ ]:


# from matplotlib import pyplot as plt

# for i in range(10):
#     plt.imshow(X_images[i], interpolation='nearest')
#     print(Y[i])
#     plt.show()


# In[ ]:


#Lets take a look at the category names supplied to us:
df_categories = pd.read_csv('../input/category_names.csv', index_col='category_id')

count_unique_cats = len(df_categories.index)

print("There are ", count_unique_cats, " unique categories to predict. E.g.")
print("")
print(df_categories.head())


# In[ ]:


#Function to return the category description from df_categories
def get_category(data, category_id,level):
    if(level in range(1,4)):
        try:
            return data.iloc[data.index == category_id[0],level-1].values[0]
        except:
            print("Error - category_id does not exist")
    else:
        print("Error - level must be between 1 - 3")

#Play around with the index and cat levels to explore the images in the test data set
index = 14
cat_desc_level = 1 # level 1 - 3
print("ID: ",X_ids[index][0], "category_id: ",Y[index][0], "category_description: ",get_category(df_categories,Y[index],cat_desc_level))
plt.imshow(X_images[index])


# In[ ]:


Y_new = []
for i in range(len(Y)):
    Y_new.append( get_category(df_categories,Y[i],1))


# In[ ]:


from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore") 

#full list of classes
# category_classes = df_categories.index.values
# category_classes = category_classes.reshape(category_classes.shape[0],1)


# In[ ]:


df_categories['category_level1'].values


# In[ ]:


le = preprocessing.LabelEncoder() 
lb = preprocessing.LabelBinarizer()
le.fit(df_categories['category_level1'].values)
y_encoded = le.transform(Y_new) # Label Encoding


# In[ ]:





# In[ ]:


# #binarizer to convert all unique category_ids to have a column for each class 
# lb.fit(y_encoded)
# Y_flat = lb.transform(y_encoded)


# In[ ]:


# Y_flat.shape


# In[ ]:


X_flat = X_images.reshape((n,-1))


# In[ ]:



# m = X_flat.shape[1]
# n = Y_flat.shape[1]

#Scale RGB data for learning
X_flat = X_flat/255
#print results
print("X Shape =", X_flat.shape, "Y Shape =",y_encoded.shape,"N_unique_label=",len(set(y_encoded)) )


# ### Try PCA for dimension red:

# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


pca = PCA(n_components=100,random_state=0)
pca.fit(X_flat)


# In[ ]:


pd.DataFrame(pca.explained_variance_).plot()


# In[ ]:


pca = PCA(n_components=20,random_state=0)
new_X = pca.fit_transform(X_flat)


# In[ ]:


new_X.shape


# In[ ]:


from sklearn.linear_model import LogisticRegression 
model = LogisticRegression(random_state=0,solver='sag',n_jobs=-1) 


# In[ ]:


model.fit(X=new_X,y=y_encoded)


# In[ ]:


pred = model.predict(new_X)


# In[ ]:


model.predict_proba(new_X)


# In[ ]:


pred = pred.tolist()
y_encoded = y_encoded.tolist()


# In[ ]:


sum([pred[i]==y_encoded[i] for i in range(len(pred))])/float(len(y_encoded))


# ### Submission

# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


from sklearn.utils import shuffle


# In[ ]:


with open('../input/test.bson', 'rb') as f:
    data = bson.decode_file_iter(f)

    counter = 0
    for c, d in enumerate(data):


        
        pred = []
        for e, pic in enumerate(d['imgs']):
            picture = imread(io.BytesIO(pic['picture']))
            picture = picture.reshape(1,-1)
            picture = picture/255.0
            picture = pca.transform(picture)
            pred1 = model.predict_proba(picture)
            pred.append(pred1)
        
        
        pred = np.array(pred).mean(axis=0) # average all single predictions
        pred = model.classes_[np.argmax(pred)]
        pred2 = le.classes_[pred]

        res_dat = df_categories[df_categories['category_level1']==pred2]
        shuffle(res_dat)
        sub.iloc[counter]['category_id'] = res_dat.index.values[0]
        if counter % 2000 == 0:
            print(counter)

# #         X_images[counter] = picture #add only the last image 

        counter+=1


# In[ ]:





# 

# 
