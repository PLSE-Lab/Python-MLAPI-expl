#!/usr/bin/env python
# coding: utf-8

# # KNN for Classification

# The dataset I will be using is the "Glass Classification" dataset found on Kaggle at: https://www.kaggle.com/uciml/glass. The study of classification of types of glass was motivated by criminological investigation. At the scene of the crime, the glass left can be used as evidence if it is correctly identified.
# 
# With the techniques I use below, I was able to take the KNN algorithm from 79% accuracy to 88% accuracy. With different random states, the model would score over 93%.

# ## 1. Get to know your dataset

# <u><b>The Features</b></u>:
# 
# <b>RI</b>: refractive index
# 
# <b>Na</b>: Sodium (unit measurement: weight percent in corresponding oxide, as are attributes 4-10)
# 
# <b>Mg</b>: Magnesium
# 
# <b>Al</b>: Aluminum
# 
# <b>Si</b>: Silicon
# 
# <b>K</b>: Potassium
# 
# <b>Ca</b>: Calcium
# 
# <b>Ba</b>: Barium
# 
# <b>Fe</b>: Iron

# <u><b>The Dependent Variable</b></u>
# 
# <b>Type of Glass</b>: (class attribute) -- 1 building_windows_float_processed -- 2 building_windows_non_float_processed -- 3 vehicle_windows_float_processed -- 4 vehicle_windows_non_float_processed (none in this database) -- 5 containers -- 6 tableware -- 7 headlamps

# In[ ]:


#import the required modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#import our data file "glass.csv" into a pandas dataframe
glass_df = pd.read_csv("../input/glass/glass.csv")


# In[ ]:


#get information about the general values of each feature
glass_df.describe()


# In[ ]:


#see the first 10 values of the dataset
glass_df.head(10)


# In[ ]:


#get info about the datatypes of each feature
glass_df.info()


# In[ ]:


#get counts of null values for each feature
glass_df.isnull().sum()


# In[ ]:


#plot a histogram of each feature
glass_df.hist(figsize=(20,20))


# In[ ]:


# Checking to see how spread out our features values are
ax = sns.boxplot(data=glass_df)


# The features don't seem to have a very large difference in range. Normalization could still help our model though.

# In[ ]:


#Get the distribution of the different classifications
glass_df.Type.value_counts().plot(kind="bar")


# We have an unequal distribution of dependent variables. We may need to use oversampling to even that out.

# ## 2. Clean the data

# We don't have any missing values. However, we will do some manipulation to the data after we run our first KNN model.

# ## 3. Load the data

# In[ ]:


# This will be the dataframe containing the features used to train our model
X = pd.DataFrame(glass_df.drop(["Type"], axis = 1),
            columns=['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe'])

# This will be the dataframe containing the labels of each data point
y=glass_df.Type


# In[ ]:


#importing train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=1/5, random_state=5, stratify = y)


# Noticed inside the train_test_split function, we used "stratify = y." The stratify parameter makes a split so that the proportion of values in the sample produced will be the same as the proportion of values provided to the "stratify" parameter. In our case, we are using it to ensure there is an equal proportion of labels in our training and testing sets.

# ## 4. Run the KNN algorithm over a range of different K values

# In[ ]:


#importing the k classifier
from sklearn.neighbors import KNeighborsClassifier


test_scores = []
train_scores = []

# we will be testing 14 different values for k, starting with 1 and ending before 15
for i in range(1,15):

    knn = KNeighborsClassifier(i)
    knn.fit(X_train,y_train)
    
    train_scores.append(knn.score(X_train,y_train))
    test_scores.append(knn.score(X_test,y_test))


# ## 5. Assess the model

# In[ ]:


# This score comes from testing on the same datapoints that were used for training
max_train_score = max(train_scores)
train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]
print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))


# In[ ]:


# This score comes from testing on the datapoints that were split in the beginning to be used for testing solely
max_test_score = max(test_scores)
test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))


# In[ ]:


#plotting the train and test score for each value of k we tested
plt.figure(figsize=(12,5))
p = sns.lineplot(range(1,15),train_scores,marker='*',label='Train Score')
p = sns.lineplot(range(1,15),test_scores,marker='o',label='Test Score')


# The k with the highest accuracy for both the training and testing data was 1.

# ## 6. Improve the model

# A few ways that we can improve the model is through normalization, which will reduce the distance between our various features, Principal Component Analysis, which will reduce our model's dimensions down to a smaller number without losing very much information, and oversampling with SMOTE which will even out the distribution of samples.

# ### Normalizing the data

# Normalizing our data will ensure that all of our features are being treated equally by our KNN model. Features that range greatly in value compared to other features can hurt the effectiveness of our KNN model. Normalizing/Standardizing our data will put each value between 0 and 1.

# In[ ]:


from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
X_minmax = min_max_scaler.fit_transform(X)
X_norm = pd.DataFrame(X_minmax)


# In[ ]:


# a peek at the new normalized features
X_norm.head()


# In[ ]:


#now all of our values fall between 0 and 1
ax = sns.boxplot(data=X_norm)


# In[ ]:


#now lets run the model on the normalized data and see if it has any effect on the accuracy
#importing train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_norm,y, test_size=1/5, random_state=5, stratify = y)

test_scores = []
train_scores = []

# we will be testing 14 different values for k, starting with 1 and ending before 15
for i in range(1,15):

    knn = KNeighborsClassifier(i)
    knn.fit(X_train,y_train)
    
    train_scores.append(knn.score(X_train,y_train))
    test_scores.append(knn.score(X_test,y_test))


# In[ ]:


# This score comes from testing on the same datapoints that were used for training
max_train_score = max(train_scores)
train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]
print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))


# In[ ]:


# This score comes from testing on the datapoints that were split in the beginning to be used for testing solely
max_test_score = max(test_scores)
test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))


# We can see that normalizing our data increased our model's accuracy by 2%. Now let's try Principal Component Analysis and see if that further improves our results.

# ### Principal Component Analysis (PCA)

# PCA is a dimensionality reduction technique that allows us to keep most of the impoortant information from our data, while reducing the total number of features our model is using to predict the Type of glass. KNN models suffer from the curse of dimensionality, which basically means that the more features you are using to predict your outcome, the less accurate the model will be because of the distance between each data point. If we reduce the dimensionality, our model might become more accurate.

# In[ ]:


from sklearn.decomposition import PCA

scores = []

for i in range(9):
    pca = PCA(n_components=(i+1))
    principalComponents = pca.fit_transform(X_norm)
    principalDf = pd.DataFrame(data = principalComponents)
    
    X_train, X_test, y_train, y_test = train_test_split(principalDf,y, test_size=1/5, random_state=5, stratify = y)

    test_scores = []
    train_scores = []

    # we will be testing 14 different values for k, starting with 1 and ending before 15
    for i in range(1,15):

        knn = KNeighborsClassifier(i)
        knn.fit(X_train,y_train)
    
        train_scores.append(knn.score(X_train,y_train))
        test_scores.append(knn.score(X_test,y_test))
        
    # This score comes from testing on the datapoints that were split in the beginning to be used for testing solely
    max_test_score = max(test_scores)
    scores.append(max_test_score)
    
for i in range(len(scores)):
    print("With {} components, our accuracy was {}.".format(i+1,scores[i]))


# In[ ]:


plt.figure(figsize=(12,5))
p = sns.lineplot(range(1,10),scores)
p.set(xlabel="Number of Principal Components", ylabel="Model Accuracy", title = "Model Accuracy per number of Principal Components")


# With only 7 components, our model performed just as well as the normalized model with 9 components. However the overall accuracy was not increased.

# In[ ]:


print("Our model had a maximum accuracy score of {:.2f}% with {} principal components.".format(max(scores)*100,scores.index(max(scores))+1))


# ### Oversampling

# If our model's dependent variables, in our case Type of Glass, have an unequal distribution, our model might benefit from an oversampling technique called SMOTE. SMOTE will even out the difference by inserting syntheticly produced rows of data which fall in between the already existing minority dependent variables.

# In[ ]:


ax = y.value_counts().plot(kind="bar")
ax.set(xlabel="Type of Glass", ylabel="Count", title = "Before SMOTE")


# As you can see, a couple of our types of glass have many more values than the rest. SMOTE will fill in the difference.

# In[ ]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
x_res, y_res = sm.fit_resample(X_norm,y)
y_res_df = pd.DataFrame(y_res)
ax = y_res_df.Type.value_counts().plot(kind="bar")
ax.set(xlabel="Type of Glass", ylabel="Count", title = "After SMOTE")


# Now our data is equally distributed accross each of our different dependent variables. Lets run our KNN algorithm over the new data.

# In[ ]:


scores = []

for i in range(1,3):
    pca = PCA(n_components=(i))
    principalComponents = pca.fit_transform(x_res)
    principalDf = pd.DataFrame(data = principalComponents)
    
    X_train, X_test, y_train, y_test = train_test_split(x_res,y_res, test_size=1/5, random_state=5, stratify = y_res)

    test_scores = []
    train_scores = []

    # we will be testing 14 different values for k, starting with 1 and ending before 15
    for i in range(1,15):

        knn = KNeighborsClassifier(i)
        knn.fit(X_train,y_train)
    
        test_scores.append(knn.score(X_test,y_test))
        
    # This score comes from testing on the datapoints that were split in the beginning to be used for testing solely
    max_test_score = max(test_scores)
    scores.append(max_test_score)
    
for i in range(len(scores)):
    print("With {} components, our accuracy was {:.2f}%.".format(i+1,scores[i]*100))


# After Normalizing, running SMOTE, and running PCA our KNN model is 88% accurate with only 1 principal component. With SMOTE you run the risk of overfitting. Our model is just as accurate with 2 components. Lets see what a 2D plot of our datapoints looks like.

# In[ ]:


finalDf = pd.concat([principalDf, y_res], axis = 1)
color_dict = dict({1:'brown',
                  2:'green',
                  3: 'orange',
                  5: 'red',
                   6: 'dodgerblue',
                  7: 'purple'})
plt.figure(figsize=(10,10))
ax = sns.scatterplot(x=0,y=1,hue="Type",data=finalDf, palette=color_dict)
ax.set(xlabel="PC1",ylabel="PC2", title = "Our Final Dataset with 2 Principal Components")


# With this visualization, you can see where the various types of glass tend to group together.

# ## Conclusion

# We were able to solve our model's problem of unequal range with normalization. We used SMOTE to solve the dependent variables unequal distribution problem. We used PCA to reduce the dimensionality of our dataset and visualize our clusters.
# 
# I hope you enjoyed this notebook. Thank you for reading.

# ## Credits

# I took inspiration from the kaggle user "Shruti_lyyer" and her KNN guide: https://www.kaggle.com/shrutimechlearn/step-by-step-diabetes-classification-knn-detailed
# 
# The kaggle user "KenTu" has also put together a good notebook on KNN, PCA, and Oversampling: https://www.kaggle.com/yundoou/knn-precision-rate-grows-from-72-to-92
