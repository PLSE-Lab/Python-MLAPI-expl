#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# This is a humble attempt to try and make any reader, especially one new to the fields of data science, a little more aware of how things work together. How you go from a simple data table to actually predicting something.
# 
# I simply started with a Kaggle Starter notebook for the Breast Cancer dataset and all the EDA you see in the beginning of this article was done by the Kaggle Bot. I start my own work, a little later.  
# 
# In this article, we will take the Breast Cancer dataset and learn about Support vector Machines and how to use them as we build our own prediction/classification model.
# 
# If you really want to support research towards Breast Cancer please visit https://www.breastcancer.org/ to learn more about the disease.

# ## Exploratory Analysis ( Thank you KaggleBot for this )
# To begin this exploratory analysis, first use `matplotlib` to import libraries and define functions for plotting the data. Depending on the data, not all plots will be made. (Hey, I'm just a kerneling bot, not a Kaggle Competitions Grandmaster!)

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns


# There is 1 csv file in the current version of the dataset:
# 

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# The next hidden code cells define functions for plotting data. Click on the "Code" button in the published kernel to reveal the hidden code.

# In[ ]:


# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()


# In[ ]:


# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()


# In[ ]:


# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()


# Now you're ready to read in the data and use the plotting functions to visualize the data.

# ### Let's check 1st file: /kaggle/input/data.csv

# In[ ]:


nRowsRead = 1000 # specify 'None' if want to read whole file
df1 = pd.read_csv('/kaggle/input/data.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'data.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')


# Let's take a quick look at what the data looks like:

# In[ ]:


df1.head(5)


# Distribution graphs (histogram/bar graph) of sampled columns:

# In[ ]:


plotPerColumnDistribution(df1, 10, 5)


# Correlation matrix:

# In[ ]:


plotCorrelationMatrix(df1, 8)


# Scatter and density plots:

# In[ ]:


plotScatterMatrix(df1, 20, 10)


# ## Personal Work
# ( From this point on, we are on our own - no more KaggleBot. )
# 
# The first thing we will do is to create a pairplot - The bot has already done that for us above but it used a complex function to do so and not only that, you can't tell the malignant from benign cases in the previous visualization.
# Here we will use Seaborn to create out plot and we will color it accordingly for the different cases. The agenda here is to get a 'bird-eye' view to maybe build intuition about what variables can really help us.

# In[ ]:


sns.pairplot(df1, hue='diagnosis',vars = ['radius_mean','concave points_mean','area_mean','smoothness_mean'])


# Here the cases are seperated into malignant and benign cases. categorized according to the color, we can already see how some of the columns like 'area_mean' and 'concave points_mean' are good classifiers for our use case because we can 'see' a clear seperation in the two variables. It can be important here, to note why and how 'smoothness_mean' is not as good of a classifier when compared to the others. Learning to read whatever plot you create is a key Data Analysis skill.
# 
# Next in line is to figure out a base model with any which variables that we prefer. The motive to build this model is not to predict anything but just to have something to compare against the final model that we will keep. However, we are better off taking some 'educated' guesses here.

# In[ ]:


# The first order of business now is to create new variables which you will then feed into the model functions
# We will create training and test sets to both train our data on and then test our model as well.

X = df1.drop(['diagnosis'],axis=1)#drop the target class from train set
Y = df1['diagnosis']#seperating out the target class
print(X.columns)


# In[ ]:


## Extra step to deal with null data
X = X.dropna(axis=1) # axis-1 is running horizontal across columns. axis=0 is downwards across rows
print(X.columns)
#NOTE : I was getting a 'null' error while making my model. I tried removing 'Unnamed: 32' just as I had 
# removed the 'diagnosis' column, however the error still crept up, so I used the old tried and trusted
# .dropna() function.


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
#notice how the train_test_split function takes the test size and not the 
#size of the train size.Also 42 is the answer to everything.


# In[ ]:


#Now that our sets to train are ready, we may just choose to look at their
#new dimensions as well
print(X_train.shape)


# In[ ]:


print(Y_train.shape)


# In[ ]:


# I love using the sns heatmap to visualize the null values
sns.heatmap(X_train.isnull())


# In[ ]:


#Now to actually building the model
from sklearn.svm import SVC #from support vector machine get support vector classifier
from sklearn.metrics import classification_report, confusion_matrix #these will make sense later

svc_model = SVC() #initialize the code module
svc_model.fit(X_train,Y_train) #fit your data variables to the model


# ****Evaluating our first test model****

# In[ ]:


y_predict = svc_model.predict(X_test) # get 'y' by predict FOR the TEST values.
cm = confusion_matrix(Y_test,y_predict) # we use the confusion matrix to check
                                        #how our predicted y compares to the
                                        # real one.
sns.heatmap(cm,annot=True)


# Here it is interesting to see that our first model has actually classified all points as 'benign'. From this point on, it will be beneficial to try and understand how we build our next models and how they contrast to this one.

# In[ ]:


print(classification_report(Y_test,y_predict))


# The classification report only pushes further the fact that our model can NOT classify and identify malignant cases.

# **Improving our model**
# 
# The 'spread' of our data is across ranges of many magnitudes. While on the one hand, we have a column with values betwwen 0.0 and 0.26, on the other hand there are columns with values ranging from 200 to 2500. Having such contrasting ranges can throw off our model's predictions.
# 
# We will use 'unity-based normalization' to solve this problem. We will essentially turn all the columns to a 0-1 range. This will allow a proper foundation for our prediction model.
# 
# https://en.m.wikipedia.org/wiki/Feature_scaling

# In[ ]:


min_train = X_train.min()
min_train


# In[ ]:


range_train = (X_train-min_train).max()
range_train


# In[ ]:


X_train_scaled = (X_train-min_train)/range_train
X_train_scaled


# Let's now visualize some of our new columns and compare with our previous X_train(without scaling)

# In[ ]:


#without scaling
sns.scatterplot(x=X_train['area_mean'],y=X_train['smoothness_mean'],hue=Y_train)


# In[ ]:


#with scaling
sns.scatterplot(x=X_train_scaled['area_mean'],y=X_train_scaled['smoothness_mean'],hue=Y_train)


# In[ ]:


#Now we need to do the same normalization for our TEST set
min_test = X_test.min()
range_test = (X_test-min_test).max()
X_test_scaled = (X_test-min_test)/range_test


# In[ ]:


#setting up our new SVC model
svc_model = SVC()
svc_model.fit(X_train_scaled,Y_train)


# In[ ]:


y_predict = svc_model.predict(X_test_scaled)
cm = confusion_matrix(Y_test,y_predict)
sns.heatmap(cm,annot=True,fmt='d')


# Just with the simple improvement of our scaled features, our model has gotten better my many magnitudes. While before we had completely mis-classified all the malignant cases. Now, our model has only mis-classified a total of two cases. Let's also print the classification report for this new model ( we will get a number for this model's precision ).

# In[ ]:


print(classification_report(Y_test,y_predict))
# we get a 99% precision already.


# **Improving our model, this time with hyperparameters.**

# All machine learning modules packaged as Python libraries, have their unique set of hyperparameters. These 'parameters' are the gateway to fine tuning our models. The two such variables which concern us here are 'c' and 'gamma'.  
# 
# Even though our model might already seem very strong, it is a good exercise to work with these fine tuning variables as well.
# 
# **Refresher**  
# 'c' - this is a measure of how much you want the model to penalize a wrong classification. A higher value of 'c' can lead to over training.  
# 'gamma' - Sometimes it is fruitful to add weight to our data points. A higher gamma value will give a heavier 'weight' to points bunched closely together and also reduce the weightage to outlier points.
# 
# Needless to say, it can be a tedious task trying to manually figure out a perfect combination of these variables. At this point, you can rest assure there is a python library/function to do everything and here we will use GridSearchCV to figure out the best set of hyperparameters for us.

# In[ ]:


param_grid = {'C':[0.1,1,10,100],'gamma':[0.1,1,0.01,0.001],'kernel':['rbf']}
from sklearn.model_selection import GridSearchCV #read GridSearchCrossValidation
grid = GridSearchCV(SVC(),param_grid,)


# Kernel Selection is a rather important skill which comes in handy for many different kinds of classification models, especially Support Vector Machines.
# https://www.kdnuggets.com/2016/06/select-support-vector-machine-kernels.html  
# This is a great article, a quick 5 minute read to clarify how and why different kernels exist and how to pick one for yourself. We could add all kernels to the GridSearch function but that would be excessive and unnecessary computing power and time.  
# 
# Reading through documentation to learn what different parameters represent is both very energy draining and also a very crucial part of the job. Here we use refit and verbose parameters with GridSearchCV. Here are two links for the same:  
# https://stackoverflow.com/questions/26962050/confused-with-repect-to-working-of-gridsearchcv  
# https://stackoverflow.com/questions/24121018/sklearn-gridsearch-how-to-print-out-progress-during-the-execution   
# Lesson learned : Documentation is not always super lear, get used to experimenting with such things to get a better feel of what they are about. Doing data science is the best way to learn data science.

# In[ ]:


grid.fit(X_train_scaled,Y_train)


# In[ ]:


grid.best_params_


# In[ ]:


grid.best_estimator_


# In[ ]:


grid_predictions = grid.predict(X_test_scaled)


# In[ ]:


cm = confusion_matrix(Y_test,grid_predictions)
sns.heatmap(cm,annot=True)


# In[ ]:


print(classification_report(Y_test,grid_predictions))


# ## The naked truth

# As it is very nicely visible, we lost prediction power in our model.  
# Lesson learned - 'Life is not all sunshine and rainbows'
# 
# The takeaway at this point is that our model can do the required task of properly classifying breast cancer cases. However if we do want to create a very fine tuned model to do our job, just to be sure, we need a lot more data. Not only that, we need a lot more experimentation time as well. We can see that our previous model where gamma was 'auto_deprecated' performed better than having a set gamma value. Even if we do decide to stick with our simpler model, we can be never sure of how it holds against a bigger dataset.
# 
# Making predictions and taking decisions upon them, has been,is and shall always be a dirty task. LOL
# 
# Happy Data Hacking!!

# ## Conclusion
# I started writing this kernel with the goal of creating something, I never had. Someone to tell me, how it all ties together.  
# 
# 'DOING' data science is very different from simply learning it, rather I might say that if you haven't DONE data science, you never really learned it.
# I hope I was able to clear somethings for the new people, I hope the old people read this and help me iron out my own mistakes.
# 
# We are all in this together!! I'll be more than happy to start a long conversation in the comments :D
