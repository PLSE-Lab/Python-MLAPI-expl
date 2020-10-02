#!/usr/bin/env python
# coding: utf-8

# # IBM HR Analytics Employee Attrition & Performance          

# ## [Please star/upvote it if you like it.]

# In[ ]:


from IPython.display import Image
Image('../input/imagesannibm/image-logo.png')
 


# 1. ######  Note that you can view the same kernel on my Github acccount also::--

# https://github.com/mrc03/IBM-HR-Analytics-Employee-Attrition-Performance

# ## CONTENTS :

# [ **1 ) Exploratory Data Analysis**](#content1)

# [ **2) Corelation b/w Features**](#content2)

# [ **3) Feature Selection**](#content3)

# [**4) Preparing Dataset**](#content4)

# [ **5) Making Predictions Using an Artificial Neural Network (ANN)**](#content5)
# 
#  Note that this notebook uses ANN. I have another notebook in which I have used traditional ML algorithms on the same dataset. To  check it out please follow the below link-->
# 
# https://www.kaggle.com/rajmehra03/imbalanceddata-predictivemodelling-by-ibm-dataset/

# [ **6) Hyperparameter Tuning**](#content6)

# [ **7)Conclusions**](#content7)

# <a id="content1"></a>
# ## 1 ) Exploratory Data Analysis

# ## 1.1 ) Importing Various Modules

# In[ ]:


# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import missingno as msno

#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#import the necessary modelling algos.
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

#model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV

from imblearn.over_sampling import SMOTE

#preprocess.
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Imputer,LabelEncoder,OneHotEncoder

# ann and dl libraraies
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical

import tensorflow as tf
import random as rn


# ## 1.2 ) Reading the data from a CSV file

# In[ ]:


df=pd.read_csv(r'../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.columns


# ## 1.3 ) Missing Values Treatment

# In[ ]:


df.info()  # no null or Nan values.


# In[ ]:


df.isnull().sum()


# In[ ]:


msno.matrix(df) # just to visualize.


# ## 1.4 ) The Features and the 'Target'

# In[ ]:


df.columns


# In[ ]:


df.head()


#  In all we have 34 features consisting of both the categorical as well as the numerical features. The target variable is the 
#  'Attrition' of the employee which can be either a Yes or a No.

# ######  Hence this is a Binary Classification problem. 

# ## 1.5 ) Univariate Analysis

# In this section I have done the univariate analysis i.e. I have analysed the range or distribution of the values that various features take. To better analyze the results I have plotted various graphs and visualizations wherever necessary.

# In[ ]:


df.describe()


#  Let us first analyze the various numeric features. To do this we can actually plot a boxplot showing all the numeric features.

# In[ ]:


sns.factorplot(data=df,kind='box',size=10,aspect=3)


# Note that all the features have pretty different scales and so plotting a boxplot is not a good idea. Instead what we can do is plot histograms of various continuously distributed features.
# 
# We can also plot a kdeplot showing the distribution of the feature. Below I have plotted a kdeplot for the 'Age' feature.
# Similarly we plot for other numeric features also. We can also use a distplot from seaborn library.

# In[ ]:


sns.kdeplot(df['Age'],shade=True,color='#ff4125')


# In[ ]:


sns.distplot(df['Age'])


# Similarly we can do this for all the numerical features. Below I have plotted the subplots for the other features.

# In[ ]:


warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

fig,ax = plt.subplots(5,2, figsize=(9,9))                
sns.distplot(df['TotalWorkingYears'], ax = ax[0,0]) 
sns.distplot(df['MonthlyIncome'], ax = ax[0,1]) 
sns.distplot(df['YearsAtCompany'], ax = ax[1,0]) 
sns.distplot(df['DistanceFromHome'], ax = ax[1,1]) 
sns.distplot(df['YearsInCurrentRole'], ax = ax[2,0]) 
sns.distplot(df['YearsWithCurrManager'], ax = ax[2,1]) 
sns.distplot(df['YearsSinceLastPromotion'], ax = ax[3,0]) 
sns.distplot(df['PercentSalaryHike'], ax = ax[3,1]) 
sns.distplot(df['YearsSinceLastPromotion'], ax = ax[4,0]) 
sns.distplot(df['TrainingTimesLastYear'], ax = ax[4,1]) 
plt.tight_layout()
plt.show()


# Let us now analyze the various categorical features. Note that in these cases the best way is to use a count plot to show the relative count of observations of different categories.

# In[ ]:


cat_df=df.select_dtypes(include='object')


# In[ ]:


cat_df.columns


# In[ ]:


def plot_cat(attr,labels=None):
    if(attr=='JobRole'):
        sns.factorplot(data=df,kind='count',size=5,aspect=3,x=attr)
        return
    
    sns.factorplot(data=df,kind='count',size=5,aspect=1.5,x=attr)


# I have made a function that accepts the name of a string. In our case this string will be the name of the column or attribute which we want to analyze. The function then plots the countplot for that feature which makes it easier to visualize.

# In[ ]:


plot_cat('Attrition')   


# ######  Note that the number of observations belonging to the 'No'  category is way greater than that belonging to 'Yes' category. Hence we have skewed classes and this is a typical example of the 'Imbalanced Classification Problem'. To handle such types of problems we need to use the over-sampling or under-sampling techniques. I shall come back to this point later.

# 

# Let us now similalry analyze other categorical features.

# In[ ]:


plot_cat('BusinessTravel')   


# The above plot clearly shows that most of the people belong to the 'Travel_Rarely' class. This indicates that most of the people did not have a job which asked them for frequent travelling.

# In[ ]:


plot_cat('OverTime')


# In[ ]:


plot_cat('Department')   


# In[ ]:


plot_cat('EducationField')


# In[ ]:


plot_cat('Gender') 


# Note that males are presnt in higher number.

# In[ ]:


plot_cat('JobRole')   


# ######  Similarly we can continue for other categorical features. 
# 
#  

# ###### Note that the same function can also be used to better analyze the numeric discrete features like 'Education' ,'JobSatisfaction' etc...  

# In[ ]:


# just uncomment the following cell.


# In[ ]:


# num_disc=['Education','EnvironmentSatisfaction','JobInvolvement','JobSatisfaction','WorkLifeBalance','RelationshipSatisfaction','PerformanceRating']
# for i in num_disc:
#     plot_cat(i)

# similarly we can intrepret these graphs.


# <a id="content2"></a>
# ## 2 ) Corelation b/w Features

# In[ ]:


#corelation matrix.
cor_mat= df.corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig=plt.gcf()
fig.set_size_inches(30,12)
sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True)


# ###### BREAKING IT DOWN
# Firstly calling .corr() method on a pandas data frame returns a corelation data frame containing the corelation values b/w the various attributes.
# now we obtain a numpy array from the corelation data frame using the np.array method.
# nextly using the np.tril_indices.from() method we set the values of the lower half of the mask numpy array to False. this is bcoz on passing the mask to heatmap function of the seaborn it plots only those squares whose mask is False. therefore if we don't do this then as the mask is by default True then no square will appear. Hence in a nutshell we obtain a numpy array from the corelation data frame and set the lower values to False so that we can visualise the corelation. In order for a full square just use the [:] operator in mask in place of tril_ind... function.
# in next step we get the refernce to the current figure using the gcf() function of the matplotlib library and set the figure size.
# in last step we finally pass the necessary parameters to the heatmap function.
# 
# DATA=the corelation data frame containing the 'CORELATION' values.
# 
# MASK= explained earlier.
# 
# vmin,vmax= range of values on side bar
# 
# SQUARE= to show each individual unit as a square.
# 
# ANNOT- whether to dispaly values on top of square or not. In order to dispaly pass it either True or the cor_mat.
# 
# CBAR= whether to view the side bar or not.

# 

# ###### SOME INFERENCES FROM THE ABOVE HEATMAP
# 
# 1. Self relation ie of a feature to itself is equal to 1 as expected.
# 
# 2. JobLevel is highly related to Age as expected as aged employees will generally tend to occupy higher positions in the company.
# 
# 3. MonthlyIncome is very strongly related to joblevel as expected as senior employees will definately earn more.
# 
# 4. PerformanceRating is highly related to PercentSalaryHike which is quite obvious.
# 
# 5. Also note that TotalWorkingYears is highly related to JobLevel which is expected as senior employees must have worked for a larger span of time.
# 
# 6. YearsWithCurrManager is highly related to YearsAtCompany.
# 
# 7. YearsAtCompany is related to YearsInCurrentRole.
# 
#   

# Note that we can drop some highly corelated features as they add redundancy to the model but since the corelation is very less in genral let us keep all the features for now. In case of highly corelated features we can use something like Principal Component Analysis(PCA) to reduce our feature space.

# In[ ]:


df.columns


# <a id="content3"></a>
# ## 3 ) Feature Selection

# ## 3.1 ) Plotting the Features against the 'Target' variable.

# ####  3.1.1 ) Age

# Note that Age is a continuous quantity and therefore we can plot it against the Attrition using a boxplot.

# In[ ]:


sns.factorplot(data=df,y='Age',x='Attrition',size=5,aspect=1,kind='box')


# Note that the median as well the maximum age of the peole with 'No' attrition is higher than that of the 'Yes' category. This shows that peole with higher age have lesser tendency to leave the organisation which makes sense as they may have settled in the organisation.

# #### 3.1.2 ) Department

# Note that both Attrition(Target) as well as the Deaprtment are categorical. In such cases a cross-tabulation is the most reasonable way to analyze the trends; which shows clearly the number of observaftions for each class which makes it easier to analyze the results.

# In[ ]:


df.Department.value_counts()


# In[ ]:


sns.factorplot(data=df,kind='count',x='Attrition',col='Department')


# In[ ]:


pd.crosstab(columns=[df.Attrition],index=[df.Department],margins=True,normalize='index') # set normalize=index to view rowwise %.


# Note that most of the observations corresspond to 'No' as we saw previously also. About 81 % of the people in HR dont want to leave the organisation and only 19 % want to leave. Similar conclusions can be drawn for other departments too from the above cross-tabulation.

# #### 3.1.3 ) Gender

# In[ ]:


pd.crosstab(columns=[df.Attrition],index=[df.Gender],margins=True,normalize='index') # set normalize=index to view rowwise %.


# About 85 % of females want to stay in the organisation while only 15 % want to leave the organisation. All in all 83 % of employees want to be in the organisation with only being 16% wanting to leave the organisation or the company.

# #### 3.1.4 ) Job Level

# In[ ]:


pd.crosstab(columns=[df.Attrition],index=[df.JobLevel],margins=True,normalize='index') # set normalize=index to view rowwise %.


# People in Joblevel 4 have a very high percent for a 'No' and a low percent for a 'Yes'. Similar inferences can be made for other job levels.

# #### 3.1.5 ) Monthly Income

# In[ ]:


sns.factorplot(data=df,kind='bar',x='Attrition',y='MonthlyIncome')


#  Note that the average income for 'No' class is quite higher and it is obvious as those earning well will certainly not be willing to exit the organisation. Similarly those employees who are probably not earning well will certainly want to change the company.

# #### 3.1.6 ) Job Satisfaction

# In[ ]:


sns.factorplot(data=df,kind='count',x='Attrition',col='JobSatisfaction')


# In[ ]:


pd.crosstab(columns=[df.Attrition],index=[df.JobSatisfaction],margins=True,normalize='index') # set normalize=index to view rowwise %.


# Note this shows an interesting trend. Note that for higher values of job satisfaction( ie more a person is satisfied with his job) lesser percent of them say a 'Yes' which is quite obvious as highly contented workers will obvioulsy not like to leave the organisation.

# #### 3.1.7 ) Environment Satisfaction 

# In[ ]:


pd.crosstab(columns=[df.Attrition],index=[df.EnvironmentSatisfaction],margins=True,normalize='index') # set normalize=index to view rowwise %.


# Again we can notice that the relative percent of 'No' in people with higher grade of environment satisfacftion.

# #### 3.1.8 ) Job Involvement

# In[ ]:


pd.crosstab(columns=[df.Attrition],index=[df.JobInvolvement],margins=True,normalize='index') # set normalize=index to view rowwise %.


# #### 3.1.9 ) Work Life Balance

# In[ ]:


pd.crosstab(columns=[df.Attrition],index=[df.WorkLifeBalance],margins=True,normalize='index') # set normalize=index to view rowwise %.


# Again we notice a similar trend as people with better work life balance dont want to leave the organisation.

# #### 3.1.10 ) RelationshipSatisfaction

# In[ ]:


pd.crosstab(columns=[df.Attrition],index=[df.RelationshipSatisfaction],margins=True,normalize='index') # set normalize=index to view rowwise %.


# ###### Notice that I have plotted just some of the important features against out 'Target' variable i.e. Attrition in our case. Similarly we can plot other features against the 'Target' variable and analye the trends i.e. how the feature effects the 'Target' variable.

# ## 3.2 ) Feature Selection

# The feature Selection is one of the main steps of the preprocessing phase as the features which we choose directly effects the model performance. While some of the features seem to be less useful in terms of the context; others seem to equally useful. The better features we use the better our model will perform.
# 
# We can also use the Recusrive Feature Elimination technique (a wrapper method) to choose the desired number of most important features.
# The Recursive Feature Elimination (or RFE) works by recursively removing attributes and building a model on those attributes that remain.
# 
# It uses the model accuracy to identify which attributes (and combination of attributes) contribute the most to predicting the target attribute.
# 
# We can use it directly from the scikit library by importing the RFE module or function provided by the scikit. But note that since it tries different combinations or the subset of features;it is quite computationally expensive.

# In[ ]:


df.drop(['BusinessTravel','DailyRate','EmployeeCount','EmployeeNumber','HourlyRate','MonthlyRate'
           ,'NumCompaniesWorked','Over18','StandardHours', 'StockOptionLevel','TrainingTimesLastYear'],axis=1,inplace=True)


# 

# <a id="content4"></a>
# ##  4 ) Preparing Dataset

# Before feeding our data into a ML model we first need to prepare the data. This includes encoding all the categorical features (either LabelEncoding or the OneHotEncoding) as the model expects the features to be in numerical form. Also for better performance we will do the feature scaling ie bringing all the features onto the same scale by using the StandardScaler provided in the scikit library.

# ## 4.1 ) Feature Encoding 

# I have used the Label Encoder from the scikit library to encode all the categorical features.

# In[ ]:


def transform(feature):
    le=LabelEncoder()
    df[feature]=le.fit_transform(df[feature])
    print(le.classes_)
    


# In[ ]:


cat_df=df.select_dtypes(include='object')
cat_df.columns


# In[ ]:


for col in cat_df.columns:
    transform(col)


# In[ ]:


df.head() # just to verify.


# ## 4.2 ) Feature Scaling

# The scikit library provides various types of scalers including MinMax Scaler and the StandardScaler. Below I have used the StandardScaler to scale the data.

# Note that the neural networks are quite sensitive towards the scale of the features. Hence it is always good to perform feature scaling on the data before feeding it into an Artificial Neural Network.

# In[ ]:


scaler=StandardScaler()
scaled_df=scaler.fit_transform(df.drop('Attrition',axis=1))
X=scaled_df
Y=df['Attrition'].as_matrix()


# ## 4.3 ) One Hot Encoding the Target 

# Note that there are two main things to watch out before feeding data into an ANN. 
# 
# The first is that our data needs to be in the form of numpy arrays (ndarray).
# 
# The second that the target variable should be one hot encoded eg 2--> 0010 (assuming 0 based indexing) and so on.. 
# In this way for a 'n' class classification problems our target variable will have n classes and hence after one hot encoding we shall have n labels with each label corressponding to a particular target class.

# In[ ]:


Y=to_categorical(Y)
Y


# ## 4.4 ) Splitting the data into training and validation sets

# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)


# <a id="content5"></a>
# ## 5 )  Making Predictions Using an Artificial Neural Network (ANN)

# ## 5.1 ) Handling the Imbalanced dataset

# Note that we have a imbalanced dataset with majority of observations being of one type ('NO') in our case. In this dataset for example we have about 84 % of observations having 'No' and only 16 % of 'Yes' and hence this is an imbalanced dataset.
# 
# To deal with such a imbalanced dataset we have to take certain measures, otherwise the performance of our model can be significantly affected. In this section I have discussed two approaches to curb such datasets.

# ## 5.1.1 ) Oversampling the Minority or Undersampling the Majority Class
#  
#   

# In an imbalanced dataset the main problem is that the data is highly skewed ie the number of observations of certain class is more than that of the other. Therefore what we do in this approach is to either increase the number of observations corressponding  to the minority class (oversampling) or decrease the number of observations for the majority class (undersampling).
# 
# Note that in our case the number of observations is already pretty low and so oversampling will be more appropriate.
# 
# Below I have used an oversampling technique known as the SMOTE(Synthetic Minority Oversampling Technique) which randomly creates some 'Synthetic' instances of the minority class so that the net observations of both the class get balanced out.
# 
# One thing more to take of is to use the SMOTE before the cross validation step; just to ensure that our model does not overfit the data; just as in the case of feature selection.

# In[ ]:


# oversampler=SMOTE(random_state=42)
# x_train_smote,  y_train_smote = oversampler.fit_sample(x_train,y_train)


# ## 5.1.2 ) Using the Right Evaluation Metric

# Another important point while dealing with the imbalanced classes is the choice of right evaluation metrics. 
# 
# Note that accuracy is not a good choice. This is because since the data is skewed even an algorithm classifying the target as that belonging to the majority class at all times will achieve a very high accuracy. 
# For  eg if we have 20 observations of one type 980 of another ; a classifier predicting the majority class at all times will also attain a accuracy of 98 % but doesnt convey any useful information.
# 
# Hence in these type of cases we may use other metrics such as -->
# 
# 
# 'Precision'-- (true positives)/(true positives+false positives)
# 
# 'Recall'-- (true positives)/(true positives+false negatives)
# 
# 'F1 Score'-- The harmonic mean of 'precision' and 'recall'
# 
# 'AUC ROC'-- ROC curve is a plot between 'senstivity' (Recall) and '1-specificity' (Specificity=Precision)
# 
# 'Confusion Matrix'-- Plot the entire confusion matrix

# ## 5.2) Setting the random seeds

# Note that in order to get exactly same results after training an artificial neural network at different instances of time we need to specify the random seed for the Keras backend engine which is TensorFlow in my case. Also I have specified the seeds for the python random module as well as for the numpy.
# 
# In order to adjust the weights oof an ANN ; the BackProp algorithm starts with a random weights and hence after a given no of epochs the results can be different if the random initialisation of weights is different in starting. 
# 
# Hence to obtain the same results it is necessary to specify the random seed to get the reproducible results.

# In[ ]:


np.random.seed(42)


# In[ ]:


rn.seed(42)


# In[ ]:


tf.set_random_seed(42)


# ## 5.3 ) Building the Keras model

# In[ ]:


model=Sequential()
model.add(Dense(input_dim=23,units=8,activation='relu'))
model.add(Dense(units=16,activation='relu'))
model.add(Dense(units=2,activation='sigmoid'))


# ####  BREAKING IT DOWN
# 
# 1. First we need to build a model. For this we use the Sequential model provided by the Keras which is nothing but a linear stack of layers.
# 
# 
# 2. Next we need to add the layers to our Sequential model. For this we use the model.add() function.
# 
# 
# 3. Note that for each layer we need to specify the number of units ( or the number of neurons) and also the activation function used by the neurons.
# 
#   Note that activation  function is used to model complex non-linear relationships and their are many choices. But generally it is preferred to use 'relu' function for the hidden layers and the 'sigmoid' or the 'logistic' function for the output layer. For a multi-class classification problem we can use the 'softmax' function as the activation function for the output layer.
#   
# 
# 4. Note that the first layer and ONLY the first layer expects the input dimensions in order to know the shape of the input numpy array.
# 
# 
# 5. Finally note that the number of units or neurons in the final layer is equal to the number of classes of the target variable. In other words for a 'n' class classification problem we shall have 'n' neurons in the output layer. 
#  
#  Each neuron represents a specific target class. The output of each neuron in the final layer thus represents the probability of given observation being classified to that target class. The observation is classified to the target class; the neuron corressponding to which has the highest value. 

# ## 5.4 ) Compiling the Keras model

# In[ ]:


model.compile(optimizer=Adam(lr=0.01),loss='binary_crossentropy',metrics=['accuracy'])


# #### BREAKING IT DOWN
# 
# 1. Now we need to compile the model. We have to specify the optimizer used by the model We have many choices like Adam, RMSprop etc.. Rfer to Keras doc for a comprehensive list of the optimizers available.
# 
# 
# 2. Next we need to specify the loss function for the neural network which we seek to minimize.
# 
#   I have used the 'binary_crossentropy' loss function since this is a binary classification problem. For a multi-class classification problems we may use the 'categorical_crossentropy'.
# 
# 
# 3. Next we need to specify the metric to evaluate our models performance. Here I have used accuracy.

# ## 5.5 ) Summary of the model

# In[ ]:


model.summary()


# Provides overall description of the model.

# ## 5.6 ) Fitting the model on the training data and testing on the validation set

# In[ ]:


History=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10,verbose=1)


# #### BREAKING IT DOWN
# 
# 1. Lastly we need to fit our model onto the training data just as we do for traditional ML algorithms.
# 
# 
# 2. We have to specify the training(x_train ,y_train) and the testing (validation_data) sets.
# 
# 
# 3. We also need to specify the 'number of epochs'. 
#    An 'epoch' is one entire cycle of 'Forward & Backward propagation' through all the training examples.
# 
# 
# 4. Verbose is an optional parameter that just ensures how the output of each epoch is displayed on the screen.
# 
# 
# 5. We have assigned it to a 'History' variable to retrieve the model performance during each epoch in the future.

# ## 5.7 ) Making Predictions

# In[ ]:


model.predict_classes(x_test)


# In[ ]:


model.predict(x_test)


# ## 5.8 ) Evaluating the Model Performance

# In[ ]:


model.evaluate(x_test,y_test)


# In[ ]:


plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()


# In[ ]:


plt.plot(History.history['acc'])
plt.plot(History.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()


# <a id="content6"></a>
# ## 6 ) Hyperparameter Tuning

# #### In order to increase the model performance of the ANN we need to tune the hyperparameters. Some of the hyperparameters include

# a) Number of hidden layers.
# 
# b) Number of neurons in a particular layer.
# 
# c) The activation function.
# 
# d) The optimizer used.
# 
# e) Number of epochs etc...

# <a id="content7"></a>
# ## 7 ) Conclusions

# ###### Hence we have completed the analysis of the data and also made predictions using an Artificial Neural Network.      

# In[ ]:





# In[ ]:


Image('../input/imagesannibm/image-hr.jpg')


# In[ ]:





# # THE END. [please star/upvote if u find it helpful.]

# In[ ]:





# In[ ]:




