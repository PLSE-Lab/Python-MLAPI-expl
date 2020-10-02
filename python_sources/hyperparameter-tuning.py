#!/usr/bin/env python
# coding: utf-8

# <h1>Hyperparameter Tuning Leads to more Accurate Model</h1>
# <img src='https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRfubyUCAM_eOPbyWe4JgEOKAXdNsFP9xEmsMn_JPdER-9gVgWw&usqp=CAU'>
# <br>
# <p>Hello Kagglers, In this Kernel I am going to Discuss a very important Subject in Machine Learning Which Leads to a better and Accurate Results, which is Hyperparameter Tuning Using GridSearchCV from sickit-learn Library in python, and we Are going to use the Famous Iris Dataset and three different ML Algorithms (Logistic Regression (Linear Model).<br>
# and the plan of this kernel is to use them without no Hyperparmeters and after that, we are going to apply our Technique and see which one will give us the better resuls.</p> 

# <h2>Plan Of Kernel</h2>
# <ul>
#     <li>Load Dataset & Getting Some informations about it</li>
#     <li>Basic-EDA</li>
#     <li>Preprocessing</li>
#     <li>Predicting Species Column Without hyperparemeters</li>
# </ul>

# <h1>Load Dataset & Getting Some informations about it</h1>

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
import os
#set matplotlib style
plt.style.use('ggplot')
#set seaborn
sns.set(context='notebook', palette='RdBu', style='darkgrid')
get_ipython().run_line_magic('matplotlib', 'inline')
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df=pd.read_csv('/kaggle/input/iris/Iris.csv')
df.head()


# <p>From the table above, we can see that the target variable is the Species Column</p>

# In[ ]:


df.info()


# In[ ]:


df.describe()


# <h1>EDA</h1>
# <br>
# <ul>
#     <li>Unique Values in Target Column</li>
#     <li>distribution of Target Column</li>
#     <li>Discover the CDF Graph of each variable for the different Species</li>
# </ul>

# In[ ]:


df.Species.unique()


# In[ ]:


sns.countplot(x='Species',
              data=df)
plt.show()


# <p>Equal Number of Species</p>

# In[ ]:


# define the CDF function
def cdf(x):
    x=np.sort(x)
    y=np.arange(1,len(x)+1)/len(x)
    return x,y


# In[ ]:


# Check Sepal Length for different Species
setosa=df[df['Species']=='Iris-setosa']['SepalLengthCm']
versicolor=df[df['Species']=='Iris-versicolor']['SepalLengthCm']
virginica=df[df['Species']=='Iris-virginica']['SepalLengthCm']
s_x, s_y=cdf(setosa)
ve_x, ve_y=cdf(versicolor)
vi_x, vi_y=cdf(virginica)
plt.plot(s_x, s_y, label='setosa', color='red', marker='.')
plt.plot(ve_x, ve_y, label='versicolor', color='blue', marker='.')
plt.plot(vi_x, vi_y, label='virginica', color='green', marker='.')
plt.title('CDF of Sepal Length')
plt.xlabel('Sepal Length')
plt.ylabel('ECDF')
plt.legend()
plt.show()


# <p>from the figure above we can see that Sepal length in virginica has the highest mean and Satndard deviation values</p>

# In[ ]:


setosa=df[df['Species']=='Iris-setosa']['SepalWidthCm']
versicolor=df[df['Species']=='Iris-versicolor']['SepalWidthCm']
virginica=df[df['Species']=='Iris-virginica']['SepalWidthCm']
s_x, s_y=cdf(setosa)
ve_x, ve_y=cdf(versicolor)
vi_x, vi_y=cdf(virginica)
plt.plot(s_x, s_y, label='setosa', color='red', marker='.')
plt.plot(ve_x, ve_y, label='versicolor', color='blue', marker='.')
plt.plot(vi_x, vi_y, label='virginica', color='green', marker='.')
plt.title('CDF of Sepal Width')
plt.xlabel('Sepal Width in cm')
plt.ylabel('ECDF')
plt.legend()
plt.show()


# <p>Setosa has the highest mean and Standard Deviation values in Sepal Width</p>

# In[ ]:


setosa=df[df['Species']=='Iris-setosa']['PetalLengthCm']
versicolor=df[df['Species']=='Iris-versicolor']['PetalLengthCm']
virginica=df[df['Species']=='Iris-virginica']['PetalLengthCm']
s_x, s_y=cdf(setosa)
ve_x, ve_y=cdf(versicolor)
vi_x, vi_y=cdf(virginica)
plt.plot(s_x, s_y, label='setosa', color='red', marker='.')
plt.plot(ve_x, ve_y, label='versicolor', color='blue', marker='.')
plt.plot(vi_x, vi_y, label='virginica', color='green', marker='.')
plt.title('CDF of Petal Length')
plt.xlabel('Petal Length in cm')
plt.ylabel('ECDF')
plt.legend()
plt.show()


# <p>Virginica also, has the highest mean and STD vales in petal Length</p>

# In[ ]:


setosa=df[df['Species']=='Iris-setosa']['PetalWidthCm']
versicolor=df[df['Species']=='Iris-versicolor']['PetalWidthCm']
virginica=df[df['Species']=='Iris-virginica']['PetalWidthCm']
s_x, s_y=cdf(setosa)
ve_x, ve_y=cdf(versicolor)
vi_x, vi_y=cdf(virginica)
plt.plot(s_x, s_y, label='setosa', color='red', marker='.')
plt.plot(ve_x, ve_y, label='versicolor', color='blue', marker='.')
plt.plot(vi_x, vi_y, label='virginica', color='green', marker='.')
plt.title('CDF of Petal width')
plt.xlabel('Petal width in cm')
plt.ylabel('ECDF')
plt.legend()
plt.show()


# <p>Virginica also, has the highest mean and STD vales in petal width</p>

# <h1>Preprocessing</h1>

# In[ ]:


def to_codes(df,col):
    df[col]=pd.Categorical(df[col]).codes
    return df
pre_df=to_codes(df, 'Species')


# In[ ]:



def min_max_scale(df, col):
    df[col]=(df[col]-df[col].min())/(df[col].max()-df[col].min())
    return df
for i in ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']:
    pre_df=min_max_scale(pre_df, i)


# In[ ]:


x=pre_df.drop(['Id', 'Species'], axis=1)
y=pre_df.Species


# <h1>Predicting Species Column Without hyperparemeters</h1>
# <br>
# <p>Before Predicting our target varaible, We first have to avoid overfitting/Underfitting Problems, so first Split your dataset into train dataframe(80%) and test dataframe</p>
# <img src='data:image/png;base64,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'>
# <br>
# <p> and use Cross validation to give use a better result of what the accuracy for test dataframe</p>
# <img src='data:image/png;base64,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'>

# In[ ]:


x_train, x_test, y_train, y_test=train_test_split(x,
                                                  y,
                                                  test_size=0.2,
                                                  random_state=123) 


# In[ ]:


def cv_score(x,y, model):
    cv=cross_val_score(model,
                       x, y,
                       cv=5,
                       scoring='accuracy',
                       n_jobs=-1)
    return np.mean(cv)
lr=LogisticRegression()
print('CV Score of Logistic Regression: ', cv_score(x_train, y_train, lr))


# In[ ]:


# prediction
lr.fit(x_train, y_train)
pred=lr.predict(x_test)
print('accuracy score of Logistic Regression: ', accuracy_score(y_test, pred))


# <h1>Predicting With Hyperparameter Tuning</h1>
# <br>
# <p>Hyperparameter Tuning  takes your model to the next level by using parameters that can develope your model's performance because in sklearn, a model's default hyperparameters are not optimal for all problems.<p>
# <h2>Approaches to hyperparameter Tuning</h2>
# <ul>
#     <li>Grid Search</li>
#     <li>Random Search</li>
#     <li>Bayesian Optimization</li>
# </ul>
# <p>but you must Know that hyperparametes Tuning is computationally expensive.<br> and we will discuss here grid search approach, which uses cross validation technique for all parameters and gives you the best score.</p>
# <img src='https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRmAAjjNWkbwwlVauGJUAnODAK6IaU2PModhxNFQ1RA4IaCnwDz&usqp=CAU'>
# <a href='https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html'>Check parameters of Logistic Regression</a>

# In[ ]:


lr.get_params


# <p>parameters used in the previous model which we can say default parameters</p>

# In[ ]:


params={
    'solver':['newton-cg', 'lbfgs', 'liblinear'],
    'penalty':['l1', 'l2', 'elasticnet'],
    'C':np.arange(0.1,3,0.01)
}
gs=GridSearchCV(lr,
                param_grid=params,
                cv=5,
                scoring='accuracy', 
                n_jobs=-1)
gs.fit(x_train, y_train)
print('best parameters: ',gs.best_params_)
print('best CV score: ',gs.best_score_)


# <p>Better CV Score </p>

# In[ ]:


lr2=LogisticRegression(C=1.7799999999999994, penalty='l2', solver= 'newton-cg')
lr2.fit(x_train, y_train)
pred2=lr2.predict(x_test)
print('accuracy score of tuned Logistic Regression: ', accuracy_score(y_test, pred))


# In[ ]:


pd.DataFrame({
    'predicted':pred2,
    'real':y_test
})

