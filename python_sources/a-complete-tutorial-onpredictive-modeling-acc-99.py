#!/usr/bin/env python
# coding: utf-8

# # Gender Recognition by Voice Kaggle [ Test Accuracy : 99.08 % ]

# In[ ]:





# ## CONTENTS::

# [ **1 )  Importing Various Modules and Loading the Dataset**](#content1)

# [ **2 ) Exploratory Data Analysis (EDA)**](#content2)

# [ **3 ) OutlierTreatment**](#content3)

# [ **4 ) Feature Engineering**](#content4)

# [ **5 ) Preparing the Data**](#content5)

# [ **6 ) Modelling**](#content6)

# [ **7 ) Parameter Tuning with GridSearchCV**](#content7)

# In[ ]:





# ## 1.1 ) Importing Various  Modules

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
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

#model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV

#preprocess.
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Imputer,LabelEncoder,OneHotEncoder


# ## 1.2 ) Loading the Dataset

# In[ ]:


train=pd.read_csv(r'../input/voice.csv')


# In[ ]:


train.head(10)


# <a id="content2"></a>
# ## 2 )  Exploratory Data Analysis (EDA)

# ## 2.1 ) The Features and the 'Target' variable

# In[ ]:


df=train.copy()


# In[ ]:


df.head(10)


# In[ ]:


df.shape


# In[ ]:


df.index   


# In[ ]:


df.columns # give a short description of each feature.


# **#A short description as on 'Data' tab on kaggle is :**

# #### 
# 
# **meanfreq**: mean frequency (in kHz)
# 
# **sd**: standard deviation of frequency
# 
# **median**: median frequency (in kHz)
# 
# **Q25**: first quantile (in kHz)
# 
# **Q75**: third quantile (in kHz)
# 
# **IQR**: interquantile range (in kHz)
# 
# **skew**: skewness (see note in specprop description)
# 
# **kurt**: kurtosis (see note in specprop description)
# 
# **sp.ent**: spectral entropy
# 
# **sfm**: spectral flatness
# 
# **mode**: mode frequency
# 
# **centroid**: frequency centroid (see specprop)
# 
# **peakf**: peak frequency (frequency with highest energy)
# 
# **meanfun**: average of fundamental frequency measured across acoustic signal
# 
# **minfun**: minimum fundamental frequency measured across acoustic signal
# 
# **maxfun**: maximum fundamental frequency measured across acoustic signal
# 
# **meandom**: average of dominant frequency measured across acoustic signal
# 
# **mindom**: minimum of dominant frequency measured across acoustic signal
# 
# **maxdom**: maximum of dominant frequency measured across acoustic signal
# 
# **dfrange**: range of dominant frequency measured across acoustic signal
# 
# **modindx**: modulation index. Calculated as the accumulated absolute difference between adjacent measurements of fundamental frequencies divided by the frequency range
# 
# **label**: male or female

# #### Note that we have 3168 voice samples  and for each of sample 20 different acoustic properties are recorded. Finally the 'label' column is the target variable which we have to predict which is the gender of the person.

# ## 2.2 ) Missing Values Treatment

# In[ ]:


# check for null values.
df.isnull().any()   


# In[ ]:


msno.matrix(df)  # just to visualize. no missing value.


# ## 2.3 ) Univariate Analysis

# In this section I have performed the univariate analysis. Note that since all of the features are 'numeric' the most reasonable way to plot them would either be a 'histogram' or a 'boxplot'.
# 
# Also note that univariate analysis is useful for outlier detection. Hence besides plotting a boxplot and a histogram for each column or feature, I have written a small utility function which tells the remaining no of observations for each feature if  we remove its outliers.

# #### To detect the outliers I have used the standard 1.5 InterQuartileRange (IQR) rule which states that any observation lesser than  'first quartile - 1.5 IQR' or greater than 'third quartile +1.5 IQR' is an outlier.

# In[ ]:


df.describe()


# In[ ]:


def calc_limits(feature):
    q1,q3=df[feature].quantile([0.25,0.75])
    iqr=q3-q1
    rang=1.5*iqr
    return(q1-rang,q3+rang)


# In[ ]:


def plot(feature):
    fig,axes=plt.subplots(1,2)
    sns.boxplot(data=df,x=feature,ax=axes[0])
    sns.distplot(a=df[feature],ax=axes[1],color='#ff4125')
    fig.set_size_inches(15,5)
    
    lower,upper = calc_limits(feature)
    l=[df[feature] for i in df[feature] if i>lower and i<upper] 
    print("Number of data points remaining if outliers removed : ",len(l))


# In[ ]:


plot('meanfreq')


# #### INFERENCES FROM THE PLOT--
# 
# 1) First of all note that the values are in compliance with that observed from describe method data frame..
# 
# 2) Note that we have a couple of outliers w.r.t. to 1.5 quartile rule (reprsented by a 'dot' in the box plot).Removing  these data points or outliers leaves us with around 3104 values.
# 
# 3) Also note from the distplot that the distribution seems to be a bit -ve skewed hence we can normalize to make the distribution a bit more symmetric. 
#  
# 4) LASTLY NOTE THAT A LEFT TAIL DISTRIBUTION HAS MORE OUTLIERS ON THE SIDE BELOW TO Q1 AS EXPECTED AND A RIGHT TAIL HAS ABOVE THE Q3.

# #### Similar other plots can be inferenced.

# In[ ]:


plot('sd')


# In[ ]:


plot('median')


# In[ ]:


plot('Q25')


# In[ ]:


plot('IQR')


# In[ ]:


plot('skew')


# In[ ]:


plot('kurt')


# In[ ]:


plot('sp.ent')


# In[ ]:


plot('sfm')


# In[ ]:


plot('meanfun')


# In[ ]:


sns.countplot(data=df,x='label')


# In[ ]:


df['label'].value_counts()


# ####  Note that  we have equal no of observations for the 'males' and the 'females'. Hence it is a balanced class problem.

# ## 2.4 ) Bivariate Analysis

# ## 2.4.1 ) Corealtion b/w Features

# In this section I have analyzed the corelation between different features. To do it I have plotted a 'heat map' which clearly visulizes the corelation between different features.

# In[ ]:


temp = []
for i in df.label:
    if i == 'male':
        temp.append(1)
    else:
        temp.append(0)
df['label'] = temp


# In[ ]:


#corelation matrix.
cor_mat= df[:].corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig=plt.gcf()
fig.set_size_inches(30,12)
sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True)


# #### SOME INFERENCES FROM THE ABOVE HEATMAP--
# 
# 1) Mean frequency is  moderately related to label.
# 
# 2) IQR and label tend to have a strong positive corelation.
# 
# 3) Spectral entropy is also quite highly corelated with the label while sfm is moderately related with label.
# 
# 4) skewness and kurtosis aren't much related with label.
# 
# 5) meanfun is highly negatively corelated with  the label.
# 
# 6) Centroid and median have a high positive corelationas expected from their formulae.
# 
# 7) ALSO NOTE THAT MEANFREQ AND CENTROID ARE EXACTLY SAME FEATURES AS PER FORMULAE AND VALUES ALSO. HENCE THEIR CORELATION IS PERFCET 1. IN THAT CASE WE CAN DROP ANY COLUMN. note that centroid in general has a high degree of corelation with most of the other features. 
# 
# SO I WILL DROP THE 'CENTROID' COLUMN.
# 
# 8) sd is highly positively related to sfm and so is sp.ent to sd.
# 
# 9) kurt and skew are also highly corelated.
# 
# 10) meanfreq is highly related to medaina s well as Q25.
# 
# 11) IQR is highly corelated to sd.
# 
# 12) Finally self relation ie of a feature to itself is equal to 1 as expected.

# #### Note that we can drop some highly corelated features as they add redundancy to the model  but  let us keep all the features for now. In case of highly corelated features we can use dimensionality reduction techniques like Principal Component Analysis(PCA) to reduce our feature space.

# In[ ]:


df.drop('centroid',axis=1,inplace=True)


# ## 2.4.2 ) Plotting the Features against the 'Target' variable

# Here I have just written a small utility function that plots the 'label' column vs the provided feature on a boxplot. In this way I have plotted some of the features against our target variable. This makes it easier to see the effect of the corressponding feature on the 'label'.

# In[ ]:


# drawing features against the target variable.

def plot_against_target(feature):
    sns.factorplot(data=df,y=feature,x='label',kind='box')
    fig=plt.gcf()
    fig.set_size_inches(7,7)


# In[ ]:


plot_against_target('meanfreq') # 0 for females and 1 for males.


# #### INFERENCES--
# 
# 1) Firstly note that 0->'female' and 1->'male'.
# 
# 2) Note that the boxpot depicts that the females in genral have higher mean frequencies than their male counterparts and which is a generally accepted fact.

# #### Again similar inferences can be drawn.

# In[ ]:


plot_against_target('sd')


# In[ ]:


plot_against_target('median')


# In[ ]:


plot_against_target('Q25')


# In[ ]:


plot_against_target('IQR')


# #### Note here that  there is a remarkable difference b/w the inter quartile ranges of males and females.This is evident from the strong relation between 'label' and the 'IQR' in the heatmap plotted above.

# In[ ]:


plot_against_target('sp.ent')


# In[ ]:


plot_against_target('sfm')


# In[ ]:


plot_against_target('meanfun')  


# #### Again high difference in females and males mean fundamental frequency. This is evident from the heat map which clearly shows the high corelation between meanfun and the 'label'.

# In[ ]:





# #### Now we move onto analyzing different features pairwise. Since all the features are continuous the most reasonable way to do this is plotting the scatter plots for each feature pair. I have also distinguished males and feamles on the same plot which makes it a bit easier to compare the variation of features within the two classes.

# In[ ]:


g = sns.PairGrid(df[['meanfreq','sd','median','Q25','IQR','sp.ent','sfm','meanfun','label']], hue = "label")
g = g.map(plt.scatter).add_legend()


# In[ ]:





# <a id="content3"></a>
# ## 3 ) Outlier Treatment

# In this section I have dealt with the outliers.  Note that we discovered the potential outliers in the **'univariate analysis' ** section. Now to remove those outliers we can either remove the corressponding data points or impute them with some other statistical quantity like median (robust to outliers) etc..

# #### For now I shall be removing all the observations or data points which are outlier to 'any' feature. Note that this substantially reduces the dataset size.

# In[ ]:


# removal of any data point which is an outlier for any fetaure.
for col in df.columns:
    lower,upper=calc_limits(col)
    df = df[(df[col] >lower) & (df[col]<upper)]


# In[ ]:


df.shape


# In[ ]:


df.head(10)


# <a id="content4"></a>
# ## 4 ) Feature Engineering.

# ## 4.1 ) Dropping the features

# I have dropped some columns which according to my analysis proved to be less useful or redundant.

# In[ ]:


temp_df=df.copy()

temp_df.drop(['skew','kurt','mindom','maxdom'],axis=1,inplace=True) # only one of maxdom and dfrange.
temp_df.head(10)
#df.head(10)


# ## 4.2 ) Creating new features

# I have done two new things. Firstly I have made 'meanfreq','median' and 'mode' to comply by the standard relation->

# ####     ......................................................................................3*Median=2*Mean +Mode.........................................................................

# ####  For this I have adjusted values in the 'median' column as shown below. You can alter values in any of the other column say the 'meanfreq' column.

# In[ ]:


temp_df['meanfreq']=temp_df['meanfreq'].apply(lambda x:x*2)
temp_df['median']=temp_df['meanfreq']+temp_df['mode']
temp_df['median']=temp_df['median'].apply(lambda x:x/3)


# In[ ]:


temp_df.head(10) 


# In[ ]:


sns.boxplot(data=temp_df,y='median',x='label') # seeing the new 'median' against the 'label'.


# The second new feature that I have added is  a new feature to mesure the 'skewness'. 

# #### For this I have used the 'Karl Pearson Coefficent' which is calculated as shown below->

# **** ..........................................................Coefficent = (Mean - Mode )/StandardDeviation......................................................****

# **You can also try some other coefficient also and see how it comapres with the target i.e. the 'label' column.**

# In[ ]:


temp_df['pear_skew']=temp_df['meanfreq']-temp_df['mode']
temp_df['pear_skew']=temp_df['pear_skew']/temp_df['sd']
temp_df.head(10)


# In[ ]:


sns.boxplot(data=temp_df,y='pear_skew',x='label') # plotting new 'skewness' against the 'label'.


# <a id="content5"></a>
# ## 5 ) Preparing the Data

# ## 5.1 ) Normalizing the Features.

# In[ ]:


scaler=StandardScaler()
scaled_df=scaler.fit_transform(temp_df.drop('label',axis=1))
X=scaled_df
Y=df['label'].as_matrix()


# ## 5.2 ) Splitting into Training and Validation sets.

# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=42)


# In[ ]:





# <a id="content6"></a>
# ## 6 ) Modelling

# #### LOGISTIC REGRESSSION

# In[ ]:


clf_lr=LogisticRegression()
clf_lr.fit(x_train,y_train)
pred=clf_lr.predict(x_test)
print(accuracy_score(pred,y_test))


# #### kNN

# In[ ]:


clf_knn=KNeighborsClassifier()
clf_knn.fit(x_train,y_train)
pred=clf_knn.predict(x_test)
print(accuracy_score(pred,y_test))


# #### Support Vector Machine (SVM)

# In[ ]:


clf_svm=SVC()
clf_svm.fit(x_train,y_train)
pred=clf_svm.predict(x_test)
print(accuracy_score(pred,y_test))


# #### DECISION TREE  

# In[ ]:


clf_dt=DecisionTreeClassifier()
clf_dt.fit(x_train,y_train)
pred=clf_dt.predict(x_test)
print(accuracy_score(pred,y_test))


# #### RANDOM FOREST

# In[ ]:


clf_rf=RandomForestClassifier()
clf_rf.fit(x_train,y_train)
pred=clf_rf.predict(x_test)
print(accuracy_score(pred,y_test))


# #### GRADIENT BOOSTING

# In[ ]:


clf_gb=GradientBoostingClassifier()
clf_gb.fit(x_train,y_train)
pred=clf_gb.predict(x_test)
print(accuracy_score(pred,y_test))


# #### We can now move onto comparing the results of various modelling algorithms. for tthis I shall combine the results of all models in a data frame and then plot using  a barplot .

# In[ ]:


models=[LogisticRegression(),LinearSVC(),SVC(kernel='rbf'),KNeighborsClassifier(),RandomForestClassifier(),
        DecisionTreeClassifier(),GradientBoostingClassifier(),GaussianNB()]
model_names=['LogisticRegression','LinearSVM','rbfSVM','KNearestNeighbors','RandomForestClassifier','DecisionTree',
             'GradientBoostingClassifier','GaussianNB']

acc=[]
d={}

for model in range(len(models)):
    clf=models[model]
    clf.fit(x_train,y_train)
    pred=clf.predict(x_test)
    acc.append(accuracy_score(pred,y_test))
     
d={'Modelling Algo':model_names,'Accuracy':acc}


# In[ ]:


acc_frame=pd.DataFrame(d)
acc_frame


# In[ ]:


sns.barplot(y='Modelling Algo',x='Accuracy',data=acc_frame)


# In[ ]:





# <a id="content7"></a>
# ## 7 ) Parameter Tuning with GridSearchCV

# 1. I have tuned only SVM Similarly other algorithms can be tuned.

# In[ ]:


params_dict={'C':[0.001,0.01,0.1,1,10,100],'gamma':[0.001,0.01,0.1,1,10,100],'kernel':['linear','rbf']}
clf=GridSearchCV(estimator=SVC(),param_grid=params_dict,scoring='accuracy',cv=10)
clf.fit(x_train,y_train)


# In[ ]:


clf.best_score_


# In[ ]:


clf.best_params_


# In[ ]:


print(accuracy_score(clf.predict(x_test),y_test))


# In[ ]:


print(precision_score(clf.predict(x_test),y_test))


# ### The precision is almost 99.5 % which is quite high.

# ###  After tuning SVM gives an amazing accuracy of around 99.1 %. Similarly tuning other algorithms parameters might give even greater accuracy !!!

# In[ ]:





# ## THE END!!!

# In[ ]:




