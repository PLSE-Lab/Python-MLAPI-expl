#!/usr/bin/env python
# coding: utf-8

# # PREDICTING A PULSAR STAR

# HTRU2 is a data set which describes a sample of pulsar candidates collected during the High Time Resolution Universe Survey . 
# 
# Pulsars are a rare type of Neutron star that produce radio emission detectable here on Earth. They are of considerable scientific interest as probes of space-time, the inter-stellar medium, and states of matter . 
# 
# As pulsars rotate, their emission beam sweeps across the sky, and when this crosses our line of sight, produces a detectable pattern of broadband radio emission. As pulsars 
# rotate rapidly, this pattern repeats periodically. Thus pulsar search involves looking for periodic radio signals with large radio telescopes. 
# 
# Each pulsar produces a slightly different emission pattern, which varies slightly with each rotation . Thus a potential signal detection known as a 'candidate', is averaged over many rotations of the pulsar, as determined by the length of an observation. In the absence of additional info, each candidate could potentially describe a real pulsar. However in practice almost all detections are caused by radio frequency interference (RFI) and noise, making legitimate signals hard to find. 
# 
# Machine learning tools are now being used to automatically label pulsar candidates to facilitate rapid analysis. Classification systems in particular are being widely adopted, 
# which treat the candidate data sets as binary classification problems. Here the legitimate pulsar examples are a minority positive class, and spurious examples the majority negative class.
# 
# The data set shared here contains 16,259 spurious examples caused by RFI/noise, and 1,639 real pulsar examples. These examples have all been checked by human annotators. 
# 
# Each row lists the variables first, and the class label is the final entry. The class labels used are 0 (negative) and 1 (positive). 

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import itertools
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')
from PIL import Image


# In[4]:


img = np.array(Image.open("../input/picture/thumb-1920-510805.jpg"))
plt.figure(figsize=(15,12))
plt.imshow(img)
plt.axis("off")
plt.show()


# # Attribute Information:

# Each candidate is described by 8 continuous variables, and a single class variable. The first four are simple statistics obtained from the integrated pulse profile (folded profile). This is an array of continuous variables that describe a longitude-resolved version of the signal that has been averaged in both time and frequency . The remaining four variables are similarly obtained from the DM-SNR curve . These are summarised below: 
# 
# 1. Mean of the integrated profile. 
# 2. Standard deviation of the integrated profile. 
# 3. Excess kurtosis of the integrated profile. 
# 4. Skewness of the integrated profile. 
# 5. Mean of the DM-SNR curve. 
# 6. Standard deviation of the DM-SNR curve. 
# 7. Excess kurtosis of the DM-SNR curve. 
# 8. Skewness of the DM-SNR curve. 
# 9. Class 
# 
# HTRU 2 Summary 
# 17,898 total examples. 
# 1,639 positive examples. 
# 16,259 negative examples.
# 
# 

# # Data

# In[5]:


data = pd.read_csv(r"../input/predicting-a-pulsar-star/pulsar_stars.csv")
data.head()


# # Data dimensions

# In[4]:


print("No of rows : ", data.shape[0])
print("No of columns : ", data.shape[1])


# # Data Information

# In[5]:


print("data info : ", data.info())


# # Missing values

# In[6]:


print(data.isnull().sum())


# In[7]:


#Renaming columns
data = data.rename(columns={' Mean of the integrated profile':"mean_profile",
       ' Standard deviation of the integrated profile':"std_profile",
       ' Excess kurtosis of the integrated profile':"kurtosis_profile",
       ' Skewness of the integrated profile':"skewness_profile", 
        ' Mean of the DM-SNR curve':"mean_dmsnr_curve",
       ' Standard deviation of the DM-SNR curve':"std_dmsnr_curve",
       ' Excess kurtosis of the DM-SNR curve':"kurtosis_dmsnr_curve",
       ' Skewness of the DM-SNR curve':"skewness_dmsnr_curve",
       })


# # Data summary

# In[9]:


plt.figure(figsize=(12,8))
sns.heatmap(data.describe()[1:].transpose(),annot=True,linewidth=2,cmap=sns.color_palette("Set1"))
plt.title("Data summary")
plt.show()


# # CORRELATION BETWEEN VARIABLES

# In[11]:


correlation = data.corr()
plt.figure(figsize=(9,7))
sns.heatmap(correlation,annot=True,cmap=sns.color_palette("magma"),linewidth=2,edgecolor="k")
plt.title("CORRELATION BETWEEN VARIABLES")
plt.show()


# # Proportion of target variable in dataset

# In[13]:


plt.figure(figsize=(12,6))
plt.subplot(121)
ax = sns.countplot(y=data["target_class"],palette=["r","g"])
for i,j in enumerate(data["target_class"].value_counts().values):
    ax.text(.7,i,j,weight="bold",fontsize=27)
plt.title("Count for target variable in datset")


plt.subplot(122)
plt.pie(data["target_class"].value_counts().values,labels=["not pulsar stars","pulsar stars"],autopct="%1.0f%%",wedgeprops={"linewidth":2,"edgecolor":"white"})
my_circ = plt.Circle((0,0),.7,color = "white")
plt.gca().add_artist(my_circ)
plt.subplots_adjust(wspace = .3)
plt.title("Proportion of target variable in dataset")
plt.show()


# ### COMPARING MEAN & STANDARD DEVIATION BETWEEN ATTRIBUTES FOR TARGET CLASSES

# In[14]:


compare = data.groupby("target_class")[['mean_profile', 'std_profile', 'kurtosis_profile', 'skewness_profile',
                                        'mean_dmsnr_curve', 'std_dmsnr_curve', 'kurtosis_dmsnr_curve',
                                        'skewness_dmsnr_curve']].mean().reset_index()


compare = compare.drop("target_class",axis =1)

compare.plot(kind="bar",width=.6,figsize=(13,6),colormap="Set2")
plt.title("COMPARING MEAN OF ATTRIBUTES FOR TARGET CLASSES")

compare1 = data.groupby("target_class")[['mean_profile', 'std_profile', 'kurtosis_profile', 'skewness_profile',
                                        'mean_dmsnr_curve', 'std_dmsnr_curve', 'kurtosis_dmsnr_curve',
                                        'skewness_dmsnr_curve']].std().reset_index()
compare1 = compare1.drop("target_class",axis=1)
compare1.plot(kind="bar",width=.6,figsize=(13,6),colormap="Set2")
plt.title("COMPARING STANDARD DEVIATION OF ATTRIBUTES FOR TARGET CLASSES")
plt.show()


# ### COMPARING MEAN & STANDARD DEVIATION BETWEEN ATTRIBUTES FOR TARGET CLASSES

# In[15]:


compare_mean = compare.transpose().reset_index()
compare_mean = compare_mean.rename(columns={'index':"features", 0:"not_star", 1:"star"})
plt.figure(figsize=(13,14))
plt.subplot(211)
sns.pointplot(x= "features",y="not_star",data=compare_mean,color="r")
sns.pointplot(x= "features",y="star",data=compare_mean,color="g")
plt.xticks(rotation =60)
plt.xlabel("")
plt.title("COMPARING MEAN OF ATTRIBUTES FOR TARGET CLASSES")

compare_std = compare1.transpose().reset_index()
compare_std = compare_std.rename(columns={'index':"features", 0:"not_star", 1:"star"})
plt.subplot(212)
sns.pointplot(x= "features",y="not_star",data=compare_std,color="r")
sns.pointplot(x= "features",y="star",data=compare_std,color="g")
plt.xticks(rotation =60)
plt.title("COMPARING STANDARD DEVIATION OF ATTRIBUTES FOR TARGET CLASSES")
plt.subplots_adjust(hspace =.4)
print ("[GREEN == STAR , RED == NOTSTAR]")
plt.show()


# In[16]:


compare_mean
plt.figure(figsize=(12,6))
plt.subplot(121)
sns.barplot(y="features",x="not_star",data=compare_mean,color="r")
sns.barplot(y="features",x="star",data=compare_mean,color="g")
plt.title("COMPARING MEAN OF ATTRIBUTES FOR TARGET CLASSES")

plt.subplot(122)
sns.barplot(y="features",x="star",data=compare_std,color="g")
sns.barplot(y="features",x="not_star",data=compare_std,color="r")
plt.title("COMPARING STANDARD DEVIATION OF ATTRIBUTES FOR TARGET CLASSES")
plt.subplots_adjust(wspace =.6)


# # DISTIBUTION OF VARIABLES IN DATA SET

# In[17]:


columns = ['mean_profile', 'std_profile', 'kurtosis_profile', 'skewness_profile',
           'mean_dmsnr_curve', 'std_dmsnr_curve', 'kurtosis_dmsnr_curve',
           'skewness_dmsnr_curve']
length  = len(columns)
colors  = ["r","g","b","m","y","c","k","orange"] 

plt.figure(figsize=(13,20))
for i,j,k in itertools.zip_longest(columns,range(length),colors):
    plt.subplot(length/2,length/4,j+1)
    sns.distplot(data[i],color=k)
    plt.title(i)
    plt.subplots_adjust(hspace = .3)
    plt.axvline(data[i].mean(),color = "k",linestyle="dashed",label="MEAN")
    plt.axvline(data[i].std(),color = "b",linestyle="dotted",label="STANDARD DEVIATION")
    plt.legend(loc="upper right")
    
print ("***************************************")
print ("DISTIBUTION OF VARIABLES IN DATA SET")
print ("***************************************")


# # PAIR PLOT BETWEEN ALL VARIABLES

# In[18]:


sns.pairplot(data,hue="target_class")
plt.title("pair plot for variables")
plt.show()


# # Scatter plot between variable for target classes

# In[19]:


plt.figure(figsize=(14,7))
plt.subplot(121)
plt.scatter(x = "kurtosis_profile",y = "skewness_profile",data=data[data["target_class"] == 1],alpha=.7,label="pulsar stars",s=30,color = "g",linewidths=.4,edgecolors="black")
plt.scatter(x = "kurtosis_profile",y = "skewness_profile",data=data[data["target_class"] == 0],alpha=.6,label="not pulsar stars",s=30,color ="r",linewidths=.4,edgecolors="black")
plt.axvline(data[data["target_class"] == 1]["kurtosis_profile"].mean(),color = "g",linestyle="dashed",label="mean pulsar star")
plt.axvline(data[data["target_class"] == 0]["kurtosis_profile"].mean(),color = "r",linestyle="dashed",label ="mean non pulsar star")
plt.axhline(data[data["target_class"] == 1]["skewness_profile"].mean(),color = "g",linestyle="dashed")
plt.axhline(data[data["target_class"] == 0]["skewness_profile"].mean(),color = "r",linestyle="dashed")
plt.legend(loc ="best")
plt.xlabel("kurtosis profile")
plt.ylabel("skewness profile")
plt.title("Scatter plot for skewness and kurtosis for target classes")
plt.subplot(122)
plt.scatter(x = "skewness_dmsnr_curve",y = 'kurtosis_dmsnr_curve',data=data[data["target_class"] == 0],alpha=.7,label="not pulsar stars",s=30,color ="r",linewidths=.4,edgecolors="black")
plt.scatter(x = "skewness_dmsnr_curve",y = 'kurtosis_dmsnr_curve',data=data[data["target_class"] == 1],alpha=.7,label="pulsar stars",s=30,color = "g",linewidths=.4,edgecolors="black")
plt.axvline(data[data["target_class"] == 1]["kurtosis_dmsnr_curve"].mean(),color = "g",linestyle="dashed",label ="mean pulsar star")
plt.axvline(data[data["target_class"] == 0]["kurtosis_dmsnr_curve"].mean(),color = "r",linestyle="dashed",label ="mean non pulsar star")
plt.axhline(data[data["target_class"] == 1]["skewness_dmsnr_curve"].mean(),color = "g",linestyle="dashed")
plt.axhline(data[data["target_class"] == 0]["skewness_dmsnr_curve"].mean(),color = "r",linestyle="dashed")
plt.legend(loc ="best")
plt.xlabel("skewness_dmsnr_curve")
plt.ylabel('kurtosis_dmsnr_curve')
plt.title("||Scatter plot for skewness and kurtosis of dmsnr_curve for target classes")
plt.subplots_adjust(wspace =.4)


# # BOXPLOT FOR VARIABLES IN DATA SET WITH TARGET CLASS

# In[20]:


columns = [x for x in data.columns if x not in ["target_class"]]
length  = len(columns)
plt.figure(figsize=(13,20))
for i,j in itertools.zip_longest(columns,range(length)):
    plt.subplot(4,2,j+1)
    sns.lvplot(x=data["target_class"],y=data[i],palette=["orangered","lime"])
    plt.title(i)
    plt.subplots_adjust(hspace=.3)
    plt.axhline(data[i].mean(),linestyle = "dashed",color ="k",label ="Mean value for data")
    plt.legend(loc="best")
    
print ("****************************************************")
print ("BOXPLOT FOR VARIABLES IN DATA SET WITH TARGET CLASS")
print ("****************************************************")


# # Area plot for attributes of pulsar stars vs non pulsar stars

# In[21]:


st = data[data["target_class"] == 1].reset_index()
nst= data[data["target_class"] == 0].reset_index()
new = pd.concat([nst,st]).reset_index()

plt.figure(figsize=(13,10))
plt.stackplot(new.index,new["mean_profile"],alpha =.5,color="b",labels=["mean_profile"])
plt.stackplot(new.index,new["std_profile"],alpha=.5,color="r",labels=["std_profile"])
plt.stackplot(new.index,new["skewness_profile"],alpha=.5,color ="g",labels=["skewness_profile"])
plt.stackplot(new.index,new["kurtosis_profile"],alpha=.5,color = "m",labels=["kurtosis_profile"])
plt.axvline(x=16259,color = "black",linestyle="dashed",label = "separating pulsars vs non pulsars")
plt.axhline(new["mean_profile"].mean(),color = "b",linestyle="dashed",label = "average mean profile")
plt.axhline(new["std_profile"].mean(),color = "r",linestyle="dashed",label = "average std profile")
plt.axhline(new["skewness_profile"].mean(),color = "g",linestyle="dashed",label = "average skewness profile")
plt.axhline(new["kurtosis_profile"].mean(),color = "m",linestyle="dashed",label = "average kurtosis profile")
plt.legend(loc="best")
plt.title("Area plot for attributes for pulsar stars vs non pulsar stars")
plt.show()


# # Area plot for dmsnr_curve attributes of pulsar stars vs non pulsar star

# In[22]:


plt.figure(figsize=(13,10))
plt.stackplot(new.index,new["mean_dmsnr_curve"],color="b",alpha=.5,labels=["mean_dmsnr_curve"])
plt.stackplot(new.index,new["std_dmsnr_curve"],color="r",alpha=.5,labels=["std_dmsnr_curve"])
plt.stackplot(new.index,new["skewness_dmsnr_curve"],color="g",alpha=.5,labels=["skewness_dmsnr_curve"])
plt.stackplot(new.index,new["kurtosis_dmsnr_curve"],color="m",alpha=.5,labels=["kurtosis_dmsnr_curve"])
plt.axvline(x=16259,color = "black",linestyle="dashed",label = "separating pulsars vs non pulsars")
plt.axhline(new["mean_dmsnr_curve"].mean(),color = "b",linestyle="dashed",label = "average mean dmsnr_curve")
plt.axhline(new["std_dmsnr_curve"].mean(),color = "r",linestyle="dashed",label = "average std dmsnr_curve")
plt.axhline(new["skewness_dmsnr_curve"].mean(),color = "g",linestyle="dashed",label = "average skewness dmsnr_curve")
plt.axhline(new["kurtosis_dmsnr_curve"].mean(),color = "m",linestyle="dashed",label = "average kurtosis dmsnr_curve")
plt.legend(loc="best")
plt.title("Area plot for dmsnr_curve attributes for pulsar stars vs non pulsar stars")
plt.show()


# ## 3D PLOT FOR MEAN_PROFILE VS STD_PROFILE VS SKEWNESS_DMSNR_CURVE

# In[23]:


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(13,13))
ax  = fig.add_subplot(111,projection = "3d")

ax.scatter(data[data["target_class"] == 1][["mean_profile"]],data[data["target_class"] == 1][["std_profile"]],data[data["target_class"] == 1][["skewness_dmsnr_curve"]],
           alpha=.5,s=80,linewidth=2,edgecolor="k",color="lime",label="Pulsar star")
ax.scatter(data[data["target_class"] == 0][["mean_profile"]],data[data["target_class"] == 0][["std_profile"]],data[data["target_class"] == 0][["skewness_dmsnr_curve"]],
           alpha=.5,s=80,linewidth=2,edgecolor="k",color="r",label=" NotPulsar star")

ax.set_xlabel("mean_profile",fontsize=15)
ax.set_ylabel("std_profile",fontsize=15)
ax.set_zlabel("skewness_dmsnr_curve",fontsize=15)
plt.legend(loc="best")
fig.set_facecolor("grey")
plt.title("3D PLOT FOR MEAN_PROFILE VS STD_PROFILE VS SKEWNESS_DMSNR_CURVE",fontsize=10)
plt.show()


# # DENSITY PLOT BETWEEN MEAN_PROFILE & STD_PROFILE

# In[24]:


sns.jointplot(data["mean_profile"],data["std_profile"],kind="kde",scale=10)
plt.show()


# # Bubble plot between mean,std for skewness and kurtosis

# In[26]:


plt.figure(figsize=(13,7))
plt.subplot(121)
plt.scatter(st["mean_profile"],st["std_profile"],alpha=.5,s=st["skewness_profile"]*3,linewidths=1,color="g",label="pulsar_star")
plt.scatter(nst["mean_profile"],nst["std_profile"],alpha=.5,s=nst["skewness_profile"]*3,linewidths=1,color="r",label="pulsar_star")
plt.legend(loc="best")
plt.xlabel("mean_profile")
plt.ylabel("std_profile")
plt.title("Bubble plot for mean,std and skewness")


plt.subplot(122)
plt.scatter(st["mean_profile"],st["std_profile"],alpha=.5,s=st["kurtosis_profile"]*5,linewidths=1,color="g",label="pulsar_star")
plt.scatter(nst["mean_profile"],nst["std_profile"],alpha=.5,s=nst["kurtosis_profile"]*5,linewidths=1,color="r",label="pulsar_star")
plt.legend(loc="best")
plt.xlabel("mean_profile")
plt.ylabel("std_profile")
plt.title("Bubble plot for mean,std and kurtosis")
plt.show()


# ### Bubble plot between mean_dmsnr_curve,std_dmsnr_curve for skewness_dmsnr_curve and kurtosis_dmsnr_curve

# In[27]:


plt.figure(figsize=(13,7))
plt.subplot(121)
plt.scatter(st["mean_dmsnr_curve"],st["std_dmsnr_curve"],alpha=.5,s=st["skewness_dmsnr_curve"],linewidths=1,color="g",label="pulsar_star")
plt.scatter(nst["mean_dmsnr_curve"],nst["std_dmsnr_curve"],alpha=.5,s=nst["skewness_dmsnr_curve"],linewidths=1,color="r",label="pulsar_star")
plt.legend(loc="best")
plt.xlabel("mean_dmsnr_curve")
plt.ylabel("std_dmsnr_curve")
plt.title("Bubble plot for mean,std and skewness of dmsnr_curve")


plt.subplot(122)
plt.scatter(st["mean_dmsnr_curve"],st["std_dmsnr_curve"],alpha=.5,s=st["kurtosis_dmsnr_curve"],linewidths=1,color="g",label="pulsar_star")
plt.scatter(nst["mean_dmsnr_curve"],nst["std_dmsnr_curve"],alpha=.5,s=nst["kurtosis_dmsnr_curve"],linewidths=1,color="r",label="pulsar_star")
plt.legend(loc="best")
plt.xlabel("mean_dmsnr_curve")
plt.ylabel("std_dmsnr_curve")
plt.title("Bubble plot for mean,std and kurtosis of dmsnr_curve")
plt.show()


# # visualizing the distribution of a variables for target class

# In[28]:


columns = [x for x in data.columns if x not in ["target_class"]]
length  = len(columns)

plt.figure(figsize=(13,25))

for i,j in itertools.zip_longest(columns,range(length)):
    plt.subplot(length/2,length/4,j+1)
    sns.violinplot(x=data["target_class"],y=data[i],palette=["Orangered","lime"],alpha=.5)
    plt.title(i)


# # Parllel coordinates plot to compare features between variables

# In[29]:


from pandas.tools.plotting import parallel_coordinates
plt.figure(figsize=(14,8))
parallel_coordinates(data,"target_class",alpha=.5)
plt.show()


# # Proportion of target class in train & test data

# In[6]:


from sklearn.model_selection import train_test_split

train , test = train_test_split(data,test_size = .3,random_state = 123)

plt.figure(figsize=(12,6))
plt.subplot(121)
train["target_class"].value_counts().plot.pie(labels = ["not star","star"],autopct = "%1.0f%%",shadow = True,explode=[0,.1])
plt.title("proportion of target class in train data")
plt.ylabel("")
plt.subplot(122)
test["target_class"].value_counts().plot.pie(labels = ["not star","star"],autopct = "%1.0f%%",shadow = True,explode=[0,.1])
plt.title("proportion of target class in test data")
plt.ylabel("")
plt.show()


# In[7]:


#Separating Predictor and target variables
train_X = train[[x for x in train.columns if x not in ["target_class"]]]
train_Y = train[["target_class"]]
test_X = test[[x for x in test.columns if x not in ["target_class"]]]
test_Y = test[["target_class"]]


# # MODEL

# In[12]:


#MODEL FUNCTION

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve,auc

def model(algorithm, dtrain_x, dtrain_y, dtest_x, dtest_y, of_type):
    print ("*****************************************************************************************")
    print ("MODEL - OUTPUT")
    print ("*****************************************************************************************")
    
    algorithm.fit(dtrain_x,dtrain_y)
    predictions = algorithm.predict(dtest_x)
    
    print(algorithm)
    print("\naccuracy_score : ", accuracy_score(dtest_y, predictions))
    print("\nclassification report:\n", (classification_report(dtest_y, predictions)))
    
    plt.figure(figsize=(13,10))
    plt.subplot(221)
    sns.heatmap(confusion_matrix(dtest_y, predictions), annot=True,fmt="d",linecolor="k",linewidths=3)
    plt.title("CONFUSION MATRIX",fontsize=20)
    
    predicting_probabilities = algorithm.predict_proba(dtest_x)[:,1]
    fpr,tpr,thresholds = roc_curve(dtest_y,predicting_probabilities)
    plt.subplot(222)
    plt.plot(fpr,tpr,label=("Area under the curve :", auc(fpr,tpr)), color="r")
    plt.plot([1,0],[1,0],linestyle="dashed",color="k")
    plt.legend(loc="best")
    plt.title("ROC - CURVE & AREA UNDER CURVE", fontsize=20)
    
    if  of_type == "feat":
        
        dataframe = pd.DataFrame(algorithm.feature_importances_,dtrain_x.columns).reset_index()
        dataframe = dataframe.rename(columns={"index":"features",0:"coefficients"})
        dataframe = dataframe.sort_values(by="coefficients",ascending = False)
        plt.subplot(223)
        ax = sns.barplot(x = "coefficients" ,y ="features",data=dataframe,palette="husl")
        plt.title("FEATURE IMPORTANCES",fontsize =20)
        for i,j in enumerate(dataframe["coefficients"]):
            ax.text(.011,i,j,weight = "bold")
    
    elif of_type == "coef" :
        
        dataframe = pd.DataFrame(algorithm.coef_.ravel(),dtrain_x.columns).reset_index()
        dataframe = dataframe.rename(columns={"index":"features",0:"coefficients"})
        dataframe = dataframe.sort_values(by="coefficients",ascending = False)
        plt.subplot(223)
        ax = sns.barplot(x = "coefficients" ,y ="features",data=dataframe,palette="husl")
        plt.title("FEATURE IMPORTANCES",fontsize =20)
        for i,j in enumerate(dataframe["coefficients"]):
            ax.text(.011,i,j,weight = "bold")
            
    elif of_type == "none" :
        return (algorithm)
    
    


# # RandomForestClassifier

# In[13]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
model(rf, train_X, train_Y, test_X, test_Y, "feat")


# # LogisticRegression

# In[14]:


from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
model(log,train_X,train_Y,test_X,test_Y,"coef")


# # DecisionTreeClassifier

# In[15]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
model(dt,train_X,train_Y,test_X,test_Y,"feat")


# # GradientBoostingClassifier

# In[16]:


from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
model(gbc,train_X,train_Y,test_X,test_Y,"feat")


# # Gaussian Naive Bayes

# In[17]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
model(nb,train_X,train_Y,test_X,test_Y,"none")


# # K- Nearest Neighbour Classifier

# In[18]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
model(knn,train_X,train_Y,test_X,test_Y,"none")


# In[ ]:




