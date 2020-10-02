#!/usr/bin/env python
# coding: utf-8

# The dataset at hand concerns a company that is concerned with its rate of attrition, our aim is to gather meaningful insights from the data of 15000 employees and hopefully come up with a reliable model that can accurately predict which employee/s are about to leave, thus increasing managerial efficiency and allowing the company to retain its talent pool.
# 

# In[ ]:


from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import binarize
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn import tree
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report , confusion_matrix
from sklearn.model_selection import cross_val_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as  plt
from sklearn.model_selection import train_test_split
from scipy import stats
from matplotlib import rc
import collections


# Fields in the dataset include:
# 1. Satisfaction Level
# 1. Last evaluation
# 1. Number of projects
# 1. Average monthly hours
# 1. Time spent at the company
# 1. Whether they have had a work accident
# 1. Whether they have had a promotion in the last 5 years
# 1. Departments (column sales)
# 1. Salary
# 1. Whether the employee has left

# In[ ]:


df = pd.read_csv('../input/HR_comma_sep.csv')


# In[ ]:


da =df[df['left']==1]


# # Exploratory Data Analysis

# To tackle the given questions we combine methods of **Exploratory Data Analysis**(EDA) and Supervised Machine Learning.
# John Tukey defined EDA as "Procedures for analyzing data, techniques for interpreting the results of such procedures, ways of planning the gathering of data to make its analysis easier, more precise or more accurate, and all the machinery and results of (mathematical) statistics which apply to analyzing data."
# Put simply EDA gives us our initial insight on how different variables in the data act with respect to each other and within themselves. I use some very common methods of EDA in the following section, keeping my methods very basic so that they maybe interpreted easily by someone with limited knowledge of the field, to that end I extensively use Python's plotting functionality to communicate results visually.

# In[ ]:


df.describe()


# An intial look at this table provides useful information:
# 1.  Approximately 24% of the employees have left the firm, which is a significant loss of talent to the firm.
# 1.  On an average the employees got a score of 0.71 which is quite good and when combined with a very low standard deviation, the numbers look quite good.
# 1.  Even though the high evaluation score might look good at the face of it, the high attrition rate   might imply that the company may be losing most of its high-performing employees.
# 
# To adress this question let's look at the numbers for only those who left the firm.    

# In[ ]:


da = df[df['left']==1]
da.describe()


# Looking at the data of only those who left tell an interesting story.
# 1. 1. The employees who left the firm have performed only marginally better than the average employee.
# 1. 2. These employees, however, have a much less level of satisfaction. 
# 1. 3. Other factors seem quite balanced, these employees:
#     1.  Spent almost the same amount of time each day in the office.
#     1.  Were alloted equally as many projects as everyone else.
# 
# So the level of satisfaction might be the single most important factor in predicting employee attrition, this follows from common sense and is also consistent with our findings.
Having looked at the numbers we now have sufficient insight to look at more intricate aspects of the data and try to extract information based on graphs.
# In[ ]:


def Histogram(data,to_plot):
    for i in range(len(to_plot)):
        plt.hist(data[to_plot[i]])
        plt.axvline(data[to_plot[i]].mean(),color='r')
        plt.xlabel(to_plot[i])
        plt.show()


# In[ ]:


to_plt =['satisfaction_level','last_evaluation','average_montly_hours']
Histogram(df,to_plt)


# Not much can be said from merely looking the histograms above, let's take it one step further and see 
# 1. How differently are the last evaluation scores of the two groups (left and stayed) distributed.
# 1. Which departments did the ones who left belong to.

# In[ ]:


sns.set(font_scale=2)
fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(20,6))
asx = df[df['left']==1]
sns.kdeplot(data=asx['last_evaluation'],color='b',ax=axs[0],shade=True,label='left')
axs[0].set_xlabel('last_evaluation')
asd = df[df['left']==0]
sns.kdeplot(data=asd['last_evaluation'],color='g',ax=axs[0],shade=True,label='stayed')
sns.countplot(x=asx['sales'],ax=axs[1],label='Department wise attrition')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# Our key takeaways from the charts are:
# 1. The employees who left are either those who performed poorly or performed highly, those in the middle are very few.
# 1.  The sales department has seen the highest level of attrition.
# 
# Let's now focus on how the features in data behave with respect to one-another, we shall do this via a co-relation heatmap.

# Let's dig in a bit deeper and look at other features in our data. To do so we check for co-relation between the features.

# In[ ]:


sns.set(font_scale=1.5)
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(),annot=True)
plt.xticks(rotation=90)
plt.show()


# Just at a glance we see:
# 1. The number of projects and the average monthly hours worked by an employee are positively co-related
# 1. The level of satisfaction and 'left' variables are negatively co-related which further lends evidence to our previous hypothesis.
# 
# Other than that none of the features show a co-relation high enough to mean something.
# An upside to this is that it simplifies our Feature Engineering process,however, we need to keep looking deeper for any actionable intelligence, the above inferences are a mere confirmation of what one could already think of intuitively.

# Let's now look at the data based on the condition of our target variable, i.e, let's look at the data seperately for the employees who left and stayed.

# In[ ]:


sns.set(color_codes=True)
plot = sns.FacetGrid(df,col='left',hue='left',size=5)
plot.map(sns.kdeplot,'satisfaction_level','last_evaluation',shade=True,cmap='Blues')
plt.show()


# The density plot reveals some really intriguing results, we now see that the employees who left exhaustively belong to three clusters
# 1. Less satisfied and Under-performers
# 1. Less satisfied and Above-average performers
# 1. Highly satisfied and Above-average performers
# 
# There is some relief for the firm in seeing that most of the people who left were under-performers as such. 
# There are some high-performers who left, no surprise, as we've already seen that low satisfaction is a cause for attrition.
# 
# It is the third group that intrigues me the most, employees with high evaluation scores and satisfaction level presumably have a bright future in any firm, why then would they leave? Let's see if we can dig in and answer this question.

# To do so let's isolate the group and look at its characteristics. It makes sense to start with examining their salaries as employees that perform well would seek jobs elsewhere if they find jobs with a higher pay.

# In[ ]:


# dt is the data of all the employees in the third group
dt=da[[all([a,b]) for a,b in zip(da['last_evaluation'] > df['last_evaluation'].mean(),da['satisfaction_level']>df['satisfaction_level'].mean())]]
# dl is the data of all the employees who had low salaries in the above group
dl = dt[dt['salary']=='low']
sns.countplot(dl['sales'])
plt.xticks(rotation=90)
plt.show()


# So this chart makes it pretty clear that within the third group of individuals who left, the sales employees showed the highest (and by a huge margin) number of people who earned low salaries, hence the high attrition.

# ## How well do the departments fare individually?
# I now look into how the employees in various departments behave, how well do leaders of each department perform in terms of retaining their talent pool and keeping their employees satisfied, at the same time not hampering the efficiency with which work is carried out.
# To do so we define our **Key Performance Indicatiors** as the aggregate of:
# 1. Level of Satisfaction
# 1. Last evaluation
# 1. Average Monthly Hours worked
# 1. Number of Projects undertaken

# In[ ]:


depts = df['sales'].unique()
avgs =[]
avgl =[]
avgn =[]
avgm =[]
for i in depts:
    mean = df['satisfaction_level'][df['sales']==i].mean()
    avgs.append(mean)       
for i in depts:
    mean = df['last_evaluation'][df['sales']==i].mean()
    avgl.append(mean)       
for i in depts:
    mean = df['number_project'][df['sales']==i].mean()
    avgn.append(mean)
for i in depts:
    mean = df['average_montly_hours'][df['sales']==i].mean()
    avgm.append(mean)


# In[ ]:


averages=pd.DataFrame({'Depts': depts,'AVGS':avgs,'AVGL':avgl,'AVGN':avgn,'AVGM':avgm},index=None)


# In[ ]:


sns.set(style="whitegrid",font_scale=2)
q = sns.PairGrid(averages.sort_values('AVGS',ascending=False),y_vars='Depts',x_vars=['AVGS','AVGN','AVGM','AVGL'],size=12,aspect=0.5)
q.map(sns.stripplot,orient='h',palette="Reds_r", edgecolor="gray",size=30,)
titles = ["Last Evaluation", "Satisfaction", "Number of Projects",'Average Monthly Hours']
plt.title('Department wise performance based on our KPI')
for ax, title in zip(q.axes.flat, titles):

    # Set a different title for each axes
    ax.set(title=title)

    # Make the grid horizontal instead of vertical
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
plt.show()


# Clearly the management department has done the best job in handling its employees not only are its employees have the highest mean evaluation, they have also taken up more projects than most departments hence indicating efficiency. I leave the scrutiny of other departments as an excercise to the reader, share your findings in the comments section below.

# In[ ]:


ds = df[df['left']==1]
x_vars=['satisfaction_level','last_evaluation','number_project']
depts = df['sales'].unique()
lavgs =[]
lavgl =[]
lavgn =[]
lavgm =[]
for i in depts:
    mean = ds['satisfaction_level'][da['sales']==i].mean()
    lavgs.append(mean)       
for i in depts:
    mean = ds['last_evaluation'][da['sales']==i].mean()
    lavgl.append(mean)       
for i in depts:
    mean = ds['number_project'][da['sales']==i].mean()
    lavgn.append(mean)       
for i in depts:
    mean = ds['average_montly_hours'][da['sales']==i].mean()
    lavgm.append(mean)


# In[ ]:


Laverages=pd.DataFrame({'Depts': depts,'AVGS':lavgs,'AVGL':lavgl,'AVGN':lavgn,'AVGM':lavgm},index=None)


# In[ ]:


sns.set(style="whitegrid",font_scale=3)
lq = sns.PairGrid(Laverages.sort_values('AVGL',ascending=False),y_vars='Depts',x_vars=['AVGL','AVGS','AVGN','AVGM'],size=12,aspect=0.5)
lq.map(sns.stripplot,orient='h',palette="Reds_r", edgecolor="black",size=30)
titles = ["Last Evaluation", "Satisfaction", "Number of Projects",'Average Monthly Hours']
plt.title('Department wise performance based on our KPI')
for ax, title in zip(lq.axes.flat, titles):

#     Set a different title for each axes
    ax.set(title=title)

    # Make the grid horizontal instead of vertical
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
plt.show()


# As is quite clear from the plots, RandD has lost the highest number of talented employees and is also accompanied by low satisfaction. 
# Again, I'd love to hear your insight about the rest in the comments.

# # Machine Learning

# We now aim to use Machine Learning to model the data so as to accurately predict which employee will leave.
# 

# ## Preprocessing
# We start by preparing our data for our models
# 

# In[ ]:


X = df.drop(['left'],axis=1)
y = df['left']
le = LabelEncoder()
X['salary']= le.fit_transform(X['salary'])
X['sales']= le.fit_transform(X['sales'])


# In[ ]:


X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=2,stratify=y)
# Creating a seperate scaled set to be used in some models to improve our results
Xscaled_train = pd.DataFrame(preprocessing.scale(X_train.values))
Xscaled_test  =  pd.DataFrame(preprocessing.scale(X_test.values))


# ## Training, Testing and Optimising
# We now start with the process of building our models.
# Throughout the section I have used the AUC score as the metric for model performance. AUC is the most appropriate choice as the classes are highly unbalanced and thus we would see a very high null accuracy. Also we need to prioritise on true positives as it's better to falsely predict that an employee will leave than to lose a good employee because of a wrong prediction.
# 
# The AUC is defined as "probability that the classifier will assign a higher score to a randomly chosen positive example than to a randomly chosen negative example"

# ### Gaussian Naive Bayes
# We start with this simple and fast model and mostly blackbox our way through without optimising for AUC score. 

# In[ ]:


gnb = GaussianNB() # Gaussian Naive Bayes
gnb.fit(Xscaled_train,y_train)
gnbpred = gnb.predict_proba(Xscaled_test)
print (roc_auc_score(y_test,gnbpred[:,1]))


# We're off to a decent start let's see how much we can improve the score.

# In[ ]:


def params_tuning(model,X_train,y_train,X_test,y_test,metrics,param_grid,clf=False,conf=False):
    """Tune parameters of the model using a grid search, this function just makes the job easier."""
    gs=GridSearchCV(model,param_grid=param_grid,scoring=metrics,cv=10)
    gs.fit(X_train,y_train)
    predicted = gs.predict(X_test)
    proba = gs.predict_proba(X_test)[:,1]
    if clf == True:
        print (classification_report(y_test,predicted))
    if conf == True:
        print (confusion_matrix(y_test,predicted))
    print (gs.best_params_)
    print (roc_auc_score(y_test,proba),'Optimised Score')


# ### K Nearest Neighbors
# We train the model and optimise for 
# 1. Number of neighbors to be considered.
# 1. Weights to be assigned.

# In[ ]:


knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
knnp = knn.predict_proba(X_test)
print (roc_auc_score(y_test,knnp[:,1]),"Initial Score")
weight_options = ['uniform','distance']
params_grid_knn = dict(n_neighbors = range(1,18) ,weights=weight_options)
params_tuning(knn,X_train,y_train,X_test,y_test,'roc_auc',params_grid_knn)


# We see a significant jump in the score post optimisation.
# This classifier yields a significantly better result and is potentially the best candidate for the final model. One reason why it works so well is that, as seen above, employees who left form very distinct clusters.

# ### Support Vector Machine
# We fit a SVM and optimise the regularisation constant 'C'. We use the scaled X_train and X_test as the model is much more efficient under the same.

# In[ ]:


svc = svm.SVC(probability=True,random_state=12)
svc.fit(Xscaled_train,y_train)
scaledp = svc.predict_proba(Xscaled_test)
print (roc_auc_score(y_test,scaledp[:,1]),'Initial Score')
svcp = dict(C=np.linspace(0.1,1,5))
params_tuning(svc,Xscaled_train,y_train,Xscaled_test,y_test,'roc_auc',svcp)


# ### Logistic Regression

# In[ ]:


logreg = LogisticRegression(random_state=12)
logreg.fit(Xscaled_train,y_train)
logp = logreg.predict_proba(Xscaled_test)
print (roc_auc_score(y_test,logp[:,1]),"Inital Score")
logp = dict(C=np.linspace(0.16,0.2,5),solver=['newton-cg', 'lbfgs','sag'])
params_tuning(logreg,Xscaled_train,y_train,Xscaled_test,y_test,'roc_auc',logp,clf=True)


# The inital score is very poor given the performance of previous models, optimisation doesn't help much. Notice a grave problem in the classification report: the model has an **extremely low sensitivity** as is reflected by the low recall score, a possible fix for this adjusting the threshold, to do so let's start by plotting ROC.

# In[ ]:


logreg= LogisticRegression(C=0.17,solver='sag')
logreg.fit(Xscaled_train,y_train)
logregprob = logreg.predict(Xscaled_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, logregprob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()


# We may continue further with adjusting the threshold values, however, notice the shape of the curve, it's slope changes at only one point, thus altering the threshold values will have no affect on the sensitivity, thus the process will be futile.

# In[ ]:


def evaluate_threshold(threshold):
    """returns sensitivity and specifity for a given threshold value"""
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])


# ## Decision Tree Classifier
# Note: Optimising this algorithm was very computationally expensive, hence I decided to do away with it. 

# In[ ]:


tr = tree.DecisionTreeClassifier(class_weight='balanced',random_state=12)
tr.fit(X_train,y_train)
predtree = tr.predict(X_test)
print (roc_auc_score(y_test,predtree))


# The Decision Tree Classifier allows us to see which features were most important in predicting the outcomes, let's see how different features ranked in terms of their importance.

# In[ ]:


def importance_plotting(data,x,y,palette,title):
    sns.set(style="whitegrid")
    ft = sns.PairGrid(data,y_vars=y,x_vars=x,size=5,aspect=1)
    ft.map(sns.stripplot,orient='h',palette=palette, edgecolor="black",size=15)
    for ax, title in zip(ft.axes.flat, titles):
    # Set a different title for each axes
        ax.set(title=title)
    # Make the grid horizontal instead of vertical
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)
    plt.show()


# In[ ]:


fo = {'Features':df.drop('left',axis=1).columns.tolist(),'Importance':tr.feature_importances_}
Importance = pd.DataFrame(fo,index=None).sort_values('Importance',ascending=False)
titles = ["Importance of the various Features in predicting the outcome"]
importance_plotting(Importance,'Importance','Features','Greens_r',titles)


# As is consistent with our previous observation, satisfaction level proved to be the most important feature. It is shocking and  also important to note how low 'salary' ranked in terms of importance

# ## Ensemble Methods

# ### Random Forest Classifier

# In[ ]:


rf = RandomForestClassifier() #Random Forest
rf.fit(X_train,y_train)
rfpred=rf.predict(X_test)
rfp = dict(n_estimators=np.arange(5,25,10))
print (roc_auc_score(y_test,rfpred),'Initial Score')
params_tuning(rf,X_train,y_train,X_test,y_test,'roc_auc',rfp)


# 1. We've finally managed to cross the 0.98 bound and are now looking at a near perfect score. Let's see which features ranked most important in the process.

# In[ ]:


ho = {'Features':df.drop('left',axis=1).columns.tolist(),'Importance':rf.feature_importances_}
ImportanceRF = pd.DataFrame(ho,index=None).sort_values('Importance',ascending=False)
importance_plotting(ImportanceRF,'Importance','Features','Greens_r',titles)


# ### Adaptive Boosting

# In[ ]:


ada = AdaBoostClassifier(algorithm='SAMME')
ada.fit(Xscaled_train,y_train)
adaproba= ada.predict_proba(Xscaled_test)
print (roc_auc_score(y_test,adaproba[:,1]),'Initial Score')
adap = dict(n_estimators=[25,50,75],learning_rate =[0.25,0.5,0.75,1])
params_tuning(ada,Xscaled_train,y_train,Xscaled_test,y_test,'roc_auc',adap,clf=True)


# Even though the  classifier produces a good score we face the problem of low sensitivity and since we wish to optimise for higher positive predictions, adjusting the threshold values is worth considering.

# In[ ]:


fpr, tpr, thresholds = metrics.roc_curve(y_test,adaproba[:,1])
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()


# In[ ]:


evaluate_threshold(0.479)
# We decrease the threshold to increase sensitivity and I reached this value through simple trial and error.
binned = (binarize(adaproba[:,1].reshape(-1,1),0.479))
print (classification_report(y_test,binned))


# Indeed we've seen an increase in recall even though precision has taken a hit.

# ### Gradient Boosting 
# The Gradient Boosting Classifier is arguably one of the best while dealing with Classification, we look to make the most of it by optimising its parameters iteratively.

# In[ ]:


gbr = GradientBoostingClassifier()
gbr.fit(Xscaled_train,y_train)
gbrp = gbr.predict_proba(Xscaled_test)
print (roc_auc_score(y_test,gbrp[:,1]),'Initial Score')


# In[ ]:


# gbr = GradientBoostingClassifier(min_samples_leaf=10,max_features='sqrt',n_estimators=82,min_samples_split=400,max_depth=15,subsample=0.85,random_state=12)
# We start by training model specific parameter n_estimators
gbr = GradientBoostingClassifier(min_samples_split=75,max_depth=8,min_samples_leaf=50,max_features='sqrt',subsample=0.8,random_state=12)
gbrd = dict(n_estimators=range(20,81,10))
params_tuning(gbr,Xscaled_train,y_train,Xscaled_test,y_test,'roc_auc',gbrd)


# In[ ]:


### tree specific parameters
gbr = GradientBoostingClassifier(n_estimators=70,min_samples_split=75,max_depth=8,min_samples_leaf=50,max_features='sqrt',subsample=0.8,random_state=12)
gbrd = dict(max_depth=range(5,16,2),min_samples_split = range(15,80,10))
params_tuning(gbr,Xscaled_train,y_train,Xscaled_test,y_test,'roc_auc',gbrd)


# In[ ]:


# subsample
gbr = GradientBoostingClassifier(min_samples_split=15,max_depth=15,n_estimators=70,min_samples_leaf=50,max_features='sqrt',random_state=12)
gbrd = dict(subsample=[0.75,0.8,0.85,0.9])
params_tuning(gbr,Xscaled_train,y_train,Xscaled_test,y_test,'roc_auc',gbrd)


# # Conclusion
# We now have modelled the data over various algorithms and the clearly Gradient Boosting Classifier provides the best result, we've also drawn relevant insights from our EDA. This information can be further used to predict which employees might leave the firm, thus saving the firm a significant amount of money over time, also the managers can now take preventive action to reduce attrition and improve the overall environment in the firm.

# In[ ]:


scores_list= pd.DataFrame(collections.OrderedDict([('Model',['Gaussian Naive Bayes','K Neighbors','Support Vector Machine','Logistic Regression','Decision Tree','Random Forest','Adaptive Boosting','Gradient Boosting']),('AUC_Score',[0.848839820268,0.977592568086,0.974202826362,0.802274102992,0.972431754854,0.988187674457,0.958517593389,0.992851230606])])) 
print (scores_list.sort_values('AUC_Score',ascending=False))

