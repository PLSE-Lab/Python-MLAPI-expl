#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv("/kaggle/input/students-performance-in-exams/StudentsPerformance.csv")


# In[ ]:


data = pd.DataFrame(data)
data.head()


# In[ ]:


plt.rcParams['figure.figsize'] = (8,5)
sns.kdeplot(data["math score"])
sns.kdeplot(data["reading score"])
sns.kdeplot(data["writing score"])
plt.show()


# In[ ]:


# Let' create a new feature called total score 
data["tot_score"] = data["math score"]+data["reading score"]+data["writing score"]


# In[ ]:


num_col = data.select_dtypes(exclude = object).columns
data[num_col].describe()


# In[ ]:


cat_col = data.select_dtypes(include = object).columns


# In[ ]:


data[cat_col]


# In[ ]:


for i in cat_col:
    print(i," \n")
    print(data[i].value_counts())
    print("==="*15)


# In[ ]:


# let's assign grade Based up on average score's obtained by .. (math,reading,writing)//3
data["grades"]  = [0]*data.shape[0]


# In[ ]:


for i in range(len(data["tot_score"])):
    if((data["tot_score"][i]//3)>79):
        data["grades"][i] = "A"
    elif((data["tot_score"][i]//3)>59 and (data["tot_score"][i]//3) <80 ):
        data["grades"][i] = "B"
    elif((data["tot_score"][i]//3)>35 and (data["tot_score"][i]//3) <60 ):
        data["grades"][i] = "C" 
    else:
        data["grades"][i] = "D"
        
        


# In[ ]:


grades = data.grades.value_counts()
print("grades counts :\n",grades)
plt.rcParams['figure.figsize'] = (8,5)
plt.pie(grades.values,labels = grades.index,shadow  = True,explode = [0.0001,0.0001,0.0001,0.0001])
plt.legend(["B","C","A","D"])
plt.title("Grades")
plt.tight_layout()
plt.show()

# Note :  Most of the Students are Having B Grade . 


# # let's do some bivariate analysis

# In[ ]:


# gender v/s different test score's 

# note : male's   are good at Math
# note : female's are good at reading and writing 

data.pivot_table(index = "gender",values = ["math score","reading score","writing score"]).plot(kind = 'bar')
plt.xlabel("Gender")
plt.ylabel("Mean Different Test Scores : ")


# In[ ]:


data.pivot_table(index = "test preparation course",values = ["math score","reading score","writing score"]).plot(kind = 'bar')
plt.xlabel("Gender")
plt.ylabel("Mean Different Test Scores : ")

# Note : Student's who are completed test preparation course are good at all three test score's compared to students who are not done test preparation


# In[ ]:


# let's see how gender affects different test score's
# This shows Regression task, we can see the linear relationship among the featues.
sns.pairplot(data,hue = "gender")


# In[ ]:


data.columns


# # Let's see  the cause for   Student Being  a topper .

# In[ ]:



topper = data[(data["math score"] >80) & (data["reading score"] >80) & (data["writing score"] >80)]
topper.shape


# In[ ]:


topper


# # Which Gender Got Highest Toppers count ?
# # Ans : Female's are Most Toppers ! 

# In[ ]:


print(topper["gender"].value_counts())
sns.countplot(topper["gender"],palette="PuBu_r")
plt.show()


# # let"s see how parental level education affects Being Topper 

# In[ ]:



plt.rcParams['figure.figsize'] = (10,7)
print(topper["parental level of education"].value_counts())
sns.countplot(topper["parental level of education"],palette="gist_stern")
plt.show()


# # let"s see how test preparation course feature affects Being Topper

# In[ ]:


plt.rcParams['figure.figsize'] = (10,7)
print(topper["test preparation course"].value_counts())
sns.countplot(topper["test preparation course"],palette="spring")
plt.show()


# # let"s see how race/ethnicity course feature affects Being Topper 

# In[ ]:


plt.rcParams['figure.figsize'] = (10,7)
print(topper["race/ethnicity"].value_counts())
sns.countplot(topper["race/ethnicity"],palette="spring_r")
plt.show()


# # Most of The Topper Student's are having Standard lunch

# In[ ]:


# let"s see how lunch  feature affects Being Topper 
plt.rcParams['figure.figsize'] = (10,7)
print(topper["lunch"].value_counts())
sns.countplot(topper["lunch"],palette="cubehelix")
plt.show()


# # Let's see the cause Behind ,student's having Very less score 

# In[ ]:


low_scorrer = data[(data["math score"] <35) & (data["reading score"] <35) & (data["writing score"] <35)]
low_scorrer.shape


# In[ ]:


low_scorrer


# # let"s see how parental level education affects Being low_scorrer 
# Insight :  Neither of the Student Parent's  are from Education level Master's, bachelor's , associate degree's

# In[ ]:


plt.rcParams['figure.figsize'] = (10,7)
print(low_scorrer["parental level of education"].value_counts())
sns.countplot(low_scorrer["parental level of education"],palette="cividis_r")
plt.show()


# # let"s see how lunch  feature affects Being Low_Scrorrer

# In[ ]:



plt.rcParams['figure.figsize'] = (10,7)
print(low_scorrer["lunch"].value_counts())
sns.countplot(low_scorrer["lunch"],palette="autumn")
plt.show()


# In[ ]:


data.columns


# # Let's See how test preparation course affects being low_scroccer 

# ##### This show's the reason Behind being Low scorrer , I.e student's who are not took test preparation course  ended up having very test's less score !

# In[ ]:


print(low_scorrer["test preparation course"].value_counts())


# # Let's See overall Passed and Failed Percentage Of Student's in all three test's 
# # Note : I have Considered Student's With "D" Grades as Failed Once's and Student's With A or B or C Grade's  as Passed Onces .

# In[ ]:


fail_count = data[data["grades"]=="D"]["grades"].count()
pass_count = data[data["grades"]!="D"]["grades"].count()
labels     = ["Failed","Passed"]
plt.pie([fail_count,pass_count],labels = labels,colors = ["red","green"])
plt.legend(["fail_count","pass_count"])
plt.tight_layout()
plt.show()


# In[ ]:


data.head()


# # Let's Build a Regression Model First In order to Predict Total Score Obtained by Student Based on Given Features.

# In[ ]:


data["tot_score"] = data["tot_score"].astype(int) 


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error


# In[ ]:


scopy = data.copy()
c2 = data.copy()


# In[ ]:


scopy =pd.get_dummies(scopy,drop_first = True)
scopy.shape


# In[ ]:


x = scopy.drop("tot_score",axis = 1)
y = scopy["tot_score"]


# In[ ]:


lr = LinearRegression()
xtrain,xtest,ytrain,ytest  = train_test_split(x,y,test_size = 0.3,random_state = 33)
lr.fit(xtrain,ytrain)


# In[ ]:


print("Train set Perfomance :  ")
n = len(xtrain)
k = xtrain.shape[1]
y_pred_train = lr.predict(xtrain)
r2 = r2_score(y_pred_train,ytrain)
adjusted_r2_score = (1- (((1 - r2)*(n-1))/(n-k-1)))
print("r2_score value : ",r2)
print("adjusted_r2_score value : ",adjusted_r2_score)
print("MSE Value : ",mean_squared_error(y_pred_train,ytrain))


# In[ ]:


print("Test set Perfomance :  ")
n = len(xtest)
k = xtest.shape[1]
y_pred_test = lr.predict(xtest)
r2 = r2_score(y_pred_test,ytest)
adjusted_r2_score = (1- (((1 - r2)*(n-1))/(n-k-1)))
print("r2_score value : ",r2)
print("adjusted_r2_score value : ",adjusted_r2_score)
print("MSE Value : ",mean_squared_error(y_pred_test,ytest))


# In[ ]:


res  = pd.DataFrame(ytest,dtype = int)
res.rename(columns = {"tot_score":"actual"},inplace = True)
res["predicted"] = y_pred_test
res["predicted"] =  res["predicted"].astype(int)


# In[ ]:


res.head()


# In[ ]:


# Let's see the result Visually .

sns.scatterplot(res["predicted"],res["actual"])

