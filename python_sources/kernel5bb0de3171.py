#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st


# So basically it is a Classification Problem as we have to know if employee will be promoted or not

# In[ ]:


train=pd.read_csv("../input/train.csv")


# In[ ]:


test=pd.read_csv("../input/test.csv")


# In[ ]:


train.shape


# In[ ]:


test.shape


# ##### Problem Statement
# 
#   Your client is a large MNC and they have 9 broad verticals across the organisation. One of the problem your client is facing is around identifying the right people for promotion (only for manager position and below) and prepare them in time. Currently the process, they are following is:
# 
#     . They first identify a set of employees based on recommendations/ past performance
#     
#     . Selected employees go through the separate training and evaluation program for each vertical. These programs are based on the required skill of each vertical
#         
#     . At the end of the program, based on various factors such as training performance, KPI completion (only employees with KPIs completed greater than 60% are considered) etc., employee gets promotion
#     
#     . For above mentioned process, the final promotions are only announced after the evaluation and this leads to delay in transition to their new roles. Hence, company needs your help in identifying the eligible candidates at a particular checkpoint so that they can expedite the entire promotion cycle. 
# 
# They have provided multiple attributes around Employee's past and current performance along with demographics. 
# 
# Now, The task is to predict whether a potential promotee at checkpoint in the test set will be promoted or not after the evaluation process.

# In[ ]:


combined=pd.concat([train,test],ignore_index=True,sort=True)


# In[ ]:


combined_backup=combined.copy()


# In[ ]:


combined.head()


# In[ ]:


combined.info()


# In[ ]:


combined.isnull().sum()


# In[ ]:


combined.employee_id.nunique()


# In[ ]:


combined.shape


# #### Age Analysis :
# People with more than54 years of age are outliers

# In[ ]:


sns.boxplot(combined.age)


# In[ ]:


sns.distplot(combined.age)
#Highly right skewed 


# In[ ]:


combined["KPIs_met >80%"].value_counts().plot(kind="bar")


# In[ ]:


sns.boxplot(combined["avg_training_score"])


# In[ ]:


sns.distplot(combined["avg_training_score"])


# average training score is grouped we can clearly see it
# 
#     most people lie on 45-50 bracket with higher intensity
#     a few people lie on 70 bracket
#     50,60,70,85

# In[ ]:


combined.columns


# In[ ]:


sns.countplot(combined["awards_won?"])

#maximum poeple has won no awards. so obviously maximum people has got no awrds,

# we can figure out which are the poeple who got awards and were they promoted or  


# In[ ]:


combined[(combined["awards_won?"]==1) & (combined["is_promoted"]==1.0)]["age"]


#          559 PEOPLE WHO GOT PROMOTED AND WON AWARDS

# In[ ]:


sns.distplot(combined[(combined["awards_won?"]==1) & (combined["is_promoted"]==1.0)]["age"])


# Most people who got promoted and won awards are of 30 years of age 

# In[ ]:


combined[(combined["awards_won?"]==1) & (combined["is_promoted"]==0.0)]


# In[ ]:


sns.distplot(combined[(combined["awards_won?"]==1) & (combined["is_promoted"]==0.0)]["age"])


# In[ ]:


combined[(combined["awards_won?"]==0) & (combined["is_promoted"]==1.0)]


# In[ ]:


#0 awards won and is getting promoted is 4109


# In[ ]:


sns.countplot(combined.department)
plt.xticks(rotation=90)


# In[ ]:


combined.department.value_counts().plot(kind="bar")


# Sales & Marketing
# 
# Operations
# 
# Procurement
# 

# In[ ]:


combined.education.value_counts().plot(kind="bar")


# In[ ]:


#most people are bachelors


# In[ ]:


combined.gender.value_counts().plot(kind="bar")


# In[ ]:


plt.figure(figsize=(10,5))
combined.length_of_service.value_counts().plot(kind="bar")
plt.xticks(rotation=90)


# #### Max people with 3 years of work ex so its quiet definited that they got promoted the most
# 

# In[ ]:


sns.countplot(combined["no_of_trainings"])


# #### most poeple have done 1 Year of training 

# In[ ]:


sns.countplot(combined.previous_year_rating)


# ##### Maximum people with rating = 3 
# ##### Minimum people with rating = 2.0

# In[ ]:


sns.countplot(combined.recruitment_channel)


# In[ ]:


combined.region.value_counts().plot(kind="bar",figsize=(17,6))


# Maximum Region id is region 2

# #####  Bi-Variate Analysis

# In[ ]:


combined.head()


# In[ ]:


#Going for Boxplots


# In[ ]:


sns.boxplot(combined["awards_won?"],combined.age)


# In[ ]:


sns.boxplot(combined["awards_won?"],combined.length_of_service)


# In[ ]:


sns.boxplot(combined["awards_won?"],combined.no_of_trainings)


# In[ ]:


sns.boxplot(combined["awards_won?"],combined.previous_year_rating)


# maximum poeple who won awards have 3 to 5 previous year rating

# In[ ]:


sns.boxplot(combined["awards_won?"],combined.recruitment_channel.value_counts())


# In[ ]:


sns.boxplot(combined["is_promoted"],combined.age)


# In[ ]:


sns.boxplot(combined["is_promoted"],combined.length_of_service)


# In[ ]:


sns.boxplot(combined["is_promoted"],combined.previous_year_rating)


# maximum poeple who got promoted had previous year rating of 3.5 to 5 
# 
# one poeple is having a rating of 1.0 who os not promoted which is true

# In[ ]:


sns.boxplot(combined["is_promoted"],combined.no_of_trainings)


# In[ ]:


#Numerical vs Numerical


# In[ ]:


plt.scatter(combined.age,combined.avg_training_score)


# In[ ]:


plt.scatter(combined.age,combined.length_of_service)
# we can clearly see with age length of service is increasing


# In[ ]:


#categorical vs categorical analysis
combined.head()


# In[ ]:


combined.groupby(["education","department"]).describe().plot(kind="bar",figsize=(20,10))


# In[ ]:


combined.groupby(["education","department"])["age"].describe().plot(kind="bar",figsize=(20,10))


# Inferences : 
#     
#     Bachelors sales and marketing with age count is very high
#     Bachleors with opetrations
#     then, bachelors in tech and procurement, analytics
#     
#     masters, S adn Marketing
#     then operations
#     tech,procurement, analytics
#     
#     Bachelors sales and marketing are most
#     

# In[ ]:


combined.groupby(["education","department","gender"]).describe().plot(kind="bar",figsize=(20,10))


# In[ ]:


combined.groupby(["education","department","gender"])["age"].describe().plot(kind="bar",figsize=(20,10))


# Sales and Marketing Male are very high
# 
# Bachelors Operations Female are very high after that procurement,technology

# In[ ]:


combined.groupby(["department","is_promoted"])["age"].describe().plot(kind="bar",figsize=(15,7))


# In[ ]:


combined.groupby(["department","education","is_promoted"])["age"].describe().plot(kind="bar",figsize=(15,7))


# People who are getiting maximum promotions
#     
#     Sales adn markerting bachelors
#     technology bachelors
#     technology masters abd above
#     Analytics, bachelors
#     operations, bachelors
#     operation s masters and above
# 

# In[ ]:


combined.groupby(["department","awards_won?","is_promoted"])["age"].describe().plot(kind="bar",figsize=(15,7))


# In[ ]:


combined.head()


# In[ ]:


pd.DataFrame(combined.groupby(["no_of_trainings","KPIs_met >80%","is_promoted"])["age"].describe())


# In[ ]:


combined.groupby(["KPIs_met >80%","is_promoted"])["age"].describe()


# In[ ]:


sns.boxplot(combined["KPIs_met >80%"],combined["is_promoted"])


# In[ ]:


combined.groupby(["KPIs_met >80%","is_promoted"])["age"].describe().plot(kind="bar")


# In[ ]:


combined.groupby(["length_of_service","is_promoted"])["age"].describe()


# In[ ]:


#Perfroming Feature Engineering


# In[ ]:


combined.head()


# In[ ]:


train.shape


# In[ ]:


combined.iloc[54807]


# In[ ]:


combined.isnull().sum()


# In[ ]:


combined.previous_year_rating.mean()


# In[ ]:


combined.previous_year_rating.mode()


# In[ ]:


combined.previous_year_rating.median()


# In[ ]:


combined.previous_year_rating.skew()


# In[ ]:


combined.previous_year_rating.kurt()


# In[ ]:


combined.loc[combined.previous_year_rating.isnull(),"previous_year_rating"]=3.0


# In[ ]:


combined.isnull().sum()


# In[ ]:


combined.education.mode()


# In[ ]:


combined.education.dropna(inplace=True)


# In[ ]:


train=combined[:54808]


# In[ ]:


test=combined[54808:]


# In[ ]:


train.head()


# In[ ]:


train.corr()


# In[ ]:


plt.figure(figsize=(10,6))
sns.heatmap(train.corr(),annot=True)


# In[ ]:


train.drop("region",axis=1,inplace=True)
test.drop("region",axis=1,inplace=True)


# In[ ]:


train.drop("employee_id",axis=1,inplace=True)
test.drop("employee_id",axis=1,inplace=True)


# In[ ]:


d={"m":1,"f":0}
train.gender=train.gender.map(d)


# In[ ]:


test.gender=test.gender.map(d)


# In[ ]:


train.head()


# In[ ]:


dummytrain=pd.get_dummies(train).drop("recruitment_channel_other",axis=1)
dummytest=pd.get_dummies(test).drop("recruitment_channel_other",axis=1)


# X and Y split

# In[ ]:


X=dummytrain.drop(["is_promoted"],axis=1)


# In[ ]:


y=dummytrain.is_promoted


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=123,test_size=0.2)


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
scaled_train=pd.DataFrame(sc.fit_transform(X_train,y_train),columns=X_train.columns)
scaled_test=pd.DataFrame(sc.transform(X_test),columns=X_test.columns)


# In[ ]:


scaled_train.shape


# In[ ]:


scaled_test.shape


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
model_rf=rf.fit(scaled_train,y_train).predict(scaled_test)


# In[ ]:


from sklearn.metrics import r2_score ,mean_squared_error,mean_absolute_error
print("The R sqaure of the model is ",r2_score(y_test,model_rf))
print("The RMSE IS", np.sqrt(mean_squared_error(y_test,model_rf)))


# In[ ]:


features = pd.DataFrame(rf.feature_importances_, index = scaled_test.columns,
            columns = ["Features"])


# In[ ]:


features.sort_values(by = "Features").plot(kind = "barh", color = "red")


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
gb=GradientBoostingClassifier()
model_gb=gb.fit(scaled_train,y_train).predict(scaled_test)
print("The R sqaure of the model is ",r2_score(y_test,model_gb))
print("The RMSE IS", np.sqrt(mean_squared_error(y_test,model_gb)))


# In[ ]:


from xgboost import XGBRFRegressor
xg=XGBRFRegressor()
model_xg=xg.fit(scaled_train,y_train).predict(scaled_test)
print("The RMSE IS ",np.sqrt(mean_squared_error(y_test,model_xg)))
print("tHE R SQAURE IS ",r2_score(y_test,model_xg))


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
ad=AdaBoostClassifier(random_state=123)
model_ad=ad.fit(scaled_train,y_train).predict(scaled_test)
print("The R sqaure of the model is ",r2_score(y_test,model_ad))
print("The RMSE IS", np.sqrt(mean_squared_error(y_test,model_ad)))


# In[ ]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.min_rows', 1000)


# In[ ]:


combined.head()


# In[ ]:


#https://stackoverflow.com/questions/11707586/how-do-i-expand-the-output-display-to-see-more-columns-of-a-pandas-dataframe


# In[ ]:


combined.age.head()


# In[ ]:


combined.head()


# In[ ]:


combined.drop(["employee_id","region"],axis=1,inplace=True)


# In[ ]:


sns.distplot(np.sqrt(combined.age))


# In[ ]:


sns.distplot(np.log1p(combined.age))


# In[ ]:


sns.distplot(np.log1p(combined.length_of_service))


# In[ ]:


combined.length_of_service=np.log1p(combined.length_of_service)


# In[ ]:


combined.age=np.log1p(combined.age)


# In[ ]:


combined.head()


# In[ ]:


d={"f":0,"m":1}
combined.gender=combined.gender.map(d)


# In[ ]:


combined.head()


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


newtrain=combined[:54808]
newtest=combined[54808:]


# In[ ]:


newtest=combined[54808:]


# In[ ]:


newtest.drop("is_promoted",axis=1,inplace=True)


# In[ ]:


dummytrain=pd.get_dummies(newtrain)
dummytest=pd.get_dummies(newtest)


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
cols = dummytrain.columns[dummytrain.columns!="is_promoted"]
scaled_train = pd.DataFrame(sc.fit_transform(dummytrain.drop("is_promoted", axis = 1)), 
             columns=cols)
scaled_test = pd.DataFrame(sc.transform(dummytest),
                          columns = dummytest.columns)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
model = rf.fit(scaled_train, dummytrain.is_promoted).predict(scaled_test)


# In[ ]:


features = pd.DataFrame(rf.feature_importances_, index = scaled_test.columns,
            columns = ["Features"])
features.sort_values(by = "Features").plot(kind = "barh", color = "red")


# In[ ]:


from sklearn.metrics import r2_score ,mean_squared_error,mean_absolute_error
print("The R sqaure of the model is ",r2_score(dummytrain.is_promoted[:23490],model))
print("The RMSE IS", np.sqrt(mean_squared_error(dummytrain.is_promoted[:23490],model)))


# In[ ]:


y_test.shape


# In[ ]:


model.shape


# In[ ]:


solution = pd.DataFrame({"employee_id":pd.read_csv("../input/test.csv").employee_id, 
                        "is_promoted":model})
solution.to_csv("RF MODEL.csv", index =False)


# In[ ]:


test.head()


# In[ ]:


x=pd.read_csv("RF MODEL.csv")


# In[ ]:


x.is_promoted=x.is_promoted.astype("int64")


# In[ ]:


x.head()


# In[ ]:


solution = pd.DataFrame({"employee_id":x.employee_id, 
                        "is_promoted":x.is_promoted})
solution.to_csv("RF MODEL2.csv", index =False)


# 0.4385

# 
# Applying GRADIENT BOOSTING

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
gb=GradientBoostingClassifier()
model_gb = gb.fit(scaled_train, dummytrain.is_promoted).predict(scaled_test)


# In[ ]:


solution = pd.DataFrame({"employee_id":pd.read_csv("../input/test.csv").employee_id, 
                        "is_promoted":model_gb})


# In[ ]:


solution.is_promoted=solution.is_promoted.astype("int64")


# In[ ]:


solution.to_csv("GB MODEL.csv", index =False)


# 0.458

# ##### Applying Xtreme GRdaient boosting

# In[ ]:


from xgboost import XGBRFClassifier
xg=XGBRFClassifier(n_estimators=3,max_depth=500)
model_xg = xg.fit(scaled_train, dummytrain.is_promoted).predict(scaled_test)


# In[ ]:


solution = pd.DataFrame({"employee_id":pd.read_csv("../input/test.csv").employee_id, 
                        "is_promoted":model_xg})


# In[ ]:


solution.is_promoted=solution.is_promoted.astype("int64")


# In[ ]:


solution.to_csv("xg MODEL.csv", index =False)


# 0.446

# #### Adaboostikng 

# In[ ]:


from sklearn.ensemble import AdaBoostRegressor
ad=AdaBoostRegressor(random_state=123)
model_ada = ad.fit(scaled_train, dummytrain.is_promoted).predict(scaled_test)


# In[ ]:


solution = pd.DataFrame({"employee_id":pd.read_csv("../input/test.csv").employee_id, 
                        "is_promoted":model_xg})


# In[ ]:


solution.is_promoted=solution.is_promoted.astype("int64")


# In[ ]:


solution.to_csv("ADA MODEL.csv", index =False)


# : 0.44629629629629636.

# In[ ]:


features = pd.DataFrame(ad.feature_importances_, index = scaled_test.columns,
            columns = ["Features"])
features.sort_values(by = "Features").plot(kind = "barh", color = "blue")


# In[ ]:


features.sort_values(by = "Features")


# In[ ]:


combined.head()


# #### Feature Engineering
# 
#     sum_performance = addition of the important factors for the promotion (awards_won;KpIs_met & previous_year_rating).
#     
#     Total nmber of training hours = avg_training_score * no_of_training
#     
#     recruitment_channel have no impact on the promotion so removed that.

# In[ ]:


combined.head()


# ### Feature Engineering 

# Recruitment Channel : 

# In[ ]:


combined.groupby(["recruitment_channel","is_promoted"]).describe().plot(kind="bar",figsize=(10,5))


# This column does have much impact on other columns

# In[ ]:


combined.drop(["recruitment_channel"],axis=1,inplace=True)


# In[ ]:


combined.head()


# Combining to make a new feature : Average Trarining Score & No of Trainings = Total hours

# In[ ]:


combined["total_hours"]=combined.avg_training_score*combined.no_of_trainings


# In[ ]:


combined.head()


# Combining Awards Won, KPI,Previous year rating

# In[ ]:


combined["total_sum"]=combined["KPIs_met >80%"]+combined["awards_won?"]+combined["no_of_trainings"]


# In[ ]:


combined.head()


# In[ ]:


plt.figure(figsize=(10,5))
sns.heatmap(combined.corr(),annot=True,cmap="ocean")


# In[ ]:


combined.drop(["total_score"],axis=1,inplace=True)


# In[ ]:


combined.head()


# In[ ]:


combined.education.unique()


# In[ ]:


combined.isnull().sum()


# In[ ]:


combined.education.mode()


# In[ ]:


combined[combined.education.isnull()]["education"]=combined.education.mode()


# In[ ]:


combined.loc[combined.education.isnull(),"education"]="Bachelor's"


# In[ ]:


combined[combined.education.isnull()]["education"]


# In[ ]:


combined.isnull().sum()


# In[ ]:


combined.head()


# In[ ]:


newtrain=combined[:54808]
newtest=combined[54808:]
newtest.drop("is_promoted",axis=1,inplace=True)
dummytrain=pd.get_dummies(newtrain)
dummytest=pd.get_dummies(newtest)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
cols = dummytrain.columns[dummytrain.columns!="is_promoted"]
scaled_train = pd.DataFrame(sc.fit_transform(dummytrain.drop("is_promoted", axis = 1)), 
             columns=cols)
scaled_test = pd.DataFrame(sc.transform(dummytest),
                          columns = dummytest.columns)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
model = rf.fit(scaled_train, dummytrain.is_promoted).predict(scaled_test)
features = pd.DataFrame(rf.feature_importances_, index = scaled_test.columns,
            columns = ["Features"])
features.sort_values(by = "Features").plot(kind = "barh", color = "red")


# In[ ]:


from sklearn.metrics import r2_score ,mean_squared_error,mean_absolute_error
print("The R sqaure of the model is ",r2_score(dummytrain.is_promoted[:23490],model))
print("The RMSE IS", np.sqrt(mean_squared_error(dummytrain.is_promoted[:23490],model)))
solution = pd.DataFrame({"employee_id":pd.read_csv("../input/test.csv").employee_id, 
                        "is_promoted":model})
solution.is_promoted=solution.is_promoted.astype("int64")
solution.to_csv("RF MODEL4_FEATURE ENG.csv", index =False)


# #### Model Accuracy on Analytics Vidhya : 0.4516129032258064.
# 

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
gb=GradientBoostingClassifier()
model_gb = gb.fit(scaled_train, dummytrain.is_promoted).predict(scaled_test)
solution = pd.DataFrame({"employee_id":pd.read_csv("../input/test.csv").employee_id, 
                        "is_promoted":model_gb})
solution.is_promoted=solution.is_promoted.astype("int64")
solution.to_csv("GB MODEL_feature reng.csv", index =False)


# In[ ]:


features = pd.DataFrame(gb.feature_importances_, index = scaled_test.columns,
            columns = ["Features"])
features.sort_values(by = "Features").plot(kind = "barh", color = "green")


# In[ ]:


get_ipython().system('pip install catboost')


# In[ ]:


from catboost import CatBoostClassifier
cb=CatBoostClassifier()


# In[ ]:


model_cb = cb.fit(scaled_train, dummytrain.is_promoted).predict(scaled_test)
solution = pd.DataFrame({"employee_id":pd.read_csv("test.csv").employee_id, 
                        "is_promoted":model_gb})
solution.is_promoted=solution.is_promoted.astype("int64")
solution.to_csv("CAT BOOST MODEL_feature reng.csv", index =False)


# In[ ]:


features = pd.DataFrame(cb.feature_importances_, index = scaled_test.columns,
            columns = ["Features"])
features.sort_values(by = "Features").plot(kind = "barh", color = "magenta")


# Accuracy :
#         
#         0.4385964912280702.

# In[ ]:





# In[ ]:


import pandas as pd
test = pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")

