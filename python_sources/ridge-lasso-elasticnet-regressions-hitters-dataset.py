#!/usr/bin/env python
# coding: utf-8

# # 1) IMPORT THE DATA

# In[ ]:


from warnings import filterwarnings
filterwarnings('ignore')
import pandas as pd
import numpy as np
Hitters=pd.read_csv("../input/hitters/Hitters.csv")
df=Hitters.copy()
df.head()


# # Data Description
# Major League Baseball Data from the 1986 and 1987 seasons.
# 

# A data frame with 322 observations of major league players on the following 20 variables.
# 
# - AtBat: Number of times at bat in 1986
# - Hits: Number of hits in 1986
# - HmRun: Number of home runs in 1986
# - Runs: Number of runs in 1986
# - RBI: Number of runs batted in in 1986
# - Walks: Number of walks in 1986
# - Years: Number of years in the major leagues
# - CAtBat: Number of times at bat during his career
# - CHits: Number of hits during his career
# - CHmRun: Number of home runs during his career
# - CRuns: Number of runs during his career
# - CRBI: Number of runs batted in during his career
# - CWalks:Number of walks during his career
# - League: A factor with levels A and N indicating player's league at the end of 1986
# - Division: A factor with levels E and W indicating player's division at the end of 1986
# - PutOuts: Number of put outs in 1986
# - Assists: Number of assists in 1986
# - Errors: Number of errors in 1986
# - Salary: 1987 annual salary on opening day in thousands of dollars
# - NewLeague: A factor with levels A and N indicating player's league at the beginning of 1987
# 
# 

# # I wanted to learn dtypes.
# 
# -  Then i saw that there are missing values in Salary variable.

# In[ ]:


df.info()


# # 2) DESCRIBE AND TRY TO FILL NA VALUES

# In[ ]:


df.isnull().sum()


# In[ ]:


df.describe().T


# In[ ]:


import seaborn as sns
sns.boxplot(x=df["Salary"]);


# - In real life, we know that there are huge differences btw baseball players' salaries.
# - Therefore, i won't evaulate the salaries which are more than 1500 as outliers.  

# In[ ]:


NAdf= df[df.isnull().any(axis=1)]
NAdf.describe().T


# In[ ]:


notNAdf=df[df.notnull().all(axis=1)]
notNAdf.describe().T


# # There isn't strong correlation btw any x variables and Y.

# In[ ]:


notNAdf.corr()


# In[ ]:


print("New League= A" ,notNAdf[notNAdf["NewLeague"]=="A"].agg({"Salary":"mean"}))
print("New League= N" ,notNAdf[notNAdf["NewLeague"]=="N"].agg({"Salary":"mean"}))
print("League= A" ,notNAdf[notNAdf["League"]=="A"].agg({"Salary":"mean"}))
print("League= N" ,notNAdf[notNAdf["League"]=="N"].agg({"Salary":"mean"}))
print("Division= E" ,notNAdf[notNAdf["Division"]=="E"].agg({"Salary":"mean"}))
print("Division= W" ,notNAdf[notNAdf["Division"]=="W"].agg({"Salary":"mean"}))


# # I will fill NA values according to these division levels salary means.
# - It is clear that League and New League levels aren't decisive in "Salary"
# - It is obvious that if the player's division level  is E, the salary is higher and if the player's division level  is W, the salary is less. 
# 

# In[ ]:


df.head(2)


# In[ ]:


df.loc[(df["Salary"].isnull())& (df['Division'] == 'E'),"Salary"]=624.27
df.head(2)


# In[ ]:


df.loc[(df["Salary"].isnull())& (df['Division'] == 'W'),"Salary"]=450.87
df.isnull().sum().sum()


# ## I filled NA Values

# In[ ]:


df[df["Salary"]<0]    


# # 3) ONE HOT ENCODING

# In[ ]:


dff = pd.get_dummies(df, columns = ['League', 'Division', 'NewLeague'], drop_first = True)
dff.head()


# #  4) DETECT THE OUTLIERS

# In[ ]:


import numpy as np
from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor(n_neighbors = 20, contamination = 0.05)
clf.fit_predict(dff)
dff_scores = clf.negative_outlier_factor_
np.sort(dff_scores)[0:20]


# In[ ]:


sns.boxplot(x=dff_scores);


# In[ ]:


threshold = np.sort(dff_scores)[13]
dff.loc[dff_scores < threshold]


# In[ ]:


dff.loc[(dff_scores < threshold)&(dff["Salary"]>1500)]


# In[ ]:


df["Salary"].describe([0.75,0.90,0.95,0.99]).T


# In[ ]:


dff.loc[(dff_scores < threshold)&(dff["Salary"]>1500),"Salary"]=1967
dff.loc[dff_scores < threshold]


# In[ ]:


dff.loc[dff_scores < threshold]


# In[ ]:


df1=dff.loc[df1_scores > esik_deger]


# In[ ]:


df1.shape


# In[ ]:


import seaborn as sns
sns.pairplot(df1)


# In[ ]:


plt.figure(figsize=(12,10))
cor = df1.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[ ]:


CHits_cor=abs(cor["CHits"])
CHits_relevant_features = CHits_cor[CHits_cor>0.9]
CHits_relevant_features


# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(df["CHits"],df["CAtBat"], alpha=0.3,
            cmap='viridis');


# In[ ]:


df1=df1.drop("CHits", axis=1)
df1=df1.drop("CAtBat", axis=1)


# In[ ]:


plt.figure(figsize=(12,10))
cor = df1.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[ ]:


df1=df1.drop("CRBI", axis=1)
df1=df1.drop("CWalks", axis=1)


# In[ ]:


plt.figure(figsize=(12,10))
cor = df1.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[ ]:


df1.describe().T


# In[ ]:


df1.shape


# Standard Scaler(df3)

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
SS=StandardScaler()
col=df1.columns
df2=SS.fit_transform(df1)
df3=pd.DataFrame(df2, columns=col)
df3.head(2)



# Normalize(df7)

# In[ ]:


df6=preprocessing.normalize(df1, axis=0)
col=df1.columns
df7=pd.DataFrame(df6, columns=col)
df7.head(2)


# # PREPROCESSING IS FINISHED HERE.

# # CLASSIC REGRESSION

# In[ ]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression


# In[ ]:


y=df1["Salary"]
X=df1.drop("Salary", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=46)
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
y_pred=reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


# In[ ]:


print("intercept(b0):",reg_model.intercept_)
print("coefficients(b1,b2..):","\n",reg_model.coef_)


# In[ ]:


reg_model.predict(X)[0:10]


# In[ ]:


y.head(10)


# In[ ]:


df1[y<0]


# In[ ]:


# df3 (Standar Scaler)

y=df3["Salary"]
X=df3.drop("Salary", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=46)
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
y_pred=reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


# In[ ]:


#df7 (Normalize)

y=df7["Salary"]
X=df7.drop("Salary", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=46)
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
y_pred=reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


# # RIDGE REGRESSION

# In[ ]:


y=df1["Salary"]
X=df1.drop("Salary", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=46)


# In[ ]:


ridge_model=Ridge().fit(X_train,y_train)
y_pred= ridge_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


# In[ ]:


alphas1 = np.random.randint(0,1000,100)
alphas2 = 10**np.linspace(10,-2,100)*0.5
alphas3 = np.linspace(0,1,1000)
ridge_cv = RidgeCV(alphas = alphas1, scoring = "neg_mean_squared_error", cv = 10, normalize = True)
ridge_cv.fit(X_train,y_train)
ridge_cv.alpha_


# In[ ]:


ridge_tuned=Ridge(alpha=ridge_cv.alpha_).fit(X_train,y_train)
y_pred=ridge_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


# # LASSO REGRESSION

# In[ ]:


y=df1["Salary"]
X=df1.drop("Salary", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                   random_state=46)
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import RidgeCV, LassoCV


# In[ ]:


lasso_model = Lasso().fit(X_train, y_train)
y_pred=lasso_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


# In[ ]:


alphas1 = np.random.randint(0,1000,100)
alphas2 = 10**np.linspace(10,-2,100)*0.5
alphas3 = np.linspace(0,1,1000)
lasso_cv_model = LassoCV(alphas = alphas1, cv = 10).fit(X_train, y_train)


# In[ ]:


print(lasso_cv_model.alpha_)


# In[ ]:


lasso_tuned = Lasso(alpha = lasso_cv_model.alpha_).fit(X_train, y_train)
y_pred = lasso_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_pred,y_test))


# # ELASTIC NET REGRESSION

# In[ ]:


from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.linear_model import RidgeCV, LassoCV,ElasticNetCV


y=df1["Salary"]
X=df1.drop("Salary", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                   random_state=46)


# In[ ]:


enet_model = ElasticNet().fit(X_train, y_train)
y_pred = enet_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


# In[ ]:


alphas1 = np.random.randint(0,1000,100)
alphas2 = 10**np.linspace(10,-2,100)*0.5
alphas3 = np.linspace(0,1,1000)
enet_cv_model = ElasticNetCV(alphas = alphas1, cv = 10).fit(X_train, y_train)


# In[ ]:


enet_cv_model.alpha_


# In[ ]:


enet_tuned = ElasticNet(alpha = enet_cv_model.alpha_).fit(X_train, y_train)
y_pred = enet_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


# In[ ]:


enet_params = {"l1_ratio": [0,0.01,0.05,0.1,0.2,0.4,0.5,0.6,0.8,1],
              "alpha":[0.1,0.01,0.001,0.2,0.3,0.5,0.8,0.9,1,2,5,7,10,13,20,45,99,100]}
enet_model = ElasticNet().fit(X, y)
from sklearn.model_selection import GridSearchCV
gs_cv_enet = GridSearchCV(enet_model, enet_params, cv = 10).fit(X, y)


# In[ ]:


gs_cv_enet.best_params_


# In[ ]:


enet_tuned = ElasticNet(**gs_cv_enet.best_params_).fit(X_train, y_train)
y_pred = enet_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


# In[ ]:


# Note for me

dff=dff.drop("CRBI", axis=1)
dff=dff.drop("CWalks", axis=1)
dff=dff.drop("CHits", axis=1)
dff=dff.drop("CAtBat", axis=1)
dff=dff.loc[df1_scores > esik_deger]
y=dff["Salary"]
X=dff.drop("Salary", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=46)


# In[ ]:




