#!/usr/bin/env python
# coding: utf-8

# # importing_libraries

# In[ ]:


from pandas import Series,DataFrame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style("whitegrid")
import statistics
from scipy import stats


# In[ ]:


from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,mean_squared_error
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV


# In[ ]:


from statsmodels.formula.api import ols
from sklearn.preprocessing import StandardScaler


# # importing_data

# In[ ]:


df=pd.read_csv("ahs-mort-odisha-sundargarh.csv")


# In[ ]:


df1=df[["rural","stratum_code","deceased_sex",
    "age_of_death_above_one_year","treatment_source","place_of_death","month_of_death","year_of_death","member_identity",
    "social_group_code","marital_status","year_of_marriage","highest_qualification","occupation_status","disability_status",
    "chew","smoke","alcohol","house_structure","drinking_water_source","household_have_electricity","lighting_source",
    "cooking_fuel","no_of_dwelling_rooms"]]


# In[ ]:


df1["year_of_death"].unique()


# In[ ]:


#df1.to_csv("anual_health_survey.csv")
df1.head(2)


# # data pre_processing

# In[ ]:


def cat(cl):
    c=pd.get_dummies(df1[cl]).columns
    n=(len(df1[cl].values))
    ar=np.zeros(n)
    for i in np.arange(len(c)):
        for j in np.arange(n):
            if c[i]==df1[cl][j]:
                ar[j]=i
    return(ar)


# In[ ]:


def replacer_mean(dff):
    dff1=dff
    r0=np.mean(dff)
    r1=r0.index
    r2=r0.values
    for i in np.arange(len(r1)):
          ri=r1[i]
          rv=r2[i]
          dff1[ri].fillna(value=rv)
          dff1[ri]=(nan_remover(dff[ri].values,rv))
    return(dff1)
def nan_remover(v,vm):
    vr=[]
    for i in np.arange(len(v)):
        if str(v[i])=="nan":
            vr=np.append(vr,vm)
        else:
            vr=np.append(vr,v[i])
    return(vr)
def numriser(a):
    a1=[]
    for i in np.arange(len(a)):
        a1=np.append(a1,round(a[i]))
    return(a1)

def substi(ar,s):
    n=len(ar)
    sum1=0
    for i in np.arange(n):
        j=ar[i]
        df_c1["interval"][j]=s
        #sum1=sum1+ar[i]
    return(df_c1)
def inret(y,m):
    return(np.where((df_c1["year_of_death"]==y) & (df_c1["month_of_death"]==m))[0])


# In[ ]:


(cat("rural"))

r1=[]
for i in np.arange(len(df1["rural"].values)):
    if df1["rural"][i]=="Rural":
        r1=np.append(r1,0)
    else:
        r1=np.append(r1,1)
r2=[]
for i in np.arange(len(df1["rural"].values)):
    if df1["stratum_code"][i]=="200<population<2000":
        r2=np.append(r2,0)
    elif df1["stratum_code"][i]=="population>=2000":
        r2=np.append(r2,2)
    else:
        r2=np.append(r2,1)
r3=[]
for i in np.arange(len(df1["rural"].values)):
    if df1["deceased_sex"][i]=="Male":
        r3=np.append(r3,1)
    else:
        r3=np.append(r3,0)
r4=[]
for i in np.arange(len(df1["treatment_source"].values)):
    if (df1["treatment_source"][i]=="At Home") or (df1["treatment_source"][i]=="No Medical attention") or (df1["treatment_source"][i]=="Others"):
        r4=np.append(r4,0)
    elif (df1["treatment_source"][i][0:9]=="Governmen") or (df1["treatment_source"][i][0:9]=="NGO or Trust Hosp/Clinic"):
        r4=np.append(r4,1)
    else:
        r4=np.append(r4,2)
r5=[]
for i in np.arange(len(df1["treatment_source"].values)):
    if (df1["treatment_source"][i]=="At home"):
        r5=np.append(r5,0)
    elif (df1["treatment_source"][i][0:9]=="In health facility"):
        r5=np.append(r5,2)
    elif (df1["treatment_source"][i][0:9]=="In-transit"):
        r5=np.append(r5,1)
    else:
        r5=np.append(r5,3)
r6=[]
for i in np.arange(len(df1["social_group_code"].values)):
    if (df1["social_group_code"][i]=="SC"):
        r6=np.append(r6,0)
    elif (df1["treatment_source"][i]=="ST"):
        r6=np.append(r6,1)
    else:
        r6=np.append(r6,2)
r7=[]
for i in np.arange(len(df1["marital_status"].values)):
    if (df1["marital_status"][i]=="Divorced"):
        r7=np.append(r7,0)
    elif (df1["marital_status"][i]=="Married and Gauna performed") or (df1["marital_status"][i]=="Married but Gauna not performed" ):
        r7=np.append(r7,1)
    elif (df1["marital_status"][i]=="Never married"):
        r7=np.append(r7,2)
    elif (df1["marital_status"][i]=="Not stated"):
        r7=np.append(r7,3)
    elif (df1["marital_status"][i]=="Remarried"):
        r7=np.append(r7,4)
    elif (df1["marital_status"][i]=="Separated"):
        r7=np.append(r7,5)
    else:
        r7=np.append(r7,6)
r8=[]
for i in np.arange(len(df1["rural"].values)):
    if (df1["highest_qualification"][i]=="Illiterate") or (df1["highest_qualification"][i]=="Literate With formal education-Below primary") or (df1["highest_qualification"][i]=="Literate Without formal education"):
        r8=np.append(r8,0)
    else:
        r8=np.append(r8,1)
r9=[]
for i in np.arange(len(df1["rural"].values)):
    if (df1["occupation_status"][i]=="Agricultural Wage labourer") or (df1["occupation_status"][i]=="Cultivator") or (df1["occupation_status"][i]=="Attending routine domestic chores etc."):
        r9=np.append(r9,0)
    elif  (df1["occupation_status"][i]=="Too old to work") or (df1["occupation_status"][i]=="Not able to work due to disability"):
        r9=np.append(r9,1)
    else:
        r9=np.append(r9,2)
r10=[]
for i in np.arange(len(df1["rural"].values)):
    if (df1["disability_status"][i]=="Hearing"):
        r10=np.append(r10,1)
    elif (df1["disability_status"][i]=="Locomotor"):
        r10=np.append(r10,2)
    elif (df1["disability_status"][i]=="Mental"):
        r10=np.append(r10,3)
    elif (df1["disability_status"][i]=="No Disability"):
        r10=np.append(r10,0)
    elif (df1["disability_status"][i]=="Speech"):
        r10=np.append(r10,1)
    else:
        r10=np.append(r10,4)
# In[ ]:


df1.columns


# In[ ]:


df1["cooking_fuel"][0:1]


# In[ ]:


txt=["rural","stratum_code","deceased_sex","treatment_source","place_of_death","social_group_code","marital_status",
       "highest_qualification","occupation_status","disability_status", "chew","smoke","alcohol","house_structure","drinking_water_source",
    "household_have_electricity","lighting_source","cooking_fuel"]


# In[ ]:


for j in (txt):
    df1[j]=cat(j)


# In[ ]:


r0=(np.mean(df1))
r1=r0.index
r2=r0.values
df2=df1
df1=replacer_mean(df2)


# In[ ]:


y=df1["age_of_death_above_one_year"].values
df1=df1.drop(["age_of_death_above_one_year","year_of_death","month_of_death"],axis=1)


# In[ ]:


for i in np.arange(len(df1.columns)):
    sc=StandardScaler()
    cl=df1.columns[i]
    sc.fit(df1[cl].values.reshape(-1,1))
    df2[cl]=sc.transform(df2[cl].values.reshape(-1,1))


# # model_fitting

# In[ ]:


x=df1.values


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)


# In[ ]:


model_le=xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=.2, gamma=0.2,
       importance_type='gain', learning_rate=0.3, max_delta_step=0,
       max_depth=15, min_child_weight=3, missing=None, n_estimators=100,
       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=None, subsample=1, verbosity=1)
model_le.fit(x_train,y_train)

model_le=xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=1, gamma=0, learning_rate=0.2,
       max_delta_step=0, max_depth=15, min_child_weight=3, missing=None,
       n_estimators=100, n_jobs=1, nthread=None,
       objective='multi:softprob', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=None,
       subsample=1, verbosity=1)
model_le.fit(x_train,y_train)
# In[ ]:


y_pre=model_le.predict(x_train)


# In[ ]:


accuracy_score(numriser(y_pre),numriser(y_train))


# In[ ]:


y_pre1=model_le.predict(x_test)
accuracy_score(numriser(y_pre1),numriser(y_test))


# In[ ]:


df_pre=DataFrame([y_pre1,y_test],index=["predicted","observed"]).T
df_pre["predicted"].plot(kind="kde",label=True,figsize=(15,5),alpha=1)
df_pre["observed"].plot(kind="kde",label=True,figsize=(15,5),alpha=1)
plt.title("predicted label of happiness",fontsize=20)
plt.xlabel("labels of happiness ",fontsize=20)
plt.ylabel("frequency",fontsize=20)
plt.legend(["predicted","observed"])


# In[ ]:


df_pre["predicted"][1:100].plot(figsize=(20,5))
df_pre["observed"][1:100].plot(figsize=(20,5))
plt.title("predicted age and observerd age classification",fontsize=20)
plt.xlabel("people ",fontsize=20)
plt.ylabel("age",fontsize=20)
plt.legend(["predicted","observed"])


# In[ ]:


(mean_squared_error(y_pre1,y_test))


# # clinical_life_table

# In[ ]:


df_c=df2[["month_of_death","year_of_death","age_of_death_above_one_year"]]


# In[ ]:


df_c["interval"]=np.zeros(len(df_c["month_of_death"]))


# In[ ]:


df_c.head(2)


# In[ ]:


mon=np.unique(df_c["month_of_death"])
year=np.unique(df_c["year_of_death"])


# In[ ]:


df_c1=df_c


# In[ ]:


s1=1
for i in year:
    for j in mon[1:]:
        indx=inret(i,j)
        df_c=substi(indx,s1)
        s1=s1+1
        #print(s1)


# In[ ]:


sum1=[]
for i in np.arange(len(df_c["interval"])):
    if df_c["interval"][i]==0:
        sum1=np.append(sum1,i)


# In[ ]:


df_c=df_c.drop(sum1)


# In[ ]:


d=[]
for i in np.arange(1,len(np.unique(df_c["interval"]))+1):
                d=np.append(d,sum(pd.get_dummies(df_c["interval"])[i]))


# In[ ]:


s=len(df_c["interval"].values)
s


# In[ ]:


s=len(df_c["interval"].values)
cs=[s]
for i in np.arange(1,49):
    cs1=s-d[i-1]
    cs=np.append(cs,cs1)
    s=cs1


# In[ ]:


clt=DataFrame([np.arange(1,49),cs,d],index=["age interval","cum_sum","deaths"]).T


# In[ ]:


qt=clt["deaths"]/clt["cum_sum"]
pt=(np.ones(len(qt))-qt)


# In[ ]:


clt["qt"]=qt
clt["pt"]=pt


# In[ ]:


clt["Pt"]=np.ones(len(qt))


# In[ ]:


for i in np.arange(0,47):
    clt["Pt"][i+1]=(clt["pt"][i]*clt["Pt"][i])


# In[ ]:


clt=clt.drop(48)


# In[ ]:


clt["Pt"].plot(figsize=(12,5))
plt.title("survival_rate",fontsize=20)
plt.xlabel("months ",fontsize=20)
plt.ylabel("prob_of_remission",fontsize=20)
plt.legend(["survival_odisha"])


# In[ ]:


df_p=pd.read_csv("ahs-mort-bihar-patna.csv")


# In[ ]:


df_cb=df_p[["month_of_death","year_of_death","age_of_death_above_one_year"]]


# In[ ]:


r0=(np.mean(df1))
r1=r0.index
r2=r0.values
df2=df_cb
df_cb=replacer_mean(df2)


# In[ ]:


mon1=np.unique(df_cb["month_of_death"])
year1=np.unique(df_cb["year_of_death"])


# In[ ]:


def substi_b(ar,s):
    n=len(ar)
    sum1=0
    for i in np.arange(n):
        j=ar[i]
        df_cb1["interval"][j]=s
        #sum1=sum1+ar[i]
    return(df_c1)
def inret_b(y,m):
    return(np.where((df_cb1["year_of_death"]==y) & (df_cb1["month_of_death"]==m))[0])


# In[ ]:


df_cb["interval"]=np.zeros(len(df_cb["month_of_death"]))

mon1=np.unique(df_cb["month_of_death"])
year1=np.unique(df_cb["year_of_death"])

df_cb1=df_cb

s1=1
for i in year:
    for j in mon[1:]:
        indx=inret_b(i,j)
        df_cb=substi_b(indx,s1)
        s1=s1+1


# In[ ]:


sum1=[]
for i in np.arange(len(df_c1["interval"])):
    if df_cb1["interval"][i]==0:
        sum1=np.append(sum1,i)


# In[ ]:


df_cb1=df_cb1.drop(sum1)


# In[ ]:


d=[]
for i in np.arange(1,len(np.unique(df_cb1["interval"]))):
                d=np.append(d,sum(pd.get_dummies(df_cb1["interval"])[i]))
s=len(df_cb1["interval"].values)
cs=[s]
for i in np.arange(1,49):
    cs1=s-d[i-1]
    cs=np.append(cs,cs1)
    s=cs1

clt_b=DataFrame([np.arange(1,49),cs,d],index=["age interval","cum_sum","deaths"]).T

qt_b=clt_b["deaths"]/clt_b["cum_sum"]
pt_b=(np.ones(len(qt_b))-qt_b)

clt_b["qt"]=qt_b
clt_b["pt"]=pt_b

clt_b["Pt"]=np.ones(len(qt_b))

for i in np.arange(0,47):
    clt_b["Pt"][i+1]=(clt_b["pt"][i]*clt_b["Pt"][i])

clt_b=clt_b.drop(48)

clt_b["Pt"].plot(figsize=(12,5))
clt["Pt"].plot(figsize=(12,5))
plt.title("survival_rate",fontsize=20)
plt.xlabel("months ",fontsize=20)
plt.ylabel("prob._of_remission",fontsize=20)
plt.legend(["survival_Bihar(patna)","survival_odisha(sundargadh)"])


# In[ ]:


clt_ob=clt_b
clt_ob["o"]=clt["Pt"].values
clt_ob.head(2)


# # EDA

# In[ ]:


#univariate_analysis


# In[ ]:


df1.columns


# In[ ]:


df1["no_of_dwelling_rooms"].plot(kind="hist",figsize=(9,5),color="red")


# In[ ]:


g=[sum(pd.get_dummies(df["smoke"])["Ex - Smoker"]),sum(pd.get_dummies(df["smoke"])["Never smoked"]),sum(pd.get_dummies(df["smoke"])["Not known"]),
                                          sum(pd.get_dummies(df["smoke"])["Occasional smoker"]),
                                          sum(pd.get_dummies(df["smoke"])["Usual smoker"])]
labels=["EX_smokers","Never smoked","not known","Occasional smoker","Occasional smoker"]
pie=plt.pie(g,radius=1.5,shadow=True,autopct='%1.1f%%')
plt.legend(pie[0], labels, loc="best")


# In[ ]:


df1["chew"].plot(kind="hist",figsize=(9,5),color="brown")


# In[ ]:


df_anova=(pd.get_dummies(df["chew"]))


# In[ ]:


#anova


# In[ ]:


import statsmodels.api as sm
from statsmodels.formula.api import ol


# In[ ]:


from statsmodels.formula.api import ols


# In[ ]:


df2.columns


# In[ ]:


lm=ols(" age_of_death_above_one_year ~ chew",data=df).fit()


# In[ ]:


table=sm.stats.anova_lm(lm)


# In[ ]:


print(table)


# In[ ]:


lm=ols(" age_of_death_above_one_year ~ smoke",data=df).fit()
table=sm.stats.anova_lm(lm)
print(table)


# In[ ]:


lm=ols(" age_of_death_above_one_year ~ cooking_fuel",data=df).fit()
table=sm.stats.anova_lm(lm)
print(table)


# In[ ]:


lm=ols(" age_of_death_above_one_year ~ alcohol",data=df).fit()
table=sm.stats.anova_lm(lm)
print(table)


# In[ ]:


lm=ols(" age_of_death_above_one_year ~ social_group_code",data=df).fit()
table=sm.stats.anova_lm(lm)
print(table)


# In[ ]:


#df1["smoke"]
df2=df1


# In[ ]:


ty=np.unique(df2["smoke"])


# In[ ]:


ty=np.unique(df2["smoke"])
ex_smoker=[]
never_smoke=[]
not_known=[]
occ_smoker=[]
usual_smoker=[]
for j in np.arange(len(df1["smoke"])):
    if int(ty[0])==int(df2["smoke"][j]):
        ex_smoker=np.append(ex_smoker,y[j])
    if int(ty[1])==int(df2["smoke"][j]):
        never_smoke=np.append(never_smoke,y[j])
    if int(ty[2])==int(df2["smoke"][j]):
        not_known=np.append(not_known,y[j])
    if int(ty[3])==int(df2["smoke"][j]):
        occ_smoker=np.append(occ_smoker,y[j])
    if int(ty[4])==int(df2["smoke"][j]):
        usual_smoker=np.append(usual_smoker,y[j])


# In[ ]:


stats.ttest_ind(usual_smoker,never_smoke,equal_var=False)


# In[ ]:


[np.mean(usual_smoker),np.mean(never_smoke)]


# so smoking effects the life expectency of human significantly
# #on an average it reduces the life time to two years

# In[ ]:


stats.ttest_ind(ex_smoker,occ_smoker,equal_var=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




