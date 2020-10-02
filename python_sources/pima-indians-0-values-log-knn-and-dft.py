#!/usr/bin/env python
# coding: utf-8

# # 1.Introduction 

# ### In This Kernel I tried to fill those 0 values by using the correalation between variables, I also ask for some comment (doctor) to make sure about the relation between varaiable, these kernel is not that visual friendly because I had to find several intervals/ each columns , instead of using the mean / columns , I checked several kernel with 90% accurancy but i got 61 max, anyway it was a great challenge because it improved my pandas and numpy skills. Feel free to provide some feedback or a better way to find or replace the values 

# In[ ]:


#Basic library
import numpy as np 
import pandas as pd 
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#visualization 
import seaborn as sns
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set_style("whitegrid")

df= pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df.tail(5)


# # 1.1. Conclusion 

# In[ ]:


from IPython.display import display
from PIL import Image
path="../input/conclusion/conclusion.png"
display(Image.open(path))


#     **From *LEFT* to *RIGHT***  BEST = "Logistical Regression" - Random Forest - KNN cluster

# In[ ]:


df.describe()[1:6]


# In[ ]:


df.info()


# ## 2. Scrub data** (filtering, extracting , replacing , handle missing values)

# ### When I was analyzing this data frame I realized that there were many 0 values,then i was checking others kenerls and found out that many people was asking if theses 0 were set in purpose including myself,because I couldnt find any guidance , I decided to fill them out. 
# 1. I decided to fill out those value that has some kind of relation between variables
# 2. I wrote a Funcion to easily chech how many Zeros are lefft to keep tracking 

# In[ ]:


def finding_zeros (frame):
    columns = frame.columns[:8]
    for i in columns:
        zeros = len(frame.loc[frame[i]==0])
        print(f'The numbers of 0 values in {i} = {zeros}')
        
finding_zeros(df)


# In[ ]:


#Finding Missing Values
NAN_value = (df.isnull().sum() / len(df)) * 100
Missing = NAN_value[NAN_value==0].index.sort_values(ascending=False)
Missing_data = pd.DataFrame({'Missing Ratio' :NAN_value})
Missing_data.head()


# # "--------  Section 2 ----------"
# 

# # 3. FINDING MISSING VALUES

# ## In this section I want to drop all those rows which as 0 Values in all the columns
# 1. Because later I want to find some relation between variable but 0 values might affect the final result
# 2. I realized I can do this with dropna.(tresh=#), but I will have to create a new Dataframe and applya pd.join
# 3. I check other forums and many user mentioned the highest accurancy was 60 % so I thought that the missing values might influence the final result 

# In[ ]:


def finding_zeros (frame):
    columns = frame.columns[:8]
    for i in columns:
        zeros = len(frame.loc[frame[i]==0])
        print(f'The numbers of 0 values in {i} = {zeros}')
        
# Let's Find all those Cells that have 0 Values
cond1= df[(df['Insulin']==0) & (df['SkinThickness']==0) & (df['Pregnancies']==0) & (df['BloodPressure']==0) & (df['BMI']==0)].index
cond2 = df[(df['Insulin']==0) & (df['SkinThickness']==0) & (df['Pregnancies']==0) & (df['BloodPressure']==0)].index
Zeros_values = cond2.append(cond1) 
df.drop(Zeros_values,inplace=True)

finding_zeros(df)


# ## 3.1. Finding the Relation Between Variable to fill out values 0
# 1. By using a heatmap, I intended to find a relation between variable (Glucose-Skinthickness-BMI-Insulin), later I will use the relation with stronger coorelation to find the missing values
# 2. Instead of using the group mean or Interpolate, I will sort the value by ranks I.e Find all People with Glucose btwn A-B, then extract the mean and use this value to fill the missing one 

# In[ ]:


f,ax = plt.subplots(figsize=(15,10))
mask = np.triu(df.corr())
sns.heatmap(df.corr(), annot=True, mask=mask,ax=ax,cmap='viridis')
bottom,top = ax.get_ylim()
ax.set_ylim(bottom+ 0.5, top - 0.5)
print("variable relationship")


# **Conclusion:** Glucose -  Insulin , Skinthickness - BMI

# # "--------  Section 3 ----------"

# ## 4. Finding Insuline Values
# 
# 1.  Isuline has 364 "0" so I will try to fill them according to Glucose and Skinthickness, but first I will create some graph to have a general view of the data Behav.
# 2.  Based on the Graph , We can notice that Insuline - Glucose has a better reationship "it might seemed a linear regresion" , the higher A is , B is as well,(please feel free to let any comment on this)
# 3. Because Glucose is gonna be my parameter I have to create some Interval to limite the values (I.e People who has glucose btwn A and B and then Extract the mean)

# In[ ]:


f,(ax1,ax2,ax3) =plt.subplots(1,3,figsize=(20,5))
y='Insulin'
sns.regplot(y=y,x='Glucose',data=df,order = 2,ax=ax1)
sns.scatterplot(y=y,x='SkinThickness',data=df , ax=ax2,hue='Outcome')
sns.boxplot(df.Glucose,ax=ax3)
print("This shows that the Higher Glucose. Higher Insuline")


# ##### Based on the Graph there is a better relation btwn Insulin-Glucose, so now I will fill them segmented by interval = 5

# ### 4.1. Finding Mean of Insulin

# In[ ]:


#Defining Intervals
lower = np.arange(26,236,5) #ax=ax Glucose we can see the min 20 , max 200
upper= np.arange(30,205,5)

#Finding the Insuline based on the Glucose/segmented 
def find_insuline (name,down,top):
    result = df.loc[(df['Insulin']>0)&(df['Glucose']>=down)&(df['Glucose']<=top)]['Insulin'].mean()
    result=np.round(result)
    Value.append(result)
    #print(f'the Mean of {name} in range {down}, {top} is :{result}')  
    
#Insuline Mean value  
Value= []
for i,j in zip(lower,upper):
    find_insuline("Insulin",i,j)
    
Value[1:9]  # <--- Here we have a lot of value but this way is more visualy friendly


# ###### The following Interval has Nan values so they will remain with 0 values 
# 1. The Mean of Insulin in range 31, 35 is :nan
# 2. The Mean of Insulin in range 36, 40 is :nan
# 3. The Mean of Insulin in range 41, 45 is :nan
# 4. The Mean of Insulin in range 46, 50 is :nan
# 5. The Mean of Insulin in range 51, 55 is :nan
# 
# ###### The Interval from 61,65 has nan value, but I can interpolate the value from rank 56-60 and 66-70

# ### 4.2. Finding Index of Insulin == 0

# In[ ]:


#Finding Insulines zero value (based on gluose) _index by range
def find_insuline_index (name,down,top):
    Index= df[(df['Insulin']==0) & (df['Glucose']>=down) & (df['Glucose']<=top)].index.values
    Indexes.append(Index)
    #print(f'the index in range {down}, {top} is :{Index}')
    
#Index
Indexes =[]
for i , j in zip(lower,upper):
    find_insuline_index("Index",i,j)   

Indexes[10:13] #It will show all the indexes with Insulin == 0


# ### 4.3. Replacing Values 

# In[ ]:


#Total
for i,j in zip (np.arange(0,36,1),Value):
    df.loc[Indexes[i],"Insulin"]=j
# Values replace nan for 0    
for i in np.arange(0,4,1):
    df.loc[Indexes[i],"Insulin"]=0
#Interval from 61,65 with 42   
df.loc[Indexes[7],"Insulin"]=42

finding_zeros(df)


# # "--------  Section 4 ----------"

# ## * 5. Finding Skin-Thickness*

# In[ ]:


f,(ax1,ax2,ax3) =plt.subplots(1,3,figsize=(20,5))
y='SkinThickness'
sns.scatterplot(y=y,x='BMI',data=df,ax=ax1)
sns.scatterplot(y=y,x='Glucose',data=df , ax=ax2,hue='Outcome')
sns.boxplot(df.BMI,ax=ax3)
print("This shows that the greater BMI ,the greater SkinThickness (it maskes sense)")


# ### 5.1. Creating Intervals
# ### 5.2. Finding SlkinThickness Values (mean)

# In[ ]:


# I wont delete the oulier(except zero) because it is not wrong data, according to the Paper they are all girls 21>Age years old
i_skin = np.arange(15,70,5)
j_skin = np.arange(20,75,5)

#Finding Value based on BMI
def finding_skin (name,down,top):
    result = df.loc[(df['SkinThickness']>0)&(df['BMI']>=down)&(df['BMI']<=top)]['SkinThickness'].mean()
    result=np.round(result,2)
    Skin_values.append(result)
    #print(f'the Mean of {name} in range {down}, {top} is :{result}')
    
Skin_values =[]
for i, j in zip (i_skin,j_skin):
    finding_skin("Thickness",i,j)
    
Skin_values


# ### 5.3. Finding Skin-Thickness Index
#     
# 

# In[ ]:


def finding_skin_index (name,down,top):
    Index = df.loc[(df['SkinThickness']==0)&(df['BMI']>=float(down))&(df['BMI']<=float(top))]['SkinThickness'].index.values
    Skin_Index.append(Index)
    #print(f'the index in range {down}, {top} is :{Index}')
    
Skin_Index = []
for i, j in zip (i_skin,j_skin):
    finding_skin_index("Thickness",i,j)
    
Skin_Index[0:2]


# ### 5.4. Replacing Values

# In[ ]:


#replacing Values    
for i,j in zip (np.arange(0,10,1),j_skin):
    df.loc[Skin_Index[i],"SkinThickness"]=j
    
finding_zeros(df)


# # "--------  Section 5 ----------"

# ## 6. Variable related with 0 Values  
# #### 6.1. Skinth Thickness ~ BMI

# In[ ]:


SKIN_BMI_ZERO= df[(df['SkinThickness']==0)&(df['BMI']==0)]
SKIN_BMI_ZERO


# #### 6.2.Glucose ~ Insulin

# In[ ]:


BLOOD_INSULINE_ZERO= df[(df['Glucose']==0)&(df['Insulin']==0)]
BLOOD_INSULINE_ZERO


# ### 6.3. Dropping Values
# 

# In[ ]:


#Index
BLOOD_INSULINE_ZERO=df[(df['Glucose']==0)&(df['Insulin']==0)].index.values
SKIN_BMI_ZERO= df[(df['SkinThickness']==0)&(df['BMI']==0)].index.values

#Droping VAlues
df.drop(BLOOD_INSULINE_ZERO, inplace=True)
df.drop(SKIN_BMI_ZERO ,inplace=True)


# ## 7. Blood Pressure

# In[ ]:


f,(ax1,ax2) =plt.subplots(1,2,figsize=(20,5))
sns.scatterplot(x='BloodPressure',y='BMI',data=df, ax=ax1)
sns.scatterplot(x='Age',y='BloodPressure',data=df, ax=ax2) 


# I can apply the same methodology to 'bloodpresurre' but after consulting with doctor (Friend of mine) He claims that blood pressure con not be associate with age , because it varies according several factors and that are not showed in the Data 

# In[ ]:


df.drop(df[df['BloodPressure']==0].index.values,inplace=True)
#Finding Missing Values
NAN_value = (df.isnull().sum() / len(df)) * 100
Missing = NAN_value[NAN_value==0].index.sort_values(ascending=False)
Missing_data = pd.DataFrame({'Missing Ratio' :NAN_value})
Missing_data.head()


# # "--------  Section 6 ----------"

# ## 8. Visualization

# In[ ]:


def boxplot (frame1,frame2,frame3):
    f,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(20,5))
    sns.boxplot(frame1,ax=ax1)
    sns.boxplot(frame2,ax=ax2)
    sns.boxplot(frame3,ax=ax3)


# In[ ]:


boxplot(df.Pregnancies,df.Glucose,df.BloodPressure)


# In[ ]:


boxplot(df.SkinThickness,df.Insulin,df.BMI)


# In[ ]:


boxplot(df.DiabetesPedigreeFunction,df.Age,df.Outcome)


# #### **We can easealy notice that there are so many outliers**

# # "--------  Section 7 ----------"

# ## 9. Outliers

# In[ ]:


preg = df.loc[df['Pregnancies']>=15]['Pregnancies'].count()
glu = df.loc[df['Glucose']<40]['Glucose'].count()
blood_1 =df[df['BloodPressure']<40]['BloodPressure'].count() 
blood_2 = df[df['BloodPressure']>100]['BloodPressure'].count()
blood = blood_1 + blood_2
skin = df[df['SkinThickness']>55]['SkinThickness'].count()
insu =df[df['Insulin']>380]['Insulin'].count()
bmi = df[df['BMI']>50]['BMI'].count()
dia = df[df['DiabetesPedigreeFunction']>1.2]['DiabetesPedigreeFunction'].count()
age= df[df['Age']>63]['Age'].count()
outliers = [preg,glu,blood,skin,insu,bmi,dia,age]
Outliers = pd.DataFrame(data=outliers, index = df.columns[0:8], columns=['Outliers'])
Outliers


# ### Dropping Outliers

# In[ ]:


preg_i = df.loc[df['Pregnancies']>=15]['Pregnancies'].index.values
glu_i = df.loc[df['Glucose']<40]['Glucose'].index.values
blood_1_i =df[df['BloodPressure']<40]['BloodPressure'].index.values 
blood_2_i= df[df['BloodPressure']>100]['BloodPressure'].index.values
skin = df[df['SkinThickness']>55]['SkinThickness'].index.values
insu =df[df['Insulin']>380]['Insulin'].index.values
bmi = df[df['BMI']>50]['BMI'].index.values
dia = df[df['DiabetesPedigreeFunction']>1.2]['DiabetesPedigreeFunction'].index.values
age= df[df['Age']>63]['Age'].index.values

ind_out = [preg_i,glu_i,blood_1_i,blood_2_i,skin,insu,bmi,dia,age]
for i in ind_out:
    df_out= df.drop(i)


# # "--------  Section 8 ----------"

# ## 10. Applying Machine Learning 
# 1. ##### Splitting Data

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,f1_score

X=df_out.drop('Outcome',axis=1)
y=df_out['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)


# #### 2. Applying Random Forest Tree

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=100)
RFC_model =RFC.fit(X_train,y_train)
y_pred_RFC= RFC.predict(X_test)
y_pred_RFC

print(classification_report(y_test,y_pred_RFC))
RFC_cm =confusion_matrix(y_test,y_pred_RFC)
print(f1_score(y_test,y_pred_RFC))
#confusion_matrix(y_pred_RFC, y_test)


# ##### 3. Applying Logistic Regression

# In[ ]:


from sklearn.preprocessing import StandardScaler

ES=StandardScaler()
ES=ES.fit_transform(df_out.drop('Outcome',axis=1))
data= pd.DataFrame(ES , columns= df.columns[:-1])
data.head(4) # data_p = Data already Processed

X_data=data
y_data=df_out['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.33, random_state=101)
data.head()


# In[ ]:


from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
log=log.fit(X_train,y_train)
y_pred_log = log.predict(X_test)

log_m = confusion_matrix(y_test,y_pred_log )
                         
print(classification_report(y_test,y_pred_log))
#print(confusion_matrix(y_test,y_pred_log))
print(f1_score(y_test,y_pred_log))


# ##### 4. Applying KNN 

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

Error_Rate = []
for i in range (1,50):
    
    KNN_Error = KNeighborsClassifier(n_neighbors=i)
    KNN_Error.fit(X_train,y_train)
    pred_i = KNN_Error.predict(X_test)
    Error_Rate.append(np.mean(pred_i != y_test))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,50), Error_Rate , color = 'blue', linestyle = 'dashed', marker = 'o')
plt.title('Error rate vs K value')
print( " K=3 is the most accurate rate because the error is closest to 0 ")    


# In[ ]:


KNN=KNeighborsClassifier(n_neighbors=3)
KNN=KNN.fit(X_train,y_train)
y_pred_KNN= KNN.predict(X_test)

KNN_cm =confusion_matrix(y_test,y_pred_KNN)
print(classification_report(y_test,y_pred_KNN))
#print(confusion_matrix(y_test,y_pred_KNN))
f1_score(y_test,y_pred_KNN)


# In[ ]:


f,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(20,5))
model= [log_m,RFC_cm,KNN_cm]
axes= [ax1,ax2,ax3]

for i,j in zip (model,axes):
    group_names = ['True Neg','False Pos','False Neg','True Pos']
        
    group_counts = ['{0:0.0f}'.format(value) for value in
                i.flatten()]
        
    group_percentages = ['{0:.2%}'.format(value) for value in
                     i.flatten()/np.sum(i)]
        
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
     zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
        
    ax =sns.heatmap(i, annot=labels, fmt='', cmap='plasma',ax=j)


#                                              **From *LEFT* to *RIGHT***  Logistical Regression - Random Forest - KNN cluster
