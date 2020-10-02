# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#getting the discriptive information about the train data
#FMCG TEAM_3
#IMPORTING THE LIBRARIES
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, norm
from scipy import stats
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2



train_data.head(10) 
train_data.shape

#droping ['id'] column from train data
train_data = train_data.loc[:, ~train_data.columns.str.contains('^Unnamed')]

print(train_data.keys())
#['PROD_CD', 'SLSMAN_CD', 'PLAN_MONTH', 'PLAN_YEAR', 'TARGET_IN_EA','ACH_IN_EA']
#are the columns name in train data
train_data.dtypes #datatype of colunms in train data
train_data.info
 
#CHECKING THE NULL VALUES
train_data.isnull().sum()#NO NULL VALUES\


#CHECKING DUPLICATES AND DRPPPING THEM
train_data.drop_duplicates(keep='first',inplace=True)   #NO DUPLICATES
#train_data.columns


#getting the discriptive information about test data
test_data.head(10)
test_data.shape

#droping ['id'] column from train data

test_data.info
print(test_data.keys())
test_data.dtypes #datatype of all columns

#ENCODING  (using this we have remove string part and kept integer in following colunms for EDA and better insights )
train_data['PROD_CD'] = train_data['PROD_CD'].str.replace(r'\D', '').astype(int)
train_data['SLSMAN_CD'] = train_data['SLSMAN_CD'].str.replace(r'\D', '').astype(int)
train_data['TARGET_IN_EA'] = train_data['TARGET_IN_EA'].str.replace(r'\D', '').astype(int)
train_data['ACH_IN_EA'] = train_data['ACH_IN_EA'].str.replace(r'\D', '').astype(int)

test_data['PROD_CD'] = test_data['PROD_CD'].str.replace(r'\D', '').astype(int)
test_data['SLSMAN_CD'] = test_data['SLSMAN_CD'].str.replace(r'\D', '').astype(int)
test_data['TARGET_IN_EA'] = test_data['TARGET_IN_EA'].str.replace(r'\D', '').astype(int)

types_train = train_data.dtypes  # all are integers
types_test = test_data.dtypes #datatype of all columns

#CHECK FOR CORRELATION
sns.pairplot((train_data),hue='ACH_IN_EA')
corr=train_data.corr()
corr
sns.heatmap(corr,annot=True)

#by heatmap we can see that target and achivement have good correlation that is 0.719321
#and month and year do not have good correlation that is -0.98

"""

                    ID   PROD_CD    ...      TARGET_IN_EA  ACH_IN_EA
ID            1.000000  0.004423    ...         -0.020581  -0.008335
PROD_CD       0.004423  1.000000    ...         -0.038098  -0.018343
SLSMAN_CD     0.999510 -0.000161    ...         -0.020497  -0.008000
PLAN_MONTH    0.009756  0.051039    ...         -0.062976   0.154697
PLAN_YEAR    -0.008052 -0.041916    ...          0.059697  -0.161209
TARGET_IN_EA -0.020581 -0.038098    ...          1.000000   0.719321
ACH_IN_EA    -0.008335 -0.018343    ...          0.719321   1.000000

"""



#GROUPBY IS USED FOR GRUOPING OF PARTICULAR PARAMETERS TO GET INSIGHTS OF PRODUCT AND SALESMAN DATA ACCORDING TO THEIR FREQUENCY IN A PERTICULAR MONTH AND YEAR 
train_data.groupby(['PROD_CD','PLAN_MONTH'])['PROD_CD'].count()
train_data.groupby(['SLSMAN_CD','PLAN_MONTH'])['SLSMAN_CD'].count()
train_data.groupby(['SLSMAN_CD','PLAN_MONTH','PLAN_YEAR'])['SLSMAN_CD'].count()

#CENTER TENDENCY
np.mean(train_data.PROD_CD)  
np.mean(train_data.SLSMAN_CD)  
np.mean(train_data.PLAN_MONTH)  
np.mean(train_data.PLAN_YEAR)  
np.mean(train_data.TARGET_IN_EA)  
np.mean(train_data.ACH_IN_EA)  

#STANDARD DEVIATION
np.std(train_data) 
#VARIANCE
np.var(train_data) 
# SKWENESS
skew(train_data) 
# KURTOSIS
kurtosis(train_data) 
   
#HISTOGRAM
plt.hist(train_data['PROD_CD']);plt.title('Histogram of PROD_CD'); plt.xlabel('PROD_CD'); plt.ylabel('Frequency')
plt.hist(train_data['SLSMAN_CD'], color = 'coral');plt.title('Histogram of SLSMAN_CD'); plt.xlabel('SLSMAN_CD'); plt.ylabel('Frequency')
plt.hist(train_data['PLAN_MONTH'], color= 'orange');plt.title('Histogram of PLAN_MONTH'); plt.xlabel('PLAN_MONTH'); plt.ylabel('Frequency')
plt.hist(train_data['TARGET_IN_EA'], color= 'brown');plt.title('Histogram of TARGET_IN_EA'); plt.xlabel('TARGET_IN_EA'); plt.ylabel('Frequency')
plt.hist(train_data['ACH_IN_EA'], color = 'violet');plt.title('Histogram of ACH_IN_EA'); plt.xlabel('ACH_IN_EA'); plt.ylabel('Frequency')

#BARPLOT
sns.barplot(x="TARGET_IN_EA", y="ACH_IN_EA", hue="PROD_CD", data=train_data)
plt.ylabel("ACH_IN_EA")
plt.title("ACHIVEMENT BASED ON TARGET")

#BOXPLOT
sns.boxplot(train_data["PROD_CD"])
sns.boxplot(train_data["SLSMAN_CD"])
sns.boxplot(train_data["PLAN_MONTH"])
sns.boxplot(train_data["PLAN_YEAR"])
sns.boxplot(train_data["TARGET_IN_EA"])
sns.boxplot(train_data["ACH_IN_EA"])

#SCATTERPLOT
sns.scatterplot(x='PROD_CD', y='ACH_IN_EA', data=train_data).set_title('Scatterplot of ACH_IN_EA & PROD_CD')
sns.scatterplot(x='SLSMAN_CD', y='ACH_IN_EA', data=train_data).set_title('Scatterplot of ACH_IN_EA & SLSMAN_CD')
sns.scatterplot(x='PLAN_MONTH', y='ACH_IN_EA', data=train_data).set_title('Scatterplot of ACH_IN_EA & PLAN_MONTH')
sns.scatterplot(x='PLAN_YEAR', y='ACH_IN_EA', data=train_data).set_title('Scatterplot of ACH_IN_EA & PLAN_YEAR')
sns.scatterplot(x='TARGET_IN_EA', y='ACH_IN_EA', data=train_data).set_title('Scatterplot of ACH_IN_EA & TARGET_IN_EA')

#COUNTPLOT
sns.countplot(train_data["PROD_CD"])
sns.countplot(train_data["SLSMAN_CD"])
sns.countplot(train_data["PLAN_MONTH"])
sns.countplot(train_data["PLAN_YEAR"])
sns.countplot(train_data["TARGET_IN_EA"])

#UNIQUE VALUES and COUNTS
train_data.PROD_CD.unique()               
train_data.PROD_CD.value_counts()                    
train_data.SLSMAN_CD.unique()
train_data.SLSMAN_CD.value_counts()
train_data.PLAN_YEAR.unique()
train_data.PLAN_YEAR.value_counts()
train_data.PLAN_MONTH.unique()
train_data.PLAN_MONTH.value_counts()
train_data.TARGET_IN_EA.unique()
train_data.TARGET_IN_EA.value_counts()
train_data.ACH_IN_EA.unique()
train_data.ACH_IN_EA.value_counts()

#HERE IS SOME INSIGHTS ABOUT HOW MUCH SALESMAN'S TARGETS AND ACHIEVEMENTS
train_data.plot(x="ACH_IN_EA",y="SLSMAN_CD")
train_data.plot(x="TARGET_IN_EA",y="SLSMAN_CD")


#we can finally plot the targets and acheivements  made for each salesman and products for each month:
fig,ax= plt.subplots(figsize =(15,7))

fig= train_data.groupby(['PROD_CD','PLAN_MONTH']).count()['ACH_IN_EA'].unstack().plot(ax=ax)

fig= train_data.groupby(['SLSMAN_CD','PLAN_MONTH'])['ACH_IN_EA'].count().unstack().plot(ax=ax)

fig= train_data.groupby(['PROD_CD','PLAN_MONTH']).count()['TARGET_IN_EA'].unstack().plot(ax=ax)

fig= train_data.groupby(['SLSMAN_CD','PLAN_MONTH'])['TARGET_IN_EA'].count().unstack().plot(ax=ax)

#ANALYIZING MORE INSGHTS
#GETTING THE ACCURING FREQUENCY OF EACH PRODUCTS AND SALESMAN IN BAR CHART FOR EACH MONTH AND YEAR  
pd.crosstab(train_data.PROD_CD,train_data.PLAN_MONTH).plot(kind="bar")
pd.crosstab(train_data.PROD_CD,train_data.PLAN_YEAR).plot(kind="bar")
pd.crosstab(train_data.SLSMAN_CD,train_data.PLAN_MONTH).plot(kind="bar")
pd.crosstab(train_data.SLSMAN_CD,train_data.PLAN_YEAR).plot(kind="bar")
#In above crosstab we can se values of Product_CD with respect to years.

#distribution plot
#By distribution plot we can see how much our data is normally distributed
sns.distplot(train_data['PROD_CD'], fit=norm, kde=False)
sns.distplot(train_data['SLSMAN_CD'], fit=norm, kde=False, color = 'coral')
sns.distplot(train_data['PLAN_MONTH'], fit=norm, kde=False, color = 'skyblue')
sns.distplot(train_data['PLAN_YEAR'], fit=norm, kde=False, color = 'orange')
sns.distplot(train_data['TARGET_IN_EA'], fit=norm, kde=False, color = 'brown')
sns.distplot(train_data['ACH_IN_EA'], fit=norm, kde=False, color = 'violet')


#VISUALISATION OF DENSITY DISTRIBUTION OF TARGETS AND ACHEIVEMENT AND COMPARIOSION BETWEEN THEM
sns.kdeplot(train_data['TARGET_IN_EA'],shade = True, bw = .5, color = "red")
sns.kdeplot(train_data['ACH_IN_EA'],shade = True, bw = .5, color = "BLUE")



#VISIUALIZING DISTRIBUTION OF DATA ACCORDING TO THE MONTH
sns.violinplot(y=train_data['PROD_CD'],x=train_data['PLAN_MONTH'])
sns.violinplot(y=train_data['SLSMAN_CD'],x=train_data['PLAN_MONTH'])
sns.violinplot(y=train_data['TARGET_IN_EA'],x=train_data['PLAN_MONTH'])
sns.violinplot(y=train_data['ACH_IN_EA'],x=train_data['PLAN_MONTH'])



#GETTING EXACTLY WHERE SALES MA COMPLETED THEIR TARGET OR ACHIEVED THE TARGET OF EVERY PRODUCT IN DATADET 
 target=list(fmcg.TARGET_IN_EA)
     achiv=list(fmcg.ACH_IN_EA)
     yn=[]     
     for x in range(22646):
         if(target[x]<=achiv[x]):
             
             #print("yes")
             yn.append(1)
             
         else:
             yn.append(0)

#HERE IT SHOWING HOW MUCH TARGETS ARE ACHIEVED OR NOT ACHIEVED
pd.crosstab(train_data.result,train_data.PLAN_YEAR).plot(kind="bar")
pd.crosstab(train_data.result,train_data.PLAN_MONTH).plot(kind="bar")

#arrays 
prod = np.array(train_data['PROD_CD'])
salesman = np.array(train_data['SLSMAN_CD'])
month = np.array(train_data['PLAN_MONTH'])
year = np.array(train_data['PLAN_YEAR'])
target = np.array(train_data['TARGET_IN_EA'])
achieved = np.array(train_data['ACH_IN_EA'])





# Normal Probability distribution 
#As we know data is not normally Distributed so we can process and form the Normal Probability Distribution of data column wise.

# ACHIEVED
x_ach = np.linspace(np.min(achieved), np.max(achieved))
y_ach = stats.norm.pdf(x_ach, np.mean(x_ach), np.std(x_ach))
plt.plot(x_ach, y_ach,); plt.xlim(np.min(x_ach), np.max(x_ach));plt.xlabel('achieved');plt.ylabel('Probability');plt.title('Normal Probability Distribution of achieved')

# Product_code
x_prod = np.linspace(np.min(prod), np.max(prod))
y_prod = stats.norm.pdf(x_prod, np.mean(x_prod), np.std(x_prod))
plt.plot(x_prod, y_prod, color = 'coral'); plt.xlim(np.min(x_prod), np.max(x_prod));plt.xlabel('prod_cd');plt.ylabel('Probability');plt.title('Normal Probability Distribution of prod_cd')

# salesman_code
x_sale = np.linspace(np.min(salesman), np.max(salesman))
y_sale = stats.norm.pdf(x_sale, np.mean(x_sale), np.std(x_sale))
plt.plot(x_sale, y_sale, color = 'coral'); plt.xlim(np.min(x_sale), np.max(x_prod));plt.xlabel('Sale_cd');plt.ylabel('Probability');plt.title('Normal Probability Distribution of sales_cd')

# target
x_target = np.linspace(np.min(target), np.max(target))
y_target = stats.norm.pdf(x_target, np.mean(x_target), np.std(x_target))
plt.plot(x_target, y_target, color = 'coral'); plt.xlim(np.min(x_target), np.max(x_target));plt.xlabel('target');plt.ylabel('Probability');plt.title('Normal Probability Distribution of target')

# Unsquish the pie.
train_data['PLAN_MONTH'].value_counts().head(10).plot.pie()
train_data['PLAN_YEAR'].value_counts().head(10).plot.pie()
plt.gca().set_aspect('equal')

#By all the plots and graphs we have come to conclusion that highest achivement is 232,000 
#by saleman code i.e SLSMAN_CD 94 in month of november 2019 product_CD 31 



X = train_data.iloc[:,:6]  #independent columns
y = train_data.iloc[:,-1]    #target column i.e price range

#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=5)
fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['imp','importance']  #naming the dataframe columns

featureScores
"""
            imp    importance
0       PROD_CD  7.657698e+04
1     SLSMAN_CD  2.116982e+05
2    PLAN_MONTH  2.996838e+04
3     PLAN_YEAR  1.313697e+00
4  TARGET_IN_EA  4.555223e+08
5     ACH_IN_EA  6.650893e+08

"""
#By using feature selection we get the importance of particular column in data.
#so by feature selection we get that product_CD and Achivement is having more importance respectively followed by target
#column with very less importance can be drop.

