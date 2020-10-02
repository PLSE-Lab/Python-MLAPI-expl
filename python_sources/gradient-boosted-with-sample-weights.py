#!/usr/bin/env python
# coding: utf-8

# This is my first competition but is an area that I'm very intersted in, using quantitative methods to solve social problems. I learned a lot by seeing how other people handled some of the challenges this dataset provided. Especially https://www.kaggle.com/skooch/lgbm-with-random-split . I also got some feature engineering ideas from https://www.kaggle.com/gaxxxx/exploratory-data-analysis-lightgbm .
# 
# As has ben mentioned in other kernels one of the problems with working with the data is the imbalance of the classes. There are different ways to account for this, including using the LightGBM balance option for class_weight, or using subsampling. Here I adopt the approach of assigning sample weights to the different target groups based on the inverse of their probability of being selected. Finally, you also want to make sure that your training and testing splits account for households and this can be done using Group Split where each idhogar is group.
# 
# Ultimately, the main problem is there is not enough information to separate the groups. There is too much overlap between the targets. Target 4 is clearly different. Target 1 is somewhat different. But 1,2,3 are very similar. I don't show this here, but examining polar plots of the variables (especially all the dummy coded variables) and you can see that targets 1-3 are very similar in their characteristics. 
# 
# Also, the data is hierarchical in nature, individuals nested in households nested in regions. A better result might be achieved using a multilevel mixed effects model to account for the correlation inherent in this data. 
# 
# You need to do some feature engineering. Without doing this you won't get a F1 score above .4 I think. Two that are consistantly important in whatever model I tried were age and education. In particular median household age, and then an Education Index for 25 and older derived from the UN's Multidimensional Poverty Index http://hdr.undp.org/en/faq-page. Other education features were pretty important too.
# 
# For the sampling weights, I initially fit separate models for each target as binary variables. For these model I used sample weights based on the individual's target. So target for was weighted the highest because it had the fewest samples. I then took a pseudo propensity score approach and assigned weights to each observation based on the probability of getting the binary model correct. These sample weights were used for the full multiclass model.
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os

import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
sns.set()


# Any results you write to the current directory are saved as output.
get_ipython().run_line_magic('matplotlib', 'inline')

labels = {'v2a1':' Monthly rent payment','hacdor':' =1 Overcrowding by bedrooms','rooms':'  number of all rooms in the house','hacapo':' =1 Overcrowding by rooms','v14a':' =1 has toilet in the household','refrig':' =1 if the household has refrigerator','v18q':' owns a tablet','v18q1':' number of tablets household owns','r4h1':' Males younger than 12 years of age','r4h2':' Males 12 years of age and older','r4h3':' Total males in the household','r4m1':' Females younger than 12 years of age','r4m2':' Females 12 years of age and older','r4m3':' Total females in the household','r4t1':' persons younger than 12 years of age','r4t2':' persons 12 years of age and older','r4t3':' Total persons in the household','tamhog':' size of the household','tamviv':' TamViv','escolari':' years of schooling','rez_esc':' Years behind in school','hhsize':' household size','paredblolad':' =1 if predominant material on the outside wall is block or brick','paredzocalo':' "=1 if predominant material on the outside wall is socket (wood  zinc or absbesto"','paredpreb':' =1 if predominant material on the outside wall is prefabricated or cement','pareddes':' =1 if predominant material on the outside wall is waste material','paredmad':' =1 if predominant material on the outside wall is wood','paredzinc':' =1 if predominant material on the outside wall is zink','paredfibras':' =1 if predominant material on the outside wall is natural fibers','paredother':' =1 if predominant material on the outside wall is other','pisomoscer':' "=1 if predominant material on the floor is mosaic   ceramic   terrazo"','pisocemento':' =1 if predominant material on the floor is cement','pisoother':' =1 if predominant material on the floor is other','pisonatur':' =1 if predominant material on the floor is  natural material','pisonotiene':' =1 if no floor at the household','pisomadera':' =1 if predominant material on the floor is wood','techozinc':' =1 if predominant material on the roof is metal foil or zink','techoentrepiso':' "=1 if predominant material on the roof is fiber cement   mezzanine "','techocane':' =1 if predominant material on the roof is natural fibers','techootro':' =1 if predominant material on the roof is other','cielorazo':' =1 if the house has ceiling','abastaguadentro':' =1 if water provision inside the dwelling','abastaguafuera':' =1 if water provision outside the dwelling','abastaguano':' =1 if no water provision','public':' "=1 electricity from CNFL   ICE   ESPH/JASEC"','planpri':' =1 electricity from private plant','noelec':' =1 no electricity in the dwelling','coopele':' =1 electricity from cooperative','sanitario1':' =1 no toilet in the dwelling','sanitario2':' =1 toilet connected to sewer or cesspool','sanitario3':' =1 toilet connected to  septic tank','sanitario5':' =1 toilet connected to black hole or letrine','sanitario6':' =1 toilet connected to other system','energcocinar1':' =1 no main source of energy used for cooking (no kitchen)','energcocinar2':' =1 main source of energy used for cooking electricity','energcocinar3':' =1 main source of energy used for cooking gas','energcocinar4':' =1 main source of energy used for cooking wood charcoal','elimbasu1':' =1 if rubbish disposal mainly by tanker truck','elimbasu2':' =1 if rubbish disposal mainly by botan hollow or buried','elimbasu3':' =1 if rubbish disposal mainly by burning','elimbasu4':' =1 if rubbish disposal mainly by throwing in an unoccupied space','elimbasu5':' "=1 if rubbish disposal mainly by throwing in river   creek or sea"','elimbasu6':' =1 if rubbish disposal mainly other','epared1':' =1 if walls are bad','epared2':' =1 if walls are regular','epared3':' =1 if walls are good','etecho1':' =1 if roof are bad','etecho2':' =1 if roof are regular','etecho3':' =1 if roof are good','eviv1':' =1 if floor are bad','eviv2':' =1 if floor are regular','eviv3':' =1 if floor are good','dis':' =1 if disable person','male':' =1 if male','female':' =1 if female','estadocivil1':' =1 if less than 10 years old','estadocivil2':' =1 if free or coupled uunion','estadocivil3':' =1 if married','estadocivil4':' =1 if divorced','estadocivil5':' =1 if separated','estadocivil6':' =1 if widow/er','estadocivil7':' =1 if single','parentesco1':' =1 if household head','parentesco2':' =1 if spouse/partner','parentesco3':' =1 if son/doughter','parentesco4':' =1 if stepson/doughter','parentesco5':' =1 if son/doughter in law','parentesco6':' =1 if grandson/doughter','parentesco7':' =1 if mother/father','parentesco8':' =1 if father/mother in law','parentesco9':' =1 if brother/sister','parentesco10':' =1 if brother/sister in law','parentesco11':' =1 if other family member','parentesco12':' =1 if other non family member','idhogar':' Household level identifier','hogar_nin':' Number of children 0 to 19 in household','hogar_adul':' Number of adults in household','hogar_mayor':' # of individuals 65+ in the household','hogar_total':' # of total individuals in the household','dependency':' Dependency rate','edjefe':' years of education of male head of household','edjefa':' years of education of female head of household','meaneduc':'average years of education for adults (18+)','instlevel1':' =1 no level of education','instlevel2':' =1 incomplete primary','instlevel3':' =1 complete primary','instlevel4':' =1 incomplete academic secondary level','instlevel5':' =1 complete academic secondary level','instlevel6':' =1 incomplete technical secondary level','instlevel7':' =1 complete technical secondary level','instlevel8':' =1 undergraduate and higher education','instlevel9':' =1 postgraduate higher education','bedrooms':' number of bedrooms','overcrowding':' # persons per room','tipovivi1':' =1 own and fully paid house','tipovivi2':' "=1 own   paying in installments"','tipovivi3':' =1 rented','tipovivi4':' =1 precarious','tipovivi5':' "=1 other(assigned   borrowed)"','computer':' =1 if the household has notebook or desktop computer','television':' =1 if the household has TV','mobilephone':' =1 if mobile phone','qmobilephone':' # of mobile phones','lugar1':' =1 region Central','lugar2':' =1 region Chorotega','lugar3':' =1 region PacÃ­fico central','lugar4':' =1 region Brunca','lugar5':' =1 region Huetar AtlÃ¡ntica','lugar6':' =1 region Huetar Norte','area1':' =1 zona urbana','area2':' =2 zona rural','age':' Age in years','SQBescolari':' escolari squared','SQBage':' age squared','SQBhogar_total':' hogar_total squared','SQBedjefe':' edjefe squared','SQBhogar_nin':' hogar_nin squared','SQBovercrowding':' overcrowding squared','SQBdependency':' dependency squared','SQBmeaned':' meaned squared','agesq':' Age squared'}


# In[ ]:


df = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
#print(df.head())

df['Target0']=df['Target']-1
locations = ['lugar1','lugar2','lugar3','lugar4','lugar5','lugar6']

le = LabelEncoder()
df['idhogarnum'] = df[['idhogar']].apply(le.fit_transform)
df_test['idhogarnum'] = df_test[['idhogar']].apply(le.fit_transform)


# In[ ]:



df = pd.concat([df,pd.get_dummies(df['Target'], prefix='Target')],axis=1)
df.head()
dataframes = [df,df_test]


# Calculate the weights for each type of target.

# In[ ]:



df.groupby('Target')['Target'].count()
#Reciprocal of the probability of selection
T1p = 755.0/9557.0
T2p = 1597.0/9557.0
T3p = 1209.0/9557.0
T4p = 5996.0/9557.0

RT1p = 1.0/T1p
RT2p = 1.0/T2p
RT3p = 1.0/T3p
RT4p = 1.0/T4p
print(RT1p,RT2p,RT3p,RT4p)

df['targetweight'] = 0.0
df.loc[df['Target']==1,'targetweight']=RT1p
df.loc[df['Target']==2,'targetweight']=RT2p
df.loc[df['Target']==3,'targetweight']=RT3p
df.loc[df['Target']==4,'targetweight']=RT4p


# Feature engineering for individual level variables.

# In[ ]:


individual_variables = ['age','mobilephone','escolari','mobilephone','male','female','dis',
                        'instlevel1','instlevel2','instlevel3','instlevel4','instlevel5',
                        'instlevel6','instlevel7','instlevel8','instlevel9','estadocivil1',
                        'estadocivil2','estadocivil3','estadocivil4','estadocivil5','estadocivil6',
                        'estadocivil7','parentesco1','parentesco2','parentesco3','parentesco4',
                        'parentesco5','parentesco6','parentesco7','parentesco8','parentesco9',
                        'parentesco10','parentesco11','parentesco12']
#Target Group 2 as a baseline


for tdf in dataframes:
    #education Index for individuals over the age of 25
    tdf['indeduindex25'] = 0
    tdf.loc[tdf['age']>=25,'indeduindex25'] = (tdf['escolari'] / 15.0 + (7/18.0))/2.0
    tdf.loc[tdf['indeduindex25'].isna(),'indeduindex25'] = 0
    individual_variables.append('indeduindex25')
    tdf['agecentschool'] = tdf['age']-7
    #calculate if 6 to 13 year olds are attending school
    tdf['attendSchool'] = 0
    tdf['attendSchool'] = np.where(tdf['escolari']>0,1,0)
    tdf['attendSchool'] = np.where((tdf['age'] >=6) &(tdf['age'] <=13), tdf['attendSchool'],0)
    individual_variables.append('attendSchool')
    #attending school but behind for age 6 to 13
    tdf['behindSchool'] = 0
    tdf['behindSchool'] = np.where(tdf['agecentschool'] >tdf['escolari'],1,0)
    tdf['behindSchool'] = np.where((tdf['age'] >=6) &(tdf['age'] <=13), tdf['behindSchool'],0)
    individual_variables.append('behindSchool')
    tdf['behindSchool18'] = 0
    tdf['behindSchool18'] = np.where(tdf['agecentschool'] >tdf['escolari'],1,0)
    tdf['behindSchool18'] = np.where((tdf['age'] >=6) &(tdf['age'] <=18), tdf['behindSchool18'],0)
    individual_variables.append('behindSchool18')
    tdf['portionofhh'] = 0
    tdf['portionofhh'] = 1.0 / tdf['hhsize']
    tdf.loc[tdf['parentesco1']==1,'portionofhh']=1.0
    individual_variables.append('portionofhh')
#df['sampweights'] = 0.0
#df.loc[df['parentesco1']==1,'sampweights'] = 1.0
#df.loc[df['Target0']==0,'sampweights'] = df[df['Target0']==0]['portionofhh']+0.9210003139060374
#df.loc[df['Target0']==1,'sampweights'] = df[df['Target0']==1]['portionofhh']+0.8328973527257507
#df.loc[df['Target0']==2,'sampweights'] = df[df['Target0']==2]['portionofhh']+0.8734958669038402
#df.loc[df['Target0']==3,'sampweights'] = df[df['Target0']==3]['portionofhh']+0.3726064664643717
    #individual_variables.append('portionofhh')
pd.set_option('display.max_columns', None)
#df[individual_variables+['agecentschool']]
individual_variables = list(set(individual_variables)) #clear out duplicates


# Feature engineering for household level variables.

# In[ ]:


household_variables = ['lugar1','lugar2','lugar3','lugar4','lugar5','lugar6','area1','area2',
                      'paredblolad','paredzocalo','paredpreb','pareddes','paredmad','paredzinc',
                       'paredfibras','paredother','pisomoscer','pisocemento','pisoother','pisonatur',
                       'pisonotiene','pisomadera','techozinc','techoentrepiso','techocane','techootro',
                       'cielorazo','abastaguadentro','abastaguafuera','abastaguano','public','planpri',
                       'noelec','coopele','sanitario1','sanitario2','sanitario3','sanitario5','sanitario6',
                       'energcocinar1','energcocinar2','energcocinar3','energcocinar4','elimbasu1','elimbasu2',
                       'elimbasu3','elimbasu4','elimbasu5','elimbasu6','epared1','epared2','epared3','etecho1',
                       'etecho2','etecho3','eviv1','eviv2','eviv3','tipovivi1','tipovivi2','tipovivi3','tipovivi4','tipovivi5',
                      'v18q1','qmobilephone','tamviv','tamhog','r4h1','r4h2','r4h3','r4m1','r4m2','r4m3','bedrooms', 'computer', 'hacapo', 'hacdor', 'hogar_adul', 'hogar_mayor', 'hogar_nin', 'r4t1',
              'r4t2','r4t3', 'refrig', 'rooms', 'television', 'v14a', 'v18q']#
#household_variables = ['lugar1','lugar2','lugar3','lugar4','lugar5','lugar6','area1','area2',
                      #'v18q1','qmobilephone','tamviv','tamhog']
for tdf in dataframes:
    #change missing rent values to 0
    tdf.loc[tdf['v2a1'].isna(),'v2a1'] = 0
    tdf.loc[tdf['v18q1'].isna(),'v18q1'] = 0
    tdf.loc[tdf['qmobilephone'].isna(),'qmobilephone'] = 0
    
    conditions = [(tdf['v2a1']==0),(tdf['v2a1']!=0)]
    choices = [0,np.log(tdf['v2a1']+.0001)]
    tdf['lv2a1']=np.select(conditions, choices)
    household_variables.append('lv2a1')
    
    #Median Household Age
    tdf['medhhage'] = tdf['age'].groupby(tdf['idhogar']).transform('median')
    household_variables.append('medhhage')
    #Create a household education index
    #Based on UN Education Index for poverty. 
    #Household members 25 years and older mean education based on average 7 years education
    tdf['eduindex'] = tdf[tdf['age']>=25]['escolari'].groupby(tdf['idhogar']).transform('mean')
    tdf['eduindex'] = (tdf['eduindex'] / 15.0 + (8.3/18.0))/2.0
    tdf['eduindex'] = tdf['eduindex'].groupby(tdf['idhogar']).transform('mean')
    tdf.loc[tdf['eduindex'].isna(),'eduindex'] = 0
    household_variables.append('eduindex')
    #At least one household member with 8 years of education
    tdf['atleast8'] = 0
    tdf['atleast8'] = tdf[tdf['escolari']>=8]['escolari'].groupby(tdf['idhogar']).transform('count')
    tdf['atleast8'] = tdf['atleast8'].groupby(tdf['idhogar']).transform('max')
    tdf['atleast8'] = np.where(tdf['atleast8']>=1, 1, 0)
    household_variables.append('atleast8')
    
    #At least one female household member with 8 years of education
    tdf['atleast8F'] = 0
    tdf['atleast8F'] = tdf[(tdf['escolari']>=6)&(tdf['female']>=1)]['escolari'].groupby(tdf['idhogar']).transform('count')
    tdf['atleast8F'] = tdf['atleast8F'].groupby(tdf['idhogar']).transform('max')
    tdf['atleast8F'] = np.where(tdf['atleast8F']>=1, 1, 0)
    household_variables.append('atleast8F')
    
    #Number of household members below 14
    tdf['nhhlt14'] = 0
    tdf['nhhlt14'] = tdf[tdf['age']<=14]['age'].groupby(tdf['idhogar']).transform('count')
    tdf['nhhlt14'] = tdf['nhhlt14'].groupby(tdf['idhogar']).transform('max')
    tdf['nhhlt14'] = np.where(tdf['nhhlt14']>=1, tdf['nhhlt14'], 0)
    household_variables.append('nhhlt14')
    
    #Number of household members below 14
    tdf['perhhlt14'] = 0.0
    tdf['perhhlt14'] = tdf['nhhlt14'] / tdf['hhsize']
    household_variables.append('perhhlt14')
    
    tdf['perschoolage'] = 0.0
    tdf['perschoolage'] = tdf[(tdf['age'] >=6) &(tdf['age'] <=13)].groupby(tdf['idhogar']).transform('count')
    tdf['perschoolage'] = tdf['perschoolage'].groupby(tdf['idhogar']).transform('max')
    tdf['perschoolage'] = np.where(tdf['perschoolage']>=1, tdf['perschoolage'], 0)
    tdf['perschoolage'] = tdf['perschoolage'] / tdf['hhsize']
    household_variables.append('perschoolage')
    #parametric household consumption units
    tdf['cupara'] = tdf['hogar_adul']+(0.5*tdf['nhhlt14'])
    household_variables.append('cupara')
    
    #no household assets
    tdf['assets2'] = 0
    tdf.loc[(tdf['v18q']<=1)&(tdf['qmobilephone']<=1)&(tdf['computer']==0)&(tdf['television']==0),'assets2'] = 1 
    household_variables.append('assets2')
    
    #some assets
    tdf['assets3'] = 0
    tdf.loc[(tdf['v18q']<=1)&(tdf['qmobilephone']>=1)&(tdf['computer']==0)&(tdf['television']==1),'assets3'] = 1 
    household_variables.append('assets3')
    #some assets
    tdf['assets3b'] = 0
    tdf.loc[(tdf['v18q']<=1)&(tdf['qmobilephone']>=1)&(tdf['computer']==0)&(tdf['television']==0),'assets3b'] = 1 
    household_variables.append('assets3b')
    
    #a lot of assets
    tdf['assets4'] = 0
    tdf.loc[(tdf['v18q']>=1)&(tdf['qmobilephone']>=1)&(tdf['computer']==1)&(tdf['television']==1),'assets4'] = 1 
    household_variables.append('assets4')
    
    #head of household single, divorced, separated, or widowed
    conditions = [(tdf['parentesco1']==1)&(tdf['estadocivil4']==1),(tdf['parentesco1']==1)&(tdf['estadocivil5']==1),
                  (tdf['parentesco1']==1)&(tdf['estadocivil6']==1),
                  (tdf['parentesco1']==1)&(tdf['estadocivil7']==1),(tdf['parentesco1']==0)]
    choices = [1,1,1,1,0]
    tdf['headmarital']=np.select(conditions, choices)
    tdf['headmarital'] = tdf['headmarital'].groupby(tdf['idhogar']).transform('count')
    tdf['headmarital'] = np.where(tdf['headmarital']>=1, 1, 0)
    household_variables.append('headmarital')
    
    #Ratio household children to Adults
    conditions = [(tdf['hogar_adul']==0),(tdf['hogar_adul']>0)]
    choices = [0, tdf['hogar_nin']/tdf['hogar_adul']]
    tdf['radultchild']=np.select(conditions, choices)
    household_variables.append('radultchild')
    
    #Ratio bedrooms to household size
    conditions = [(tdf['tamhog']==0),(tdf['tamhog']>0)]
    choices = [0, tdf['bedrooms']/tdf['tamhog']]
    tdf['bedperperson']=np.select(conditions, choices)
    household_variables.append('bedperperson')
    
    conditions = [(tdf['rooms']==0),(tdf['rooms']>0)]
    choices = [0, tdf['bedrooms']/tdf['rooms']]
    tdf['perbedrooms']=np.select(conditions, choices)
    household_variables.append('perbedrooms')
    
    #ratio rooms to household size
    conditions = [(tdf['rooms']==0),(tdf['rooms']>0)]
    choices = [0, tdf['tamhog']/tdf['rooms']]
    tdf['rperperson']=np.select(conditions, choices)
    household_variables.append('rperperson')
    
    #ratio bedrooms to rooms
    conditions = [(tdf['rooms']==0),(tdf['rooms']>0)]
    choices = [0, tdf['bedrooms']/tdf['rooms']]
    tdf['roomstobedrooms']=np.select(conditions, choices)
    household_variables.append('roomstobedrooms')
    
    #ratio rent to adults
    conditions = [(tdf['hogar_adul']==0),(tdf['hogar_adul']>0)]
    choices = [0, tdf['lv2a1']/tdf['hogar_adul']]
    tdf['rentpadult']=np.select(conditions, choices)
    household_variables.append('rentpadult')
    
    #ratio phones to adults
    conditions = [(tdf['hogar_adul']==0),(tdf['hogar_adul']>0)]
    choices = [0, tdf['qmobilephone']/tdf['hogar_adul']]
    tdf['phoneperadult']=np.select(conditions, choices)
    household_variables.append('phoneperadult')
    
    #ratio phones to hhsize
    conditions = [(tdf['hhsize']==0),(tdf['hhsize']>0)]
    choices = [0, tdf['qmobilephone']/tdf['hhsize']]
    tdf['phoneperhhsize']=np.select(conditions, choices)
    household_variables.append('phoneperhhsize')
    
    #ratio of rent to rooms
    conditions = [(tdf['rooms']==0),(tdf['rooms']>0)]
    choices = [0, tdf['lv2a1']/tdf['rooms']]
    tdf['rentproom']=np.select(conditions, choices)
    household_variables.append('rentproom')
    
    #Dependency in the household
    conditions = [(tdf['dependency']=='yes'),(tdf['dependency']=='no'),(tdf['dependency']!='yes')&(tdf['dependency']!='no')]
    choices = [1,0,0]
    tdf['dphhyesno']=np.select(conditions, choices)
    household_variables.append('dphhyesno')
    
    #Dependency rate
    conditions = [(tdf['dependency']=='yes'),(tdf['dependency']=='no'),(tdf['dependency']!='yes')&(tdf['dependency']!='no')]
    choices = [0,0,tdf['dependency']]
    tdf['dpnumeric']=np.select(conditions, choices)
    household_variables.append('dpnumeric')
    
    #Female head of household
    tdf['femhead'] = 0
    tdf.loc[(tdf['parentesco1']==1)&(tdf['female']==1),'femhead'] = 1
    tdf['femhead'] = tdf['femhead'].groupby(tdf['idhogar']).transform('count')
    tdf['femhead'] = np.where(tdf['femhead']>=1, 1, 0)
    household_variables.append('femhead')
    
    #Get the age of the head of household and the number of schooling missing values cleaned up later
    tdf['hhheadeduindex'] = 0.0
    tdf['hhheadeduindex'] = tdf[tdf['parentesco1']==1]['escolari'].groupby(tdf['idhogar']).transform('mean')
    tdf['hhheadeduindex'] = (tdf['hhheadeduindex'] / 15.0 + (7.2/18.0))/2.0
    tdf.loc[tdf['hhheadeduindex'].isna(),'hhheadeduindex'] = 0
    tdf['hhheadeduindex'] = tdf['hhheadeduindex'].groupby(tdf['idhogar']).transform('mean')
    tdf.loc[tdf['hhheadeduindex'].isna(),'hhheadeduindex'] = 0
    household_variables.append('hhheadeduindex')

    tdf['hhheadage'] =0
    tdf['hhheadage'] = tdf[tdf['parentesco1']==1]['age'].groupby(tdf['idhogar']).transform('mean')
    tdf['hhheadage'] = tdf['hhheadage'].groupby(tdf['idhogar']).transform('mean')
    tdf.loc[tdf['hhheadage'].isna(),'hhheadage'] = 0
    household_variables.append('hhheadage')
    
    
    #missing head of household
    tdf['misshead'] = 0
    tdf['misshead'] = tdf['parentesco1'].groupby(tdf['idhogar']).transform('sum')
    tdf['misshead'] = np.where(tdf['misshead']==0, 1, 0)
    household_variables.append('misshead')
    #get the missing value information for male head of household schooling
    conditions = [(tdf['edjefe']=='yes'),(tdf['edjefe']=='no'),(tdf['edjefe']!='yes')&(tdf['edjefe']!='no')]
    choices = [0,1,0]
    tdf['edjefemiss']=np.select(conditions, choices)
    household_variables.append('edjefemiss')
    
    conditions = [(tdf['edjefe']=='yes'),(tdf['edjefe']=='no'),(tdf['edjefe']!='yes')&(tdf['edjefe']!='no')]
    choices = [0,0,tdf['edjefe']]
    tdf['edjefeval']=np.select(conditions, choices) 
    household_variables.append('edjefeval')
    
    #get the missing value information for male head of household schooling
    conditions = [(tdf['edjefa']=='yes'),(tdf['edjefa']=='no'),(tdf['edjefa']!='yes')&(tdf['edjefa']!='no')]
    choices = [0,1,0]
    tdf['edjefamiss']=np.select(conditions, choices)
    household_variables.append('edjefamiss')
    
    conditions = [(tdf['edjefa']=='yes'),(tdf['edjefa']=='no'),(tdf['edjefa']!='yes')&(tdf['edjefa']!='no')]
    choices = [0,0,tdf['edjefa']]
    tdf['edjefaval']=np.select(conditions, choices) 
    household_variables.append('edjefaval')
    
    tdf['region'] = tdf['region'] = tdf[locations].idxmax(axis = 1)
    
    #credit for thse variables = https://www.kaggle.com/gaxxxx/exploratory-data-analysis-lightgbm
    tdf['r4h1permale'] = tdf['r4h1'] / tdf['r4h3']
    tdf.loc[tdf['r4h1permale'].isna(),'r4h1permale'] =0
    household_variables.append('r4h1permale')
    tdf['r4m1perfemale'] = tdf['r4m1'] / tdf['r4m3']
    tdf.loc[tdf['r4m1perfemale'].isna(),'r4m1perfemale'] =0
    household_variables.append('r4m1perfemale')
    tdf['r4h1pertotal'] = tdf['r4h1'] / tdf['hhsize']
    household_variables.append('r4h1pertotal')
    tdf['r4m1pertotal'] = tdf['r4m1'] / tdf['hhsize']
    household_variables.append('r4m1pertotal')
    tdf['r4t1pertotal'] = tdf['r4t1'] / tdf['hhsize']
    household_variables.append('r4t1pertotal')

    
    
    #Assumed access to clean water
    tdf['cleanwater'] = 0
    tdf.loc[(tdf['abastaguafuera']==1)|(tdf['abastaguano']==1),'cleanwater'] = 1 
    household_variables.append('cleanwater')
    
    #Assumed access to improved sanitation
    tdf['improvedsan'] = 0
    tdf.loc[(tdf['sanitario2']==1)|(tdf['sanitario3']==1)|(tdf['sanitario5']==1),'improvedsan'] = 1 
    household_variables.append('improvedsan')
    
    #Assumed access to consistent electricity
    tdf['electricity'] = 0
    tdf.loc[(tdf['public']==1)|(tdf['planpri']==1)|(tdf['coopele']==1),'electricity'] = 1 
    household_variables.append('electricity')
    
    #Combined flooring in household
    tdf['flooring'] = 0
    tdf.loc[(tdf['pisoother']==1)|(tdf['pisonatur']==1)|(tdf['pisonotiene']==1)|(tdf['pisomadera']==1),'flooring'] = 1 
    household_variables.append('flooring')
    
    #Combined wall in household
    tdf['wall'] = 0
    tdf.loc[(tdf['paredzocalo']==1)|(tdf['pareddes']==1)|(tdf['paredfibras']==1)|(tdf['paredother']==1)|(tdf['paredmad']==1),'wall'] = 1 
    household_variables.append('wall')
    
    #Combined roof in household
    tdf['roof'] = 0
    tdf.loc[(tdf['techocane']==1)|(tdf['techootro']==1)|(tdf['cielorazo']==0),'roof'] = 1 
    household_variables.append('roof')
    
    #Cooking material combined
    tdf['cooking'] = 0
    tdf.loc[(tdf['energcocinar4']==1)|(tdf['energcocinar1']==1),'cooking'] = 1 
    household_variables.append('cooking')
    
    tdf['rubbish'] = 0
    tdf.loc[(tdf['elimbasu3']==1)|(tdf['elimbasu4']==1)|(tdf['elimbasu5']==1)|(tdf['elimbasu6']==1),'rubbish'] = 1 
    household_variables.append('rubbish')
    
    tdf['condition'] = 0
    tdf.loc[(tdf['epared1']==1)&(tdf['etecho1']==1)&(tdf['eviv1']==1),'condition'] = 1 
    household_variables.append('condition')
    
    tdf['condition2'] = 0
    tdf.loc[(tdf['epared2']==1)&(tdf['etecho2']==1)&(tdf['eviv2']==1),'condition2'] = 1 
    household_variables.append('condition2')
    
    tdf['condition3'] = 0
    tdf.loc[(tdf['epared3']==1)&(tdf['etecho3']==1)&(tdf['eviv3']==1),'condition3'] = 1 
    household_variables.append('condition3')
    
household_variables = list(set(household_variables)) #clear out duplicates
#df[['behindSchool','perhhlt14','perschoolage','perbedrooms','assets3','assets4','assets3b','phoneperhhsize','phoneperadult']]


# Feature engineering for regional level variables. Test and Train datasets were combined to calculate these.

# In[ ]:


#Create regional variables from all the data
df_regional = df.copy()
df_test_regional = df_test.copy()
df_test_regional["Target"]=0
df_test_regional["Target0"]=0
df_all = pd.concat([df_regional,df_test_regional],sort=False)
#reindex to avoid problems
df_all.index = range(0,len(df_all['idhogar']))


regional_variables = []

df_all['regionalMeanLogRent'] = 0.0

df_all['regionalMeanLogRent'] = df_all[df_all['lv2a1']>0][['lv2a1','idhogar','region']].groupby(['idhogar','region']).transform('mean').groupby(df_all['region']).transform('mean')
df_all.loc[df_all['regionalMeanLogRent'].isna(),'regionalMeanLogRent'] = 0
regional_variables.append('regionalMeanLogRent')

#Median Age for the region
df_all['rmedianage'] = df_all[['age','idhogar','region']].groupby(['region']).transform('median')
regional_variables.append('rmedianage')

#Mean household education index for the region
df_all['reduindex'] = df_all[['eduindex','idhogar','region']].groupby(['idhogar','region']).transform('mean').groupby(df_all['region']).transform('mean')
regional_variables.append('reduindex')

#print(pd.pivot_table(df_all,values=['reduindex'],index=['region']))
df_all['rnhhlt14'] = df_all[['nhhlt14','idhogar','region']].groupby(['idhogar','region']).transform('mean').groupby(df_all['region']).transform('mean')
regional_variables.append('rnhhlt14')

#mean household size for the region
#print(pd.pivot_table(df_all,values=['rhogar_adul'],index=['region']))
df_all['rhogar_total'] = df_all[['hogar_total','idhogar','region']].groupby(['idhogar','region']).transform('mean').groupby(df_all['region']).transform('mean')
regional_variables.append('rhogar_total')

#mean number of bedrooms for the region
#print(pd.pivot_table(df_all,values=['rhogar_total'],index=['region']))
df_all['rbedrooms'] = df_all[['bedrooms','idhogar','region']].groupby(['idhogar','region']).transform('mean').groupby(df_all['region']).transform('mean')
regional_variables.append('rbedrooms')
#print(pd.pivot_table(df_all,values=['rbedrooms'],index=['region']))

#percent of urban households for the region
df_all['urbanhouseholds']=df_all[['area1','idhogar','region']].groupby(['idhogar','region']).transform('mean').groupby(df_all['region']).transform('mean')
regional_variables.append('urbanhouseholds')

for fld in regional_variables:
    tempPiv = pd.pivot_table(df_all, values=fld, index=['region'], aggfunc=np.mean)
    tempPiv.columns = [fld]
    df = df.merge(tempPiv,on='region',how='left')
    del tempPiv
for fld in regional_variables:
    tempPiv = pd.pivot_table(df_all, values=fld, index=['region'], aggfunc=np.mean)
    tempPiv.columns = [fld]
    df_test = df_test.merge(tempPiv,on='region',how='left')
    del tempPiv


# Check for any final missing or NAN values.

# In[ ]:


for tdf in dataframes:
    print (tdf.columns[tdf.isna().any()].tolist())
    for f in tdf.columns[tdf.isna().any()].tolist():
        if f in regional_variables:
            print(f)
    print (tdf.columns[tdf.isnull().any()].tolist())
    for f in tdf.columns[tdf.isnull().any()].tolist():
        if f in regional_variables:
            print(f)


# Develop the model using a gradient boosted classifier.

# In[ ]:


from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold
from  sklearn.model_selection import KFold
from sklearn.base import TransformerMixin, BaseEstimator
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
#from mlxtend.classifier import EnsembleVoteClassifier


#column ranges
#column ranges
combo = individual_variables+household_variables+regional_variables
combo = list(set(combo))
index_individual_variables = list(range(0,len(individual_variables)))
index_household_variables = list(range(len(individual_variables),len(individual_variables)+len(household_variables)))
index_indiv_and_hh_variables = index_individual_variables+index_household_variables


# This iterates through each target to create a separate model. The weights are derived from these models. The sampling uses GroupShuffleSplit to maintain houshold groups.

# In[ ]:


keeplst = []
indvmodels = {}
for t in [1,2,3,4]:
    params = {'n_estimators': 500,
                  'max_depth':None,
                  'min_samples_split':2,
                  'min_samples_leaf':90,
                  'max_features':'auto',
                  'loss':'deviance',
                 "learning_rate":.01,
                 "subsample":.5}

    currentTarget = "Target_"+str(t)
    print(currentTarget)
    currentTarget_v = t
    df_sub = df[df['parentesco1']==1][combo+[currentTarget,"idhogarnum",'targetweight']].copy()
    X = df_sub[combo].values
    y = df_sub[currentTarget].values.ravel()
    weights = df_sub['targetweight'].values.ravel()
    Y_strat = df_sub['idhogarnum'].values
    gss = GroupShuffleSplit(test_size=.1, n_splits=1,random_state=314)
    for train_index, test_index in gss.split(X, y, groups=Y_strat):
        break
    #print(train_index)
    #print(test_index)
    X_train, X_test = X[train_index],X[test_index]
    y_train, y_test = y[train_index],y[test_index]
    #w_train = weights[train_index]
    w_train = weights[train_index]
    strat_train = Y_strat[train_index]
    print(X_train.shape)
    print(X_test.shape)
    reg = GradientBoostingClassifier(**params)
    reg.fit(X_train,y_train,sample_weight=w_train)
    indvmodels[t] = reg
    y_pred = reg.predict(X_test)
    print(f1_score(y_test, y_pred, average='macro'))
    print(classification_report(y_test, y_pred))
    fi = list(zip(range(0,len(reg.feature_importances_)),reg.feature_importances_))
    fi.sort(key=lambda x: x[1])
    vals = [x[1] for x in fi]
    for i,important in fi:
        if important >=np.percentile(reg.feature_importances_, 75):
            keeplst.append(combo[i])
            print("%s: %s"%(combo[i],important))
        #if important==0:
            #removelst.append(combo[i])


    probabilities = pd.DataFrame(reg.predict_proba(X),columns=["P0","P1"])
    df[currentTarget+"_p"]=0.1
    
    df.loc[df['parentesco1']==1,currentTarget+"_p"]= probabilities["P1"].values
print(list(set(keeplst)))


# Calculates the weight for each training sample. I also divide the weights by the sum of the weights, but this doesn't seem to make much of a difference.

# In[ ]:


df['pscore'] = 0.0
#wghts={1:0.9210003139060374,2:0.8328973527257507,3:0.8734958669038402,4:0.3726064664643717}
for t in [1,2,3,4]:
    df.loc[df['Target']==t,'pscore'] = df.loc[df['Target']==t]["Target_"+str(t)+'_p']
print(df[df['Target']==1].head())
df['indsampweight']= 100.0*df['pscore']
df['targetpscore'] = df['targetweight']*df['pscore']
df['targetpscore'] = df['targetpscore'] / df['targetpscore'].sum()

print(df[df['Target']==1].head())
from sklearn.cluster import KMeans
y_pred = KMeans(n_clusters=4, random_state=314).fit_predict(df['pscore'].values.reshape(-1, 1))
df['strata'] = y_pred
df[['Target','pscore','strata']]
df[df['strata']==1][['Target','pscore','strata']]
pd.pivot_table(df,values='pscore',index='strata',aggfunc="mean")


# Final model

# In[ ]:


keeplst = list(set(keeplst))


df_sub = df[combo+["Target0","idhogarnum",'strata','indsampweight','targetweight','targetpscore']].copy()
X = df_sub[combo].values
y = df_sub["Target0"].values.ravel()
Y_strat = df_sub["idhogarnum"]
weights_target = df_sub['targetweight'].values.ravel()
weights_all = df_sub['indsampweight'].values.ravel()
weights_tp = df_sub['targetpscore'].values.ravel()
gss = GroupShuffleSplit(test_size=.1, n_splits=1,random_state=314)
gss = ShuffleSplit(n_splits=1, test_size=.1, random_state=314)

#for train_index, test_index in gss.split(X, y,Y_strat):
    #break
for train_index, test_index in gss.split(X, y):
    break


X_train, X_test = X[train_index],X[test_index]
y_train, y_test = y[train_index],y[test_index]



wtar_train = weights_target[train_index]
wall_train = weights_all[train_index]
wtp_train = weights_tp[train_index]
strat_train = Y_strat[train_index]
print(X_train.shape)
print(X_test.shape)
params = {'n_estimators': 500,
                  'max_depth': None,
                  'min_samples_split':10,
                  'min_samples_leaf':90,
                  'max_features':'auto',
                  'loss':'deviance',
                 "learning_rate":.01,
                 "subsample":.5}

reg = GradientBoostingClassifier(**params)
reg.fit(X_train,y_train,sample_weight=wtp_train)
y_pred = reg.predict(X_test)
print(f1_score(y_test, y_pred, average='macro'))
print(classification_report(y_test, y_pred))


# Output.

# In[ ]:


X_fin = df_test[combo].values
df_test['TargetP'] = reg.predict(X_fin)+1
print(df_test.TargetP)
df_output = pd.DataFrame({"Id":df_test['Id'],"Target":df_test['TargetP']})
df_output.to_csv("outputTargetPropensity.csv",index=False)


# In[ ]:


#y_indiv = np.zeros([X_test.shape[0], 4], dtype=float)
#for t in [1,2,3,4]:
#    reg = indvmodels[t]
#    df_prob = pd.DataFrame(reg.predict_proba(X_test),columns=['B0','B1'])
#    #print(df_prob['B1'].values)
#    y_indiv[:, t-1] = df_prob['B1'].values
#    del df_prob
#y_pred= np.zeros(X_test.shape[0])
#for i in range(X_test.shape[0]):
#    y_pred[i]=np.argmax(y_indiv[i])

#print(f1_score(y_test, y_pred, average='macro'))
#print(classification_report(y_test, y_pred))


# ## Prediction using separate Binary models
# 
# You can also use the individual models to predict the class. Which ever class has the highest prediction probability is assigned to that sample. This doesn't perform as well, but I few tries put it at about .420.

# In[ ]:


X_fin = df_test[combo].values
y_indiv = np.zeros([X_fin.shape[0], 4], dtype=float)
for t in [1,2,3,4]:
    reg = indvmodels[t]
    df_prob = pd.DataFrame(reg.predict_proba(X_fin),columns=['B0','B1'])
    #print(df_prob['B1'].values)
    y_indiv[:, t-1] = df_prob['B1'].values
    del df_prob
y_pred= np.zeros(X_fin.shape[0])
for i in range(X_fin.shape[0]):
    y_pred[i]=np.argmax(y_indiv[i])
y_pred = np.array(y_pred,dtype=int)
#print(f1_score(y_test, y_pred, average='macro'))
#print(classification_report(y_test, y_pred))
df_test['TargetI'] = y_pred+1
print(df_test.TargetI)
df_output = pd.DataFrame({"Id":df_test['Id'],"Target":df_test['TargetI']})
df_output.to_csv("outputTargetIndiv.csv",index=False)


# In[ ]:




