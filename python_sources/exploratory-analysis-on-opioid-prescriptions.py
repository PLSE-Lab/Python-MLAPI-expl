#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import re

from bokeh.io import output_notebook
from bokeh.sampledata import us_states
from bokeh.plotting import figure, show, output_file, ColumnDataSource
from bokeh.models import HoverTool, Range1d

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


# # 1. Introduction
# 
# Natural derivatives of Opium like heroin are called Opiates which are illegal. Similar synthetically synthesized drugs have been put under the class of Opioids which are legally available. Opioids are prescribed primarily as pain relievers despite a high risk of addiction and overdose. The increase in deaths caused by the risks involved with the consumption of opioids was alarming and declared an epidemic.
# 
# Current status of the opioid epidemic is that it is still a crisis (31st January 2018) and tweets have been pouring in [(recent tweets related to Opioid Crisis)](https://twitter.com/search?q=opioid+crisis&ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Esearch), talking about the issue.
# 
# An artcle explaining the crisis: [The opioid epidemic may be even deadlier than we think (Vox)](https://www.vox.com/science-and-health/2017/4/26/15425972/opioid-epidemic-overdose-deadlier-study)
# 
# Recent news on the epidemic: [Fast facts on Opioid Crisis](https://edition.cnn.com/2017/09/18/health/opioid-crisis-fast-facts/index.html)
# 
# # 2. Objective
# The objective of this notebook is to perform exploratory data analysis on opioid crisis and gather insights
# 
# # 3. Data 
# 
# There are 3 datasets available
# 1. opioids - Contains a list of drug names and its generic name
# 2. prescribers - Data on prescribers having the count of each drug which they prescribed
# 3. overdoses - Contains state population and deaths due to opioids

# In[ ]:


opioids = pd.read_csv('../input/opioids.csv')
overdoses = pd.read_csv('../input/overdoses.csv')
prescribers = pd.read_csv('../input/prescriber-info.csv')
overdoses['Deaths'] = overdoses['Deaths'].apply(lambda x: float(re.sub(',', '', x)))
overdoses['Population'] = overdoses['Population'].apply(lambda x: float(re.sub(',', '', x)))


# # 4. Exploratory Analysis
# 
# ## 4.1 Prescribers data description
# 
# **Sample:**

# In[ ]:


prescribers.head()


# In[ ]:


prescribers.describe()


# **Prescribed Opioids in prescriber-info data:**

# In[ ]:


ops = list(re.sub(r'[-\s]','.',x) for x in opioids.values[:,0])
prescribed_ops = list(set(ops) & set(prescribers.columns))

for i,drug in enumerate(prescribed_ops):
    print (i+1,drug)


# **Insights:**
# 
#     1. There are 11 opioid drugs out of the 250 drugs mentioned in the prescribers data
#     2. 60% of the prescribers on this list are opioid prescribers 

# ## 4.2 Prescribed Opioids vs Total Prescriptions 
# 
# **No. of opioid prescribers out of the total prescribers:**

# In[ ]:


# % of Opiod Prescribers
print (float(prescribers['Opioid.Prescriber'].sum())*100/prescribers.shape[0],"%")


# In[ ]:


prescribers['NumOpioids'] = prescribers.apply(lambda x: sum(x[prescribed_ops]),axis=1)
prescribers['NumPrescriptions'] = prescribers.apply(lambda x: sum(x.iloc[5:255]),axis=1)
prescribers['OpiodPrescribedVsPrescriptions'] = prescribers.apply(lambda x: float(x['NumOpioids'])/x['NumPrescriptions'],axis=1)


# In[ ]:


N = prescribers['NumOpioids'].shape[0]
x = prescribers['NumPrescriptions']
y = prescribers['NumOpioids']
colors = (192/255,192/255,192/255)
# area = np.pi*3
 
# Plot
# plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.scatter(x, y, c=colors, alpha=0.5)

m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b, '-')

plt.title('Opiods Prescribed vs Number of Prescriptions')
plt.xlabel('Number of Opiods Prescribed')
plt.ylabel('Number of Prescriptions')
plt.show()


# **Opiods Prescribed / Number of Preciptions:**

# In[ ]:


mu, sigma = np.mean(prescribers['OpiodPrescribedVsPrescriptions']), np.std(prescribers['OpiodPrescribedVsPrescriptions'])


# In[ ]:


n, bins, patches = plt.hist( prescribers['OpiodPrescribedVsPrescriptions'], 20, facecolor='grey', alpha=0.75)

plt.xlabel('Fraction of Opiods out of total drugs prescribed')
plt.ylabel('Number of Prescribers')
plt.title(r'$\mathrm{(Opiods Prescribed / Number of Preciptions)}\ \mu='+str(round(mu,2))+',\ \sigma='+str(round(sigma,2))+'$')
plt.grid(True)

plt.show()


# **(Opiods Prescribed / Number of Preciptions) for precribers who prescribed opioids:**

# In[ ]:


OpioidPrescriber_OpioidFrac = prescribers.loc[prescribers['Opioid.Prescriber']>0,'OpiodPrescribedVsPrescriptions']

mu, sigma = np.mean(OpioidPrescriber_OpioidFrac), np.std(OpioidPrescriber_OpioidFrac)
n, bins, patches = plt.hist( OpioidPrescriber_OpioidFrac, 20, facecolor='grey', alpha=0.75)

plt.xlabel('Fraction of Opiods out of total drugs prescribed')
plt.ylabel('Number of Prescribers')
plt.title(r'$\mathrm{(Opiods Prescribed / Number of Preciptions)}\ \mu='+str(round(mu,2))+',\ \sigma='+str(round(sigma,2))+'$')
plt.grid(True)

plt.show()


# **Fraction of Opiods Prescribed by Opioid Prescribers:**

# In[ ]:


fig = plt.figure(figsize=(7, 5))
axes = fig.add_subplot(1,1,1)
axes.boxplot( OpioidPrescriber_OpioidFrac, 0, 'rs', 0, 0.75, widths=[0.75])
plt.subplots_adjust(left=0.1, right=0.9, top=0.6, bottom=0.4)

plt.yticks([1],['Fraction of Opiods Prescribed by Opioid Prescribers'])
plt.show()


# **Insights:**
#     
#     An increment in the number of prescriptions by a Prescriber is likely to increase the chances of Opioid prescription by an average of 10%

# ## 4.3 Opioid Prescribers by Gender

# In[ ]:


genderCount = np.array(list(prescribers[['Gender','NPI']].groupby('Gender').count()['NPI']))

genderCount = np.append([genderCount],[list(prescribers.loc[prescribers['Opioid.Prescriber']>0,['Gender','NPI']].groupby('Gender').count()['NPI'])], axis=0)

genderCount[0] = genderCount[0]-genderCount[1]

fig = plt.gcf()
fig.set_size_inches( 7, 5)

configs = genderCount[0]
N = configs.shape[0]
ind = np.arange(N)
width = 0.4

p1 = plt.bar(ind, genderCount[0], width, color='b')
p2 = plt.bar(ind, genderCount[1], width, bottom=genderCount[0], color='r')

# plt.ylim([0,120])
plt.yticks(fontsize=12)
plt.ylabel("Number of Prescribers", fontsize=12)
plt.xticks(ind,["Female","Male"])
plt.xlabel('Gender', fontsize=12)
plt.title("Opioid Prescribers by Gender")
plt.legend([p1[0], p2[0]], ["Did not prescribe opioids","Prescribed opioids"], fontsize=12, fancybox=True)
plt.show()


# **Insights:**
#     
#     The number of non opioid prescribers is similar in the case of Male and Female, though Prescribed Opioids is higher in the case of Male. This can be subject to the kind of Specialties Male and Female Prescribers prefer.

# ## 4.4 Opioid Prescribers by State

# In[ ]:


stateCount = pd.DataFrame(prescribers[['State','NPI']].groupby('State').count())

stateCount.reset_index(level=0, inplace=True)

stateCount.columns = ['State', 'Total_Prescribers']

stateCount_PrescribedOpiods = pd.DataFrame(prescribers.loc[prescribers['Opioid.Prescriber']>0,['State','NPI']].groupby('State').count())
stateCount_PrescribedOpiods.reset_index(level=0, inplace=True)
stateCount_PrescribedOpiods.columns = ['State', 'Opiod_Prescribers']
stateCount = pd.merge(stateCount, stateCount_PrescribedOpiods,  how='left', on="State")

stateCount = stateCount.fillna(0)

stateCount = stateCount.sort_values('Total_Prescribers')

fig = plt.gcf()
fig.set_size_inches( 20, 15)

N = stateCount.shape[0]
ind = np.arange(N)
width = 0.6

p1 = plt.bar(ind, stateCount['Total_Prescribers']-stateCount['Opiod_Prescribers'], width, color='b')
p2 = plt.bar(ind, stateCount['Opiod_Prescribers'], width, bottom=stateCount['Total_Prescribers']-stateCount['Opiod_Prescribers'], color='r')

# plt.ylim([0,120])
plt.yticks(fontsize=12)
plt.ylabel("Number of Prescribers", fontsize=15)
plt.xticks(ind,stateCount['State'], fontsize=15, rotation=70)
plt.xlabel('States', fontsize=15)
plt.title("Opioid Prescribers by State", fontsize=15)
plt.legend([p1[0], p2[0]], ["Did not prescribe opioids","Prescribed opioids"], fontsize=12, fancybox=True)
plt.show()


# **Insights:**
#     
#     States like CA, NY, FL, TX have higher opioid prescribers which corresponds with high death rates reported due to opioid overdose from analysis in section 4.1 

# ## 4.5 Opioid Prescribers by Specialty

# In[ ]:


SpecialtyCount = pd.DataFrame(prescribers[['Specialty','NPI']].groupby('Specialty').count())

SpecialtyCount.reset_index(level=0, inplace=True)

SpecialtyCount.columns = ['Specialty', 'Total_Prescribers']

SpecialtyCount_PrescribedOpiods = pd.DataFrame(prescribers.loc[prescribers['Opioid.Prescriber']>0,['Specialty','NPI']].groupby('Specialty').count())
SpecialtyCount_PrescribedOpiods.reset_index(level=0, inplace=True)

SpecialtyCount_PrescribedOpiods.columns = ['Specialty', 'Opiod_Prescribers']
SpecialtyCount = pd.merge(SpecialtyCount, SpecialtyCount_PrescribedOpiods,  how='left', on="Specialty")

SpecialtyCount = SpecialtyCount.fillna(0)

SpecialtyCount = SpecialtyCount.sort_values('Total_Prescribers')

SpecialtyCount = SpecialtyCount[-30::]

fig = plt.gcf()
fig.set_size_inches( 20, 10)

N = SpecialtyCount.shape[0]
ind = np.arange(N)
width = 0.6

p1 = plt.bar(ind, SpecialtyCount['Total_Prescribers']-SpecialtyCount['Opiod_Prescribers'], width, color='b')
p2 = plt.bar(ind, SpecialtyCount['Opiod_Prescribers'], width, bottom=SpecialtyCount['Total_Prescribers']-SpecialtyCount['Opiod_Prescribers'], color='r')

# plt.ylim([0,120])
plt.yticks(fontsize=12)
plt.ylabel("Number of Prescribers", fontsize=15)
plt.xticks(ind,SpecialtyCount['Specialty'], fontsize=15, rotation=90)
plt.xlabel('Specialty', fontsize=15)
plt.title("Opioid Prescribers by Specialty (Top 30)", fontsize=15)
plt.legend([p1[0], p2[0]], ["Did not prescribe opioids","Prescribed opioids"], fontsize=12, fancybox=True)
plt.show()


# **Sorted by num of Opioids prescribed:**

# In[ ]:


SpecialtyCount = pd.DataFrame(prescribers[['Specialty','NPI']].groupby('Specialty').count())

SpecialtyCount.reset_index(level=0, inplace=True)

SpecialtyCount.columns = ['Specialty', 'Total_Prescribers']

SpecialtyCount_PrescribedOpiods = pd.DataFrame(prescribers.loc[prescribers['Opioid.Prescriber']>0,['Specialty','NPI']].groupby('Specialty').count())
SpecialtyCount_PrescribedOpiods.reset_index(level=0, inplace=True)

SpecialtyCount_PrescribedOpiods.columns = ['Specialty', 'Opiod_Prescribers']
SpecialtyCount = pd.merge(SpecialtyCount, SpecialtyCount_PrescribedOpiods,  how='left', on="Specialty")

SpecialtyCount = SpecialtyCount.fillna(0)

SpecialtyCount = SpecialtyCount.sort_values('Opiod_Prescribers')

SpecialtyCount = SpecialtyCount[-30::]

fig = plt.gcf()
fig.set_size_inches( 20, 10)

N = SpecialtyCount.shape[0]
ind = np.arange(N)
width = 0.6

p1 = plt.bar(ind, SpecialtyCount['Total_Prescribers']-SpecialtyCount['Opiod_Prescribers'], width, color='b')
p2 = plt.bar(ind, SpecialtyCount['Opiod_Prescribers'], width, bottom=SpecialtyCount['Total_Prescribers']-SpecialtyCount['Opiod_Prescribers'], color='r')

# plt.ylim([0,120])
plt.yticks(fontsize=12)
plt.ylabel("Number of Prescribers", fontsize=15)
plt.xticks(ind,SpecialtyCount['Specialty'], fontsize=15, rotation=90)
plt.xlabel('Specialty', fontsize=15)
plt.title("Opioid Prescribers by Specialty (Top 30)", fontsize=15)
plt.legend([p1[0], p2[0]], ["Did not prescribe opioids","Prescribed opioids"], fontsize=12, fancybox=True)
plt.show()


# **Insights:**
# 
#     Use of opioids is higher in specialties which involve the use of Pain Killers/Inhibitors

# # 5. Principal Component Analysis
# 
# ### Objective:
# This analysis aims at dimensionality reduction for bringing down 358 columns to a few factors (or columns a.k.a variables) to describe the data. From these new factors only those will be selected which explain 80% of the total variability in data to classify opioid Prescribers. Lastly the weightage of original factors in the newly formed factors will be observed to gather insights for a driver analysis.

# In[ ]:


specialty = pd.DataFrame(prescribers.groupby(['Specialty']).count()['NPI']).sort_values('NPI')

specialty.loc[specialty['NPI']<40].shape


rareSpecialty = list(specialty.loc[specialty['NPI']<40].index)


prescribers.loc[prescribers['Specialty'].isin(rareSpecialty),'Specialty'] = prescribers.loc[prescribers['Specialty'].isin(rareSpecialty),'Specialty'].apply(lambda x: 'Surgery' if 'Surgery' in list(x.split( )) else 'Other')

prescribersData = prescribers.drop( ['NPI','Credentials'], axis=1)

prescribersData = pd.get_dummies(prescribersData, columns=['Gender','Specialty','State'], drop_first=True)


# In[ ]:


#convert it to numpy arrays
X= prescribersData.values

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

pca = PCA(n_components=300)

pca.fit(X_scaled)

#The amount of variance that each PC explains
var= pca.explained_variance_ratio_

#Cumulative Variance explains
cum_var = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)


# In[ ]:


plt.plot(var, color='y')
# plt.plot(cum_var, color='r')
# plt.xticks()
plt.ylabel('Principal Components')
plt.ylabel('Proportion of Variance Explained')
plt.title('Scree Plot')


# In[ ]:


#Looking at above plot I'm taking 20 variables
pca = PCA(n_components=20)

# pca.fit(X_scaled)
X1=pca.fit_transform(X_scaled)


# **Variance Explained by each Factor:**

# In[ ]:


print ("Explained variance by component: %s" % pca.explained_variance_ratio_)


# In[ ]:


print ("Variance explained by first 10 factors: %s" % (pca.explained_variance_ratio_[0:9].sum()/pca.explained_variance_ratio_.sum()))
print ("Since these explain ~80% of the variance they are selected for further analysis")


# In[ ]:


newFactors = pd.DataFrame(pca.components_,columns=prescribersData.columns)


# **Factor Loadings**
# 
# Selecting the top 10 factors in terms of explained variance

# In[ ]:


newFactors = newFactors.loc[0:9]


# In[ ]:


newFactors


# **Important factors which constitute higher weightage in the newly discovered factors/components:**

# In[ ]:


impFactors = list(set(pd.DataFrame(newFactors.max())[pd.DataFrame(newFactors.max()> 0.2)[0]].index).union(set(pd.DataFrame(newFactors.min())[pd.DataFrame(newFactors.min()< -0.2)[0]].index)))


# In[ ]:


pd.DataFrame(impFactors)


# In[ ]:


newFactors_ = newFactors[impFactors]


# **Factor Loadings Heatmap**

# In[ ]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(rc={'figure.figsize':(12,25)})
# sns.heatmap(newFactors.T, cmap='RdYlGn', linewidths=0.5, annot=True)
sns.heatmap(newFactors_.T, cmap='RdYlGn', linewidths=0.5)
plt.plot()


# ## 5.1 Insights from PCA
# 
# ### Factor 5 (Glaucoma Treatment)
# 
# 1. Latanoprost - Latanoprost, sold under the brand name Xalatan among others, is a medication used to treat increased pressure inside the eye.
# 
# 2. Travatan.Z - Once-a-day eye drop you take in the evening to help reduce the elevated pressure inside your eye. It belongs to a class of drugs called prostaglandin analogs. These drugs work by increasing the drainage of fluid inside the eye.
# 
# 3. Dorzolamide.Timolol - Dorzolamide/timolol eye drops are used to lower intraocular pressure in the eye to normal pressure and as a treatment for glaucoma.
# 
# 4. Timolol.Maleate - Timolol is a medication used either by mouth or as eye drops. As eye drops it is used to treat increased pressure inside the eye such as in ocular hypertension and glaucoma.
# 
# 5. Brimonidine.Tartarate - This medication is used to treat open-angle glaucoma or high fluid pressure in the eye. Lowering high fluid pressure in the eye reduces the risk of vision loss, nerve damage, or blindness. 
# 
# 6. Lumigan - Bimatoprost is a prostaglandin analog used topically to control the progression of glaucoma and in the management of ocular hypertension. It reduces intraocular pressure by increasing the outflow of aqueous fluid from the eyes.
# 
# Note: Definitions directly quoted from online references
# 
# All the above drugs have a positive contribution to Factor 5 and also are vital to the describing prescription of Opioids. All the drugs mentioned above are related to treatment of Glaucoma and somewhat related to the problem of increased pressure in the eye. On digging deeper into the matter it was found that Opoids are indeed being used in the treatment of Glaucoma and the use of Opioids is considered as an advancement in such cases.
# 
# ["Patients with chronic open angle glaucoma showed a significant decrease in intraocular pressure after conjunctival instillation of morphine solution. It is concluded that intraocular opiate receptors are involved in the regulation of intraocular pressure in animals and humans." - Effects of opiates and opioids on intraocular pressure of rabbits and humans.](https://www.ncbi.nlm.nih.gov/pubmed/4006315).
# 
# ["Despite their drawbacks, opioids represent a largely untapped resource in ophthalmology(*). In addition to established evidence of effective ocular anesthesia, preliminary studies suggest that opioids may be viable intraocular pressure-lowering agents and offer a useful alternative to current topical glaucoma therapies."-Finding New Uses For Ancient Drugs](https://www.reviewofophthalmology.com/article/finding-new-uses-for-ancient-drugs)
# 
# (*An ophthalmologist is a specialist in medical and surgical eye disease)
# 
# ["Kappa opioid receptors (KORs) are present on cell membranes in human nonpigmented ciliary epithelial and TM (HTM-3) cells. Activation of these KORs by the selective KOR agonist spiradoline resulted in increases in NO production in both cell types that were inhibited by KOR antagonists."-Advances in Glaucoma Treatment and Management: Outflow Drugs](http://iovs.arvojournals.org/article.aspx?articleid=2127003)
# 
# ### Factor 2 (Opioids)
# 
# 1. NumOpioids - Number of Opioids (obvious)
# 
# 2. Morphine.Sulphate - Morphine is a pain medication of the opiate variety which is found naturally in a number of plants and animals
# 
# 3. Oxycodone.Hcl - Oxycodone is a semisynthetic opioid synthesized from thebaine, an opioid alkaloid found in the Persian poppy. (Opioid) 
# 
# 4. Oxycodone.Acetaminophen - The combination oxycodone/paracetamol is a combined opioid/non-opioid pain reliever used to treat moderate to severe acute pain. 
# 
# Note: Definitions directly quoted from online references
# 
# All the above drugs are opioids and hence is a good indicator of the validity of the Analysis.
# 
# ### Factor 6 (Glucose and Not Endocrine Disorders)
# 
# 1. Specialty Endocrinology - Endocrinology is a branch of biology and medicine dealing with the endocrine system, its diseases, and its specific secretions known as hormones.
# 
# 2. Levemir.Flexpen - Insulin detemir, a long-acting human insulin analogue for maintaining the basal level of insulin.
# 
# 3. Lantus Solostar - Insulin glargine, marketed under the names Lantus, among others, is a long-acting basal insulin analogue, given once daily to help control the blood sugar level of those with diabetes.
# 
# 4. Novolog, Novolog.Flexpen & Humalog  - Humalog is the brand-name version of insulin lispro, and Novolog is the brand-name version of insulin aspart. These drugs both help control blood glucose (sugar) in people with type 1 and type 2 diabetes. Humalog and Novolog are both rapid acting.
# 
# 5. BD.Ultra.Fine pin needle - Short, thin needles
# 
# Drugs in 1-5 are related to treatment of Diabetes (i.e. controlling blood glucose (sugar) in people). 
# 
# ["Opioid use, acute and chronic, is also associated with weight gain, glycemic dysregulation, and dental pathology. The literature supporting the connection between opiate use and development of preference for sweet tastes is reviewed, and further association with dental pathology, weight gain, and loss of glycemic control are considered."-The relationship between opioid and sugar intake: Review of evidence and clinical applications](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3109725/)
# 
# ["In the basal state, among opioid-dependent individuals it has been observed that insulin responses to intravenous glucose was markedly reduced and they had low glucose disappearance rates when compared to the controls."-Opioid use and diabetes: An overview](http://www.joshd.net/article.asp?issn=2321-0656;year=2016;volume=4;issue=1;spage=6;epage=10;aulast=Sharma)
# 
# (Negative)
# 1. Finasteride - Finasteride, sold under the brand names Proscar and Propecia among others, is a medication used mainly to treat an enlarged prostate or scalp hair loss in men.
# 
# 2. Avodart - Dutasteride, sold under the brand name Avodart among others, is a medication used primarily to treat enlarged prostate in men.
# 
# The above two mentioned drugs are used for treatment of enlarged prostate which may worsen in cases where opioids are used. Hence, there is negative relation with prescribtion of opioids.
# 
# [Pain meds may worsen symptoms of enlarged prostate](https://www.reuters.com/article/us-meds-prostate/pain-meds-may-worsen-symptoms-of-enlarged-prostate-idUSKUA17347020070921)
# 
# ["Opioids have been used for medicinal and analgesic purposes for centuries. However, their negative effects on the endocrine system, which have been known for some times, are barely discussed in modern medicine."-The impact of opioids on the endocrine system](https://www.ncbi.nlm.nih.gov/pubmed/19333165)
# 
# ### Factor 8 & 9 (Epilepsy/Seizures)
# 
# 1. Specialty Neurology - Neurology deals with the diagnosis and treatment of all categories of conditions and disease involving the central and peripheral nervous system (and its subdivisions, the autonomic nervous system and the somatic nervous system); including their coverings, blood vessels, and all effector tissue, such as muscle.
# 
# 2. Primidone - Primidone belongs to a class of drugs known as barbiturate anticonvulsants. It works by controlling the abnormal electrical activity in the brain that occurs during a seizure.
# 
# ["There may be an increased risk of drowsiness if primidone is taken in combination with pain killers like opioids"-Primidone is a medicine used to treat epilepsy. It stabilises electrical activity in the brain.](http://www.netdoctor.co.uk/medicines/brain-and-nervous-system/a8155/primidone/)
# 
# 
# # Inference
# 
# In a nutshell people undergoing treatment for Glaucoma, Diabetes and related diseases have a high risk of being exposed to treatment using Opioids. Such cases are likely to increase the probability of Prescriber to prescribe Opioids. While in the case of epilepsy or seizures, and enlarged prostate treatment with other endocrine disorders, the Prescriber would avoid prescribing opioids. Thus, higher values for Factor 2, 5, 6 would mean higher probability of Opioid Prescribtion while higher values of Factors 8 & 9 would mean the opposite i.e. less likelihood of Opioid Prescribtion. 
